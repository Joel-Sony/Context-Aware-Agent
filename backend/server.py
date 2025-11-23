from flask import Flask, request, jsonify
from flask_cors import CORS               
import os 
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import uuid
from datetime import date
from collections import deque               
import numpy as np
from pprint import pprint                 
import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

## Mood Predictor Model
model_path = "./mood_prediction_model" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

label_map = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "shame",
    7: "surprise"
}


def predict_mood(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    predicted_mood = label_map[predicted_class]
    return predicted_mood, confidence


def get_turn_id(cur_namespace):
    """Retrieves number of vectors in an index.
    
    Since we insert two rows (user + assistant) at every turn,
    (turn_id is half of total vectors + 1).

    Args:
        namespace (str): The unique identifier for the user's conversation.

    Returns:
        int: The next turn ID for the conversation.
    """

    for namespace in index.list_namespaces():
            cur_namespace = str(cur_namespace)
            if(namespace.name == cur_namespace):
                return (int(namespace.record_count)//2) + 1
    
    return 1


class ShortTermMemory:
    """Saves n number of recent user-assistant messages,
    -Will keep context-relevant recent user-assistant interactions in memory.
    -Rows are stored in the same way as they are stored in the db.
    -Will prevent querying db everytime user sends a message.
    -Uses a deque
    """
     
    def __init__(self, size=30):
        self.cache = deque(maxlen=size)

    def add(self, turn_id, role, text, embedding):
        self.cache.append({
            "turn_id": turn_id,
            "role": role,
            "text": text,
            "embedding": np.array(embedding)
        })

    def find_similar(self, embedding, threshold=0.8):
        if not self.cache:
            return None

        # cosine similarity
        sims = [
            (msg, np.dot(msg["embedding"], embedding) /
                  (np.linalg.norm(msg["embedding"]) * np.linalg.norm(embedding)))
            for msg in self.cache
        ]
        # best match
        best_msg, best_sim = max(sims, key=lambda x: x[1])
        return best_msg if best_sim >= threshold else None


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_relevant_context(query_embedding, memory, top_k=3, threshold=0.4):
    """
    Search short-term memory for the most relevant entries to the query.
    
    Args:
        query_embedding (np.ndarray): Embedding of the user's current query.
        memory (deque): ShortTermMemory.cache.
        top_k (int): Number of top results to return.
        threshold (float): Minimum cosine similarity to consider.
    
    Returns:
        list of dict: Relevant memory entries with high similarity.
    """
    if not memory:
        print("NO MEMORY")
        return []

    scored = []
    for item in memory:
        sim = cosine_similarity(query_embedding, item["embedding"])
        if sim >= threshold:
            scored.append((sim, item))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def build_prompt_with_context(user_query, query_embedding, STM, base_prompt, namespace, threshold=0.4):
    """
    Builds the final LLM prompt including relevant short-term memory.

    Case 1: If STM not full, include all STM entries.
    Case 2: If STM full, drop oldest N entries and replace with contextually
            similar ones from Pinecone.
    """

    context_items = []
    query_db = False

    query_norm = query_embedding / np.linalg.norm(query_embedding)
    trigger_norm = triggerEmbeddings / np.linalg.norm(triggerEmbeddings, axis=1, keepdims=True)
    sims = trigger_norm @ query_norm
    
    query_db = np.any(sims >= threshold)
            
    # Case 1: STM not full → use everything
    if len(STM.cache) < STM.cache.maxlen and query_db == False:
        context_items = list(STM.cache)

    else:
        # Case 2: STM is full → drop oldest N
        N = 10
        recent_items = list(STM.cache)[N:]  # keep most recent (maxlen - N)

        # Similarity search in Pinecone to refill N slots
        result = index.query(
            namespace=namespace,     
            vector=query_embedding.tolist(),
            top_k=N,
            include_metadata=True
        )

        retrieved_items = []
        for match in result.matches:
            if match.score >= threshold:
                retrieved_items.append({
                    "turn_id": match.metadata.get("turn_id"),
                    "role": match.metadata.get("role"),
                    "text": match.metadata.get("text"),
                    "embedding": None  # no need here, just for prompt
                })

        # Combine: recent STM (after dropping oldest) + retrieved from DB
        combined = recent_items + retrieved_items

        seen = set()                   #if fetched rows are same as the ones already present then its removed
        deduped = []
        for item in combined:
            key = (item.get("turn_id"), item.get("role"))
            if key not in seen:
                seen.add(key)
                deduped.append(item)

        context_items = deduped

    predicted_mood, confidence = predict_mood(user_query)

    # Build string for LLM
    context_str = "\n".join(
        [f"{item['role'].capitalize()}: {item['text']}" for item in context_items]
    )
    
    final_prompt = f"""{base_prompt}
Relevant conversation context:
{context_str}

Detected user emotion: {predicted_mood}
Emotion confidence: {confidence:.2f}

User: {user_query}"""
    
    print(final_prompt)
    return final_prompt

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

#Pinecone and index initialisation
pc = Pinecone(api_key = PINECONE_API_KEY)
index = pc.Index(host=INDEX_HOST)

#loading the triggerEmbeddings to check if db needs to be queried in advance
triggerEmbeddings = np.load("triggerEmbeddings.npy")

# index.delete(delete_all=True, namespace='123')    #to delete all rows

# #flask app initialisation
app = Flask(__name__)
CORS(app)  

#Short term memory cache
STM = ShortTermMemory(size=30)

@app.route("/submit_message" , methods = ["POST"])
def submit_message():
    """Handles the submission of a user's message, processes it with a language model, and stores the conversation in Pinecone.

    This function acts as an API endpoint. It expects a JSON payload containing a 'user_id' and a 'message'.
    It uses the 'user_id' as the namespace for Pinecone and sends the 'message' to a language model
    to generate a reply. Both the user's message and the bot's reply are then stored in Pinecone
    with associated metadata, including a unique 'turn_id' to track the conversation flow.

    Args:
        None (uses Flask's 'request' object to get data).

    Returns:
        A JSON response containing the generated reply from the language model.
    """

    base_prompt = """
    You are a Lenni: a kind, empathetic mental health companion. 
Always listen without judgment, validate feelings, 
and respond with warmth, compassion, and supportive guidance. 
Keep advice practical, gentle, and caring.
"""
    data = request.get_json()
    user_id = data.get("user_id")
    message = data.get("message") 


    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )

    user_convo_id = str(uuid.uuid4())              #to generate unique id for user row
    assistant_convo_id = str(uuid.uuid4())         #to generate unique id for assistant row
    today = str(date.today())
    turn_id = get_turn_id(user_id)              #2 rows will be inserted with same turn_id but different roles(assistant/user)

    start = time.time()
    index.upsert_records(
        user_id,
        [
            {
                "id":user_convo_id,
                "text":message,
                "date":today,                   #everything after id and text is part of metadata and can used to filter data later
                "turn_id":turn_id,  
                "role":"user"
            }
        ]
    ) 

    end = time.time()

    retries = 10                                                  
    delay = 0.1
    res = None

    for i in range(retries):                
        res = index.fetch(ids=[user_convo_id], namespace=user_id)
        if res.vectors:  # found it
            print('\n Fetch successfull!')
            break
        time.sleep(delay * (2 ** i))  # exponential backoff
    else:
        raise RuntimeError("Vector not available after retries")
          
    # Access vectors dict from FetchResponse
    vectors = res.vectors

    # Grab the vector embedding for the user’s message
    user_vector = np.array(vectors[user_convo_id].values)
    user_id = str(user_id)
    prompt = build_prompt_with_context(                             #builds prompt with relevant entries from short-term-memory
        message,
        user_vector,
        STM,
        base_prompt,
        threshold=0.7,
        namespace=user_id
    )
    

    STM.add(turn_id,"user",message,user_vector)

    completion = client.chat.completions.create(
        model="google/gemma-3n-e2b-it:free",
        messages=[{"role": "assistant", "content": prompt}]
    )

    reply = completion.choices[0].message.content

    index.upsert_records(
        user_id,
        [
            {
                "id":assistant_convo_id,
                "text":reply,
                "date":today,
                "turn_id":turn_id,
                "role":"assistant"
            }
        ]
    ) 
    
    return jsonify({"reply":reply})  
    

@app.route('/get_messages', methods = ['POST'])
def get_messages():
    data = request.get_json()
    user_id = data["user_id"]

    chunk_ids = index.list(namespace=user_id)
    record_ids = list(chunk_ids)

    all_messages = []

    if record_ids:
        for all_ids in record_ids:
            all_records = index.fetch(ids=all_ids, namespace=user_id)

            for id in all_records.vectors:
                vector = all_records.vectors.get(id) 
            
                if vector:
                    turn_id = vector["metadata"]["turn_id"]
                    role = vector["metadata"]["role"]
                    text = vector["metadata"]["text"]
                    all_messages.append({"sender": role, "text": text, "turn_id":turn_id})
    all_messages.sort(key = lambda m : (m["turn_id"], 0 if m["sender"] == "user" else 1))
    all_messages = [{"sender": m["sender"], "text": m["text"]} for m in all_messages]
    return jsonify({"messages":all_messages})
    

if __name__ == "__main__":
    app.run(debug=True)
