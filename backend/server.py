from flask import Flask, request, jsonify
from flask_cors import CORS                 #when flask and react server run on different ports
import os 
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import uuid
from datetime import date
from collections import deque               
import numpy as np
from pprint import pprint                  #for pretty printing


# In-memory turn tracker per namespace
namespace_turn_ids = {}  # {namespace: current_turn_id}

def get_turn_id(namespace):
    """Retrieves and increments the turn ID for a given user conversation namespace.

    This function uses an in-memory dictionary to keep track of the current turn ID
    for each user's conversation. It first checks if a turn ID exists for the
    given namespace. If not, it initializes it by calling 'initialize_turn_id'.
    It then increments the ID and returns the new value.

    Args:
        namespace (str): The unique identifier for the user's conversation.

    Returns:
        int: The next turn ID for the conversation.
    """
    if namespace not in namespace_turn_ids:
        namespace_turn_ids[namespace] = initialize_turn_id(namespace)
    namespace_turn_ids[namespace] += 1
    return namespace_turn_ids[namespace]


def initialize_turn_id(namespace):
    """Initializes the turn ID tracker in Pinecone for a new namespace.

    This function queries the Pinecone index to check if a "turn_tracker"
    record already exists for the given namespace. If a tracker is found,
    it returns the last known turn ID. If no tracker exists, it creates one
    with a starting ID of 0 and then returns 0.

    Args:
        namespace (str): The unique identifier for the user's conversation.

    Returns:
        int: The initial turn ID (0) or the last known turn ID if a tracker exists.
    """
    result = index.query(
        namespace=namespace,
        vector=[0.0]*1024,  # dummy vector, just need metadata
        top_k=1,
        include_metadata=True,
        filter={"role": {"$eq": "turn_tracker"}}
    )

    if result.matches:
        return result.matches[0].metadata["turn_id"]

    # No tracker row exists → create one
    index.upsert_records(
        namespace,
        [
            {
                "id": "turn_tracker",
                "text": "x",                        #i have set text = x cuz text can't be empty
                "turn_id": 0,
                "role": "turn_tracker"
            }
        ]
    )
    return 0

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

def get_relevant_context(query_embedding, memory, top_k=3, threshold=0.7):
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
        return []

    scored = []
    for item in memory:
        sim = cosine_similarity(query_embedding, item["embedding"])
        if sim >= threshold:
            scored.append((sim, item))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [item for _, item in scored[:top_k]]


def build_prompt_with_context(user_query, query_embedding, STM, base_prompt, top_k=3, threshold=0.7):
    """
    Builds the final LLM prompt including relevant short-term memory.
    
    Args:
        user_query (str): The latest user message.
        query_embedding (np.ndarray): Embedding of the user query.
        STM (ShortTermMemory): Short-term memory instance.
        base_prompt (str): The base system/assistant prompt.
        top_k (int): Number of context turns to include.
        threshold (float): Minimum similarity threshold.
    
    Returns:
        str: Final prompt to send to the LLM.
    """
    relevant_items = get_relevant_context(query_embedding, STM.cache, top_k, threshold)

    context_str = "\n".join(
        [f"{item['role'].capitalize()}: {item['text']}" for item in relevant_items]
    )

    if context_str:
        context_str = f"Relevant past turns:\n{context_str}\n\n"

    final_prompt = f"""{base_prompt}
{context_str}
User: {user_query}
Assistant:"""

    return final_prompt

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

#Pinecone and index initialisation
pc = Pinecone(api_key = PINECONE_API_KEY)
index = pc.Index(host=INDEX_HOST)


#flask app initialisation
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
    You are a kind, empathetic mental health companion. 
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

    completion = client.chat.completions.create(
    extra_body={},
    model="moonshotai/kimi-k2:free",
    messages=[
        {
        "role": "assistant",
        "content": base_prompt + message
        }
    ]
    )
    reply = completion.choices[0].message.content

    user_convo_id = str(uuid.uuid4())              #to generate unique id for user row
    assistant_convo_id = str(uuid.uuid4())         #to generate unique id for assistant row
    today = str(date.today())
    turn_id = get_turn_id(user_id)              #2 rows will be inserted with same turn_id but different roles(assistant/user)
    
    index.upsert_records(
        user_id,
        [
            {
                "id":user_convo_id,
                "text":message,
                "date":today,                   #everything after id and text is part of metadata and can used to filter data later
                "turn_id":turn_id,  
                "role":"user"
            },
            {
                "id":assistant_convo_id,
                "text":reply,
                "date":today,
                "turn_id":turn_id,
                "role":"assistant"
            }
        ]
    ) 
    
    res = index.fetch(ids=[user_convo_id], namespace=user_id)       #returns FetchResponse object

    # Access vectors dict from FetchResponse
    vectors = res.vectors

    # Grab the vector embedding for the user’s message
    user_vector = np.array(vectors[user_convo_id].values)

    prompt = build_prompt_with_context(                             #builds prompt with relevant entries from short-term-memory
        message,
        user_vector,
        STM,
        base_prompt,
        top_k=3,
        threshold=0.7
    )

    print(STM.cache)
    completion = client.chat.completions.create(
        model="moonshotai/kimi-k2:free",
        messages=[{"role": "assistant", "content": prompt}]
    )

    return jsonify({"reply":reply})



if __name__ == "__main__":
    app.run(debug=True)
