from flask import Flask, request, jsonify
from flask_cors import CORS
import os 
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import uuid
from datetime import date


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

    

@app.route("/submit_message" , methods = ["POST"])
def submit_message():
    """Handles the submission of a user's message, processes it with a language model, and stores the conversation in Pinecone.

    This function acts as an API endpoint. It expects a JSON payload containing a `user_id` and a `message`.
    It uses the `user_id` as the namespace for Pinecone and sends the `message` to a language model
    to generate a reply. Both the user's message and the bot's reply are then stored in Pinecone
    with associated metadata, including a unique `turn_id` to track the conversation flow.

    Args:
        None (uses Flask's `request` object to get data).

    Returns:
        A JSON response containing the generated reply from the language model.
    """

    base_prompt = """
    You are a kind, empathetic mental health companion. 
Always listen without judgment, validate feelings, 
and respond with warmth, compassion, and supportive guidance. 
Keep advice practical, gentle, and caring. User says: 
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

    # Upsert records into a namespace
    # `text` fields are converted to dense vectors

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
                "date":today,                   #everything after id and text is part of metadata and can be filtered later(have to confirm)
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

    return jsonify({"reply":reply})

# In-memory turn tracker per namespace
namespace_turn_ids = {}  # {namespace: current_turn_id}

def get_turn_id(namespace):
    """Retrieves and increments the turn ID for a given user conversation namespace.

    This function uses an in-memory dictionary to keep track of the current turn ID
    for each user's conversation. It first checks if a turn ID exists for the
    given namespace. If not, it initializes it by calling `initialize_turn_id`.
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
                "text": "a",
                "turn_id": 0,
                "role": "turn_tracker"
            }
        ]
    )
    return 0



if __name__ == "__main__":
    app.run(debug=True)
