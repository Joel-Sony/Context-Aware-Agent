from pymongo import MongoClient
import cohere       #provides api for embedding
import os 
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone  #vector db 
from pinecone import ServerlessSpec


load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

# MONGO_URI = "mongodb+srv://joelsony:0tpFgnGzupIuI66L@cluster0.oaqctgy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# client = MongoClient(MONGO_URI)
# db = client["test"]
# collection = db["documents"]

#==============================
#=======Embedding part=========
#==============================

# co = cohere.ClientV2(api_key=COHERE_API_KEY)

# text = "minecraft"

# text_inputs = [
#     {
#         "content": [
#             {"type": "text", "text": text},
#         ]
#     },
# ]

# response = co.embed(
#     inputs=text_inputs,
#     model="embed-v4.0",
#     input_type="classification",
#     embedding_types=["float"],
# )


# vector = response.embeddings.float[0]

#==============================
#=======Inserting part=========
#==============================

# doc = {
#     "user_id": "123",          
#     "text": text,              # raw text
#     "embedding": vector       # vector embedding
# }

# collection.insert_one(doc)


#==============================
#=======Index-Creation=========
#==============================

from pinecone import Pinecone

pc = Pinecone(api_key = PINECONE_API_KEY)


index = pc.Index(host=INDEX_HOST)

# Upsert records into a namespace
# `text` fields are converted to dense vectors
# `category` fields are stored as metadata

# index.upsert_records(
#     "example-namespace",
#     [
#         {
#             "id":"rec5",
#             "user_id":1234,
#             "date":'august-25',
#             "text":"I ate fish"
#         }
#     ]
# ) 

search_with_text = index.search(
    namespace="example-namespace", 
    query={
        "inputs": {"text": "Disease prevention"}, 
        "top_k": 4
    },
    fields=["category", "text"],
    rerank={
        "model": "bge-reranker-v2-m3",
        "top_n": 2,
        "rank_fields": ["text"] # Specified field must also be included in 'fields'
    }
)

print(search_with_text)