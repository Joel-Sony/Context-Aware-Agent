from flask import Flask, request, jsonify
from flask_cors import CORS
import os 
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = Flask(__name__)
CORS(app)  


@app.route("/chat" , methods = ["POST"])
def index():
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
        "role": "user",
        "content": base_prompt + message
        }
    ]
    )
    reply = completion.choices[0].message.content

    return jsonify({"reply":reply})

if __name__ == "__main__":
    app.run(debug=True)   # starts the development server