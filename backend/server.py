from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import uuid
import subprocess
from datetime import date
from collections import deque
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import librosa
import numpy as np
from collections import defaultdict
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification
)
from faster_whisper import WhisperModel

# ============= FLASK APP =============
app = Flask(__name__)
CORS(app)


# ============= CONFIGURATION =============

UPLOAD_FOLDER = 'temp_audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# -----------------------------
# Model setup (load once at startup)
# -----------------------------
print("Loading Whisper model...")
whisper_model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

print("Loading emotion recognition model...")
EMOTION_MODEL_NAME = "r-f/wav2vec-english-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(EMOTION_MODEL_NAME)
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME)
emotion_model.eval()

LABEL_MAP = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprise": "Surprise"
}

print("Models loaded successfully!")

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=INDEX_HOST)

# Load mood prediction model
model_id = "LurkingMango/Mood_Predictor"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

MOOD_LABELS = {
    0: "anger", 1: "disgust", 2: "fear", 3: "joy",
    4: "neutral", 5: "sadness", 6: "shame", 7: "surprise"
}

# Load trigger embeddings (for checking if DB query is needed)
triggerEmbeddings = np.load("triggerEmbeddings.npy")

# Base prompt for Lenni
BASE_PROMPT = """You are Lenni — a warm, non-judgmental mental health companion.
Keep your responses concise, gentle, and supportive.
Ask simple, curious questions to understand the user better.
Validate feelings without overexplaining.
Offer practical, grounded guidance when appropriate.
Avoid long paragraphs, lectures, or clinical language.
calm, human, attentive.."""


# ============= SHORT-TERM MEMORY =============
class ShortTermMemory:
    """Stores recent conversation messages to avoid querying DB every time."""
    
    def __init__(self, size=30):
        self.cache = deque(maxlen=size)

    def add(self, turn_id, role, text, embedding):
        self.cache.append({
            "turn_id": turn_id,
            "role": role,
            "text": text,
            "embedding": np.array(embedding)
        })

    def get_all(self):
        return list(self.cache)

    def is_empty(self):
        return len(self.cache) == 0

    def is_full(self):
        return len(self.cache) >= self.cache.maxlen


# Initialize memory
STM = ShortTermMemory(size=30)


# ============= HELPER FUNCTIONS =============
def predict_mood(text):
    """Predicts user's mood from text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    return MOOD_LABELS[predicted_class], confidence


def get_turn_id(namespace):
    """Gets the next turn ID for the conversation."""
    for ns in index.list_namespaces():
        if ns.name == str(namespace):
            return (int(ns.record_count) // 2) + 1
    return 1


def cosine_similarity(a, b):
    """Calculates cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def should_query_database(query_embedding, threshold=0.7):
    """Checks if we need to query the database based on trigger embeddings."""
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    trigger_norm = triggerEmbeddings / np.linalg.norm(triggerEmbeddings, axis=1, keepdims=True)
    similarities = trigger_norm @ query_norm
    return np.any(similarities >= threshold)


def build_context(user_query, query_embedding, namespace):
    """Builds conversation context from memory or database."""
    
    # If memory is empty, query database
    if STM.is_empty():
        return query_from_database(query_embedding, namespace)
    
    # If memory not full, use all cached messages
    if not STM.is_full():
        return STM.get_all()
    
    # Memory is full - check if we need fresh context from DB
    if should_query_database(query_embedding):
        return get_hybrid_context(query_embedding, namespace)
    
    # Use existing memory
    return STM.get_all()


def query_from_database(query_embedding, namespace, top_k=10):
    """Retrieves relevant messages from Pinecone."""
    result = index.query(
        namespace=namespace,
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    return [
        {
            "turn_id": match.metadata.get("turn_id"),
            "role": match.metadata.get("role"),
            "text": match.metadata.get("text"),
        }
        for match in result.matches
    ]


def get_hybrid_context(query_embedding, namespace):
    """Combines recent memory with database results."""
    DROP_OLDEST = 10
    
    # Keep most recent messages from memory
    recent_messages = list(STM.cache)[DROP_OLDEST:]
    
    # Get relevant old messages from database
    db_messages = query_from_database(query_embedding, namespace, top_k=DROP_OLDEST)
    
    # Combine and remove duplicates
    combined = recent_messages + db_messages
    seen = set()
    deduplicated = []
    
    for msg in combined:
        key = (msg.get("turn_id"), msg.get("role"))
        if key not in seen:
            seen.add(key)
            deduplicated.append(msg)
    
    return deduplicated


def build_prompt(user_query, query_embedding, namespace):
    """Builds the final prompt for the LLM."""
    
    # Get conversation context
    context_messages = build_context(user_query, query_embedding, namespace)
    
    # Predict user's mood
    mood, confidence = predict_mood(user_query)
    
    # Format context as conversation
    context_text = "\n".join([
        f"{msg['role'].capitalize()}: {msg['text']}" 
        for msg in context_messages
    ])
    
    # Build final prompt
    return f"""{BASE_PROMPT}

Relevant conversation context:
{context_text}

Detected user emotion: {mood}
Emotion confidence: {confidence:.2f}

User: {user_query}"""


def wait_for_vector(vector_id, namespace, max_retries=10):
    """Waits for vector to be available in Pinecone after insertion."""
    delay = 0.1
    
    for i in range(max_retries):
        result = index.fetch(ids=[vector_id], namespace=namespace)
        if result.vectors:
            return result.vectors[vector_id].values
        time.sleep(delay * (2 ** i))  # Exponential backoff
    
    raise RuntimeError("Vector not available after retries")

def convert_webm_to_wav(webm_path, wav_path):
    """Convert WebM audio to WAV format."""
    try:
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio = audio.set_channels(1)  # mono
        audio = audio.set_frame_rate(16000)  # 16kHz
        audio.export(wav_path, format="wav")
        return True
    except Exception as e:
        print(f"Error converting audio: {e}")
        return False


def transcribe_audio(wav_path):
    """Transcribe audio using Whisper."""
    segments, info = whisper_model.transcribe(wav_path)
    
    full_text = ""
    for segment in segments:
        full_text += segment.text + " "
    
    return full_text.strip(), info.language


def predict_emotion_from_audio(audio_np, sr=16000):
    """Predict emotion from audio numpy array."""
    inputs = feature_extractor(
        audio_np,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    pred_id = torch.argmax(probs, dim=-1).item()
    raw_label = emotion_model.config.id2label[pred_id]
    confidence = probs[0, pred_id].item()

    return LABEL_MAP[raw_label], confidence


def analyze_audio_emotions(wav_path, chunk_duration=3.0, sr=16000):
    """Analyze emotions in audio by chunking."""
    audio, _ = librosa.load(wav_path, sr=sr)

    chunk_size = int(chunk_duration * sr)
    num_chunks = int(np.ceil(len(audio) / chunk_size))

    chunk_results = []
    emotion_scores = defaultdict(float)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(audio))
        chunk_audio = audio[start:end]

        # Skip very short chunks
        if len(chunk_audio) < sr * 0.5:
            continue

        emotion, confidence = predict_emotion_from_audio(chunk_audio, sr)

        chunk_results.append({
            "chunk_index": i,
            "start_time": round(start / sr, 2),
            "end_time": round(end / sr, 2),
            "emotion": emotion,
            "confidence": round(confidence, 3)
        })

        emotion_scores[emotion] += confidence

    # Get dominant emotion
    if emotion_scores:
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
    else:
        dominant_emotion = "Neutral"

    return chunk_results, dominant_emotion


def process_message_logic(user_id, message):
    """
    Core chatbot logic - processes message and returns reply.
    """
    from datetime import date
    import uuid
    from openai import OpenAI
    
    # Get conversation metadata
    turn_id = get_turn_id(user_id)
    today = str(date.today())
    user_message_id = str(uuid.uuid4())
    
    # Store user message in Pinecone
    index.upsert_records(
        user_id,
        [{
            "id": user_message_id,
            "text": message,
            "date": today,
            "turn_id": turn_id,
            "role": "user"
        }]
    )
    
    # Wait for vector to be available and retrieve it
    user_vector = np.array(wait_for_vector(user_message_id, user_id))
    
    # Add to short-term memory
    STM.add(turn_id, "user", message, user_vector)
    
    # Build prompt with context
    prompt = build_prompt(message, user_vector, user_id)
    
    # Generate AI response
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
    
    completion = client.chat.completions.create(
        model="google/gemma-3n-e2b-it:free",
        messages=[{"role": "assistant", "content": prompt}]
    )
    
    reply = completion.choices[0].message.content
    
    # Store assistant response in Pinecone
    assistant_message_id = str(uuid.uuid4())
    index.upsert_records(
        user_id,
        [{
            "id": assistant_message_id,
            "text": reply,
            "date": today,
            "turn_id": turn_id,
            "role": "assistant"
        }]
    )
    
    return reply

@app.route("/submit_message", methods=["POST"])
def submit_message():
    """Handles user message submission and generates AI response."""
    data = request.get_json()
    user_id = str(data.get("user_id"))
    message = data.get("message")
    
    reply = process_message_logic(user_id, message)
    
    return jsonify({"reply": reply})


@app.route('/submit_voice_message', methods=['POST'])
def submit_voice_message():
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        user_id = request.form.get('user_id')
        
        if audio_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save the WebM file
        filename = secure_filename(f"user_{user_id}_{audio_file.filename}")
        webm_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(webm_path)
        
        # Convert to WAV
        wav_filename = filename.rsplit('.', 1)[0] + '.wav'
        wav_path = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)
        
        if not convert_webm_to_wav(webm_path, wav_path):
            return jsonify({'error': 'Failed to convert audio format'}), 500
        
        # Transcribe audio
        transcribed_text, language = transcribe_audio(wav_path)
        
        # Analyze emotions
        emotion_chunks, dominant_emotion = analyze_audio_emotions(wav_path)
        
        # Clean up temporary files
        try:
            os.remove(webm_path)
            os.remove(wav_path)
        except:
            pass
        
        # Process the transcribed text using the same chatbot logic
        message_with_emotion = f"[Emotion: {dominant_emotion}] {transcribed_text}"
        
        # Call the core chatbot logic (same as text messages)
        reply = process_message_logic(user_id, message_with_emotion)
        
        # Return response with additional voice metadata
        return jsonify({
            'reply': reply,
            'transcription': transcribed_text,
            'language': language,
            'dominant_emotion': dominant_emotion,
            'emotion_timeline': emotion_chunks
        })
    
    except Exception as e:
        print(f"Error processing voice message: {e}")
        return jsonify({'error': str(e)}), 500

    

@app.route('/get_messages', methods=['POST'])
def get_messages():
    """Retrieves all messages for a user."""
    
    data = request.get_json()
    user_id = str(data["user_id"])
    
    # Get all message IDs
    chunk_ids = index.list(namespace=user_id)
    record_ids = list(chunk_ids)
    
    all_messages = []
    
    # Fetch all messages
    if record_ids:
        for ids_chunk in record_ids:
            records = index.fetch(ids=ids_chunk, namespace=user_id)
            
            for record_id in records.vectors:
                vector = records.vectors.get(record_id)
                if vector:
                    all_messages.append({
                        "sender": vector["metadata"]["role"],
                        "text": vector["metadata"]["text"],
                        "turn_id": vector["metadata"]["turn_id"]
                    })
    
    # Sort by turn_id and role (user messages first)
    all_messages.sort(key=lambda m: (m["turn_id"], 0 if m["sender"] == "user" else 1))
    
    # Remove turn_id from response
    formatted_messages = [
        {"sender": m["sender"], "text": m["text"]} 
        for m in all_messages
    ]
    
    return jsonify({"messages": formatted_messages})


if __name__ == "__main__":
    app.run(debug=True)