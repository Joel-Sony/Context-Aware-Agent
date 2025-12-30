from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import uuid
import subprocess
from datetime import date
from collections import deque, defaultdict
import numpy as np
import time
import torch
import librosa
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from faster_whisper import WhisperModel

# ============= FLASK APP =============
app = Flask(__name__)
CORS(app)


# ============= CONFIGURATION =============

UPLOAD_FOLDER = 'temp_audio'
AUDIO_STORAGE = 'stored_audio'  # Permanent storage for audio files
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_STORAGE, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_STORAGE'] = AUDIO_STORAGE
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
def count_recent_followups():
    """
    Counts how many consecutive assistant messages were follow-up questions.
    """
    count = 0
    for msg in reversed(STM.get_all()):
        if msg["role"] == "assistant" and msg["text"].endswith("?"):
            count += 1
        else:
            break
    return count


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

def get_recent_assistant_questions():
    return [
        msg["text"].lower()
        for msg in STM.get_all()
        if msg["role"] == "assistant" and msg["text"].endswith("?")
    ]


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

def process_message_logic(user_id, message, audio_filename=None, audio_duration=None):
    """
    Core chatbot logic - processes message and returns reply.
    """

    # ---- Setup ----
    turn_id = get_turn_id(user_id)
    today = str(date.today())
    user_message_id = str(uuid.uuid4())

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )

    # ---- Store USER message ----
    metadata = {
        "id": user_message_id,
        "text": message,
        "date": today,
        "turn_id": turn_id,
        "role": "user",
        "message_type": "voice" if audio_filename else "text"
    }

    if audio_filename:
        metadata["audio_filename"] = audio_filename
        metadata["audio_duration"] = audio_duration

    index.upsert_records(user_id, [metadata])

    user_vector = np.array(wait_for_vector(user_message_id, user_id))
    STM.add(turn_id, "user", message, user_vector)

    # ---- Agent decision ----
    decision = triage_agent_decision(message)

    # ---- Prevent duplicate follow-ups ----
    if decision["action"] == "ASK_FOLLOWUP":
        asked_questions = get_recent_assistant_questions()
        proposed_q = decision.get("question", "").lower()

        if any(proposed_q[:25] in q for q in asked_questions):
            decision = {
                "action": "RETRIEVE_GUIDELINE",
                "reason": "Follow-up question already asked and answered"
            }

    # ---- Pattern-based guideline override (non-emergency only) ----
    GUIDELINE_PATTERNS = [
        ["fever", "rash"],
        ["cough", "fever"],
        ["cold", "fever"],
        ["lightheaded", "fever"],
        ["depressed", "nausea"],
        ["sad", "nausea"]
    ]

    text = message.lower()
    if decision["action"] != "ESCALATE":
        if any(all(p in text for p in pattern) for pattern in GUIDELINE_PATTERNS):
            decision = {
                "action": "RETRIEVE_GUIDELINE",
                "reason": "Recognized common non-emergency symptom pattern"
            }

    # ---- Hard stop after 2 follow-ups ----
    if count_recent_followups() >= 2 and decision["action"] == "ASK_FOLLOWUP":
        decision = {
            "action": "RETRIEVE_GUIDELINE",
            "reason": "Sufficient information gathered from follow-ups"
        }

    action = decision["action"]

    # ---- ASK FOLLOW-UP ----
    if action == "ASK_FOLLOWUP":
        question = decision.get(
            "question",
            "Could you tell me a bit more about what you're experiencing?"
        )

        # STORE assistant question 
        assistant_id = str(uuid.uuid4())
        index.upsert_records(
            user_id,
            [{
                "id": assistant_id,
                "text": question,
                "date": today,
                "turn_id": turn_id,
                "role": "assistant",
                "message_type": "text"
            }]
        )

        STM.add(turn_id, "assistant", question, user_vector)
        return question

    # ---- ESCALATE ----
    if action == "ESCALATE":
        reply = (
            "Based on what you've shared, this could be serious. "
            "Please seek immediate medical attention or contact emergency services."
        )

        STM.add(turn_id, "assistant", reply, user_vector)
        return reply

    # ---- RETRIEVE GUIDELINE ----
    if action == "RETRIEVE_GUIDELINE":
        guideline = retrieve_guideline(user_vector)

        guideline_prompt = f"""
You are responding using the following medical safety guidance.

STRICT RULES:
- Do NOT ask further diagnostic questions
- Do NOT escalate unless symptoms clearly worsen
- Provide calm, general guidance
- Explain when medical care is needed

Medical guidance:
{guideline}

User message:
{message}

Respond clearly and reassuringly.
"""

        completion = client.chat.completions.create(
            model="google/gemma-3n-e2b-it:free",
            messages=[{"role": "assistant", "content": guideline_prompt}]
        )

        reply = completion.choices[0].message.content

        assistant_id = str(uuid.uuid4())
        index.upsert_records(
            user_id,
            [{
                "id": assistant_id,
                "text": reply,
                "date": today,
                "turn_id": turn_id,
                "role": "assistant",
                "message_type": "text"
            }]
        )

        STM.add(turn_id, "assistant", reply, user_vector)
        return reply

    # ---- Default Lenni response ----
    prompt = build_prompt(message, user_vector, user_id)

    completion = client.chat.completions.create(
        model="google/gemma-3n-e2b-it:free",
        messages=[{"role": "assistant", "content": prompt}]
    )

    reply = completion.choices[0].message.content
    STM.add(turn_id, "assistant", reply, user_vector)
    return reply

def triage_agent_decision(user_text):
    """
    Decides what action to take before responding.
    """

    system_prompt = """
You are a medical triage decision agent.
Your job is to decide the NEXT ACTION.

Allowed actions:
- ASK_FOLLOWUP
- RETRIEVE_GUIDELINE
- ESCALATE

Rules:
- Do NOT give medical advice
- Do NOT diagnose
- Choose exactly ONE action
- Respond ONLY in valid JSON
- If action is ASK_FOLLOWUP, include ONE clear question

Escalate ONLY for TRUE medical emergencies.
"""

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )

    completion = client.chat.completions.create(
        model="google/gemma-3n-e2b-it:free",
        messages=[
            {
                "role": "user",
                "content": f"{system_prompt}\n\nUser message:\n{user_text}"
            }
        ],
        temperature=0
    )

    import json

    raw = completion.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        decision = json.loads(raw)
    except Exception:
        return {
            "action": "ASK_FOLLOWUP",
            "reason": "Unable to interpret symptoms safely",
            "question": "Could you tell me a bit more about what you're experiencing?"
        }

    # ---- Normalize action ----
    if decision.get("action") not in {"ASK_FOLLOWUP", "RETRIEVE_GUIDELINE", "ESCALATE"}:
        return {
            "action": "ASK_FOLLOWUP",
            "reason": "Invalid action from agent",
            "question": "Could you clarify your symptoms a bit more?"
        }

    text = user_text.lower()

    # ---- Hard safety guard for escalation ----
    RED_FLAG_KEYWORDS = [
        "chest pain",
        "can't breathe",
        "cannot breathe",
        "shortness of breath",
        "passed out",
        "unconscious",
        "fainted",
        "seizure",
        "confusion",
        "slurred speech",
        "paralyzed",
        "kill myself",
        "suicide"
    ]

    if decision["action"] == "ESCALATE":
        if not any(k in text for k in RED_FLAG_KEYWORDS):
            return {
                "action": "RETRIEVE_GUIDELINE",
                "reason": "No clear emergency red flags detected"
            }

    # ---- Mental health handling (allow ONE follow-up) ----
    MENTAL_HEALTH_TRIGGERS = [
        "depressed",
        "sad",
        "hopeless",
        "disinterest",
        "lost interest",
        "numb",
        "empty"
    ]

    if any(m in text for m in MENTAL_HEALTH_TRIGGERS):
        if decision["action"] == "ASK_FOLLOWUP":
            return decision  # allow one clarifying question
        return {
            "action": "RETRIEVE_GUIDELINE",
            "reason": "Mental health symptoms identified"
        }

    # ---- Common physical symptom patterns ----
    COMMON_PATTERNS = [
        ["cough", "fever"],
        ["cold", "fever"],
        ["flu", "fever"],
        ["fever", "rash"],
        ["lightheaded", "fever"]
    ]

    if any(all(p in text for p in pattern) for pattern in COMMON_PATTERNS):
        return {
            "action": "RETRIEVE_GUIDELINE",
            "reason": "Recognized common non-emergency symptom pattern"
        }

    # ---- Default: trust agent ----
    return decision



@app.route("/submit_message", methods=["POST"])
def submit_message():
    """Handles user message submission and generates AI response."""
    data = request.get_json()
    user_id = str(data.get("user_id"))
    message = data.get("message")
    
    reply = process_message_logic(user_id, message)
    
    return jsonify({"reply": reply})

def retrieve_guideline(query_embedding, top_k=1):
    result = index.query(
        namespace="medical_guidelines",
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    if result.matches:
        return result.matches[0].metadata["text"]
    return None


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
        
        # Get turn_id before processing
        turn_id = get_turn_id(user_id)
        
        # Save the WebM file temporarily
        filename = secure_filename(f"temp_{user_id}_{audio_file.filename}")
        webm_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(webm_path)
        
        # Convert to WAV
        wav_filename = filename.rsplit('.', 1)[0] + '.wav'
        wav_path = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)
        
        if not convert_webm_to_wav(webm_path, wav_path):
            return jsonify({'error': 'Failed to convert audio format'}), 500
        
        # Get audio duration
        audio_data = AudioSegment.from_wav(wav_path)
        duration_seconds = len(audio_data) / 1000.0  # Convert ms to seconds
        
        # Transcribe audio
        transcribed_text, language = transcribe_audio(wav_path)
        
        # Analyze emotions
        emotion_chunks, dominant_emotion = analyze_audio_emotions(wav_path)
        
        # Create permanent audio filename with turn_id
        permanent_audio_filename = f"user_{user_id}_turn_{turn_id}.webm"
        permanent_audio_path = os.path.join(app.config['AUDIO_STORAGE'], permanent_audio_filename)
        
        # Copy original webm to permanent storage
        import shutil
        shutil.copy2(webm_path, permanent_audio_path)
        
        # Clean up temporary files
        try:
            os.remove(webm_path)
            os.remove(wav_path)
        except:
            pass
        
        # Process the transcribed text using the same chatbot logic
        message_with_emotion = f"[Emotion: {dominant_emotion}] {transcribed_text}"
        
        # Call the core chatbot logic with audio metadata
        reply = process_message_logic(
            user_id, 
            message_with_emotion,
            audio_filename=permanent_audio_filename,
            audio_duration=duration_seconds
        )
        
        # Return response with additional voice metadata
        return jsonify({
            'reply': reply,
            'transcription': transcribed_text,
            'language': language,
            'dominant_emotion': dominant_emotion,
            'emotion_timeline': emotion_chunks,
            'audio_filename': permanent_audio_filename,
            'duration': duration_seconds
        })
    
    except Exception as e:
        print(f"Error processing voice message: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve stored audio files."""
    return send_from_directory(app.config['AUDIO_STORAGE'], filename)


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
                    message_data = {
                        "sender": vector["metadata"]["role"],
                        "text": vector["metadata"]["text"],
                        "turn_id": vector["metadata"]["turn_id"],
                        "message_type": vector["metadata"].get("message_type", "text")
                    }
                    
                    # Add audio metadata if it's a voice message
                    if message_data["message_type"] == "voice":
                        message_data["audio_filename"] = vector["metadata"].get("audio_filename")
                        message_data["audio_duration"] = vector["metadata"].get("audio_duration")
                    
                    all_messages.append(message_data)
    
    # Sort by turn_id and role (user messages first)
    all_messages.sort(key=lambda m: (m["turn_id"], 0 if m["sender"] == "user" else 1))
    
    # Format messages for frontend
    formatted_messages = []
    for m in all_messages:
        msg = {
            "sender": m["sender"],
            "text": m["text"],
            "type": m["message_type"]
        }
        
        # Add audio info if voice message
        if m["message_type"] == "voice":
            msg["audio_filename"] = m.get("audio_filename")
            msg["audio_duration"] = m.get("audio_duration")
        
        formatted_messages.append(msg)
    
    return jsonify({"messages": formatted_messages})


if __name__ == "__main__":  
    app.run(debug=True)