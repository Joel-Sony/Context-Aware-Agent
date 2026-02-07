from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import uuid
import subprocess
import json 
import datetime
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
import logging

# Disable the standard Werkzeug request logs FOR SEEMA MISS
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)



class Logger:
    @staticmethod
    def log_event(category, message, data=None):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # --- TERMINAL LOGGING (Visual/Colors) ---
        prefix = f"[{timestamp}] \033[96m[{category.upper()}]\033[0m"
        print(f"\n{prefix} {message}") # Added leading newline for spacing
        
        # Helper to handle non-serializable objects in JSON
        def json_serializable(obj):
            if isinstance(obj, np.ndarray): return "[Vector Data]" 
            if isinstance(obj, deque): return list(obj)
            if isinstance(obj, (datetime.date, datetime.datetime)): return obj.isoformat()
            return str(obj)

        formatted_data = ""
        if data:
            try:
                # indent=4 makes the JSON readable (pretty-print)
                formatted_data = json.dumps(data, indent=4, default=json_serializable)
                print(f"      \033[90m{formatted_data}\033[0m")
            except Exception as e:
                print(f"      \033[91m>> [Logger Error]: {e}\033[0m")
        
        # --- FILE LOGGING (Readable Structure) ---
        try:
            with open("conversation_trace.log", "a", encoding="utf-8") as f:
                # 1. Add a clear separator between events
                f.write("\n" + "="*80 + "\n") 
                
                # 2. Add Header with Category and Time
                f.write(f"EVENT: {category.upper()}\n")
                f.write(f"TIME:  {timestamp}\n")
                f.write(f"INFO:  {message}\n")
                
                # 3. Add indented data block if it exists
                if data:
                    f.write("-" * 40 + "\n")
                    f.write(formatted_data + "\n")
                
                f.write("="*80 + "\n")
        except Exception as e:
            print(f"Failed to write to file: {e}")

logger = Logger()


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
# Model setup 
# -----------------------------
whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model for the first time...")
        whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return whisper_model


print("Loading emotion recognition model...")
EMOTION_MODEL_NAME = "r-f/wav2vec-english-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(EMOTION_MODEL_NAME)
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME)
emotion_model.eval()

# Define label mapping
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

def get_conversation_state(user_id):
    try:
        state = index.fetch(ids=["__state__"], namespace=user_id)
        if state.vectors:
            return state.vectors["__state__"]["metadata"]["state"]
    except:
        pass
    return "NORMAL"


def set_conversation_state(user_id, state):
    index.upsert_records(
        user_id,
        [{
            "id": "__state__",
            "state": state
        }]
    )

    
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
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    trigger_norm = triggerEmbeddings / np.linalg.norm(triggerEmbeddings, axis=1, keepdims=True)
    similarities = trigger_norm @ query_norm
    
    max_sim = np.max(similarities)
    logger.log_event("VECTOR_MATH", "Trigger check similarity score", {
        "max_similarity": round(float(max_sim), 4),
        "threshold_required": threshold,
        "is_triggered": bool(max_sim >= threshold)
    })
    
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


def build_prompt(user_query, query_embedding, namespace, mood, confidence):
    """Builds the final prompt for the LLM using pre-calculated mood."""
    
    # Get conversation context (STM or Pinecone)
    context_messages = build_context(user_query, query_embedding, namespace)
    
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
    model = get_whisper_model()
    segments, info = model.transcribe(wav_path)
    
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
    reply = "I'm sorry, I encountered an error."
    turn_id = get_turn_id(user_id)
    today = str(date.today())
    user_message_id = str(uuid.uuid4())
    
    # --- LOG: ENTRY & STATE ---
    current_state = get_conversation_state(user_id)
    logger.log_event("SESSION_START", "New message received", {
        "user_id": user_id,
        "turn_id": turn_id,
        "current_state": current_state,
        "raw_input": message
    })

    # --- LOG: TEXT ANALYSIS ---
    mood, confidence = predict_mood(message)
    logger.log_event("EMOTION_ENGINE", "Text mood analysis complete", {
        "mood": mood,
        "confidence": round(confidence, 4)
    })

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    # --- LOG: DATABASE UPSERT ---
    metadata = {
        "id": user_message_id,
        "text": message,
        "role": "user",
        "mood": mood,
        "turn_id": turn_id
    }
    index.upsert_records(user_id, [metadata])
    user_vector = np.array(wait_for_vector(user_message_id, user_id))
    logger.log_event("PINECONE", "Vector stored and retrieved", {"vector_id": user_message_id})

    # --- LOG: SHORT TERM MEMORY ---
    STM.add(turn_id, "user", message, user_vector)
    logger.log_event("MEMORY_STM", "Current Short-Term Memory Cache", STM.get_all())

    # --- LOG: TRIAGE AGENT ---
    last_assistant_msg = None
    recent_memory = STM.get_all()
    for m in reversed(recent_memory):
        if m["role"] == "assistant":
            last_assistant_msg = m["text"]
            break

    # 2. Pass it to the triage agent
    decision = triage_agent_decision(message, last_assistant_msg)
    logger.log_event("AGENT_TRIAGE", "Contextual Triage Decision", decision)
    logger.log_event("AGENT_TRIAGE", "Triage reasoning", decision)

    # (Logic Overrides - Keeping your existing logic)
    action = decision["action"]

    # --- ROUTING & PROMPT LOGGING ---
    if action == "ASK_FOLLOWUP":
        reply = decision.get("question", "Tell me more.")
        logger.log_event("ACTION_EXECUTION", "Executing ASK_FOLLOWUP")

    elif action == "ESCALATE":
        reply = "Please seek immediate medical attention."
        logger.log_event("ACTION_EXECUTION", "Executing ESCALATE - High Risk Detected")

    elif action == "RETRIEVE_GUIDELINE":
        # 1. Fetch the rich metadata from Pinecone
        guideline_metadata = retrieve_guideline(user_vector)
        
        if guideline_metadata:
            # 2. Build a "Grounded" prompt for Gemma
            # We explicitly tell the AI to use the measures and follow the instruction
            guideline_prompt = f"""
            SYSTEM CONTEXT: The user is showing signs of {guideline_metadata.get('category')}.
            CLINICAL DEFINITION: {guideline_metadata.get('text')}
            PROPER MEASURES TO FACILITATE: {guideline_metadata.get('measures')}
            AI BEHAVIOR INSTRUCTION: {guideline_metadata.get('ai_instruction')}

            USER MESSAGE: {message}

            LENNI'S TASK: Acknowledge the feeling warmly, then gently guide the user through the PROPER MEASURES mentioned above. 
            """
        else:
            # Fallback if no specific guideline matched well enough
            guideline_prompt = f"The user is distressed but no specific protocol matched. Be supportive and empathetic.\nUser: {message}"
        
        logger.log_event("LLM_PROMPT", "Grounded RAG Prompt sent to Gemma", {"prompt": guideline_prompt})
        
        completion = client.chat.completions.create(
            model="google/gemma-3n-e2b-it:free",
            messages=[{"role": "assistant", "content": guideline_prompt}]
        )
        reply = completion.choices[0].message.content

    else:
        # Default Lenni response
        full_prompt = build_prompt(message, user_vector, user_id, mood, confidence)
        
        # LOG THE FULL CONTEXTUAL PROMPT
        logger.log_event("LLM_PROMPT", "Full Contextual Prompt sent to Lenni", {"prompt": full_prompt})
        
        completion = client.chat.completions.create(
            model="google/gemma-3n-e2b-it:free",
            messages=[{"role": "assistant", "content": full_prompt}]
        )
        reply = completion.choices[0].message.content

    # --- LOG: ASSISTANT REPLY & FINAL SAVE ---
    assistant_id = str(uuid.uuid4())
    logger.log_event("LLM_RESPONSE", "Lenni's Final Reply", {"reply": reply})

    index.upsert_records(user_id, [{
        "id": assistant_id,
        "text": reply,
        "turn_id": turn_id,
        "role": "assistant"
    }])
    STM.add(turn_id, "assistant", reply, user_vector)
    
    logger.log_event("SESSION_END", "Turn complete. Database synchronized.")
    
    return reply

def triage_agent_decision(user_text, last_assistant_text=None):
    """
    Decides the next action based on current input and previous context.
    Prevents the 'Tell me more' loop by recognizing task completion.
    """
    text_lower = user_text.lower()
    last_text_lower = last_assistant_text.lower() if last_assistant_text else ""
    
    # ---- 1. EXTREME EMERGENCY CHECK (ESCALATE) ----
    EXTREME_CRITICAL = ["kill myself", "suicide", "end my life", "ending it all", "better off without me"]
    if any(phrase in text_lower for phrase in EXTREME_CRITICAL):
        return {
            "action": "ESCALATE",
            "reason": "Immediate self-harm risk detected",
            "question": "Please contact a crisis hotline immediately."
        }

    # ---- 2. TASK CONTINUITY CHECK (The Fix) ----
    # If Lenni just suggested a task (Grounding, Breathing, 5-Second Rule)
    # and the user is responding, we MUST stay in RETRIEVE_GUIDELINE mode.
    TASK_KEYWORDS = ["5-4-3-2-1", "grounding", "breathe", "count", "5-second", "task"]
    if any(k in last_text_lower for k in TASK_KEYWORDS):
        # If the user's response isn't just a tiny word like "ok", assume they are doing the task
        if len(user_text.split()) > 1 or any(char.isdigit() for char in user_text):
            return {
                "action": "RETRIEVE_GUIDELINE",
                "reason": "User is actively participating in a therapeutic task."
            }

    # ---- 3. SYMPTOM-BASED CHECK ----
    PANIC_TRIGGERS = ["can't breathe", "heart is racing", "panic", "terrified", "chest pain"]
    MENTAL_HEALTH_TRIGGERS = ["depressed", "sad", "hopeless", "disinterest", "numb", "anxious", "paralyzed"]
    if any(p in text_lower for p in PANIC_TRIGGERS + MENTAL_HEALTH_TRIGGERS):
        return {
            "action": "RETRIEVE_GUIDELINE",
            "reason": "Psychological distress detected."
        }

    # ---- 4. LLM-BASED TRIAGE (With Context) ----
    system_prompt = f"""
    You are a medical triage agent. 
    PREVIOUS CONTEXT: The assistant previously said: "{last_assistant_text if last_assistant_text else "None"}"
    
    Decide the NEXT ACTION:
    - RETRIEVE_GUIDELINE: If user shows symptoms OR is answering a previous exercise/task.
    - ASK_FOLLOWUP: Only if user is making small talk or is very vague.
    - ESCALATE: For clear self-harm intent.
    
    Respond ONLY in JSON.
    """

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    try:
        completion = client.chat.completions.create(
            model="google/gemma-3n-e2b-it:free",
            messages=[{"role": "user", "content": f"{system_prompt}\n\nUser Message: {user_text}"}],
            temperature=0 
        )
        raw = completion.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        decision = json.loads(raw)
    except:
        decision = {"action": "ASK_FOLLOWUP", "question": "Could you tell me more about how that feels?"}

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
    """
    Retrieves the most relevant mental health guideline from Pinecone.
    Returns the metadata dictionary if a strong match is found.
    """
    result = index.query(
        namespace="medical-guidelines",
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    # Check if we have a match and if the confidence score is high enough (e.g., > 0.70)
    if result.matches and result.matches[0].score > 0.40:
        match = result.matches[0]
        logger.log_event("PINE_CONE_RAG", "Specific Guideline Triggered", {
            "category": match.metadata.get("category"),
            "score": round(match.score, 4)
        })
        return match.metadata
    
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
        logger.log_event("AUDIO_PROCESSOR", "Emotion analysis complete", {
            "dominant": dominant_emotion,
            "confidence": f"{emotion_chunks[0]['confidence'] if emotion_chunks else 0}"
        })

        # After transcription
        logger.log_event("WHISPER_TRANSCRIPT", f"Text extracted: {transcribed_text}")
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
    """Retrieves all messages for a user with safety fallbacks."""
    data = request.get_json()
    user_id = str(data["user_id"])
    
    try:
        # Get all message IDs for this specific user namespace
        chunk_ids = index.list(namespace=user_id)
        record_ids = list(chunk_ids)
        
        all_messages = []
        
        if record_ids:
            for ids_chunk in record_ids:
                records = index.fetch(ids=ids_chunk, namespace=user_id)
                
                for record_id in records.vectors:
                    vector = records.vectors.get(record_id)
                    if not vector or "metadata" not in vector:
                        continue
                    
                    meta = vector["metadata"]
                    
                    
                    role = meta.get("role")
                    if not role:
                        continue 
                    
                    message_data = {
                        "sender": role,
                        "text": meta.get("text", "[Empty Message]"),
                        "turn_id": meta.get("turn_id", 0),
                        "message_type": meta.get("message_type", "text"),
                        "audio_filename": meta.get("audio_filename"),
                        "audio_duration": meta.get("audio_duration")
                    }
                    
                    all_messages.append(message_data)
        
        # Sort by turn_id, then ensure user messages appear before bot replies
        all_messages.sort(key=lambda m: (m["turn_id"], 0 if m["sender"] == "user" else 1))
        
        # Format final JSON for frontend
        formatted_messages = []
        for m in all_messages:
            msg = {
                "sender": m["sender"],
                "text": m["text"],
                "type": m["message_type"]
            }
            if m["message_type"] == "voice":
                msg["audio_filename"] = m["audio_filename"]
                msg["audio_duration"] = m["audio_duration"]
            formatted_messages.append(msg)
        
        return jsonify({"messages": formatted_messages})

    except Exception as e:
        print(f"Error in get_messages: {e}")
        return jsonify({"messages": [], "error": str(e)}), 500
    

if __name__ == "__main__":  
    app.run(debug=True, use_reloader=False)