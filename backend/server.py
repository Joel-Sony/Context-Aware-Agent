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
BASE_PROMPT = """You are Lenni, a warm mental health companion.
STRICT RULES:
1. Response length: Max 3 sentences.
2. Context Awareness: Before asking a question, check the 'Relevant conversation context'. If the user asks 'who', 'what', or 'do you remember', you MUST answer using the history provided.
3. No Filler: Do not use phrases like "I understand" or "It sounds like".
4. Interaction: Provide one brief supportive observation followed by ONE simple question.
Tone: Calm, human, and highly attentive to details."""


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
    """Builds context and primes STM if it only contains the current message."""
    
    # 1. NEW SESSION DETECTION: 
    # If STM only has 1 message, it means we just added the current user query, 
    # but have no prior session history in the cache.
    current_stm_content = STM.get_all()
    
    if len(current_stm_content) <= 1:
        logger.log_event("MEMORY_STM", "First message of session detected. Priming STM from Pinecone.")
        
        # Fetch the last 10 messages from previous sessions
        past_messages = query_from_database(query_embedding, namespace, top_k=10)
        
        if past_messages:
            # Sort chronologically
            past_messages.sort(key=lambda x: x.get("turn_id", 0))
            
            # Temporary storage to rebuild STM
            new_stm_entries = []
            
            for msg in past_messages:
                # Avoid duplicating the current message if Pinecone is very fast
                if msg["text"] == user_query:
                    continue
                    
                new_stm_entries.append({
                    "turn_id": msg["turn_id"],
                    "role": msg["role"],
                    "text": msg["text"],
                    "embedding": np.zeros(768) # Use your model's actual dim (Gemma is usually 768/2048/3584)
                })

            # Re-initialize STM with past messages + the current message
            # We clear and re-add to maintain chronological order in the deque
            current_user_msg = current_stm_content[0] if current_stm_content else None
            STM.cache.clear()
            
            for entry in new_stm_entries:
                STM.add(entry["turn_id"], entry["role"], entry["text"], entry["embedding"])
            
            # Re-add the current message at the very end
            if current_user_msg:
                STM.add(
                    current_user_msg["turn_id"], 
                    current_user_msg["role"], 
                    current_user_msg["text"], 
                    current_user_msg["embedding"]
                )
            
            logger.log_event("MEMORY_STM", f"STM Primed. Total cache size: {len(STM.get_all())}")
        
        return STM.get_all()
    
    # 2. If memory exists but isn't full, just return it
    if not STM.is_full():
        return STM.get_all()
    
    # 3. If memory is full, check if we need to do a "Deep Retrieval" (Hybrid)
    if should_query_database(query_embedding):
        return get_hybrid_context(query_embedding, namespace)
    
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


Detected user emotion: {mood}
Emotion confidence: {confidence:.2f}

Relevant conversation context:
{context_text}

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
    user_message_id = str(uuid.uuid4())
    
    # EVENT: SESSION_START
    logger.log_event("SESSION_START", "New message received", {
        "user_id": user_id,
        "turn_id": turn_id,
        "current_state": get_conversation_state(user_id),
        "raw_input": message
    })

    # 1. ANALYZE MOOD (EVENT: EMOTION_ENGINE)
    mood, confidence = predict_mood(message)
    logger.log_event("EMOTION_ENGINE", "Text mood analysis complete", {
        "mood": mood,
        "confidence": round(float(confidence), 4)
    })
    
    # 2. LOCAL LLM SETUP
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    LOCAL_MODEL = "llama3.1:8b"
    
    # EVENT: PINECONE
    index.upsert_records(user_id, [{
        "id": user_message_id, "text": message, "role": "user", "mood": mood, "turn_id": turn_id
    }])
    logger.log_event("PINECONE", "Vector stored and retrieved", {
        "vector_id": user_message_id
    })
    
    # Wait for vector and store in STM
    user_vector = np.array(wait_for_vector(user_message_id, user_id))
    STM.add(turn_id, "user", message, user_vector)

    # EVENT: MEMORY_STM
    logger.log_event("MEMORY_STM", "Current Short-Term Memory Cache", STM.get_all())

    # 3. TRIAGE DECISION (EVENT: AGENT_TRIAGE)
    decision = triage_agent_decision(user_text=message, mood=mood, confidence=confidence)
    action = decision.get("action", "ASK_FOLLOWUP")
    logger.log_event("AGENT_TRIAGE", "Contextual Triage Decision", {
        "action": action,
        "reason": decision.get("reason", "N/A")
    })

    # 4. GENERATE REPLY BASED ON ACTION
    if action == "CHAT":
        full_prompt = build_prompt(message, user_vector, user_id, mood, confidence)
        logger.log_event("LLM_PROMPT", "Contextual Chat Prompt sent to Gemma", {"prompt": full_prompt})
        
        completion = client.chat.completions.create(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=120,
            temperature=0.7
        )
        reply = completion.choices[0].message.content

    elif action == "ASK_FOLLOWUP":
        full_prompt = build_prompt(message, user_vector, user_id, mood, confidence)
        prompt_content = f"{full_prompt}\nKeep it as a brief follow-up question."
        logger.log_event("LLM_PROMPT", "Exploratory Prompt sent to Gemma", {"prompt": prompt_content})
        
        completion = client.chat.completions.create(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": prompt_content}],
            max_tokens=50
        )
        reply = completion.choices[0].message.content

    elif action == "ESCALATE":
        reply = "I'm concerned about your safety. Please reach out to a professional or a crisis hotline immediately (Call/Text 988)."
        logger.log_event("CRISIS_MGMT", "Safety escalation triggered")

    elif action == "RETRIEVE_GUIDELINE":
        # Note: log_event for PINE_CONE_RAG is handled inside the retrieve_guideline function itself
        guideline_metadata = retrieve_guideline(user_vector)
        
        if guideline_metadata:
            prompt = f"""SYSTEM CONTEXT: The user is showing signs of {guideline_metadata.get('category')}.
            CLINICAL DEFINITION: {guideline_metadata.get('text')}
            PROPER MEASURES TO FACILITATE: {guideline_metadata.get('measures')}
            AI BEHAVIOR INSTRUCTION: {BASE_PROMPT}

            USER MESSAGE: {message}

            LENNI'S TASK: Acknowledge the feeling warmly, then gently guide the user through the PROPER MEASURES mentioned above."""
        else:
            prompt = f"{BASE_PROMPT}\nUser is distressed. Be supportive.\nUser: {message}"
        
        logger.log_event("LLM_PROMPT", "Grounded RAG Prompt sent to Gemma", {"prompt": prompt})
        
        completion = client.chat.completions.create(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        reply = completion.choices[0].message.content

    # 5. SAVE ASSISTANT REPLY
    assistant_id = str(uuid.uuid4())
    index.upsert_records(user_id, [{
        "id": assistant_id, "text": reply, "turn_id": turn_id, "role": "assistant"
    }])
    STM.add(turn_id, "assistant", reply, user_vector)
    
    return {"reply": reply, "action": action}


def triage_agent_decision(user_text, mood=None, confidence=0.0):
    text = user_text.lower()

    # 1. CRITICAL CRISIS (Highest Priority)
    ESCALATE_SIGNALS = [
        "kill myself", "suicide", "end my life", 
        "better off without me", "don't want to be here"
    ]
    if any(p in text for p in ESCALATE_SIGNALS):
        return {"action": "ESCALATE", "reason": "Crisis detected"}

    # 2. ACUTE MENTAL DISTRESS (Requires specific grounding/guidelines)
    ACUTE_MENTAL = [
        "panic", "racing mind", "can't breathe", "dying",
        "rage", "explosive", "scream", "paralyzed", "stuck"
    ]
    if any(p in text for p in ACUTE_MENTAL):
        return {"action": "RETRIEVE_GUIDELINE", "reason": "Acute distress"}

    # 3. NEW: CONTEXTUAL CHAT (Priority over generic follow-up)
    # This catches questions (why, how) or references to previous context (it, that, job)
    CHAT_SIGNALS = [
        "why", "how", "do you think", "is it because", "maybe", 
        "reason", "because", "remember", "yesterday", "earlier"
    ]
    # Also trigger CHAT if the user asks a question (ends with ?)
    if any(p in text for p in CHAT_SIGNALS) or text.strip().endswith("?"):
        return {"action": "CHAT", "reason": "Contextual conversation requested"}

    # 4. PERSISTENT LOW MOOD (RAG for depressive symptoms)
    LOW_ENERGY_SIGNALS = ["pointless", "hopeless", "exhausted", "drained", "no motivation"]
    if mood == "sadness" and (confidence > 0.70 or any(p in text for p in LOW_ENERGY_SIGNALS)):
        return {"action": "RETRIEVE_GUIDELINE", "reason": "Low mood symptoms"}

    # 5. DEFAULT: EXPLORATORY
    return {"action": "ASK_FOLLOWUP", "reason": "Generic exploratory turn"}


@app.route("/submit_message", methods=["POST"])
def submit_message():
    """Handles user message submission and generates AI response."""
    data = request.get_json()
    user_id = str(data.get("user_id"))
    message = data.get("message")
    
    result = process_message_logic(user_id, message)

    return jsonify(result)

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
    """Retrieves all messages for a user using the correct Pinecone Vector object attributes."""
    data = request.get_json()
    user_id = str(data.get("user_id"))
    
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    try:
        all_messages = []
        
        # index.list() returns a generator that yields batches of IDs
        for ids_chunk in index.list(namespace=user_id):
            if not ids_chunk:
                continue
            
            # Fetch the actual vector data for these IDs
            fetch_response = index.fetch(ids=ids_chunk, namespace=user_id)
            
            # Correctly iterate through the fetch response
            for record_id, vector_obj in fetch_response.vectors.items():
                # Skip internal state vectors
                if record_id == "__state__":
                    continue
                
                # FIX: Access the metadata attribute directly
                # In Pinecone SDK v3+, vector_obj is a 'Vector' class instance
                meta = vector_obj.metadata if hasattr(vector_obj, 'metadata') and vector_obj.metadata else {}
                
                role = meta.get("role")
                if role:
                    all_messages.append({
                        "sender": role,
                        "text": meta.get("text", "[Empty Message]"),
                        "turn_id": int(meta.get("turn_id", 0)),
                        "type": meta.get("message_type", "text"),
                        "audio_filename": meta.get("audio_filename"),
                        "audio_duration": meta.get("audio_duration")
                    })

        # Sort by turn_id, ensuring user messages appear before assistant replies in the same turn
        all_messages.sort(key=lambda m: (m["turn_id"], 0 if m["sender"] == "user" else 1))

        # Final cleanup for frontend delivery
        formatted_messages = []
        for m in all_messages:
            msg = {
                "sender": m["sender"],
                "text": m["text"],
                "type": m["type"]
            }
            # Include audio fields only if it's a voice message
            if m["type"] == "voice" or m.get("audio_filename"):
                msg["type"] = "voice"
                msg["audio_filename"] = m["audio_filename"]
                msg["audio_duration"] = m["audio_duration"]
            formatted_messages.append(msg)

        return jsonify({"messages": formatted_messages})

    except Exception as e:
        # Detailed logging for debugging
        print(f"Error in get_messages: {str(e)}")
        return jsonify({"messages": [], "error": str(e)}), 500

if __name__ == "__main__":  
    app.run(debug=True, use_reloader=False)