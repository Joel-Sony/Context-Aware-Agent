import torch
import librosa
import numpy as np
from collections import defaultdict
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification
)
from faster_whisper import WhisperModel

model = WhisperModel(
    "tiny",          # start here
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe("sample.wav")

print("Language:", info.language)
for s in segments:
    print(s.text)
# -----------------------------
# Model setup
# -----------------------------
MODEL_NAME = "r-f/wav2vec-english-speech-emotion-recognition"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

LABEL_MAP = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprise": "Surprise"
}

# -----------------------------
# Emotion prediction on raw audio array
# -----------------------------
def predict_emotion_from_audio(audio_np, sr=16000):
    inputs = feature_extractor(
        audio_np,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    pred_id = torch.argmax(probs, dim=-1).item()
    raw_label = model.config.id2label[pred_id]
    confidence = probs[0, pred_id].item()

    return LABEL_MAP[raw_label], confidence


# -----------------------------
# Chunk audio + aggregate emotions
# -----------------------------
def analyze_audio_emotions(
    audio_path,
    chunk_duration=3.0,   # seconds
    sr=16000
):
    audio, _ = librosa.load(audio_path, sr=sr)

    chunk_size = int(chunk_duration * sr)
    num_chunks = int(np.ceil(len(audio) / chunk_size))

    chunk_results = []
    emotion_scores = defaultdict(float)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(audio))
        chunk_audio = audio[start:end]

        # Skip very short chunks (mostly silence)
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

        # confidence-weighted aggregation
        emotion_scores[emotion] += confidence

    # dominant emotion
    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]

    return chunk_results, dominant_emotion


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    chunks, dominant = analyze_audio_emotions("sample.wav")

    for c in chunks:
        print(c)

    print("\nDominant emotion:", dominant)
