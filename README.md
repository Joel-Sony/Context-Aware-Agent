# RAG-Powered Medical Agent: Hybrid Memory & Triage Agent

This project implements a context-aware conversational system with short-term and long-term memory, semantic retrieval using vector search, speech processing, and a safety-first medical triage agent. It combines LLM reasoning with deterministic control flow to reduce hallucinations and handle sensitive health-related conversations reliably.

---

## Features

- **Dual-Stream Emotion Intelligence:**
    - Vocal Sentiment: wav2vec2 analysis of audio chunks for emotional tone.
    - Textual Sentiment: NLP-based emotion detection from transcribed user input.
- **Vector-based long-term memory** using Pinecone (dense embeddings)
- **Short-term memory cache** with similarity-based eviction and refill
- **Explicit memory recall triggers** (e.g. “you said”, “remember”, topic shifts)
- **Context-aware medical triage agent**:
  - Follow-up questioning
  - Guideline retrieval (vector-backed, hallucination-safe)
  - Emergency escalation with red-flag guards
- **Speech-to-text** using fast-whisper
- **Emotion detection** using wav2vec-based emotion recognition
- **Rule + LLM hybrid control flow** (prevents infinite loops and unsafe behavior)

---

## Architecture Overview

- **Vector DB**: Pinecone 
- **Short-term memory**: In-memory deque cache
- **Long-term memory**: Pinecone namespace per user
- **Backend**: Flask
- **Frontend**: React
- **STT**: fast-whisper
- **Emotion Model**: wav2vec-based classifier

---

## Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- Pinecone account
- Cohere API key

### Backend Setup
```bash
git clone https://github.com/your-username/your-repo.git
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend Setup 
```Bash
# From the project root
cd frontend
npm install
npm start
```

--- 

### Contributors  
<img src="https://github.com/Kevinjose102.png" width="60px" style="border-radius:50%" alt="Kevinjose102"/>
</a>
<a href="https://github.com/Giga4byte">
  <img src="https://github.com/Giga4byte.png" width="60px" style="border-radius:50%" alt="Giga4byte"/>
</a>
<a href="https://github.com/SNEHA-REJI">
  <img src="https://github.com/SNEHA-REJI.png" width="60px" style="border-radius:50%" alt="SNEHA-REJI"/>
</a>

  Contributions are welcome!

- Fork the repository  
- Create a new branch  
- Submit a pull request


