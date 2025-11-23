# 🩺 Multilingual Medical Support Chatbot

A lightweight, retrieval-augmented medical Q&A chatbot with **multilingual input/output**, **automatic translation**, and **context-grounded answers**.  
Built with **Streamlit**, **Mistral** (chat), **Pinecone** (vector search), and **HuggingFace embeddings**.

> ⚠️ **Medical safety notice**  
> This app is for **educational support only** and is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## ✨ Features

- **Multilingual Q&A (16 languages)**  
  Automatically detects the language of the user’s question, translates it to English, answers using medical context, and translates the answer back.

  Supported languages: en, hi, ta, te, ml, kn, mr, bn, gu, ur, es, fr, de, ar, zh-cn, ja.

- **Medical-only behavior**  
  - Answers only health-related questions.  
  - Politely refuses non-medical questions.  
  - Adds a short medical disclaimer on every reply.

- **Retrieval-Augmented Generation (RAG)**  
  - Uses Pinecone index (`medical-chatbot`).  
  - Embeddings: `BAAI/bge-small-en-v1.5`.

- **Model fallback & retry logic**  
  - Primary model: `MISTRAL_MODEL` (from `.env`).  
  - Fallback order: `[MISTRAL_MODEL] → open-mixtral-8x7b → mistral-small-latest`.

- **Small-talk detection**  
  - Short greetings (hi, hello, thanks) handled smartly.

- **Streamlit chat UI**  
  - Chat history  
  - Shows detected language, translated question, English answer, and model used.

---

## 🧱 Architecture

1. Detect language  
2. Translate → English  
3. Retrieve medical context from Pinecone  
4. Mistral ChatCompletion with safety prompt  
5. Translate back to user language  
6. Show result + disclaimer

---

## 🗂️ Repo Structure

```
.
├─ app.py
├─ requirements.txt
├─ .env
├─ .streamlit/
│   └─ config.toml
└─ Data/
```

---

## 🔑 Environment Variables

```
PINECONE_API_KEY=
MISTRAL_API_KEY=
INDEX_NAME=medical-chatbot
MISTRAL_MODEL=mistral-small-latest
```

---

## ▶️ Running Locally

```
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deployment Notes (AWS EC2)

- Runs via systemd service  
- Reverse-proxied through Nginx  
- HTTPS via Certbot + Let’s Encrypt

---

## ⚠️ Limitations

- Not a replacement for medical professionals  
- Accuracy depends on your Pinecone index  
- Only supports the listed languages  

---

## 🚀 Future Enhancements

- Add more languages  
- Add FAQ section  
- Add minor symptom forms  
