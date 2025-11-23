# 🩺 Multilingual Medical Support Chatbot

A lightweight, RAG-style medical Q&A chatbot with **multilingual input/output**, **translation**, and **context-grounded answers**.  
Built with **Streamlit**, **Mistral** (chat), **Pinecone** (vector search), and **HuggingFace embeddings**.  
Deployed on **AWS EC2** with **Nginx**.

> ⚠️ **Medical safety notice**: This app is for educational support only and **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## ✨ Features

- **Ask in 90+ languages** → auto-detect language → translate to English → ground answers with your docs → translate back
- **Retrieval-Augmented Generation (RAG)** with Pinecone (uses your existing index)
- **Model fallback & retry** for free-tier capacity errors (e.g., 429)
- Streamlit **chat UI** showing:
  - Detected language (code + readable name)
  - **User input translated to English**
  - **Assistant’s English answer**
  - **Which model produced the answer**

---

## 🧱 Architecture (high level)

1. **User message** (any language)  
2. `langdetect` → detect language code  
3. `deep-translator` → translate to English  
4. Pinecone retriever (`k=5`) over your existing index (`BAAI/bge-small-en-v1.5`)  
5. **Mistral** model generates answer **only from context** (system prompt enforces) with **retry + fallback**  
6. Translate answer back to user’s language  
7. Streamlit shows answer + language details + model used

---

## 🗂️ Repo structure (key files)

```
.
├─ app.py                     # Streamlit app (UI + pipeline)
├─ requirements.txt           # Python deps
├─ .env                       # (optional) local dev secrets
├─ .streamlit/config.toml     # Headless server settings
└─ Data/                      # (optional) PDFs used to build the index
```

---

## 🧰 Tech stack

- **Frontend / App**: Streamlit
- **LLM**: Mistral (`mistral-small-latest`, fallback `open-mixtral-8x7b`)
- **Vector DB**: Pinecone
- **Embeddings**: `BAAI/bge-small-en-v1.5` (HuggingFace)
- **Language**: `langdetect`, `deep-translator` (Google)
- **Infra**: AWS EC2, Nginx.

---

Now visit : https://medi-bot.duckdns.org/
