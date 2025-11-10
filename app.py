import os
from dotenv import load_dotenv

from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from mistralai import Mistral

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import time

import streamlit as st

load_dotenv()
DetectorFactory.seed = 0

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY", "")
INDEX_NAME       = os.getenv("INDEX_NAME", "medical-chatbot")
MODEL_NAME       = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing")
if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY is missing")

# --- language helpers ---
_LANG_NAMES = {
    "en":"English","es":"Spanish","hi":"Hindi","ta":"Tamil","te":"Telugu","ml":"Malayalam","kn":"Kannada",
    "mr":"Marathi","bn":"Bengali","gu":"Gujarati","pa":"Punjabi","ur":"Urdu","fa":"Persian","ar":"Arabic",
    "fr":"French","de":"German","it":"Italian","pt":"Portuguese","ru":"Russian","ja":"Japanese",
    "ko":"Korean","zh-cn":"Chinese (Simplified)","zh-tw":"Chinese (Traditional)","id":"Indonesian",
    "he":"Hebrew","tr":"Turkish","vi":"Vietnamese","th":"Thai"
}

def language_name(code: str) -> str:
    return _LANG_NAMES.get(code.lower(), code)

def normalize_lang(code: str) -> str:
    mapping = {"zh-cn": "zh-CN", "zh-tw": "zh-TW", "iw": "he", "in": "id"}
    return mapping.get(code.lower(), code.lower())

def detect_lang(text: str) -> str:
    try:
        return detect(text) or "en"
    except Exception:
        return "en"

def translate(text: str, target: str, source: str = "auto") -> str:
    if not text or target.lower() == source.lower():
        return text
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception:
        return text
    
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "medical-chatbot"

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

SYSTEM_PROMPT = (
    "You are a careful, multilingual medical support assistant.\n"
    "Answer ONLY using the provided context; if the answer is not in the context, say you don't know.\n"
    "Be concise and clear. Include a short safety disclaimer at the end."
)

def _format_context(docs):
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        snippet = d.page_content
        parts.append(f"{snippet}\n[source: {src}]")
    return "\n\n".join(parts)

MODEL_CANDIDATES = [
    os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
    "open-mixtral-8x7b",
    "mistral-small-latest",
]

def ask_mistral_with_context(question_en: str, docs) -> tuple[str, str]:
    context = _format_context(docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question_en}\n\nContext:\n{context}\n\nAnswer:"
        },
    ]

    max_attempts_per_model = 3
    base_delay = 1.0  # seconds
    last_err = None

    for model_name in MODEL_CANDIDATES:
        for attempt in range(1, max_attempts_per_model + 1):
            try:
                with Mistral(api_key=MISTRAL_API_KEY) as client:
                    resp = client.chat.complete(
                        model=model_name,
                        messages=messages,
                        stream=False,
                    )
                return resp.choices[0].message.content.strip(), model_name
            except Exception as e:
                err_msg = str(e).lower()
                last_err = e
                if "429" in err_msg or "capacity" in err_msg or "rate" in err_msg:
                    time.sleep(base_delay * (2 ** (attempt - 1)))  # backoff
                    continue
                break  # move to next model if not a retryable error

    raise RuntimeError(f"Mistral request failed across models; last error: {last_err}")

def answer_user_query(user_text: str, k: int = 3):
    # 1) Detect & translate to English
    src_lang = detect_lang(user_text)
    src_norm = normalize_lang(src_lang)
    q_en = translate(user_text, target="en", source=src_norm)

    # 2) Retrieve top-k context
    docs = retriever.invoke(q_en)

    # 3) LLM answer in English + which model produced it
    answer_en, model_used = ask_mistral_with_context(q_en, docs)

    # 4) Translate back to user's language
    answer_user_lang = translate(answer_en, target=src_norm, source="en")

    # 5) Return final answer + useful details (no sources)
    extras = {
        "detected_lang_code": src_norm,
        "detected_lang_name": language_name(src_norm),
        "question_en": q_en,
        "answer_en": answer_en,
        "model_used": model_used,
    }
    return answer_user_lang, extras


# === Streamlit UI ===
st.set_page_config(page_title="🩺 Multilingual Medical Support Chatbot", page_icon="🩺", layout="centered")
st.title("🩺 Multilingual Medical Support Chatbot")
st.caption("Educational support only — not a substitute for professional medical advice.")

# chat history
if "history" not in st.session_state:
    st.session_state.history = []

# render past messages
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

# input
user_text = st.chat_input("Ask your medical question in any language…")
if user_text:
    st.session_state.history.append(("user", user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("Thinking…")
        try:
            final_answer, extras = answer_user_query(user_text)
        except Exception as e:
            final_answer = f"Sorry, I hit an error: `{e}`"
            extras = None

        # Main assistant reply (translated to user's language)
        placeholder.markdown(final_answer)

        # Details expander: language, translations, model
        if extras:
            with st.expander("🔎 Details (language, translations, model)"):
                st.write(f"**Detected language:** {extras['detected_lang_name']}  \n`{extras['detected_lang_code']}`")
                st.write(f"**Model used:** {extras['model_used']}")
                st.markdown("**Input translated to English:**")
                st.code(extras["question_en"])
                st.markdown("**Assistant answer in English:**")
                st.code(extras["answer_en"])

        st.caption("⚠️ Educational information only. Not medical advice. For symptoms or emergencies, seek professional care.")

    # limit history length
    st.session_state.history = st.session_state.history[-10:]
