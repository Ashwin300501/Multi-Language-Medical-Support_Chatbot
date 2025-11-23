import os
from dotenv import load_dotenv

from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from mistralai import Mistral

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import time

import re

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
    # --- Common languages ---
    "af": "Afrikaans",
    "sq": "Albanian",
    "am": "Amharic",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "eu": "Basque",
    "be": "Belarusian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "ny": "Chichewa",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "co": "Corsican",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "eo": "Esperanto",
    "et": "Estonian",
    "tl": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Frisian",
    "gl": "Galician",
    "ka": "Georgian",
    "de": "German",
    "el": "Greek",
    "gu": "Gujarati",
    "ht": "Haitian Creole",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "iw": "Hebrew",
    "he": "Hebrew",
    "hi": "Hindi",
    "hmn": "Hmong",
    "hu": "Hungarian",
    "is": "Icelandic",
    "ig": "Igbo",
    "id": "Indonesian",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "km": "Khmer",
    "rw": "Kinyarwanda",
    "ko": "Korean",
    "ku": "Kurdish (Kurmanji)",
    "ky": "Kyrgyz",
    "lo": "Lao",
    "la": "Latin",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "lb": "Luxembourgish",
    "mk": "Macedonian",
    "mg": "Malagasy",
    "ms": "Malay",
    "ml": "Malayalam",
    "mt": "Maltese",
    "mi": "Maori",
    "mr": "Marathi",
    "mn": "Mongolian",
    "my": "Myanmar (Burmese)",
    "ne": "Nepali",
    "no": "Norwegian",
    "or": "Odia (Oriya)",
    "ps": "Pashto",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "pa": "Punjabi",
    "ro": "Romanian",
    "ru": "Russian",
    "sm": "Samoan",
    "gd": "Scots Gaelic",
    "sr": "Serbian",
    "st": "Sesotho",
    "sn": "Shona",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "es": "Spanish",
    "su": "Sundanese",
    "sw": "Swahili",
    "sv": "Swedish",
    "tg": "Tajik",
    "ta": "Tamil",
    "tt": "Tatar",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "tk": "Turkmen",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "ug": "Uyghur",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "cy": "Welsh",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zu": "Zulu",
}

def language_name(code: str) -> str:
    if not code:
        return "Unknown"
    code = code.lower()
    mapping = {"zh-cn": "zh-cn", "zh-tw": "zh-tw", "iw": "he", "in": "id"}
    normalized = mapping.get(code, code)
    return _LANG_NAMES.get(normalized, _LANG_NAMES.get(code.split("-")[0], "Unknown"))

def normalize_lang(code: str) -> str:
    mapping = {"zh-cn": "zh-CN", "zh-tw": "zh-TW", "iw": "he", "in": "id"}
    return mapping.get(code.lower(), code.lower())

# Only support a subset of popular languages
POPULAR_LANG_CODES = [
    "en",      # English
    "hi",      # Hindi
    "ta",      # Tamil
    "te",      # Telugu
    "ml",      # Malayalam
    "kn",      # Kannada
    "mr",      # Marathi
    "bn",      # Bengali
    "gu",      # Gujarati
    "ur",      # Urdu
    "es",      # Spanish
    "fr",      # French
    "de",      # German
    "ar",      # Arabic
    "zh-cn",   # Chinese (Simplified)
    "ja",      # Japanese
]
POPULAR_LANG_CODES_LOWER = {c.lower() for c in POPULAR_LANG_CODES}

# common English small-talk / greeting phrases
_ENGLISH_SMALL_TALK = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "how r you",
    "how are u",
    "how r u",
    "whats up",
    "what's up",
    "how's it going",
    "hows it going",
    "ok thanks",
    "okay thanks",
    "thank you",
    "thanks",
    "thank u",
]

def detect_lang(text: str) -> str:
    """
    Improved language detection:
    - If the text is short and clearly looks like English small-talk, force 'en'
    - Otherwise, use langdetect
    """
    t = (text or "").strip()
    if not t:
        return "en"

    t_lower = t.lower()

    # If it's short, ASCII-ish, and matches English small-talk phrases → force English
    if (
        len(t) <= 40
        and re.fullmatch(r"[A-Za-z0-9\s'?.,!]+", t)
        and any(kw in t_lower for kw in _ENGLISH_SMALL_TALK)
    ):
        return "en"

    # Otherwise fall back to langdetect
    try:
        return detect(t) or "en"
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
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL)

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "medical-chatbot"

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

SYSTEM_PROMPT = (
        "You are a careful medical assistant. Answer ONLY from the provided medical context.\n"
    "If unsure, say you don't know. Keep the answer concise and patient-friendly.\n"
    "If the user's question is clearly not about health or medicine (sports, celebrities, geography, history, etc.), "
    "explain briefly that you are a medical chatbot and cannot answer non-medical questions.\n"
    "Give a brief medical disclaimer at the end."
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

def is_small_talk(text: str) -> bool:
    if not text:
        return False

    t = text.lower().strip()
    for ch in [".", "!", "?", ","]:
        t = t.replace(ch, "")

    greeting_keywords = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "how r you", "how are u", "how r u",
        "what's up", "whats up", "how's it going", "hows it going",
    ]

    thanks_keywords = [
        "thank you", "thanks", "thank u", "ok thanks", "okay thanks"
    ]

    for kw in greeting_keywords:
        if t == kw or kw in t:
            return True
    for kw in thanks_keywords:
        if t == kw or t.startswith(kw):
            return True

    return False

def answer_user_query(user_text: str, k: int = 3):
    raw = (user_text or "").strip()

    # 0) Detect language of original input
    src_lang = detect_lang(raw)
    src_norm = normalize_lang(src_lang)

    # 0.5) Enforce popular language whitelist
    if src_norm.lower() not in POPULAR_LANG_CODES_LOWER:
        supported_list = ", ".join(sorted({language_name(c) for c in POPULAR_LANG_CODES}))
        answer_en = (
            f"Sorry, I currently support only these languages: {supported_list}.\n"
            "Please ask your question in one of these languages."
        )
        # Keep reply in English so it's understandable even if detection was weird
        answer_user_lang = answer_en
        extras = {
            "detected_lang_code": src_norm,
            "detected_lang_name": language_name(src_norm),
            "question_en": raw,
            "answer_en": answer_en,
            "model_used": "language-not-supported",
        }
        return answer_user_lang, extras

    # 1) Translate to English
    q_en = translate(raw, target="en", source=src_norm)

    # 2) Small-talk handling (on English version)
    if is_small_talk(q_en):
        answer_en = (
            "I'm just a medical chatbot, but I'm functioning well 😊. "
            "How can I help you with your health-related questions today?"
        )
        answer_user_lang = translate(answer_en, target=src_norm, source="en")
        extras = {
            "detected_lang_code": src_norm,
            "detected_lang_name": language_name(src_norm),
            "question_en": q_en,
            "answer_en": answer_en,
            "model_used": "small-talk",
        }
        return answer_user_lang, extras

    # 3) Regular medical RAG
    docs = retriever.invoke(q_en)
    answer_en, model_used = ask_mistral_with_context(q_en, docs)

    # 4) Translate back
    answer_user_lang = translate(answer_en, target=src_norm, source="en")
    
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
