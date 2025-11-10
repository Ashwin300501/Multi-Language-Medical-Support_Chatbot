import os
from dotenv import load_dotenv

from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from mistralai import Mistral

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

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

def ask_mistral_with_context(question_en: str, docs) -> str:
    context = _format_context(docs)
    with Mistral(api_key=MISTRAL_API_KEY) as client:
        resp = client.chat.complete(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {question_en}\n\nContext:\n{context}\n\nAnswer:"
                },
            ],
            stream=False,
        )
    return resp.choices[0].message.content.strip()

def answer_user_query(user_text: str, k: int = 3):
    # 1) Detect & translate to English
    src_lang = detect_lang(user_text)
    src_norm = normalize_lang(src_lang)
    q_en = translate(user_text, target="en", source=src_norm)

    # 2) Retrieve top-k context
    docs = retriever.invoke(q_en)

    # 3) LLM answer in English
    answer_en = ask_mistral_with_context(q_en, docs)

    # 4) Translate back to user's language
    answer_user_lang = translate(answer_en, target=src_norm, source="en")

    # 5) Minimal sources for UI
    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "preview": (d.page_content[:180].replace("\n", " ") + ("..." if len(d.page_content) > 180 else ""))
        }
        for d in docs
    ]
    return answer_user_lang, sources

if __name__ == "__main__":
    sample_q = "what is hypertension?"
    ans, srcs = answer_user_query(sample_q)
    print("Answer:\n", ans)
    print("\nSources:")
    for s in srcs:
        print("-", s["source"], ":", s["preview"])