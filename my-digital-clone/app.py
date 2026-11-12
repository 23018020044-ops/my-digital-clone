import os
import glob
import re
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- CONFIGURATION ---
YOUR_NAME = "Saher Nazir"
YOUR_DESC = "a CS student passionate about AI and software engineering"
CHUNK_SIZE = 250
OVERLAP = 40
TOP_K = 4

st.set_page_config(page_title=f"{YOUR_NAME}'s Digital Clone", page_icon="🤖", layout="centered")

# --- HELPER: TOPIC-STYLE PASSAGE PREVIEW (NON-VERBATIM & NEVER EMPTY) ---
def clean_snippet(text, source=""):
    cleaned = re.sub(r"[#*`_>-]+", " ", text)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)
    words = cleaned.split()

    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "have", "are",
        "was", "were", "been", "being", "into", "about", "what", "when",
        "where", "which", "who", "whom", "whose", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "nor",
        "not", "only", "own", "same", "than", "too", "very", "can", "will",
        "just", "should", "now", "also", "like", "love", "enjoy", "prefer",
        "work", "working", "thing", "things", "really", "often", "usually"
    }

    keywords = []
    for w in words:
        lw = w.lower()
        if len(lw) > 3 and lw not in stopwords and lw not in [k.lower() for k in keywords]:
            keywords.append(w.capitalize())
        if len(keywords) == 5:
            break

    topic_name = source.replace(".md", "").replace(".txt", "").replace("_", " ").title()
    if keywords:
        return f"{topic_name} — Key themes: {', '.join(keywords)}..."
    return f"Relevant notes extracted from {topic_name} section."

# --- 1. LOAD & CHUNK ---
def load_and_chunk(data_dir="data"):
    chunks = []
    files = sorted(glob.glob(os.path.join(data_dir, "*.*")))
    for filepath in files:
        if not filepath.endswith((".txt", ".md")):
            continue
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            words = f.read().split()
        
        step = CHUNK_SIZE - OVERLAP
        chunk_id = 0
        for i in range(0, len(words), step):
            chunk_words = words[i:i + CHUNK_SIZE]
            if len(chunk_words) < 20:
                continue
            chunks.append({
                "source": filename,
                "chunk_id": f"#{chunk_id}",
                "text": " ".join(chunk_words)
            })
            chunk_id += 1
    return chunks

# --- 2. CACHE MODELS & INDEXES ---
@st.cache_resource
def init_retriever():
    chunks = load_and_chunk()
    tokenized_corpus = [re.findall(r"\w+", c["text"].lower()) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = np.load("embeddings.npy")
    return chunks, bm25, embed_model, doc_embeddings

chunks, bm25, embed_model, doc_embeddings = init_retriever()

# --- 3. RETRIEVAL & HYBRID FUSION (RRF) ---
def retrieve(query, mode="Hybrid", top_k=TOP_K):
    tokens = re.findall(r"\w+", query.lower())
    bm25_scores = bm25.get_scores(tokens)
    bm25_ranked = np.argsort(bm25_scores)[::-1]

    q_emb = embed_model.encode([query], normalize_embeddings=True)[0]
    dense_scores = np.dot(doc_embeddings, q_emb)
    dense_ranked = np.argsort(dense_scores)[::-1]

    results = []
    if mode == "BM25 only":
        for idx in bm25_ranked[:top_k]:
            results.append((chunks[idx], float(bm25_scores[idx])))

    elif mode == "Dense only":
        for idx in dense_ranked[:top_k]:
            results.append((chunks[idx], float(dense_scores[idx])))

    else:  # Hybrid (Reciprocal Rank Fusion: 1 / (60 + rank))
        rrf_scores = {}
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(dense_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank + 1)

        hybrid_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for idx, score in hybrid_sorted:
            results.append((chunks[idx], float(score)))

    return results

# --- 4. GENERATION (GROQ) ---
def generate_answer(question, retrieved):
    context_blocks = []
    for c, _ in retrieved:
        context_blocks.append(f"[{c['source']} | {c['chunk_id']}]\n{c['text']}")
    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are the AI clone of {YOUR_NAME}, {YOUR_DESC}.
Speak in the first person, in a friendly and natural tone.

Rules:
1. Answer ONLY using the CONTEXT below. It comes from my own documents.
2. If the context does not contain the answer, say: "I don't have that in my notes." Do not guess or invent facts about me.
3. Never share phone numbers, addresses, ID numbers or other private details.
4. If asked whether you are the real person, say you are an AI clone of {YOUR_NAME}.

CONTEXT:
{context_str}

QUESTION:
{question}"""

    groq_api_key = ""
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = os.environ.get("GROQ_API_KEY", "")

    if not groq_api_key:
        return "⚠️ Error: GROQ_API_KEY secret is missing.", "None"

    try:
        client = Groq(api_key=groq_api_key)
        available = [
            m.id for m in client.models.list().data
            if "whisper" not in m.id.lower() and "guard" not in m.id.lower() and "tts" not in m.id.lower()
        ]
        preferred = ["openai/gpt-oss-20b", "llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3-32b"]
        models_to_try = [m for m in preferred if m in available] + [m for m in available if m not in preferred]

        last_err = ""
        for model_id in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=350,
                    temperature=0.3
                )
                return response.choices[0].message.content, model_id
            except Exception as e:
                last_err = str(e)
                continue
        return f"API Error: {last_err}", "None"
    except Exception as e:
        return f"API Error: {str(e)}", "None"

# --- 5. STREAMLIT CHAT UI ---
st.title(f"🤖 {YOUR_NAME}'s Digital Clone")
st.caption("⚠️ Disclaimer: This is an AI clone, not the real person.")

with st.sidebar:
    st.header("⚙️ Retrieval Settings")
    mode = st.radio("Retrieval Mode:", ["Hybrid", "BM25 only", "Dense only"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "model" in msg:
            st.caption(f"⚡ Groq ({msg['model']})")
            with st.expander("🔍 Retrieved Knowledge Sources & Scores", expanded=False):
                st.dataframe(msg["sources_df"], use_container_width=True, hide_index=True)

if prompt_input := st.chat_input("Ask me anything about my projects, research, experience, skills, hobbies..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.write(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved = retrieve(prompt_input, mode=mode, top_k=TOP_K)
            answer, used_model = generate_answer(prompt_input, retrieved)

        st.write(answer)
        st.caption(f"⚡ Groq ({used_model})")

        table_rows = []
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            table_rows.append({
                "Rank": f"#{rank}",
                "Source File": chunk["source"],
                "Chunk ID": chunk["chunk_id"],
                "RRF Score": f"{score:.4f}",
                "Passage Snippet": clean_snippet(chunk["text"], chunk["source"])
            })
        sources_df = pd.DataFrame(table_rows)

        with st.expander("🔍 Retrieved Knowledge Sources & Scores", expanded=True):
            st.dataframe(sources_df, use_container_width=True, hide_index=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "model": used_model,
        "sources_df": sources_df
    })
