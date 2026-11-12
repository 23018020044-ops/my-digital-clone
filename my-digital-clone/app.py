import os
import glob
import re
import numpy as np
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

st.set_page_config(page_title=f"{YOUR_NAME}'s Digital Clone", page_icon="🤖", layout="wide")

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
                "chunk_id": f"{filename}_chunk_{chunk_id}",
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
        src = c["source"]
        cid = c["chunk_id"]
        txt = c["text"]
        context_blocks.append(f"[{src} | {cid}]\n{txt}")

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
        return "⚠️ Error: GROQ_API_KEY secret is missing. Please add it in Streamlit App Settings -> Secrets."

    try:
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"

# --- 5. STREAMLIT UI ---
st.title(f"🤖 Chat with {YOUR_NAME}'s Digital Clone")
st.warning("⚠️ **Disclaimer: This is an AI clone, not the real person.**")

col1, col2 = st.columns([1.2, 1])

with col1:
    question = st.text_area("Ask me anything about myself:", placeholder="e.g., What projects have you worked on?", height=100)
    mode = st.radio("Select Retrieval Mode:", ["Hybrid", "BM25 only", "Dense only"], horizontal=True)
    ask_btn = st.button("Ask Clone", type="primary")

    if ask_btn:
        if not question.strip():
            st.error("Please enter a question first.")
        else:
            with st.spinner("Searching notes & thinking..."):
                retrieved = retrieve(question, mode=mode, top_k=TOP_K)
                answer = generate_answer(question, retrieved)
            st.subheader("Clone's Answer")
            st.info(answer)

with col2:
    st.subheader("📄 Retrieved Chunks (Sources & Scores)")
    if ask_btn and question.strip():
        for chunk, score in retrieved:
            with st.expander(f"Source: {chunk['source']} | ID: {chunk['chunk_id']} | Score: {score:.4f}", expanded=True):
                st.write(chunk["text"])
    else:
        st.caption("Ask a question to see the retrieved passages, sources, and scores here.")
