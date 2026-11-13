import streamlit as st
import os
import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Saher's Digital Clone", page_icon="🤖")

# Groq API Key setup (from Streamlit Secrets)
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("⚠️ Please add GROQ_API_KEY to your Streamlit Secrets.")
    st.stop()

# ==========================================
# 2. CACHING MODELS & DATA
# ==========================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

@st.cache_data
def process_knowledge_base():
    # Bulletproof dynamic path for cloud deployment
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    chunks_meta = []
    
    if not os.path.exists(data_dir):
        return [], None, []
        
    for filename in os.listdir(data_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                
                # Simple Word-based Chunking (Size: 250, Overlap: 40)
                words = text.split()
                chunk_size = 250
                overlap = 40
                
                chunk_id = 0
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[i : i + chunk_size]
                    if not chunk_words:
                        break
                    chunk_text = " ".join(chunk_words)
                    
                    chunks_meta.append({
                        "text": chunk_text,
                        "file": filename,
                        "chunk_id": f"#{chunk_id}"
                    })
                    chunk_id += 1

    if not chunks_meta:
        return [], None, []

    # Prepare BM25
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks_meta]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Prepare Dense Embeddings
    corpus_texts = [chunk["text"] for chunk in chunks_meta]
    dense_embeddings = embedding_model.encode(corpus_texts, convert_to_tensor=True)
    
    return chunks_meta, bm25, dense_embeddings

chunks_meta, bm25_model, dense_embeddings = process_knowledge_base()

# ==========================================
# 3. HYBRID RAG & RETRIEVAL LOGIC
# ==========================================
def retrieve(query, mode="Hybrid", top_k=3):
    if not chunks_meta:
        return []

    # BM25 Search
    tokenized_query = query.lower().split()
    bm25_scores = bm25_model.get_scores(tokenized_query)
    bm25_ranks = np.argsort(bm25_scores)[::-1]

    # Dense Search
    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), dense_embeddings)
    dense_scores = cos_scores.cpu().numpy()
    dense_ranks = np.argsort(dense_scores)[::-1]

    # Reciprocal Rank Fusion
    rrf_scores = {i: 0 for i in range(len(chunks_meta))}
    k_rrf = 60

    if mode in ["BM25 only", "Hybrid"]:
        for rank, doc_idx in enumerate(bm25_ranks):
            if bm25_scores[doc_idx] > 0:
                rrf_scores[doc_idx] += 1 / (k_rrf + rank)

    if mode in ["Dense only", "Hybrid"]:
        for rank, doc_idx in enumerate(dense_ranks):
            rrf_scores[doc_idx] += 1 / (k_rrf + rank)

    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    
    results = []
    for idx in sorted_indices:
        if rrf_scores[idx] > 0:
            results.append((chunks_meta[idx], rrf_scores[idx]))
            
    return results

# ==========================================
# 4. LLM GENERATION & GUARDRAILS
# ==========================================
def generate_response(query, context):
    system_prompt = (
        "You are Saher Nazir's AI Digital Clone. Answer questions accurately based ONLY on the provided context. "
        "If the context does not contain the answer, say EXACTLY: 'I don't have that in my notes.' "
        "Do not hallucinate, invent facts, or answer questions outside the scope of the context."
    )
    
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b", # <-- Image wale model ka exact naam
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"

# ==========================================
# 5. STREAMLIT UI & CHAT INTERFACE
# ==========================================
# Sidebar
with st.sidebar:
    st.header("⚙️ RAG Settings")
    selected_mode = st.radio("Select Retrieval Mode:", ["Hybrid", "BM25 only", "Dense only"])
    if not chunks_meta:
        st.error("⚠️ Data folder not found or empty.")
    else:
        st.caption("Loaded Chunks: " + str(len(chunks_meta)))

# Add Welcome Message to prevent blank screen
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi! I am Saher Nazir's AI Digital Clone. Ask me anything about my projects, research, experience, skills, or hobbies!"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            st.caption("⚡ Groq (openai/gpt-oss-20b)") # <-- UI mein show karne ke liye update kar diya hai
            with st.expander("🔍 Retrieved Knowledge Sources & Scores"):
                st.table(message["sources"])

# Input box
prompt = st.chat_input("Ask me anything...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        retrieved_items = retrieve(prompt, mode=selected_mode, top_k=3)
        context_text = "\n\n".join([item[0]["text"] for item in retrieved_items])
        
        answer = generate_response(prompt, context_text)
        
        table_data = []
        for i, (chunk_dict, score) in enumerate(retrieved_items):
            raw_text = chunk_dict["text"].replace("#", "").replace("*", "").replace("`", "").replace("-", "")
            clean_text = re.sub(r'\s+', ' ', raw_text).strip()
            snippet = " ".join(clean_text.split()[:15]) + "..."
            
            table_data.append({
                "Rank": f"#{i+1}",
                "Source File": chunk_dict["file"],
                "Chunk ID": chunk_dict["chunk_id"],
                "RRF Score": f"{score:.4f}",
                "Passage Snippet": snippet
            })

    with st.chat_message("assistant"):
        st.markdown(answer)
        if table_data:
            st.caption("⚡ Groq (openai/gpt-oss-20b)") # <-- Yahan bhi same name 
            with st.expander("🔍 Retrieved Knowledge Sources & Scores"):
                st.table(table_data)
                
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": table_data if table_data else []
    })
