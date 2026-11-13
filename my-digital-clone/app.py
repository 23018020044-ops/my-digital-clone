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
    st.error("Please add GROQ_API_KEY to your Streamlit Secrets.")
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
    data_dir = "./data"
    chunks_meta = []
    
    # Read files
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
                    
                    # Store chunk with its metadata
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

    # 1. BM25 Search
    tokenized_query = query.lower().split()
    bm25_scores = bm25_model.get_scores(tokenized_query)
    bm25_ranks = np.argsort(bm25_scores)[::-1]

    # 2. Dense Search (Cosine Similarity)
    query_emb = embedding_model.encode(query, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), dense_embeddings)
    dense_scores = cos_scores.cpu().numpy()
    dense_ranks = np.argsort(dense_scores)[::-1]

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {i: 0 for i in range(len(chunks_meta))}
    k_rrf = 60

    if mode in ["BM25 only", "Hybrid"]:
        for rank, doc_idx in enumerate(bm25_ranks):
            if bm25_scores[doc_idx] > 0:
                rrf_scores[doc_idx] += 1 / (k_rrf + rank)

    if mode in ["Dense only", "Hybrid"]:
        for rank, doc_idx in enumerate(dense_ranks):
            rrf_scores[doc_idx] += 1 / (k_rrf + rank)

    # Sort and get Top K
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
    system
