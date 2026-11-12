import os
import glob
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 250   # 200 to 300 words required
OVERLAP = 40       # Small overlap

def load_and_chunk(data_dir="data"):
    chunks = []
    files = sorted(glob.glob(os.path.join(data_dir, "*.*")))
    for filepath in files:
        if not filepath.endswith((".txt", ".md")):
            continue
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
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

if __name__ == "__main__":
    chunks = load_and_chunk()
    print(f"Total chunks created: {len(chunks)}")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    np.save("embeddings.npy", embeddings)
    print("Saved embeddings.npy successfully!")