"""Real vector store: sentence-transformers embeddings + FAISS indexing."""
from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, model_name: str = _MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.docs = []

    def build(self, docs: list):
        self.docs = docs
        texts = [d["text"] for d in docs]
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        return embeddings

    def search(self, query: str, k: int = 5):
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        scores, idx = self.index.search(q_emb, k)
        results = [(self.docs[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        return results
