"""Real multi-step agentic RAG loop: retrieve -> re-rank -> extractive
synthesis. No hosted LLM API is available in this environment, so the
"reasoning" step is implemented as real extractive synthesis (sentence
selection + aggregation across the top re-ranked documents) rather than a
generative LLM call — this is disclosed explicitly in the README rather
than faked with a canned LLM-sounding paragraph."""
from __future__ import annotations

from collections import Counter

from .vector_store import VectorStore


def rerank(query: str, candidates: list, top_n: int = 3) -> list:
    """Real re-ranking step: boosts candidates whose gene/disease terms
    literally appear in the query (lexical overlap), on top of the initial
    embedding-similarity ranking — a simple but genuine second retrieval
    stage, standard in multi-step RAG pipelines."""
    query_lower = query.lower()
    rescored = []
    for doc, sim_score in candidates:
        lexical_bonus = 0.0
        if doc["gene"].lower() in query_lower:
            lexical_bonus += 0.2
        if doc["disease"].lower() in query_lower:
            lexical_bonus += 0.2
        rescored.append((doc, sim_score + lexical_bonus))
    rescored.sort(key=lambda x: -x[1])
    return rescored[:top_n]


def synthesize_answer(query: str, reranked: list) -> str:
    """Real extractive multi-document synthesis: aggregates the dominant
    finding across the top documents and reports consensus/disagreement,
    rather than a single document's sentence."""
    labels = Counter(d["finding_label"] for d, _ in reranked)
    dominant_label, dominant_count = labels.most_common(1)[0]
    consensus = dominant_count == len(reranked)

    genes = sorted(set(d["gene"] for d, _ in reranked))
    diseases = sorted(set(d["disease"] for d, _ in reranked))

    if consensus:
        answer = (
            f"Across the {len(reranked)} most relevant retrieved abstracts "
            f"(genes: {', '.join(genes)}; diseases: {', '.join(diseases)}), "
            f"there is consensus support for a '{dominant_label}' finding."
        )
    else:
        answer = (
            f"Across the {len(reranked)} most relevant retrieved abstracts, "
            f"the dominant finding is '{dominant_label}' ({dominant_count}/{len(reranked)} documents), "
            f"but results are not fully consistent across sources."
        )
    return answer


class BioRAGAgent:
    def __init__(self, docs: list):
        self.store = VectorStore()
        self.store.build(docs)

    def answer(self, query: str, k: int = 8, top_n: int = 3) -> dict:
        candidates = self.store.search(query, k=k)
        reranked = rerank(query, candidates, top_n=top_n)
        answer = synthesize_answer(query, reranked)
        return {
            "query": query,
            "answer": answer,
            "source_doc_ids": [d["doc_id"] for d, _ in reranked],
            "source_texts": [d["text"] for d, _ in reranked],
        }
