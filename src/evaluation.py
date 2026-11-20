"""Real RAGAS-style faithfulness metric: measures what fraction of the
answer's claim tokens (genes, diseases, finding-type keywords) are actually
supported by the retrieved source documents, a standard token/claim-overlap
proxy for faithfulness used when no hosted LLM judge is available."""
from __future__ import annotations

import re

FINDING_KEYWORDS = {
    "upregulation": ["upregulated"],
    "downregulation": ["downregulated"],
    "loss_of_function": ["loss-of-function", "mutations"],
    "favorable_prognosis": ["improved", "survival"],
    "poor_prognosis": ["poor", "survival"],
    "therapeutic_target": ["therapeutic", "target"],
}


def _extract_claim_tokens(answer: str, genes: list, diseases: list) -> set:
    tokens = set()
    answer_lower = answer.lower()
    for g in genes:
        if g.lower() in answer_lower:
            tokens.add(("gene", g.lower()))
    for d in diseases:
        if d.lower() in answer_lower:
            tokens.add(("disease", d.lower()))
    for label, kws in FINDING_KEYWORDS.items():
        if any(kw in answer_lower for kw in kws):
            tokens.add(("finding", label))
    return tokens


def faithfulness_score(answer: str, source_texts: list, genes: list, diseases: list) -> float:
    claim_tokens = _extract_claim_tokens(answer, genes, diseases)
    if not claim_tokens:
        return 1.0

    combined_source = " ".join(source_texts).lower()
    supported = 0
    for kind, value in claim_tokens:
        if kind in ("gene", "disease"):
            if value in combined_source:
                supported += 1
        else:
            kws = FINDING_KEYWORDS[value]
            if any(kw in combined_source for kw in kws):
                supported += 1
    return supported / len(claim_tokens)


def evaluate_queries(agent, queries: list) -> dict:
    scores = []
    for q in queries:
        result = agent.answer(q["text"])
        genes = [q.get("gene")] if q.get("gene") else []
        diseases = [q.get("disease")] if q.get("disease") else []
        score = faithfulness_score(result["answer"], result["source_texts"], genes, diseases)
        scores.append(score)
    return {
        "n_queries": len(queries),
        "mean_faithfulness": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "scores": [round(s, 4) for s in scores],
    }
