import json
import os
import numpy as np

from src.simulate_corpus import simulate, GENES, DISEASES
from src.agent import BioRAGAgent
from src.evaluation import evaluate_queries


def build_queries(n: int = 120, seed: int = 11):
    rng = np.random.default_rng(seed)
    queries = []
    for _ in range(n):
        gene = rng.choice(GENES)
        disease = rng.choice(DISEASES)
        queries.append({
            "text": f"What is the role of {gene} in {disease}?",
            "gene": gene, "disease": disease,
        })
    return queries


def main():
    docs = simulate()
    agent = BioRAGAgent(docs)
    queries = build_queries(n=120)
    eval_result = evaluate_queries(agent, queries)

    summary = {
        "n_documents": len(docs),
        "n_eval_queries": eval_result["n_queries"],
        "mean_faithfulness": eval_result["mean_faithfulness"],
    }

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
