import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulate_corpus import simulate
from src.agent import BioRAGAgent
from src.evaluation import evaluate_queries, faithfulness_score


def test_at_least_800_documents():
    docs = simulate()
    assert len(docs) >= 800


def test_agent_retrieves_relevant_docs():
    docs = simulate(n=200)
    agent = BioRAGAgent(docs)
    result = agent.answer("What is the role of TP53 in breast cancer?")
    assert len(result["source_doc_ids"]) > 0
    assert any(d["gene"] == "TP53" for d in [doc for doc in docs if doc["doc_id"] in result["source_doc_ids"]])


def test_answer_mentions_queried_gene():
    docs = simulate(n=400)
    agent = BioRAGAgent(docs)
    result = agent.answer("What is the role of EGFR in lung adenocarcinoma?")
    assert "EGFR" in result["answer"] or any("EGFR" in t for t in result["source_texts"])


def test_faithfulness_score_is_bounded():
    docs = simulate(n=100)
    agent = BioRAGAgent(docs)
    result = agent.answer("What is the role of KRAS in colorectal cancer?")
    score = faithfulness_score(result["answer"], result["source_texts"], ["KRAS"], ["colorectal cancer"])
    assert 0.0 <= score <= 1.0


def test_evaluate_queries_runs_end_to_end():
    docs = simulate(n=150)
    agent = BioRAGAgent(docs)
    queries = [{"text": "What is the role of BRAF in melanoma?", "gene": "BRAF", "disease": "melanoma"}]
    result = evaluate_queries(agent, queries)
    assert result["n_queries"] == 1
    assert "mean_faithfulness" in result
