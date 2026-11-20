"""Simulate 800+ synthetic biomedical-publication abstracts.

These are NOT real PubMed records. Each abstract is built from a template
combining a gene, a disease, and a finding-type, so that every abstract has
a known, extractable ground-truth claim the retrieval/synthesis pipeline
can later be checked against honestly.
"""
from __future__ import annotations

import numpy as np

N_ABSTRACTS = 820
SEED = 303

GENES = ["TP53", "BRCA1", "EGFR", "KRAS", "PTEN", "MYC", "BRAF", "ALK", "PIK3CA", "NOTCH1"]
DISEASES = ["breast cancer", "lung adenocarcinoma", "colorectal cancer", "melanoma",
            "pancreatic cancer", "glioblastoma", "ovarian cancer", "leukemia"]
FINDINGS = [
    ("is significantly upregulated in", "upregulation"),
    ("is significantly downregulated in", "downregulation"),
    ("harbors recurrent loss-of-function mutations in", "loss_of_function"),
    ("is associated with improved survival outcomes in patients with", "favorable_prognosis"),
    ("is associated with poor survival outcomes in patients with", "poor_prognosis"),
    ("is a validated therapeutic target in", "therapeutic_target"),
]


def simulate(n: int = N_ABSTRACTS, seed: int = SEED):
    rng = np.random.default_rng(seed)
    abstracts = []
    for i in range(n):
        gene = rng.choice(GENES)
        disease = rng.choice(DISEASES)
        finding_text, finding_label = FINDINGS[rng.integers(0, len(FINDINGS))]
        n_sample = rng.integers(40, 500)

        abstract = (
            f"In a cohort of {n_sample} patients, we investigated the role of {gene} "
            f"in {disease}. Gene expression and mutation profiling revealed that {gene} "
            f"{finding_text} {disease}. These findings were validated by independent "
            f"replication and suggest that {gene} may serve as a biomarker in {disease}. "
            f"Further functional studies are warranted to confirm the mechanistic role of "
            f"{gene} in disease progression."
        )
        abstracts.append({
            "doc_id": f"ABS{i:04d}",
            "gene": gene,
            "disease": disease,
            "finding_label": finding_label,
            "n_sample": int(n_sample),
            "text": abstract,
        })
    return abstracts
