"""
train.py
---------
Offline training script.

What it does
------------
1. Loads sample resume + JD pairs from dataset/sample_data.csv
2. Preprocesses text with TextPreprocessor
3. Fits a TF-IDF vectoriser on the combined corpus
4. Saves the fitted vectoriser to models/tfidf_vectorizer.joblib
5. Evaluates similarity on the training pairs and prints a report

Run this before the Streamlit app to pre-fit the vectoriser on your
domain corpus. The app also works without it (fits on-the-fly).

Usage
-----
    python train.py

Author: Resume Screener ML Pipeline
"""

import os
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.preprocessor import TextPreprocessor, SkillExtractor

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("dataset/sample_data.csv")
MODEL_DIR  = Path("models")
MODEL_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
EVAL_PATH  = MODEL_DIR / "eval_results.json"


# ── Sample data (fallback if CSV not found) ───────────────────────────────────
SAMPLE_PAIRS = [
    {
        "resume": """
        John Doe — Software Engineer
        Skills: Python, Machine Learning, Deep Learning, TensorFlow, PyTorch,
        scikit-learn, SQL, Pandas, NumPy, Git, Docker, AWS.
        Experience: 2 years internship at tech startup, built NLP pipeline for
        sentiment analysis, deployed model to AWS SageMaker.
        Education: B.Tech Computer Science, 8.5 CGPA.
        """,
        "jd": """
        ML Engineer Position — Amazon
        Required: Python, Machine Learning, TensorFlow, PyTorch, scikit-learn,
        Deep Learning, NLP, SQL, AWS, Docker, Git.
        Preferred: Kubernetes, Spark, MLflow.
        """,
        "label": 0.85,
    },
    {
        "resume": """
        Jane Smith — Data Analyst
        Skills: Excel, SQL, Tableau, Power BI, basic Python, statistics.
        Experience: 1 year data entry, 6 months as analyst intern.
        Education: B.Com, certificate in data analytics.
        """,
        "jd": """
        Senior ML Engineer — Amazon
        Required: Python, Machine Learning, TensorFlow, PyTorch, scikit-learn,
        Deep Learning, NLP, SQL, AWS, Docker, Kubernetes, Spark.
        5+ years experience required.
        """,
        "label": 0.20,
    },
    {
        "resume": """
        Alice Wang — NLP Researcher
        Skills: Python, NLP, BERT, Transformers, HuggingFace, spaCy, NLTK,
        PyTorch, TensorFlow, scikit-learn, Git, Linux, Docker.
        Publications: 2 NLP papers. Experience: Research assistant 2 years.
        Education: M.Tech AI, 9.1 CGPA.
        """,
        "jd": """
        Applied Scientist — NLP — Amazon Alexa
        Required: Python, NLP, Transformers, BERT, PyTorch, scikit-learn,
        HuggingFace, Machine Learning, statistical modelling.
        Preferred: AWS, Kubernetes, publication record.
        """,
        "label": 0.90,
    },
    {
        "resume": """
        Bob Johnson — Backend Developer
        Skills: Java, Spring Boot, REST API, MySQL, PostgreSQL, Redis,
        Docker, Kubernetes, Jenkins, Git, Linux.
        Experience: 3 years backend at fintech company.
        """,
        "jd": """
        Data Scientist — Amazon
        Required: Python, Machine Learning, Statistics, scikit-learn,
        Pandas, SQL, A/B testing, hypothesis testing.
        Preferred: R, Spark, AWS.
        """,
        "label": 0.25,
    },
    {
        "resume": """
        Carol Chen — Data Scientist
        Skills: Python, R, scikit-learn, Machine Learning, Statistics,
        Pandas, NumPy, matplotlib, seaborn, SQL, Tableau, Git, AWS.
        Experience: 1.5 years DS at e-commerce company, built recommendation engine.
        A/B testing, hypothesis testing. Kaggle Expert (top 5% globally).
        """,
        "jd": """
        Data Scientist — Amazon
        Required: Python, Machine Learning, Statistics, scikit-learn,
        Pandas, SQL, A/B testing, hypothesis testing.
        Preferred: R, Spark, AWS, recommendation systems.
        """,
        "label": 0.88,
    },
]


# ── Helper functions ──────────────────────────────────────────────────────────
def load_data() -> list[dict]:
    """Load dataset from CSV or fall back to built-in samples."""
    if DATA_PATH.exists():
        log.info("Loading data from %s", DATA_PATH)
        df = pd.read_csv(DATA_PATH)
        pairs = df[["resume", "jd", "label"]].dropna().to_dict("records")
        log.info("Loaded %d pairs from CSV", len(pairs))
        return pairs
    else:
        log.warning("dataset/sample_data.csv not found — using built-in samples.")
        return SAMPLE_PAIRS


def preprocess_corpus(pairs: list[dict]) -> tuple[list[str], list[str]]:
    """
    Preprocess all resume and JD texts.

    Returns
    -------
    (resume_texts, jd_texts) — both preprocessed.
    """
    preprocessor = TextPreprocessor()
    resumes = [preprocessor.preprocess(p["resume"]) for p in pairs]
    jds     = [preprocessor.preprocess(p["jd"])     for p in pairs]
    return resumes, jds


def fit_vectorizer(corpus: list[str]) -> TfidfVectorizer:
    """Fit TF-IDF on the combined corpus."""
    vectorizer = TfidfVectorizer(
        ngram_range  = (1, 2),
        max_features = 10_000,
        sublinear_tf = True,
        min_df       = 1,
        analyzer     = "word",
        token_pattern= r"(?u)\b[\w\+\#\.]{2,}\b",
    )
    vectorizer.fit(corpus)
    log.info("Fitted TF-IDF on %d documents | vocab size: %d",
             len(corpus), len(vectorizer.vocabulary_))
    return vectorizer


def evaluate(
    vectorizer: TfidfVectorizer,
    resumes: list[str],
    jds:     list[str],
    labels:  list[float],
) -> dict:
    """
    Compute cosine similarity for each (resume, jd) pair and compare with labels.

    Returns evaluation metrics dict.
    """
    mae_list, mse_list = [], []
    predictions = []

    for r_text, jd_text, label in zip(resumes, jds, labels):
        vecs  = vectorizer.transform([r_text, jd_text])
        sim   = float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])
        error = abs(sim - label)
        mae_list.append(error)
        mse_list.append(error ** 2)
        predictions.append(sim)

    metrics = {
        "MAE":  round(float(np.mean(mae_list)), 4),
        "RMSE": round(float(np.sqrt(np.mean(mse_list))), 4),
        "mean_predicted": round(float(np.mean(predictions)), 4),
        "mean_label":     round(float(np.mean(labels)), 4),
        "n_samples":      len(labels),
    }
    return metrics, predictions


def save_model(vectorizer: TfidfVectorizer) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODEL_PATH)
    log.info("Saved vectorizer → %s", MODEL_PATH)


def save_eval(metrics: dict) -> None:
    with open(EVAL_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved eval metrics → %s", EVAL_PATH)


def create_sample_csv() -> None:
    """Write sample_data.csv so users can see the expected format."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        rows = [{"resume": p["resume"], "jd": p["jd"], "label": p["label"]}
                for p in SAMPLE_PAIRS]
        pd.DataFrame(rows).to_csv(DATA_PATH, index=False)
        log.info("Created sample dataset → %s", DATA_PATH)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 55)
    log.info("  Resume Screener — Training Pipeline")
    log.info("=" * 55)

    # 1. Create sample CSV if missing
    create_sample_csv()

    # 2. Load data
    pairs  = load_data()
    labels = [p.get("label", 0.5) for p in pairs]

    # 3. Preprocess
    log.info("Preprocessing %d resume-JD pairs …", len(pairs))
    resumes, jds = preprocess_corpus(pairs)

    # 4. Fit TF-IDF on the entire corpus (resumes + JDs)
    combined_corpus = resumes + jds
    vectorizer = fit_vectorizer(combined_corpus)

    # 5. Evaluate
    log.info("Evaluating …")
    metrics, preds = evaluate(vectorizer, resumes, jds, labels)
    log.info("Evaluation results:")
    for k, v in metrics.items():
        log.info("  %-22s %s", k, v)

    # 6. Save
    save_model(vectorizer)
    save_eval(metrics)

    log.info("=" * 55)
    log.info("Training complete!  Run:  streamlit run app.py")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
