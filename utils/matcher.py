"""
utils/matcher.py
-----------------
Core ML module: TF-IDF vectorisation + cosine similarity matching.

Pipeline
--------
  raw text → TextPreprocessor → TfidfVectorizer → cosine_similarity → score

The matcher is *stateless*: call `match()` with any resume / JD pair
without fitting on a training corpus (unsupervised, always fresh).

Author: Resume Screener ML Pipeline
"""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

from utils.preprocessor import TextPreprocessor, SkillExtractor

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/tfidf_vectorizer.joblib")

# TF-IDF hyper-parameters (tuned for short resume-length documents)
TFIDF_CONFIG = dict(
    ngram_range=(1, 2),   # unigrams + bigrams
    max_features=10_000,
    sublinear_tf=True,    # replace TF with 1 + log(TF) — smooths high freq words
    min_df=1,             # keep rare terms (important in small corpora)
    analyzer="word",
    token_pattern=r"(?u)\b[\w\+\#\.]{2,}\b",
)


# ── ResumeJobMatcher ─────────────────────────────────────────────────────────
class ResumeJobMatcher:
    """
    Computes a match score between a resume and a job description.

    Usage
    -----
    >>> matcher = ResumeJobMatcher()
    >>> result  = matcher.match(resume_text, jd_text)
    >>> print(result["match_score"])  # 0–100 float
    """

    def __init__(self):
        self.preprocessor    = TextPreprocessor()
        self.skill_extractor = SkillExtractor()
        self.vectorizer      = TfidfVectorizer(**TFIDF_CONFIG)

    # ── Public API ──────────────────────────────────────────────────────────
    def match(self, resume_text: str, jd_text: str) -> dict:
        """
        Compute full match analysis between one resume and one JD.

        Returns
        -------
        dict with keys:
          match_score      float  0-100
          matched_skills   set[str]
          missing_skills   set[str]
          resume_skills    set[str]
          jd_skills        set[str]
          tfidf_score      float  raw cosine similarity
          recommendation   str    "Highly Recommended" | "Recommended" | "Not Recommended"
        """
        # 1. Preprocess
        clean_resume = self.preprocessor.preprocess(resume_text)
        clean_jd     = self.preprocessor.preprocess(jd_text)

        # 2. TF-IDF vectorise both documents together
        #    (fit on [resume, jd] so the vocabulary is shared)
        tfidf_matrix = self.vectorizer.fit_transform([clean_resume, clean_jd])

        # 3. Cosine similarity between resume vector (row 0) and JD vector (row 1)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # 4. Skill extraction (on raw text — before heavy preprocessing)
        resume_skills  = self.skill_extractor.extract(resume_text)
        jd_skills      = self.skill_extractor.extract(jd_text)
        matched_skills = self.skill_extractor.get_matching_skills(resume_skills, jd_skills)
        missing_skills = self.skill_extractor.get_missing_skills(resume_skills, jd_skills)

        # 5. Composite score
        #    70 % TF-IDF cosine  +  30 % skill-overlap ratio
        skill_score = (
            len(matched_skills) / len(jd_skills) if jd_skills else 0.0
        )
        composite_raw  = 0.70 * similarity + 0.30 * skill_score
        match_score    = round(min(composite_raw * 100, 100.0), 2)

        # 6. Recommendation tier
        if match_score >= 70:
            recommendation = "✅ Highly Recommended"
        elif match_score >= 45:
            recommendation = "⚠️ Recommended"
        else:
            recommendation = "❌ Not Recommended"

        return {
            "match_score":    match_score,
            "tfidf_score":    round(float(similarity), 4),
            "skill_score":    round(skill_score * 100, 2),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "resume_skills":  resume_skills,
            "jd_skills":      jd_skills,
            "recommendation": recommendation,
        }

    def rank_candidates(
        self, resumes: list[dict[str, str]], jd_text: str
    ) -> list[dict]:
        """
        Rank multiple resumes against one JD.

        Parameters
        ----------
        resumes  : list of {"name": str, "text": str}
        jd_text  : str   Raw job description text.

        Returns
        -------
        list[dict]  Sorted by match_score descending.
                    Each dict = match() result + "name" + "rank".
        """
        results = []
        for resume in resumes:
            result         = self.match(resume["text"], jd_text)
            result["name"] = resume["name"]
            results.append(result)

        # Sort descending by match_score
        results.sort(key=lambda x: x["match_score"], reverse=True)

        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

    # ── Persistence helpers ─────────────────────────────────────────────────
    def save_vectorizer(self, path: str = str(MODEL_PATH)) -> None:
        """Persist the fitted TF-IDF vectoriser to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)

    def load_vectorizer(self, path: str = str(MODEL_PATH)) -> None:
        """Load a previously fitted TF-IDF vectoriser from disk."""
        if Path(path).exists():
            self.vectorizer = joblib.load(path)
        else:
            raise FileNotFoundError(f"No saved vectorizer at {path}")
