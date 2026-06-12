"""
utils/preprocessor.py
----------------------
Text preprocessing pipeline for resume and job description analysis.
Handles cleaning, tokenization, stopword removal, and lemmatization.

Author: Resume Screener ML Pipeline
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── Download required NLTK data (runs once) ────────────────────────────────
def download_nltk_resources():
    """Download all required NLTK corpora silently."""
    resources = [
        ("tokenizers/punkt",           "punkt"),
        ("tokenizers/punkt_tab",       "punkt_tab"),
        ("corpora/stopwords",          "stopwords"),
        ("corpora/wordnet",            "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

download_nltk_resources()

# ── Pre-built tech skill lexicon ────────────────────────────────────────────
TECH_SKILLS = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "golang", "rust",
    "scala", "kotlin", "swift", "r", "matlab", "php", "ruby", "perl",

    # ML / AI
    "machine learning", "deep learning", "neural networks", "nlp",
    "natural language processing", "computer vision", "reinforcement learning",
    "transfer learning", "bert", "gpt", "transformers", "llm",

    # ML frameworks
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "xgboost",
    "lightgbm", "catboost", "huggingface", "fastai", "spacy", "nltk",

    # Data
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "apache spark", "hadoop", "kafka", "airflow", "dbt",

    # Cloud / DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "ci/cd", "terraform",
    "jenkins", "github actions", "mlflow", "kubeflow", "sagemaker",

    # Web / APIs
    "flask", "django", "fastapi", "rest api", "graphql", "react", "nodejs",

    # General
    "git", "linux", "bash", "data structures", "algorithms", "system design",
    "statistics", "probability", "linear algebra", "calculus",
    "data analysis", "data visualization", "feature engineering",
    "model deployment", "a/b testing", "agile", "scrum",

    # Soft skills (for completeness)
    "communication", "leadership", "teamwork", "problem solving",
    "critical thinking", "time management",
}

# ── Core preprocessor class ─────────────────────────────────────────────────
class TextPreprocessor:
    """
    End-to-end NLP preprocessing pipeline.

    Steps applied (in order):
      1. Lowercase
      2. URL / email removal
      3. Special character stripping
      4. Extra whitespace collapse
      5. Tokenization
      6. Stopword removal
      7. Lemmatization
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Keep these words even though NLTK marks them as stopwords
        # — they carry meaning in resumes.
        important_words = {"not", "no", "nor", "but", "with", "without"}
        self.stop_words -= important_words

    # ── Internal helpers ────────────────────────────────────────────────────
    def _clean_text(self, text: str) -> str:
        """Remove noise: URLs, emails, special chars, numbers."""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+",    "", text)   # URLs
        text = re.sub(r"\S+@\S+",           "", text)   # emails
        text = re.sub(r"[^a-z\s\+\#\.]",   " ", text)  # keep +, #, .
        text = re.sub(r"\s+",               " ", text).strip()
        return text

    def _tokenize(self, text: str) -> list[str]:
        return word_tokenize(text)

    def _remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def _lemmatize(self, tokens: list[str]) -> list[str]:
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    # ── Public API ──────────────────────────────────────────────────────────
    def preprocess(self, text: str) -> str:
        """
        Full pipeline → returns a clean token string ready for TF-IDF.

        Parameters
        ----------
        text : str  Raw resume / JD text.

        Returns
        -------
        str  Space-joined preprocessed tokens.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        cleaned  = self._clean_text(text)
        tokens   = self._tokenize(cleaned)
        filtered = self._remove_stopwords(tokens)
        lemmas   = self._lemmatize(filtered)
        return " ".join(lemmas)

    def get_tokens(self, text: str) -> list[str]:
        """Return list of preprocessed tokens (useful for skill matching)."""
        return self.preprocess(text).split()


# ── Skill extraction ─────────────────────────────────────────────────────────
class SkillExtractor:
    """
    Extracts skills from text using a curated lexicon + bigram scanning.

    Strategy
    --------
    • Single-word skills: direct set membership after normalisation.
    • Multi-word skills (e.g. "machine learning"): bigram / trigram window scan.
    """

    def __init__(self):
        self.skills_db    = TECH_SKILLS
        # Pre-split multi-word skills for fast lookup
        self._multi_word  = {s for s in self.skills_db if " " in s}
        self._single_word = {s for s in self.skills_db if " " not in s}

    def extract(self, text: str) -> set[str]:
        """
        Extract skills present in *text*.

        Returns
        -------
        set[str]  Lower-cased skill names found in the text.
        """
        if not text:
            return set()

        text_lower = text.lower()
        found: set[str] = set()

        # Multi-word first (exact substring match)
        for skill in self._multi_word:
            if skill in text_lower:
                found.add(skill)

        # Single-word via token scan
        tokens = re.findall(r"\b[\w\+\#\.]+\b", text_lower)
        for token in tokens:
            if token in self._single_word:
                found.add(token)

        return found

    def get_missing_skills(
        self, resume_skills: set[str], jd_skills: set[str]
    ) -> set[str]:
        """Skills required by the JD but absent from the resume."""
        return jd_skills - resume_skills

    def get_matching_skills(
        self, resume_skills: set[str], jd_skills: set[str]
    ) -> set[str]:
        """Skills that appear in both the resume and the JD."""
        return resume_skills & jd_skills
