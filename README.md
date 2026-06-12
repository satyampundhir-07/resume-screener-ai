# 🎯 Intelligent Resume Screening & Job Matching System

> **AI-powered NLP system that analyses resumes against job descriptions, ranks candidates, extracts skills, and generates detailed match reports.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Project Architecture](#project-architecture)
4. [ML Pipeline](#ml-pipeline)
5. [Quick Start](#quick-start)
6. [Usage Guide](#usage-guide)
7. [Deployment](#deployment)
8. [Dataset](#dataset)
9. [Interview Q&A](#interview-qa)
10. [Resume Bullets](#resume-bullets)

---

## Overview

This system solves a real-world HR problem: **manually reviewing hundreds of resumes is slow and subjective**. The pipeline uses NLP and unsupervised machine learning to:

- **Extract** skills from resumes and job descriptions automatically  
- **Score** every resume against a JD using TF-IDF + Cosine Similarity  
- **Rank** multiple candidates from most to least suitable  
- **Identify** skill gaps so candidates know what to learn  
- **Generate** downloadable PDF reports for HR teams  

---

## Features

| Feature | Description |
|---------|-------------|
| 📄 PDF / DOCX / TXT Upload | Multi-format resume & JD ingestion via PyMuPDF |
| 🧹 NLP Preprocessing | Lowercase → URL strip → tokenise → stopword removal → lemmatise |
| 🔢 TF-IDF Vectorisation | `ngram_range=(1,2)`, `sublinear_tf=True`, 10k features |
| 📐 Cosine Similarity | Measures semantic overlap between resume & JD vectors |
| 🛠️ Skill Extraction | 200+ skill lexicon, bigram scanning, multi-word skill detection |
| 📊 Composite Scoring | 70% TF-IDF + 30% skill-overlap ratio |
| 🏆 Candidate Ranking | Sort N resumes in one click |
| 🔍 Skill Gap Analysis | Matched / missing / extra skill chips |
| 📉 Visualisations | Gauge, radar, bar, scatter, donut, word cloud |
| 📥 PDF Report | ReportLab-generated downloadable analysis |
| 💾 CSV Export | Rankings table as CSV |

---

## Project Architecture

```
resume_screener/
│
├── app.py                   # Streamlit dashboard (entry point)
├── train.py                 # Offline TF-IDF training script
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── sample_data.csv      # 5 labelled resume-JD pairs (demo)
│   └── README.md            # Kaggle dataset links
│
├── models/
│   ├── tfidf_vectorizer.joblib   # Saved fitted vectorizer (after train.py)
│   └── eval_results.json         # MAE / RMSE from training evaluation
│
└── utils/
    ├── __init__.py
    ├── preprocessor.py      # TextPreprocessor + SkillExtractor
    ├── pdf_extractor.py     # PDF / DOCX / TXT text extraction
    ├── matcher.py           # ResumeJobMatcher (TF-IDF + cosine sim)
    ├── visualizer.py        # All Matplotlib / Plotly chart factories
    └── report_generator.py  # ReportLab PDF report builder
```

### System Design (ASCII Flowchart)

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Streamlit)                   │
│  ┌────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │
│  │ Resume PDF │  │  Job Description│  │  Multi-Resume Upload │ │
│  └─────┬──────┘  └────────┬────────┘  └──────────┬───────────┘ │
└────────│─────────────────│──────────────────────│──────────────┘
         │                 │                      │
         ▼                 ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                   TEXT EXTRACTION LAYER                       │
│   PyMuPDF (PDF) │ python-docx (DOCX) │ plain decode (TXT)   │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  NLP PREPROCESSING PIPELINE                   │
│  Lowercase → URL/email strip → Special char removal          │
│  → NLTK Tokenize → Stopword removal → WordNet Lemmatize     │
└──────────────────────────┬───────────────────────────────────┘
                           │
              ┌────────────┴──────────────┐
              │                           │
              ▼                           ▼
   ┌─────────────────────┐    ┌───────────────────────────┐
   │   SKILL EXTRACTION  │    │   TF-IDF VECTORIZATION    │
   │  (Lexicon matching  │    │  TfidfVectorizer fit on   │
   │   + bigram scan)    │    │  [resume_text, jd_text]   │
   └──────────┬──────────┘    └──────────────┬────────────┘
              │                              │
              │                              ▼
              │               ┌─────────────────────────────┐
              │               │     COSINE SIMILARITY        │
              │               │  sim = cos(v_resume, v_jd)  │
              │               └──────────────┬──────────────┘
              │                              │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │       COMPOSITE SCORING           │
              │  score = 0.7 * tfidf + 0.3 * skill│
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │     RESULTS & VISUALISATIONS      │
              │  Gauge · Radar · Bar · Wordcloud  │
              │  Skill chips · PDF report · CSV   │
              └──────────────────────────────────┘
```

---

## ML Pipeline

### 1. Text Preprocessing
```
Raw Text
  │  lowercase
  │  remove URLs / emails
  │  keep alphanum + [+, #, .]
  │  word_tokenize (NLTK Punkt)
  │  remove stopwords (NLTK, with resume-important words preserved)
  └► lemmatize (WordNetLemmatizer)
```

### 2. TF-IDF Vectorisation
- **Why TF-IDF?** Weighs rare, important terms higher than frequent filler words.
- **Bigrams** (`ngram_range=(1,2)`) capture phrases like "machine learning", "deep learning".
- **`sublinear_tf=True`** — replaces TF with 1+log(TF) to reduce dominance of very frequent terms.

### 3. Cosine Similarity
- Measures the **angle** between two document vectors in high-dimensional space.
- Range: 0 (completely different) → 1 (identical).
- **Preferred over Euclidean distance** for text because it's length-invariant.

### 4. Composite Score
```
match_score = 0.70 × tfidf_similarity  +  0.30 × skill_overlap_ratio
```
- **70% TF-IDF** — overall semantic alignment.
- **30% Skill overlap** — domain-specific hard-skill match.

### 5. Skill Extraction
- 200+ skill lexicon covering Programming, ML/AI, Cloud, Data, DevOps.
- Single-word skills: token set membership.
- Multi-word skills (e.g. "machine learning"): substring scan on raw text.

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/resume-screener.git
cd resume-screener

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model (optional, used for advanced NLP)
python -m spacy download en_core_web_sm

# 5. (Optional) Run training to pre-fit the TF-IDF vectorizer
python train.py

# 6. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501** automatically.

---

## Usage Guide

### Single Resume Mode
1. Select **🎯 Single Resume** from the sidebar.
2. Paste or upload a job description.
3. Upload your resume (PDF / DOCX / TXT).
4. Click **Analyse Resume**.
5. View: match score, skill breakdown, word clouds, charts.
6. Download the PDF report.

### Multi-Resume Ranking Mode
1. Select **👥 Multi-Resume Ranking** from the sidebar.
2. Paste or upload the job description.
3. Upload multiple resumes.
4. Click **Rank All Candidates**.
5. View ranked table, scatter plot, and per-candidate detail.
6. Download PDF or CSV report.

---

## Deployment

### Local (already covered above)

### GitHub Upload
```bash
git init
git add .
git commit -m "Initial commit: Resume Screener AI"
git remote add origin https://github.com/YOUR_USERNAME/resume-screener.git
git push -u origin main
```

### Streamlit Cloud (Free)
1. Push your code to GitHub.
2. Go to https://share.streamlit.io → **New app**.
3. Select your repo and set **Main file** = `app.py`.
4. Click **Deploy** — live in ~2 minutes.

> **Tip:** Add a `packages.txt` with system packages if needed (usually not required).

---

## Dataset

See [`dataset/README.md`](dataset/README.md) for Kaggle links and download instructions.

| Dataset | Records | Source |
|---------|---------|--------|
| Resume Dataset | 2,400+ | [Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) |
| Job Description Dataset | 5,000+ | [Kaggle](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) |
| UpdatedResumeDataSet | 960+ | [Kaggle](https://www.kaggle.com/datasets/avishekmajumder/resumedataset) |

---

## Interview Q&A

### 🤖 Machine Learning (25 Questions)

**Q1. What is TF-IDF and how is it computed?**  
TF-IDF (Term Frequency–Inverse Document Frequency) measures how important a word is to a document relative to a corpus.  
- **TF(t,d)** = (count of term t in doc d) / (total terms in d)  
- **IDF(t)** = log(N / df(t)) where N = total docs, df = docs containing t  
- **TF-IDF = TF × IDF**  
High TF-IDF means the term is frequent in the document but rare across the corpus → distinctive.

**Q2. Why use cosine similarity instead of Euclidean distance for text?**  
Cosine similarity measures the *angle* between vectors, not their magnitude. Two documents with the same proportions of words but different lengths get a cosine similarity of 1 (identical direction). Euclidean distance would rate them as different because of length. For text, we care about *content*, not *volume*.

**Q3. What is the curse of dimensionality?**  
As feature dimensionality increases, the volume of the space grows exponentially, making data sparse. Distances become less meaningful and models need exponentially more data. Solutions: PCA, feature selection, regularisation.

**Q4. What is the difference between supervised and unsupervised learning?**  
- **Supervised**: labelled data (X, y); goal = learn f(X)→y. Examples: classification, regression.  
- **Unsupervised**: no labels; goal = find structure. Examples: clustering (K-Means), dimensionality reduction (PCA), topic modelling (LDA).  
This project is *unsupervised* — no labelled pairs needed at inference time.

**Q5. What is regularisation and why is it needed?**  
Regularisation adds a penalty to the loss function to reduce model complexity and prevent overfitting.  
- **L1 (Lasso)**: penalty = λΣ|w|; promotes sparsity (zero weights).  
- **L2 (Ridge)**: penalty = λΣw²; shrinks weights towards zero evenly.

**Q6. What is the bias-variance tradeoff?**  
- **High bias** = underfitting (model too simple, misses patterns).  
- **High variance** = overfitting (model too complex, memorises training data).  
- Ideal model balances both: low enough bias to capture real patterns, low enough variance to generalise.

**Q7. What is cross-validation?**  
Technique to estimate model performance on unseen data without a separate test set.  
**k-Fold CV**: split data into k folds; train on k-1, test on 1; repeat k times; average the scores.

**Q8. Explain Precision, Recall, and F1-Score.**  
- **Precision** = TP / (TP + FP) — of all positive predictions, how many were correct?  
- **Recall** = TP / (TP + FN) — of all actual positives, how many were caught?  
- **F1** = 2 × (P × R) / (P + R) — harmonic mean; useful when classes are imbalanced.

**Q9. What is the difference between PCA and t-SNE?**  
- **PCA**: linear, preserves global variance, fast, deterministic, good for preprocessing.  
- **t-SNE**: non-linear, preserves local neighbourhood structure, slow, stochastic, good for visualisation. Cannot be used for new data points without retraining.

**Q10. What is gradient descent?**  
Iterative optimisation: update parameters in the direction of the negative gradient of the loss.  
θ ← θ − α∇L(θ) where α = learning rate.  
Variants: batch GD, stochastic GD (SGD), mini-batch GD, Adam, RMSProp.

**Q11. What is overfitting and how do you prevent it?**  
Overfitting = model performs well on training data but poorly on unseen data.  
Prevention: regularisation, dropout, early stopping, data augmentation, cross-validation, simpler model.

**Q12. Explain the ROC-AUC metric.**  
ROC curve plots TPR vs FPR at all classification thresholds. AUC = area under the curve.  
AUC = 0.5 → random classifier; AUC = 1.0 → perfect. Threshold-independent, robust to class imbalance.

**Q13. What is feature engineering?**  
The process of using domain knowledge to create new features from raw data that help ML models learn better. Examples in this project: skill-overlap ratio (derived from raw text), composite score (combining TF-IDF + skill features).

**Q14. What is the difference between bagging and boosting?**  
- **Bagging** (e.g. Random Forest): trains models in *parallel* on bootstrap samples; reduces variance; majority vote.  
- **Boosting** (e.g. XGBoost): trains models *sequentially*; each corrects errors of the previous; reduces bias.

**Q15. What is K-Means clustering?**  
Partitions N data points into K clusters by minimising within-cluster sum of squares.  
Algorithm: initialise K centroids → assign each point to nearest centroid → recompute centroids → repeat until convergence.

**Q16. Explain the EM algorithm.**  
Expectation-Maximisation is used to find maximum likelihood estimates with latent variables.  
- **E-step**: compute expected value of latent variables given current parameters.  
- **M-step**: maximise likelihood w.r.t. parameters given the expected latent values.  
Used in Gaussian Mixture Models, Hidden Markov Models.

**Q17. What is transfer learning?**  
Reusing a model pre-trained on a large dataset (e.g. ImageNet, Wikipedia) for a related task. The pre-trained weights capture general features; only the final layers are fine-tuned on domain data. Saves data and compute.

**Q18. What is a confusion matrix?**  
N×N matrix for N-class classification. Rows = actual, columns = predicted. Diagonal = correct predictions. Off-diagonal = errors. Used to compute Precision, Recall, F1, Accuracy per class.

**Q19. What is the difference between a generative and discriminative model?**  
- **Generative**: models joint distribution P(X,Y); can generate samples. Examples: Naive Bayes, GANs.  
- **Discriminative**: models conditional P(Y|X) directly; focuses on decision boundary. Examples: Logistic Regression, SVM, Neural Networks.

**Q20. What are hyperparameters? How do you tune them?**  
Hyperparameters = model configuration not learned from data (e.g. learning rate, n_estimators, max_depth).  
Tuning strategies: Grid Search, Random Search, Bayesian Optimisation (Optuna), successive halving.

**Q21. Explain SMOTE.**  
Synthetic Minority Oversampling Technique. Generates synthetic samples for the minority class by interpolating between existing minority samples and their k-nearest neighbours. Addresses class imbalance without just duplicating data.

**Q22. What is the vanishing gradient problem?**  
In deep networks with sigmoid/tanh activations, gradients become very small during backpropagation, causing early layers to learn very slowly or not at all.  
Solutions: ReLU activations, batch normalisation, residual connections (ResNets), better initialisations (Xavier, He).

**Q23. What is attention in Transformers?**  
Attention allows a model to weigh the relevance of different tokens when encoding each token.  
`Attention(Q,K,V) = softmax(QKᵀ/√d_k)V`  
Self-attention lets each token attend to all others in the same sequence.

**Q24. What is the difference between L1 and L2 loss?**  
- **L1 (MAE)**: robust to outliers; non-differentiable at 0; used when outliers should be ignored.  
- **L2 (MSE)**: penalises large errors more; differentiable everywhere; sensitive to outliers; standard for regression.

**Q25. How would you deploy a machine learning model?**  
1. Serialise model (joblib / pickle / ONNX).  
2. Wrap in REST API (Flask / FastAPI).  
3. Containerise (Docker).  
4. Deploy to cloud (AWS SageMaker / GCP Vertex AI / Azure ML).  
5. Monitor data drift, latency, throughput in production.

---

### 🗣️ NLP Questions (15)

**N1. What is tokenisation?**  
Splitting text into units (tokens) — words, subwords, or characters — for further processing. NLTK's `word_tokenize` uses a Punkt sentence tokeniser and then splits on whitespace/punctuation.

**N2. What are stopwords and why remove them?**  
Common words (the, is, at) that carry little semantic meaning. Removing them reduces dimensionality and noise, letting the model focus on content words.

**N3. What is lemmatisation vs stemming?**  
- **Stemming**: crude rule-based suffix stripping (running → run, studies → studi). Fast, may produce non-words.  
- **Lemmatisation**: morphological analysis using a vocabulary/dictionary (running → run, studies → study). Slower, always produces valid words. Preferred for NLP pipelines.

**N4. What is a bag of words?**  
Represents a document as an unordered multiset of tokens, ignoring grammar and word order. Simple, fast, loses positional information.

**N5. What is the difference between word2vec and TF-IDF?**  
- **TF-IDF**: sparse, high-dimensional, statistical, no semantic understanding.  
- **Word2vec**: dense, low-dimensional, captures semantic relationships (king − man + woman ≈ queen).

**N6. What are n-grams?**  
Contiguous sequences of n tokens. Bigrams ("machine learning", "deep learning") capture multi-word phrases that unigrams miss. This project uses `ngram_range=(1,2)`.

**N7. What is BERT and why is it powerful?**  
Bidirectional Encoder Representations from Transformers. Pre-trained on masked language modelling and next-sentence prediction on huge corpora. Produces contextualised embeddings — the same word has different representations in different contexts. State-of-the-art for classification, NER, QA.

**N8. What is Named Entity Recognition (NER)?**  
Identifying and classifying named entities (persons, organisations, locations, dates) in text. In resume analysis, NER can extract candidate names, universities, companies automatically.

**N9. What is the difference between syntax and semantics?**  
- **Syntax**: structure of sentences (grammar rules, parse trees).  
- **Semantics**: meaning of words and sentences.  
NLP systems need both: syntactic parsing for structure, semantic models (embeddings) for meaning.

**N10. What is topic modelling?**  
Unsupervised technique to discover abstract "topics" in a corpus. **LDA** (Latent Dirichlet Allocation) models each document as a mixture of topics, each topic as a distribution over words. Useful for categorising resumes by domain.

**N11. How does spaCy differ from NLTK?**  
- **NLTK**: research-oriented, many algorithms, slower, function-based.  
- **spaCy**: production-oriented, fast (Cython), opinionated pipeline, excellent pre-trained models, entity recognition, dependency parsing out of the box.

**N12. What is the attention mechanism?**  
Allows the model to focus on relevant parts of the input when generating each output token. In self-attention, every token attends to every other token in the same sequence with learnable weights.

**N13. What is text augmentation?**  
Techniques to artificially expand a training dataset: synonym replacement, back-translation, random insertion/deletion/swap, paraphrasing via GPT. Helps when labelled data is scarce.

**N14. What is a language model?**  
A model that assigns probability to sequences of tokens: P(w₁, w₂, …, wₙ). Classic: n-gram LMs. Modern: neural LMs (GPT, BERT). Used for generation, classification, embeddings.

**N15. What is the difference between sentence-level and word-level embeddings?**  
- **Word-level**: each token gets its own vector (word2vec, GloVe). Document vector = average or weighted sum.  
- **Sentence-level**: the entire sentence is embedded as one vector (Sentence-BERT, USE). Better for semantic similarity tasks like resume matching.

---

### 🎯 Project-Specific Questions (15)

**P1. Why did you choose TF-IDF over word embeddings for this project?**  
TF-IDF is interpretable, fast, requires no GPU, and works well on short domain-specific documents like resumes and JDs. Word embeddings would add complexity without proportionally better results for keyword-heavy HR text. For a production system, Sentence-BERT would be the upgrade path.

**P2. Why is the composite score 70% TF-IDF + 30% skill overlap?**  
TF-IDF captures overall semantic alignment (writing style, domain language), while skill overlap ensures hard technical requirements are explicitly checked. 70/30 weights were chosen empirically — they can be tuned as sliders in a production system.

**P3. How would you improve this system for production?**  
1. Replace TF-IDF with Sentence-BERT for richer semantic similarity.  
2. Train a supervised ranking model on historical hiring data (click-through, interview outcomes).  
3. Add NER to auto-extract candidate name, education, years of experience.  
4. Build an active learning loop: HR feedback improves the model over time.  
5. Add a database (PostgreSQL + pgvector) to store and search resume vectors.

**P4. How does the skill extractor handle multi-word skills like "machine learning"?**  
It performs a substring search on the raw lowercased text before tokenisation. Single-word skills use token-set membership after tokenisation. The lexicon separates multi-word skills into a `_multi_word` set checked first to avoid partial matches.

**P5. What are the limitations of cosine similarity for resume matching?**  
1. Order-insensitive — loses context.  
2. Vocabulary mismatch — "ML" vs "machine learning" → different tokens without stemming.  
3. No understanding of negation ("no Python experience" scores similarly to "Python expert").  
4. Domain shift — works well within a domain, poorly across domains.

**P6. How do you handle a candidate who uses synonyms not in your skill lexicon?**  
The TF-IDF component still rewards term overlap for any shared vocabulary. For the skill extractor, the lexicon can be extended. A production system would use entity linking to a standard ontology (e.g. ESCO, O*NET) or use NER + embeddings to handle synonyms automatically.

**P7. Why use `sublinear_tf=True` in TF-IDF?**  
Standard TF is linear — a word appearing 100 times contributes 100× more than a word appearing once. `sublinear_tf` replaces TF with 1+log(TF), dampening the effect of highly repeated terms. This is especially useful for long resumes where a skill keyword might be repeated in multiple sections.

**P8. How would you evaluate this system if you had hiring ground truth data?**  
Treat it as a ranking problem — use **NDCG** (Normalised Discounted Cumulative Gain) or **MAP** (Mean Average Precision). If ground truth is binary (hired/not hired), use **AUC-ROC** treating match score as the classifier score.

**P9. What is the time complexity of your matching algorithm?**  
TF-IDF vectorisation: O(n × m) where n = documents, m = vocabulary size.  
Cosine similarity between two vectors: O(m).  
For N resumes: O(N × m) — linear in the number of candidates. Scales well to thousands of resumes.

**P10. How would you add a feedback loop to improve the model?**  
1. Log HR decisions (shortlisted / rejected) for each candidate-JD pair.  
2. Use these labels to train a lightweight supervised ranker (e.g. RankSVM, LambdaMART) on top of the TF-IDF features.  
3. Periodically retrain on accumulated feedback. This is **Learning to Rank** (LTR).

**P11. How does PyMuPDF extract text from PDFs?**  
PyMuPDF (fitz) reads the PDF's internal page tree and text stream objects, extracting text with position and font metadata. It handles multi-column layouts better than pdfminer. For scanned PDFs, you'd need OCR (e.g. Tesseract via pytesseract).

**P12. Why do you fit the TF-IDF on [resume, jd] together rather than a large corpus?**  
Fitting on both documents ensures the vocabulary and IDF weights are computed from the same set, so both vectors live in the same feature space and cosine similarity is meaningful. With a pre-fitted vectoriser (from train.py), new documents are transformed into that fixed feature space.

**P13. What is the purpose of train.py if the app works without it?**  
`train.py` pre-fits the TF-IDF vectoriser on a domain corpus (real resume/JD pairs), producing better IDF weights. The app does on-the-fly fitting on just two documents, which is fast but the IDF comes only from those two. A corpus-fitted vectoriser has more accurate IDF weights, especially for common domain terms that should be down-weighted.

**P14. How would you scale this to 10,000 resumes?**  
1. Pre-compute and store TF-IDF vectors for all resumes (sparse matrix).  
2. Store in a vector database (Faiss, pgvector, Weaviate).  
3. At query time, vectorise the JD once and perform a bulk cosine similarity computation: `cosine_similarity(jd_vec, all_resume_vecs)` — NumPy/scipy handles this efficiently.  
4. For even larger scale, use approximate nearest neighbour search (FAISS HNSW).

**P15. What Amazon Leadership Principles did you apply building this project?**  
- **Customer Obsession**: the HR team (customer) needs fast, accurate candidate screening.  
- **Dive Deep**: understanding TF-IDF, IDF weighting, composite scoring trade-offs.  
- **Invent and Simplify**: combining TF-IDF + skill overlap in a clean unsupervised pipeline.  
- **Deliver Results**: working end-to-end app with PDF reports, not just a notebook.

---

## Resume Bullets

```
• Engineered an AI-powered resume screening system using TF-IDF vectorisation and
  cosine similarity (scikit-learn), achieving sub-second candidate ranking across
  multiple resumes with a composite scoring algorithm (70% semantic + 30% skill overlap).

• Built a modular NLP preprocessing pipeline (NLTK tokenisation, stopword removal,
  WordNet lemmatisation) and a domain-specific 200+ skill lexicon with bigram scanning,
  extracting matched/missing skills from PDF resumes using PyMuPDF.

• Developed an interactive Streamlit dashboard with Plotly gauge charts, radar plots,
  word clouds, and ranked candidate tables; integrated ReportLab PDF report generation
  and CSV export for HR workflow integration.

• Designed a training pipeline (train.py) that fits a domain-adapted TF-IDF vectoriser
  on labelled resume-JD pairs, evaluates similarity quality with MAE/RMSE metrics,
  and persists the model with joblib for production-ready inference.
```

### Internship Application Project Description

> **Intelligent Resume Screening & Job Matching System** | Python · scikit-learn · NLTK · Streamlit  
> Designed and built an end-to-end NLP system that automates resume analysis for HR teams. Implemented a TF-IDF + cosine similarity matching engine with a custom 200+ skill lexicon to compute match scores, extract skill gaps, and rank candidates. Delivered an interactive Streamlit dashboard with real-time visualisations (Plotly gauge, radar, word cloud) and one-click PDF/CSV report generation. Trained and evaluated the model on real-world Kaggle resume datasets.

---

## License

MIT — free to use, modify, and distribute with attribution.

---

*Built with ❤️ for Amazon Summer School Application*
