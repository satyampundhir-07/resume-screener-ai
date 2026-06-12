"""
utils/visualizer.py
--------------------
All Matplotlib / Plotly chart factories for the Streamlit dashboard.

Every function returns a Matplotlib Figure or Plotly Figure so
Streamlit can render it with st.pyplot() or st.plotly_chart().

Author: Resume Screener ML Pipeline
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for Streamlit

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud

# ── Colour palette ────────────────────────────────────────────────────────────
AMAZON_ORANGE = "#FF9900"
AMAZON_DARK   = "#232F3E"
ACCENT_GREEN  = "#2ecc71"
ACCENT_RED    = "#e74c3c"
ACCENT_BLUE   = "#3498db"
GREY_LIGHT    = "#f0f2f6"


# ── 1. Match Score Gauge ──────────────────────────────────────────────────────
def plot_match_gauge(score: float, candidate_name: str = "Candidate") -> go.Figure:
    """
    Plotly gauge chart showing match percentage (0–100).

    Parameters
    ----------
    score          : float  0–100
    candidate_name : str
    """
    colour = (
        ACCENT_GREEN  if score >= 70 else
        AMAZON_ORANGE if score >= 45 else
        ACCENT_RED
    )

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = score,
        delta = {"reference": 70, "increasing": {"color": ACCENT_GREEN}},
        title = {"text": f"Match Score — {candidate_name}", "font": {"size": 18}},
        gauge = {
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": colour},
            "steps": [
                {"range": [0,  45], "color": "#fde8e8"},
                {"range": [45, 70], "color": "#fff3cd"},
                {"range": [70, 100],"color": "#d4edda"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 70,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=10, l=10, r=10))
    return fig


# ── 2. Skills Venn-style bar chart ────────────────────────────────────────────
def plot_skills_comparison(
    matched: set[str], missing: set[str], extra: set[str]
) -> plt.Figure:
    """
    Horizontal grouped bar showing matched / missing / extra skills count.

    Parameters
    ----------
    matched : set  Skills in both resume and JD.
    missing : set  Skills in JD but not resume.
    extra   : set  Skills in resume but not JD (bonus skills).
    """
    categories = ["Matched Skills", "Missing Skills", "Extra Skills"]
    values     = [len(matched), len(missing), len(extra)]
    colours    = [ACCENT_GREEN, ACCENT_RED, ACCENT_BLUE]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(categories, values, color=colours, edgecolor="white", height=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontweight="bold", fontsize=12,
        )

    ax.set_xlabel("Count", fontsize=11)
    ax.set_title("Skills Breakdown", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlim(0, max(values) + 3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


# ── 3. Candidate Ranking Bar Chart ───────────────────────────────────────────
def plot_candidate_ranking(results: list[dict]) -> plt.Figure:
    """
    Horizontal bar chart ranking all candidates by match score.

    Parameters
    ----------
    results : list[dict]  Output from ResumeJobMatcher.rank_candidates()
    """
    names  = [r["name"] for r in results]
    scores = [r["match_score"] for r in results]
    colours = [
        ACCENT_GREEN  if s >= 70 else
        AMAZON_ORANGE if s >= 45 else
        ACCENT_RED
        for s in scores
    ]

    fig, ax = plt.subplots(figsize=(9, max(3, len(names) * 0.7)))
    bars = ax.barh(names[::-1], scores[::-1], color=colours[::-1],
                   edgecolor="white", height=0.6)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}%", va="center", fontsize=10, fontweight="bold",
        )

    ax.axvline(70, color="green",  linewidth=1.5, linestyle="--", alpha=0.6, label="Recommended (70)")
    ax.axvline(45, color="orange", linewidth=1.5, linestyle="--", alpha=0.6, label="Borderline (45)")
    ax.set_xlabel("Match Score (%)", fontsize=11)
    ax.set_title("Candidate Ranking", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlim(0, 110)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


# ── 4. Score Breakdown Radar Chart ───────────────────────────────────────────
def plot_score_breakdown(tfidf_score: float, skill_score: float) -> go.Figure:
    """
    Plotly radar chart: TF-IDF score vs Skill match score.
    """
    categories = ["TF-IDF Score", "Skill Match", "Overall Balance"]
    values     = [
        tfidf_score,
        skill_score,
        (tfidf_score + skill_score) / 2,
    ]
    # Close the loop
    categories_plot = categories + [categories[0]]
    values_plot     = values     + [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r     = values_plot,
        theta = categories_plot,
        fill  = "toself",
        line_color = AMAZON_ORANGE,
        fillcolor  = f"rgba(255,153,0,0.25)",
        name       = "Score",
    ))
    fig.update_layout(
        polar  = {"radialaxis": {"visible": True, "range": [0, 100]}},
        title  = "Score Breakdown",
        height = 350,
        margin = dict(t=60, b=10, l=60, r=60),
    )
    return fig


# ── 5. Word Cloud ─────────────────────────────────────────────────────────────
def plot_wordcloud(text: str, title: str = "Word Cloud") -> plt.Figure:
    """
    Matplotlib word cloud from raw text.

    Parameters
    ----------
    text  : str  Raw or preprocessed text.
    title : str  Chart title.
    """
    if not text.strip():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No text available", ha="center", va="center")
        return fig

    wc = WordCloud(
        width=800, height=350,
        background_color="white",
        colormap="YlOrBr",
        max_words=100,
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


# ── 6. Score Distribution (multi-candidate) ──────────────────────────────────
def plot_score_distribution(results: list[dict]) -> go.Figure:
    """
    Plotly scatter plot: candidates ranked by score, coloured by tier.

    Parameters
    ----------
    results : list[dict]  rank_candidates() output.
    """
    names  = [r["name"] for r in results]
    scores = [r["match_score"] for r in results]
    tiers  = [
        "Highly Recommended" if s >= 70 else
        "Recommended"        if s >= 45 else
        "Not Recommended"
        for s in scores
    ]
    colour_map = {
        "Highly Recommended": ACCENT_GREEN,
        "Recommended":        AMAZON_ORANGE,
        "Not Recommended":    ACCENT_RED,
    }

    fig = go.Figure()
    for tier, colour in colour_map.items():
        idx = [i for i, t in enumerate(tiers) if t == tier]
        if not idx:
            continue
        fig.add_trace(go.Scatter(
            x    = [names[i]  for i in idx],
            y    = [scores[i] for i in idx],
            mode = "markers+text",
            name = tier,
            marker = dict(color=colour, size=14, symbol="circle"),
            text   = [f"{scores[i]:.1f}%" for i in idx],
            textposition = "top center",
        ))

    fig.add_hline(y=70, line_dash="dash", line_color="green",  annotation_text="Highly Recommended (70%)")
    fig.add_hline(y=45, line_dash="dash", line_color="orange", annotation_text="Borderline (45%)")

    fig.update_layout(
        title  = "Score Distribution — All Candidates",
        xaxis_title = "Candidate",
        yaxis_title = "Match Score (%)",
        yaxis_range = [0, 105],
        height = 420,
        legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── 7. Skills Pie Chart ───────────────────────────────────────────────────────
def plot_skills_pie(matched: set, missing: set, extra: set) -> go.Figure:
    """
    Plotly donut chart: matched / missing / extra skills.
    """
    labels = ["Matched", "Missing", "Extra"]
    values = [len(matched), len(missing), len(extra)]
    colours = [ACCENT_GREEN, ACCENT_RED, ACCENT_BLUE]

    fig = go.Figure(go.Pie(
        labels    = labels,
        values    = values,
        hole      = 0.4,
        marker    = dict(colors=colours),
        textinfo  = "label+percent+value",
    ))
    fig.update_layout(
        title  = "Skills Distribution",
        height = 350,
        margin = dict(t=50, b=10, l=10, r=10),
    )
    return fig
