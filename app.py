"""
app.py
-------
Intelligent Resume Screening & Job Matching System
Streamlit Dashboard — Entry Point

Run:
    streamlit run app.py

Author: Resume Screener ML Pipeline
"""

import io
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from utils.pdf_extractor    import extract_text
from utils.matcher          import ResumeJobMatcher
from utils.visualizer       import (
    plot_match_gauge,
    plot_skills_comparison,
    plot_candidate_ranking,
    plot_score_breakdown,
    plot_wordcloud,
    plot_score_distribution,
    plot_skills_pie,
)
from utils.report_generator import generate_report

# ═══════════════════════════════════════════════════════════════════════════════
# Page configuration (MUST be the first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "Resume Screener AI",
    page_icon  = "🎯",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global font ───────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── Top header bar ─────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #232F3E 0%, #37475A 100%);
    padding: 1.4rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    color: white;
}
.app-header h1 { color: #FF9900; margin: 0; font-size: 2rem; }
.app-header p  { color: #ccc;    margin: 0.2rem 0 0; font-size: 0.95rem; }

/* ── Score cards ─────────────────────────────────────── */
.score-card {
    background: white;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    text-align: center;
    border-top: 4px solid #FF9900;
}
.score-card .value { font-size: 2rem; font-weight: 700; color: #232F3E; }
.score-card .label { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }

/* ── Section headers ─────────────────────────────────── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #232F3E;
    border-left: 4px solid #FF9900;
    padding-left: 0.6rem;
    margin: 1.2rem 0 0.6rem;
}

/* ── Skill chips ─────────────────────────────────────── */
.skill-matched { background:#d4edda; color:#155724; border-radius:20px;
                 padding:3px 10px; margin:3px; display:inline-block;
                 font-size:0.82rem; }
.skill-missing { background:#fde8e8; color:#721c24; border-radius:20px;
                 padding:3px 10px; margin:3px; display:inline-block;
                 font-size:0.82rem; }
.skill-extra   { background:#d1ecf1; color:#0c5460; border-radius:20px;
                 padding:3px 10px; margin:3px; display:inline-block;
                 font-size:0.82rem; }

/* ── Candidate rank card ─────────────────────────────── */
.rank-card {
    background: white;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin: 0.4rem 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-left: 5px solid #FF9900;
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] { background: #f7f8fa; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_matcher() -> ResumeJobMatcher:
    """Cached matcher — instantiated once per session."""
    return ResumeJobMatcher()


def score_colour(score: float) -> str:
    if score >= 70: return "#2ecc71"
    if score >= 45: return "#FF9900"
    return "#e74c3c"


def skill_chips(skills: set[str], css_class: str) -> str:
    if not skills:
        return "<i style='color:#aaa'>None detected</i>"
    return "".join(
        f'<span class="{css_class}">{s}</span>' for s in sorted(skills)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar — mode selector & info
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
        width=120,
    )
    st.markdown("---")
    mode = st.radio(
        "**Select Mode**",
        ["🎯 Single Resume", "👥 Multi-Resume Ranking"],
        index=0,
    )
    st.markdown("---")
    st.markdown("""
**How it works**
1. Upload your resume(s) (PDF/DOCX/TXT)
2. Paste or upload a job description
3. The NLP pipeline extracts skills & computes TF-IDF similarity
4. Get a match score, skill gap analysis & ranking

**Tech Stack**
- TF-IDF · Cosine Similarity
- NLTK · scikit-learn
- Streamlit · Plotly
""")
    st.markdown("---")
    st.caption("Resume Screener AI · v1.0")


# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <h1>🎯 Intelligent Resume Screener</h1>
  <p>AI-powered resume analysis · TF-IDF + Cosine Similarity · NLP Skill Extraction</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Job Description Input (shared between modes)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📋 Job Description</div>', unsafe_allow_html=True)

jd_col1, jd_col2 = st.columns([3, 1])
with jd_col1:
    jd_text_input = st.text_area(
        "Paste the job description here:",
        height=180,
        placeholder="e.g. We are looking for a Machine Learning Engineer with experience in Python, TensorFlow…",
    )
with jd_col2:
    jd_file = st.file_uploader("Or upload JD (PDF/DOCX/TXT)", type=["pdf","docx","txt"],
                                key="jd_file")
    if jd_file:
        jd_text_input = extract_text(jd_file.read(), jd_file.name)
        st.success("✅ JD extracted!")

# Use the textarea content as the final JD
jd_text = jd_text_input.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Single Resume Analysis
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "🎯 Single Resume":
    st.markdown('<div class="section-title">📄 Resume Upload</div>', unsafe_allow_html=True)

    r_col1, r_col2 = st.columns([1, 1])
    with r_col1:
        resume_file = st.file_uploader(
            "Upload your resume (PDF / DOCX / TXT)",
            type=["pdf", "docx", "txt"],
        )
    with r_col2:
        candidate_name = st.text_input("Candidate Name (optional)", value="Candidate")

    resume_text = ""
    if resume_file:
        with st.spinner("Extracting text from resume…"):
            resume_text = extract_text(resume_file.read(), resume_file.name)
        if resume_text.startswith("["):
            st.error(resume_text)
        else:
            with st.expander("📃 Extracted Resume Text (click to expand)"):
                st.text(resume_text[:3000] + ("…" if len(resume_text) > 3000 else ""))

    # ── Analyse button ──────────────────────────────────────────────────────
    st.markdown("---")
    analyse_btn = st.button("🚀 Analyse Resume", type="primary", use_container_width=True)

    if analyse_btn:
        # Validation
        if not jd_text:
            st.warning("⚠️ Please provide a Job Description first.")
            st.stop()
        if not resume_text:
            st.warning("⚠️ Please upload a resume first.")
            st.stop()

        matcher = get_matcher()

        with st.spinner("Running NLP pipeline… ⚙️"):
            time.sleep(0.4)  # small UX pause
            result = matcher.match(resume_text, jd_text)

        # ── Top metric cards ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### Analysis Results — *{candidate_name}*")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="score-card">
              <div class="value" style="color:{score_colour(result['match_score'])}">
                {result['match_score']:.1f}%
              </div>
              <div class="label">Overall Match</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="score-card">
              <div class="value">{result['tfidf_score']*100:.1f}%</div>
              <div class="label">TF-IDF Similarity</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="score-card">
              <div class="value">{len(result['matched_skills'])}</div>
              <div class="label">Matched Skills</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="score-card">
              <div class="value">{len(result['missing_skills'])}</div>
              <div class="label">Missing Skills</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"**Recommendation:** {result['recommendation']}", unsafe_allow_html=True)

        # ── Charts row 1 ────────────────────────────────────────────────────
        st.markdown("---")
        ch1, ch2 = st.columns([1, 1])

        with ch1:
            gauge = plot_match_gauge(result["match_score"], candidate_name)
            st.plotly_chart(gauge, use_container_width=True)

        with ch2:
            radar = plot_score_breakdown(
                result["tfidf_score"] * 100,
                result["skill_score"],
            )
            st.plotly_chart(radar, use_container_width=True)

        # ── Skills analysis ──────────────────────────────────────────────────
        st.markdown('<div class="section-title">🔍 Skills Analysis</div>',
                    unsafe_allow_html=True)

        extra_skills = result["resume_skills"] - result["jd_skills"]

        tab1, tab2, tab3 = st.tabs([
            f"✅ Matched ({len(result['matched_skills'])})",
            f"❌ Missing ({len(result['missing_skills'])})",
            f"➕ Extra ({len(extra_skills)})",
        ])
        with tab1:
            st.markdown(
                skill_chips(result["matched_skills"], "skill-matched"),
                unsafe_allow_html=True,
            )
        with tab2:
            st.markdown(
                skill_chips(result["missing_skills"], "skill-missing"),
                unsafe_allow_html=True,
            )
            if result["missing_skills"]:
                st.info(
                    "💡 **Tip:** Consider adding these skills to strengthen your profile: "
                    + ", ".join(sorted(result["missing_skills"])[:8])
                )
        with tab3:
            st.markdown(
                skill_chips(extra_skills, "skill-extra"),
                unsafe_allow_html=True,
            )

        # ── Charts row 2 ────────────────────────────────────────────────────
        ch3, ch4 = st.columns([1, 1])
        with ch3:
            bar = plot_skills_comparison(
                result["matched_skills"],
                result["missing_skills"],
                extra_skills,
            )
            st.pyplot(bar, use_container_width=True)

        with ch4:
            pie = plot_skills_pie(
                result["matched_skills"],
                result["missing_skills"],
                extra_skills,
            )
            st.plotly_chart(pie, use_container_width=True)

        # ── Word clouds ──────────────────────────────────────────────────────
        st.markdown('<div class="section-title">☁️ Word Clouds</div>',
                    unsafe_allow_html=True)
        wc1, wc2 = st.columns(2)
        with wc1:
            fig_wc = plot_wordcloud(resume_text, "Resume Keywords")
            st.pyplot(fig_wc, use_container_width=True)
        with wc2:
            fig_wc2 = plot_wordcloud(jd_text, "Job Description Keywords")
            st.pyplot(fig_wc2, use_container_width=True)

        # ── Download report ──────────────────────────────────────────────────
        st.markdown("---")
        report_bytes = generate_report(
            [{**result, "name": candidate_name, "rank": 1}],
            jd_preview=jd_text[:400],
        )
        st.download_button(
            label     = "📥 Download PDF Report",
            data      = report_bytes,
            file_name = f"resume_analysis_{candidate_name.replace(' ','_')}.pdf",
            mime      = "application/pdf",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Multi-Resume Ranking
# ═══════════════════════════════════════════════════════════════════════════════
elif mode == "👥 Multi-Resume Ranking":
    st.markdown('<div class="section-title">📂 Upload Multiple Resumes</div>',
                unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload one or more resumes (PDF / DOCX / TXT)",
        type   = ["pdf", "docx", "txt"],
        accept_multiple_files = True,
    )

    rank_btn = st.button(
        "🚀 Rank All Candidates", type="primary", use_container_width=True
    )

    if rank_btn:
        if not jd_text:
            st.warning("⚠️ Please provide a Job Description first.")
            st.stop()
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one resume.")
            st.stop()

        # Extract text from all uploaded files
        resumes_data: list[dict] = []
        extraction_bar = st.progress(0, text="Extracting resume text…")
        for i, f in enumerate(uploaded_files):
            text = extract_text(f.read(), f.name)
            # Remove file extension for display name
            name = Path(f.name).stem.replace("_", " ").replace("-", " ").title()
            resumes_data.append({"name": name, "text": text})
            extraction_bar.progress((i + 1) / len(uploaded_files),
                                    text=f"Extracted: {name}")

        extraction_bar.empty()

        # Run matching pipeline
        matcher = get_matcher()
        with st.spinner(f"Analysing {len(resumes_data)} resumes… 🧠"):
            results = matcher.rank_candidates(resumes_data, jd_text)

        # ── Top candidate highlight ──────────────────────────────────────────
        best = results[0]
        st.success(
            f"🏆 **Top Candidate:** {best['name']}  |  "
            f"Score: **{best['match_score']:.1f}%**  |  "
            f"{best['recommendation']}"
        )

        # ── Ranking bar chart ────────────────────────────────────────────────
        st.markdown('<div class="section-title">📊 Candidate Rankings</div>',
                    unsafe_allow_html=True)
        ranking_fig = plot_candidate_ranking(results)
        st.pyplot(ranking_fig, use_container_width=True)

        scatter_fig = plot_score_distribution(results)
        st.plotly_chart(scatter_fig, use_container_width=True)

        # ── Rank table ───────────────────────────────────────────────────────
        st.markdown('<div class="section-title">📋 Results Table</div>',
                    unsafe_allow_html=True)
        table_rows = []
        for r in results:
            table_rows.append({
                "Rank":           r["rank"],
                "Candidate":      r["name"],
                "Match Score (%)": r["match_score"],
                "TF-IDF (%)":     round(r["tfidf_score"] * 100, 1),
                "Matched Skills": len(r["matched_skills"]),
                "Missing Skills": len(r["missing_skills"]),
                "Recommendation": r["recommendation"],
            })
        df = pd.DataFrame(table_rows)
        st.dataframe(
            df.style.background_gradient(subset=["Match Score (%)"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True,
        )

        # ── Per-candidate expandable details ─────────────────────────────────
        st.markdown('<div class="section-title">🔍 Individual Breakdowns</div>',
                    unsafe_allow_html=True)

        for r in results:
            colour = score_colour(r["match_score"])
            with st.expander(
                f"#{r['rank']}  {r['name']}  —  "
                f"{r['match_score']:.1f}%  {r['recommendation']}"
            ):
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    gauge = plot_match_gauge(r["match_score"], r["name"])
                    st.plotly_chart(gauge, use_container_width=True)
                with col_b:
                    extra = r["resume_skills"] - r["jd_skills"]
                    bar   = plot_skills_comparison(
                        r["matched_skills"], r["missing_skills"], extra
                    )
                    st.pyplot(bar, use_container_width=True)

                st.markdown("**✅ Matched Skills**")
                st.markdown(
                    skill_chips(r["matched_skills"], "skill-matched"),
                    unsafe_allow_html=True,
                )
                st.markdown("**❌ Missing Skills**")
                st.markdown(
                    skill_chips(r["missing_skills"], "skill-missing"),
                    unsafe_allow_html=True,
                )

        # ── Download full report ─────────────────────────────────────────────
        st.markdown("---")
        report_bytes = generate_report(results, jd_preview=jd_text[:400])
        st.download_button(
            label     = "📥 Download Full PDF Report",
            data      = report_bytes,
            file_name = "candidate_ranking_report.pdf",
            mime      = "application/pdf",
            use_container_width=True,
        )

        # ── CSV download ─────────────────────────────────────────────────────
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "📊 Download Results CSV",
            data      = csv_data,
            file_name = "candidate_rankings.csv",
            mime      = "text/csv",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.82rem; padding:0.5rem'>
  🎯 Intelligent Resume Screener  ·  
  TF-IDF + Cosine Similarity  ·  
  NLTK · scikit-learn · Streamlit  ·  
  Built for Amazon Summer School
</div>
""", unsafe_allow_html=True)
