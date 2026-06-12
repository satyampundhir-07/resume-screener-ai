"""
utils/report_generator.py
--------------------------
Generates a downloadable PDF analysis report using ReportLab.

Author: Resume Screener ML Pipeline
"""

from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# ── Colour constants ──────────────────────────────────────────────────────────
AMAZON_ORANGE = colors.HexColor("#FF9900")
AMAZON_DARK   = colors.HexColor("#232F3E")
LIGHT_GREY    = colors.HexColor("#f0f2f6")


def generate_report(results: list[dict], jd_preview: str = "") -> bytes:
    """
    Build a PDF report for all matched candidates.

    Parameters
    ----------
    results     : list[dict]  Output of ResumeJobMatcher.rank_candidates()
    jd_preview  : str         First 300 chars of the job description.

    Returns
    -------
    bytes  PDF file as raw bytes (ready for st.download_button).
    """
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer,
        pagesize    = A4,
        rightMargin = 2 * cm,
        leftMargin  = 2 * cm,
        topMargin   = 2 * cm,
        bottomMargin= 2 * cm,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Custom styles ─────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title2",
        parent    = styles["Title"],
        textColor = AMAZON_DARK,
        fontSize  = 22,
        spaceAfter= 6,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent    = styles["Normal"],
        textColor = AMAZON_ORANGE,
        fontSize  = 12,
        alignment = TA_CENTER,
        spaceAfter= 4,
    )
    section_style = ParagraphStyle(
        "Section",
        parent    = styles["Heading2"],
        textColor = AMAZON_DARK,
        fontSize  = 13,
        spaceBefore = 14,
        spaceAfter  = 6,
    )
    normal = styles["Normal"]

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Intelligent Resume Screening Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=AMAZON_ORANGE))
    story.append(Spacer(1, 0.4 * cm))

    # ── JD preview ────────────────────────────────────────────────────────────
    if jd_preview:
        story.append(Paragraph("Job Description (preview)", section_style))
        preview = jd_preview[:400].replace("\n", " ")
        story.append(Paragraph(f"<i>{preview}…</i>", normal))
        story.append(Spacer(1, 0.3 * cm))

    # ── Summary table ─────────────────────────────────────────────────────────
    story.append(Paragraph("Candidate Rankings Summary", section_style))

    header = ["Rank", "Candidate", "Match Score", "Matched Skills",
              "Missing Skills", "Recommendation"]
    rows   = [header]
    for r in results:
        rows.append([
            str(r.get("rank", "—")),
            r.get("name", "Unknown"),
            f"{r['match_score']:.1f}%",
            str(len(r.get("matched_skills", set()))),
            str(len(r.get("missing_skills", set()))),
            r.get("recommendation", "—").replace("✅ ", "").replace("⚠️ ", "").replace("❌ ", ""),
        ])

    col_widths = [1.2*cm, 4.5*cm, 2.5*cm, 2.8*cm, 2.8*cm, 4*cm]
    tbl = Table(rows, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  AMAZON_DARK),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  10),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("FONTSIZE",    (0, 1), (-1, -1), 9),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.5 * cm))

    # ── Per-candidate details ─────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=AMAZON_ORANGE))
    story.append(Paragraph("Detailed Candidate Analysis", section_style))

    for r in results:
        name = r.get("name", "Unknown")
        score = r["match_score"]

        # Candidate header row
        story.append(Spacer(1, 0.3 * cm))
        candidate_data = [[
            Paragraph(f"<b>#{r.get('rank','?')}  {name}</b>", normal),
            Paragraph(f"<b>Score: {score:.1f}%</b>", normal),
            Paragraph(r.get("recommendation",""), normal),
        ]]
        ct = Table(candidate_data, colWidths=[7*cm, 3.5*cm, 7*cm])
        ct.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
            ("BOX",        (0, 0), (-1, 0), 0.5, AMAZON_ORANGE),
            ("VALIGN",     (0, 0), (-1, 0), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("BOTTOMPADDING",(0,0),(-1, 0), 6),
        ]))
        story.append(ct)

        # Matched skills
        matched = sorted(r.get("matched_skills", set()))
        if matched:
            story.append(Paragraph(
                f"<b>Matched Skills ({len(matched)}):</b> " + ", ".join(matched),
                normal,
            ))

        # Missing skills
        missing = sorted(r.get("missing_skills", set()))
        if missing:
            story.append(Paragraph(
                f"<b>Missing Skills ({len(missing)}):</b> " + ", ".join(missing),
                normal,
            ))

        story.append(Spacer(1, 0.2 * cm))

    # ── Footer note ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Paragraph(
        "<i>Generated by Intelligent Resume Screener · "
        "TF-IDF + Cosine Similarity · NLP Pipeline</i>",
        ParagraphStyle("footer", parent=normal, fontSize=8,
                       textColor=colors.grey, alignment=TA_CENTER),
    ))

    doc.build(story)
    return buffer.getvalue()
