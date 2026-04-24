"""
generate_documentation.py

Generates the project documentation PDF.
Run with: python outputs/generate_documentation.py
"""

import os
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
NAVY       = colors.HexColor("#1a2744")
DARK       = colors.HexColor("#2c3e50")
GRAY       = colors.HexColor("#7f8c8d")
LIGHT_BLUE = colors.HexColor("#f0f4ff")
WHITE      = colors.white


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _build_styles() -> dict:
    base = getSampleStyleSheet()

    def ps(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    return {
        "title": ps(
            "TitleStyle",
            fontSize=24, fontName="Helvetica-Bold",
            alignment=TA_CENTER, textColor=NAVY, spaceAfter=12,
        ),
        "subtitle": ps(
            "SubtitleStyle",
            fontSize=14, fontName="Helvetica",
            alignment=TA_CENTER, textColor=GRAY, spaceAfter=12,
        ),
        "section": ps(
            "SectionHeader",
            fontSize=16, fontName="Helvetica-Bold",
            textColor=NAVY, spaceBefore=12, spaceAfter=8, alignment=TA_LEFT,
        ),
        "subsection": ps(
            "SubsectionHeader",
            fontSize=13, fontName="Helvetica-Bold",
            textColor=DARK, spaceBefore=8, spaceAfter=6,
        ),
        "body": ps(
            "BodyText",
            fontSize=11, fontName="Helvetica",
            alignment=TA_JUSTIFY, spaceBefore=6, spaceAfter=6, leading=16,
        ),
        "bullet": ps(
            "BulletStyle",
            fontSize=11, fontName="Helvetica",
            alignment=TA_LEFT, leftIndent=20, bulletIndent=0,
            spaceBefore=3, spaceAfter=3, leading=16,
        ),
        "centered_small": ps(
            "CenteredSmall",
            fontSize=12, fontName="Helvetica",
            alignment=TA_CENTER, spaceAfter=6,
        ),
        "caption": ps(
            "Caption",
            fontSize=9, fontName="Helvetica-Oblique",
            alignment=TA_CENTER, textColor=GRAY, spaceAfter=6,
        ),
    }


def _table_style(num_rows: int, header_navy: bool = False) -> TableStyle:
    cmds = [
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]
    start = 0
    if header_navy:
        cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
        start = 1
    for i in range(start, num_rows):
        bg = LIGHT_BLUE if (i - start) % 2 == 0 else WHITE
        cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
    return TableStyle(cmds)


def _para_row(row: list[str], S: dict, style_key: str = "body") -> list:
    return [Paragraph(cell, S[style_key]) for cell in row]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_pdf(output_path: str = "outputs/documentation.pdf") -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=LETTER,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )

    S = _build_styles()
    story = []
    PAGE_W = 6.5 * inch

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 1 — Title Page
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("Research Claim Auditor", S["title"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "An Agentic RAG System for Academic Citation Integrity", S["subtitle"]
    ))
    story.append(Spacer(1, 0.2 * inch))
    story.append(HRFlowable(width=PAGE_W, thickness=2, color=NAVY))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Final Project — Generative AI Discovery", S["centered_small"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Name: Gomathy Selvamuthiah", S["centered_small"]))
    story.append(Paragraph("NUID: 002410534", S["centered_small"]))
    story.append(Paragraph(
        datetime.now(timezone.utc).strftime("%B %d, %Y"), S["centered_small"]
    ))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 2 — Section 1: System Architecture
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("1. System Architecture", S["section"]))
    story.append(Paragraph(
        "The Research Claim Auditor is a four-component agentic pipeline that detects "
        "citation distortion in academic papers by comparing cited claims against retrieved "
        "source passages using retrieval-augmented generation.",
        S["body"],
    ))
    story.append(Spacer(1, 0.2 * inch))

    diagram_path = "outputs/architecture_diagram.png"
    try:
        # PNG is 3217×1626 px; constrain width to fit inside the 6.33-in frame
        _img_w = 6.0 * inch
        _img_h = _img_w * (1626 / 3217)
        story.append(Image(diagram_path, width=_img_w, height=_img_h))
    except Exception:
        story.append(Paragraph(
            "[Architecture diagram — run outputs/architecture_diagram.py to generate]",
            S["caption"],
        ))
    story.append(Paragraph(
        "Figure 1: Research Claim Auditor System Architecture", S["caption"]
    ))
    story.append(Spacer(1, 0.2 * inch))

    comp_data = [
        [
            Paragraph("<b>Claim Extractor</b>", S["body"]),
            Paragraph(
                "Parses introduction text using Claude Haiku. Extracts cited claims with "
                "classification of claim strength (speculative/suggestive/assertive/definitive) "
                "and hedging language. Normalizes claims to plain-language queries to reduce "
                "vocabulary mismatch during retrieval.",
                S["body"],
            ),
        ],
        [
            Paragraph("<b>Source Retriever</b>", S["body"]),
            Paragraph(
                "Chunks source documents into 300-word overlapping windows (50-word overlap). "
                "Embeds using sentence-transformers all-MiniLM-L6-v2. Stores in a "
                "FAISS-compatible cosine similarity index. Retrieves top-3 passages per claim.",
                S["body"],
            ),
        ],
        [
            Paragraph("<b>Distortion Classifier</b>", S["body"]),
            Paragraph(
                "Applies a 5-type distortion taxonomy via Claude Haiku. Includes the Silent "
                "Failure Test to distinguish topical match from logical support. Returns "
                "distortion type, severity (0–4), confidence, and adequacy score.",
                S["body"],
            ),
        ],
        [
            Paragraph("<b>Retraction Checker</b>", S["body"]),
            Paragraph(
                "Deterministic CSV lookup against Retraction Watch database. Matches by exact "
                "DOI first, then fuzzy title matching (SequenceMatcher, threshold 0.85). "
                "No LLM involved — fully deterministic.",
                S["body"],
            ),
        ],
    ]
    comp_table = Table(comp_data, colWidths=[1.8 * inch, 4.7 * inch])
    comp_table.setStyle(_table_style(len(comp_data)))
    story.append(comp_table)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 3 — Section 2: Implementation Details
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("2. Implementation Details", S["section"]))

    story.append(Paragraph("2.1 Core Components Implemented", S["subsection"]))
    for b in [
        "RAG Pipeline: Vector retrieval using sentence-transformers embeddings and cosine "
        "similarity search. Source documents chunked with 300-word windows and 50-word overlap "
        "to preserve context across chunk boundaries.",

        "Prompt Engineering: Structured distortion taxonomy prompt with 5 labeled categories, "
        "confidence scoring, and a few-shot Silent Failure example that teaches the model to "
        "distinguish topical relevance from logical support.",

        "Retraction Detection: Deterministic lookup requiring zero LLM calls. DOI exact match "
        "with fuzzy title fallback ensures high recall even when DOIs are missing.",
    ]:
        story.append(Paragraph(f"• {b}", S["bullet"]))

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("2.2 Prompting Strategy", S["subsection"]))
    story.append(Paragraph(
        "The distortion classifier uses a structured system prompt defining all five distortion "
        "types with concrete examples. A few-shot Silent Failure example is embedded directly in "
        "the prompt: a scope_inflation case where the source explicitly warns against "
        "generalization and the citing paper does exactly what the source warns against. This "
        "teaches the model to check logical contradiction, not just topical similarity. The "
        "adequacy_score field (0.0–1.0) captures this distinction: a passage can score "
        "high on similarity but low on adequacy if it contradicts the claim.",
        S["body"],
    ))

    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("2.3 Vocabulary Mismatch Mitigation", S["subsection"]))
    story.append(Paragraph(
        "Before retrieval, each extracted claim is passed through a normalization step that "
        "rewrites it as a plain-language query, stripping domain jargon. This reduces false "
        "negatives from vocabulary mismatch — for example, ‘cardiovascular "
        "mortality’ and ‘cardiac death rates’ describe the same concept but "
        "share no overlapping tokens.",
        S["body"],
    ))

    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("2.4 Real Data Sources", S["subsection"]))
    data_src_rows = [
        ["<b>Data Source</b>", "<b>Description</b>", "<b>Volume</b>"],
        [
            "CrossRef API",
            "Real retracted papers fetched via open API. Used for deterministic retraction detection.",
            "200 records",
        ],
        [
            "OpenAlex Academic Graph",
            "Real paper abstracts across 8 research domains. Used as RAG knowledge base.",
            "80 papers, 128 chunks",
        ],
        [
            "Evaluation Set",
            "30 real-paper-grounded pairs + 60 synthetic = 90 labeled pairs for classifier evaluation.",
            "90 pairs",
        ],
    ]
    ds_data = [_para_row(r, S) for r in data_src_rows]
    ds_table = Table(ds_data, colWidths=[1.6 * inch, 3.4 * inch, 1.5 * inch])
    ds_table.setStyle(_table_style(len(ds_data), header_navy=True))
    story.append(ds_table)
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "All data sources are open-access and require no authentication. The knowledge base is "
        "pre-built via scripts/build_knowledge_base.py and persisted to data/cache/. The "
        "retraction checker auto-detects and uses the real CrossRef data when available.",
        S["body"],
    ))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 4 — Section 3: Performance Metrics
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("3. Performance Metrics", S["section"]))
    story.append(Paragraph(
        "The system is evaluated against a manually labeled set of 50–100 citing-claim / "
        "source-passage pairs. Ground truth is derived from Greenberg (2009) citation distortion "
        "annotations and Retraction Watch labels.",
        S["body"],
    ))
    story.append(Spacer(1, 0.15 * inch))

    metrics_rows = [
        ["<b>Metric</b>",                    "<b>Target</b>", "<b>Rationale</b>"],
        ["Distortion Detection Precision",    "≥ 85%",
         "Of all flagged distortions, what % are genuine"],
        ["Distortion Detection Recall",       "≥ 80%",
         "Of all true distortions, what % were caught"],
        ["RAG Faithfulness (RAGAS)",          "≥ 0.80",
         "Is the finding grounded in retrieved source text"],
        ["Retraction Detection Accuracy",     "≥ 95%",
         "Deterministic — DOI/title lookup against Retraction Watch"],
        ["Silent Failure Rate",               "≤ 5%",
         'Of "accurate" predictions, what % are wrong on human review'],
        ["Human Audit Agreement (κ)",    "≥ 0.75",
         "Cohen’s kappa vs expert annotator"],
    ]
    m_data = [_para_row(r, S) for r in metrics_rows]
    m_table = Table(m_data, colWidths=[2.5 * inch, 1.0 * inch, 3.0 * inch])
    m_table.setStyle(_table_style(len(m_data), header_navy=True))
    story.append(m_table)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 5 — Section 4: Challenges and Solutions
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("4. Challenges and Solutions", S["section"]))

    challenge_rows = [
        ["<b>Challenge</b>", "<b>Solution</b>", "<b>Status</b>"],
        [
            "Silent Failure (topical match ≠ logical support)",
            "Adequacy score field in ClassificationResult + few-shot example in system prompt "
            "that explicitly demonstrates a contradiction case",
            "Implemented",
        ],
        [
            "Vocabulary Mismatch",
            "Claim normalization step rewrites jargon-dense claims to plain-language queries "
            "before FAISS retrieval",
            "Implemented",
        ],
        [
            "Paywall-Inaccessible Sources",
            'Abstract-only fallback with explicit "unverifiable" label in output '
            "rather than guessing",
            "Implemented",
        ],
        [
            "Semantic Scholar Rate Limiting",
            "Switched to OpenAlex API (same academic graph, 100 req/s limit, no auth) "
            "for paper fetching",
            "Implemented",
        ],
    ]
    c_data = [_para_row(r, S) for r in challenge_rows]
    c_table = Table(c_data, colWidths=[2.0 * inch, 3.5 * inch, 1.0 * inch])
    c_table.setStyle(_table_style(len(c_data), header_navy=True))
    story.append(c_table)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 6 — Sections 5 & 6
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("5. Future Improvements", S["section"]))
    for b in [
        "Discussion section analysis (currently scoped to introduction only)",
        "Fine-tuning on Greenberg-annotated corpus for domain-specific distortion detection",
        "Multi-document citation chain tracing (claim propagation across papers)",
        "Browser extension for real-time citation checking while reading",
        "Integration with Zotero and Mendeley reference managers",
        "Full-text retrieval for open-access papers via Semantic Scholar S2ORC API",
    ]:
        story.append(Paragraph(f"• {b}", S["bullet"]))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("6. Ethical Considerations", S["section"]))
    for b in [
        "Data sources are entirely open-access (Semantic Scholar, Open Citations, Retraction "
        "Watch). No proprietary, paywalled, or patient data used.",

        "Geographic and linguistic bias: Semantic Scholar overrepresents English-language, "
        "high-income country research. All audit outputs include a disclaimer noting this "
        "limitation.",

        "System outputs are advisory, not authoritative. Every distortion flag includes the "
        "retrieved evidence so humans can verify the classification.",

        "The system does not store or transmit user-submitted paper text beyond the current "
        "session.",

        "Not a replacement for expert peer review. Intended as a first-pass screening tool "
        "to prioritize human review effort.",
    ]:
        story.append(Paragraph(f"• {b}", S["bullet"]))

    # ── Build ────────────────────────────────────────────────────────────────
    doc.build(story)


if __name__ == "__main__":
    out = "outputs/documentation.pdf"
    build_pdf(out)
    print(f"Saved: {out}")
