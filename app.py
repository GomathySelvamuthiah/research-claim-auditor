"""
app.py

Streamlit web interface for the Research Claim Auditor.

Run with:
    streamlit run app.py

Provides a text-area for pasting a paper section, file uploaders for reference
PDFs/text files, and displays the audit report inline with distortion highlights
and a retraction warning panel.
"""

import dataclasses
import json
import pathlib

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Embedding model — cached once across all reruns
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def _get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Pipeline import (isolated so a missing API key doesn't crash the whole UI)
# ---------------------------------------------------------------------------
try:
    from src.pipeline import run_audit
    from src.report_generator import AuditReport, ReportGenerator
    _PIPELINE_AVAILABLE = True
    _PIPELINE_ERROR: str | None = None
except Exception as _exc:
    _PIPELINE_AVAILABLE = False
    _PIPELINE_ERROR = str(_exc)

# ---------------------------------------------------------------------------
# Page config — must come before any other st call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Claim Auditor",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "audit_report" not in st.session_state:
    st.session_state.audit_report = None
if "source_docs" not in st.session_state:
    st.session_state.source_docs: list[dict] = []


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def _load_sample_text() -> str:
    try:
        return pathlib.Path("data/sample_papers/sample_intro.txt").read_text()
    except Exception:
        return ""


@st.cache_data
def _load_eval_pairs() -> list[dict]:
    try:
        return json.loads(
            pathlib.Path("data/evaluation_set/labeled_pairs.json").read_text()
        )
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔬 Research Claim Auditor")
    st.markdown("Detect citation distortion in academic papers using Agentic RAG.")
    st.divider()
    with st.expander("DISTORTION TYPES"):
        st.markdown(
            "- **Certainty Inflation** — \"suggests\" becomes \"proves\"\n"
            "- **Causal Overclaim** — correlation reported as causation\n"
            "- **Scope Inflation** — study of 40 students generalized to \"all humans\"\n"
            "- **Cherry Picking** — mixed results cited for positive arm only\n"
            "- **Retracted Source** — source has since been retracted"
        )
    st.divider()
    st.subheader("📊 Data Sources")
    try:
        import os as _os
        if _os.path.exists("data/retraction_watch_real.csv"):
            rw_df = pd.read_csv("data/retraction_watch_real.csv")
            st.metric("Retraction Records", f"{len(rw_df):,}", help="CrossRef API — live data")
        if _os.path.exists("data/real_papers.json"):
            with open("data/real_papers.json") as _f:
                rp = json.load(_f)
            st.metric("Knowledge Base Papers", f"{len(rp.get('papers', rp)):,}", help="OpenAlex API — real abstracts")
        if _os.path.exists("data/evaluation_set/labeled_pairs.json"):
            with open("data/evaluation_set/labeled_pairs.json") as _f:
                ep = json.load(_f)
            _pairs = ep if isinstance(ep, list) else ep.get("pairs", [])
            real_count = sum(1 for p in _pairs if p.get("is_real_paper"))
            st.metric("Eval Pairs", f"{len(_pairs)} ({real_count} real)", help="Grounded in real abstracts")
    except Exception:
        pass
    st.divider()
    st.markdown("Built by **Gomathy Selvamuthiah** | Generative AI Discovery")
    st.markdown("[GitHub](https://github.com)")


# ---------------------------------------------------------------------------
# Pre-load embedding model once (cached across reruns)
# ---------------------------------------------------------------------------
_get_embedding_model()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔬 Run Audit", "📊 Evaluation Results", "💡 How It Works"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Run Audit
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    if not _PIPELINE_AVAILABLE:
        st.error(f"Pipeline modules could not be loaded: {_PIPELINE_ERROR}")
    else:
        # ── Inputs ──────────────────────────────────────────────────────────
        paper_title = st.text_input(
            "Paper Title",
            placeholder="Enter the paper title",
        )

        input_method = st.radio("Choose input method:", ["📝 Paste Text", "📄 Upload PDF"], horizontal=True)
        if input_method == "📄 Upload PDF":
            uploaded_file = st.file_uploader("Upload academic paper PDF", type=["pdf"])
            if uploaded_file is not None:
                try:
                    from src.pdf_extractor import pdf_to_audit_text
                    full_text, intro_text = pdf_to_audit_text(uploaded_file.read())
                    st.success(f"✅ PDF processed — {len(intro_text.split())} words extracted from introduction")
                    with st.expander("Preview extracted text"):
                        st.text(intro_text[:500] + "...")
                    paper_text = intro_text
                except Exception as e:
                    st.error(f"PDF extraction failed: {e}")
                    paper_text = ""
            else:
                paper_text = ""
        else:
            paper_text = st.text_area("Paste Introduction Section", value=_load_sample_text(), height=300)

        # Optional source documents
        with st.expander("➕ Add Source Documents (optional)"):
            if "src_form_counter" not in st.session_state:
                st.session_state.src_form_counter = 0

            counter = st.session_state.src_form_counter

            src_text = st.text_area(
                "Source document text",
                key=f"src_text_{counter}",
                height=150,
            )
            src_doi = st.text_input("DOI (optional)", key=f"src_doi_{counter}")
            src_title_val = st.text_input("Title", key=f"src_title_{counter}")

            if st.button("Add Source"):
                if src_text.strip():
                    st.session_state.source_docs.append({
                        "full_text": src_text,
                        "doi": src_doi.strip() or "",
                        "title": src_title_val.strip() or "Untitled Source",
                        "year": 2024,
                    })
                    st.session_state.src_form_counter += 1
                    st.rerun()
                else:
                    st.warning("Source text cannot be empty.")

            if st.session_state.source_docs:
                st.info(f"{len(st.session_state.source_docs)} source document(s) added.")

        # ── Run button ───────────────────────────────────────────────────────
        st.caption(
            "💡 **How to get best results:** Paste the introduction section of a paper. "
            "The system will extract cited claims, retrieve matching passages from the 80-paper "
            "knowledge base, and classify any distortions. "
            "For citations not in the knowledge base, add source documents manually below."
        )
        if st.button(
            "🔍 Run Citation Audit",
            type="primary",
            disabled=not paper_text.strip(),
        ):
            try:
                with st.spinner("Analyzing citations... This may take 30-60 seconds."):
                    report = run_audit(
                        paper_text=paper_text,
                        paper_title=paper_title.strip() or "Unknown Paper",
                        source_docs=st.session_state.source_docs or None,
                        save_outputs=True,
                    )
                st.session_state.audit_report = report
            except Exception as exc:
                st.error(f"Audit failed: {exc}")

        # ── Results ──────────────────────────────────────────────────────────
        if st.session_state.audit_report is not None:
            try:
                report: AuditReport = st.session_state.audit_report
                st.divider()

                # Metrics row
                n_distorted = sum(
                    v
                    for k, v in report.distortion_counts.items()
                    if k not in ("none", "unverifiable")
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Claims", report.total_claims)
                c2.metric("Distortions Found", n_distorted)
                c3.metric("Retracted Sources", report.retraction_flags)
                c4.metric("Integrity Score", f"{report.integrity_score:.0f}/100")

                # Risk badge
                if report.overall_risk == "low":
                    st.success("✅ Low Risk")
                elif report.overall_risk == "medium":
                    st.warning("⚠️ Medium Risk")
                else:
                    st.error("🔴 High Risk")

                # Unverifiable claims notice
                n_unverifiable = sum(1 for e in report.entries if e.distortion_type == "unverifiable")
                if n_unverifiable > len(report.entries) * 0.5:
                    st.info(
                        f"ℹ️ **{n_unverifiable} of {report.total_claims} claims are unverifiable.** "
                        "This means the cited papers are not in the knowledge base. "
                        "To get full results, add the cited source documents using "
                        "'➕ Add Source Documents' above, or run `python scripts/build_knowledge_base.py` "
                        "to index more papers."
                    )

                # Distortion breakdown
                st.subheader("Distortion Breakdown")
                chart_data = {
                    k: v
                    for k, v in report.distortion_counts.items()
                    if k not in ("none", "unverifiable") and v > 0
                }
                if chart_data:
                    df_chart = pd.DataFrame(
                        {"Count": chart_data},
                    )
                    st.bar_chart(df_chart)
                else:
                    st.info("No distortions detected.")

                # Claim-level findings
                st.subheader("Claim-Level Findings")
                for entry in report.entries:
                    is_red = entry.is_retracted or entry.severity >= 3
                    is_warn = not is_red and entry.distortion_type != "none"
                    icon = "🔴" if is_red else ("⚠️" if is_warn else "✅")

                    with st.expander(f"{icon} [{entry.claim_id}] {entry.distortion_type}"):
                        st.markdown(f"**{entry.claim_text}**")
                        st.markdown(
                            f"**Distortion type:** `{entry.distortion_type}`  "
                            f"&nbsp;|&nbsp;  **Severity:** {entry.severity_label}"
                        )
                        st.progress(
                            min(max(entry.confidence, 0.0), 1.0),
                            text=f"Confidence: {entry.confidence:.0%}",
                        )
                        if entry.explanation:
                            st.markdown(f"**Explanation:** {entry.explanation}")
                        if entry.problematic_phrase:
                            st.warning(
                                f'Problematic phrase: "{entry.problematic_phrase}"'
                            )
                        if (
                            entry.what_source_actually_says
                            and entry.distortion_type != "none"
                        ):
                            st.info(
                                f"Source actually says: {entry.what_source_actually_says}"
                            )
                        if entry.is_retracted:
                            st.error(
                                f"⚠️ RETRACTED SOURCE — "
                                f"Reason: {entry.retraction_reason or 'Unknown'}"
                            )

                # Download buttons
                try:
                    rg = ReportGenerator()
                    json_bytes = json.dumps(
                        dataclasses.asdict(report), indent=2
                    ).encode("utf-8")
                    text_bytes = rg.generate_text_summary(report).encode("utf-8")

                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button(
                            "📥 Download JSON Report",
                            data=json_bytes,
                            file_name="audit_report.json",
                            mime="application/json",
                        )
                    with dl2:
                        st.download_button(
                            "📄 Download Text Summary",
                            data=text_bytes,
                            file_name="audit_summary.txt",
                            mime="text/plain",
                        )
                except Exception as exc:
                    st.error(f"Download preparation failed: {exc}")

                st.subheader("📊 Visual Audit Report")
                try:
                    from src.visual_report import generate_audit_infographic
                    chart_bytes = generate_audit_infographic(report)
                    st.image(chart_bytes, caption="Claim-by-claim audit results", use_container_width=True)
                    st.download_button("📊 Download Chart", data=chart_bytes, file_name="audit_chart.png", mime="image/png")
                except Exception as e:
                    st.error(f"Chart generation failed: {e}")

            except Exception as exc:
                st.error(f"Error rendering results: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Evaluation Results
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    try:
        pairs = _load_eval_pairs()
        if isinstance(pairs, dict):
            pairs = pairs.get("pairs", [])
        if pairs:
            filter_type = st.selectbox(
                "Filter by distortion type:",
                ["All", "accurate", "certainty_inflation", "causal_overclaim",
                 "scope_inflation", "cherry_picking"],
            )
            filtered = pairs if filter_type == "All" else [
                p for p in pairs if p.get("distortion_type") == filter_type
            ]
            df_display = pd.DataFrame([
                {
                    "distortion_type": p.get("distortion_type", ""),
                    "citing_claim": (
                        p["citing_claim"][:70] + "..."
                        if len(p.get("citing_claim", "")) > 70
                        else p.get("citing_claim", "")
                    ),
                    "source_passage": (
                        p["source_passage"][:70] + "..."
                        if len(p.get("source_passage", "")) > 70
                        else p.get("source_passage", "")
                    ),
                    "is_real_paper": p.get("is_real_paper", False),
                }
                for p in filtered
            ])
            st.dataframe(df_display, use_container_width=True)
            real_count = int(df_display["is_real_paper"].sum()) if "is_real_paper" in df_display.columns else 0
            st.info(
                f"📋 {len(df_display)} pairs shown — {real_count} grounded in real paper abstracts "
                f"(OpenAlex), {len(df_display) - real_count} synthetic (GPT-4o-mini)"
            )
        else:
            st.warning("Evaluation pairs not found.")
    except Exception as exc:
        st.error(f"Failed to load evaluation data: {exc}")

    st.subheader("Target Performance Metrics")
    try:
        metrics_df = pd.DataFrame([
            {
                "Metric": "Distortion Detection Precision",
                "Target": "≥ 85%",
                "Notes": "Of flagged, what % are genuine",
            },
            {
                "Metric": "Distortion Detection Recall",
                "Target": "≥ 80%",
                "Notes": "Of all true distortions, what % caught",
            },
            {
                "Metric": "RAG Faithfulness",
                "Target": "≥ 0.80",
                "Notes": "RAGAS score",
            },
            {
                "Metric": "Retraction Detection Accuracy",
                "Target": "≥ 95%",
                "Notes": "Deterministic lookup",
            },
            {
                "Metric": "Silent Failure Rate",
                "Target": "≤ 5%",
                "Notes": 'False "accurate" labels',
            },
            {
                "Metric": "Human Audit Agreement (κ)",
                "Target": "≥ 0.75",
                "Notes": "Cohen's kappa vs expert",
            },
        ])
        st.table(metrics_df)
    except Exception as exc:
        st.error(f"Failed to render metrics table: {exc}")

    st.info(
        "ℹ️ Full evaluation requires running tests/run_evaluation.py with a live API key."
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — How It Works
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("System Architecture")
    st.markdown(
        "```\n"
        "📄 Input Paper → 🔍 Claim Extractor → 📚 Source Retriever (FAISS)"
        " → 🤖 Distortion Classifier → 📊 Report Generator\n"
        "                                           ↕\n"
        "                                      🗄️ Retraction Checker (CSV)\n"
        "```"
    )

    with st.expander("Step 1 — Claim Extraction"):
        st.markdown(
            "The system parses the introduction, identifies sentences with citation markers, "
            "and extracts structured claim objects including claim strength "
            "(speculative/suggestive/assertive/definitive) and hedging language."
        )

    with st.expander("Step 2 — Source Retrieval"):
        st.markdown(
            "Each claim is normalized to a plain-language query to reduce vocabulary mismatch. "
            "Source documents are chunked into 300-word overlapping windows, embedded using "
            "sentence-transformers (all-MiniLM-L6-v2), and stored in a FAISS-compatible vector "
            "index. Top-3 passages are retrieved per claim."
        )

    with st.expander("Step 3 — Distortion Classification"):
        st.markdown(
            "Claude analyzes each (claim, retrieved passages) pair against a 5-type distortion "
            "taxonomy. The Silent Failure Test checks whether retrieved passages are logically "
            "supportive — not just topically related. An adequacy score captures this distinction. "
            "If no sufficiently relevant passage is found (similarity < 0.15), the claim is marked "
            "unverifiable — meaning the system cannot assess it, not that it is accurate. "
            "This is the honest response when source documents are unavailable."
        )

    with st.expander("Step 4 — Retraction Check"):
        st.markdown(
            "All citation keys are checked deterministically against the Retraction Watch database "
            "using exact DOI matching and fuzzy title matching (threshold ≥ 0.85). "
            "No LLM involved — 100% deterministic."
        )

    st.subheader("The Citation Distortion Problem")
    try:
        stat1, stat2, stat3 = st.columns(3)
        with stat1:
            st.metric("Papers on Semantic Scholar", "200M+")
        with stat2:
            st.metric("Papers propagating one false Alzheimer's claim", "242")
            st.caption("Greenberg, 2009")
        with stat3:
            st.metric("Manual verification time", "15–30 min/citation")
    except Exception as exc:
        st.error(f"Failed to render stat cards: {exc}")
