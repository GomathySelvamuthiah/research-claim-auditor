"""
pipeline.py

Orchestrates the end-to-end Research Claim Auditor workflow, coordinating all
sub-modules into a single callable agentic pipeline.

Workflow:
    1. claim_extractor   — extract claims + citation keys from input text
    2. retraction_checker — flag any retracted sources immediately
    3. source_retriever  — retrieve source passages for each claim
    4. distortion_classifier — classify each (claim, passage) pair
    5. report_generator  — compile and render the final audit report

Exposes a single `run_audit(text: str, reference_docs: list[str]) -> AuditResult`
entry point consumed by both app.py (Streamlit) and the CLI.
"""

import difflib
import json
import logging
import os
import pathlib

from dotenv import load_dotenv

from src.claim_extractor import Claim, ClaimExtractor
from src.distortion_classifier import ClassificationResult, DistortionClassifier
from src.report_generator import AuditReport, ReportGenerator
from src.retraction_checker import RetractionChecker
from src.source_retriever import RetrievalResult, SourceRetriever

logger = logging.getLogger(__name__)


def _find_doi_for_citation(citation_key: str, papers: list[dict]) -> str | None:
    """
    Fuzzy-match a citation key against known paper titles.
    Returns the DOI if the best title match ratio >= 0.6, else None.
    """
    if not papers or not citation_key:
        return None
    norm_key = citation_key.lower().strip()
    best_ratio = 0.0
    best_doi: str | None = None
    for paper in papers:
        title = (paper.get("title") or "").lower().strip()
        if not title:
            continue
        ratio = difflib.SequenceMatcher(None, norm_key, title).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_doi = paper.get("doi") or None
    return best_doi if best_ratio >= 0.6 else None


def run_audit(
    paper_text: str,
    paper_title: str = "Unknown Paper",
    source_docs: list[dict] | None = None,
    retraction_db_path: str | None = None,
    save_outputs: bool = True,
) -> AuditReport:
    load_dotenv()

    extractor = ClaimExtractor()
    retriever = SourceRetriever()
    classifier = DistortionClassifier()
    checker = RetractionChecker(db_path=retraction_db_path)
    report_generator = ReportGenerator()

    # Step 1: Extract claims
    claims: list[Claim] = extractor.extract(paper_text)
    if not claims:
        logger.warning("No claims extracted from paper text — returning empty report")
        from datetime import datetime, timezone
        return AuditReport(
            paper_title=paper_title,
            audit_timestamp=datetime.now(timezone.utc).isoformat(),
            total_claims=0,
            distortion_counts={},
            retraction_flags=0,
            integrity_score=100.0,
            full_text_coverage=1.0,
            entries=[],
            overall_risk="low",
        )

    # Auto-load real papers as knowledge base if available and no docs provided
    if source_docs is None:
        real_papers_path = "data/real_papers.json"
        if os.path.exists(real_papers_path):
            with open(real_papers_path) as f:
                data = json.load(f)
            source_docs = data.get("papers", [])
            logger.info("Auto-loaded %d real papers from %s", len(source_docs), real_papers_path)

    # Step 2: Index source documents (skip if the persisted index was already loaded)
    if source_docs and not retriever._chunks:
        n_chunks = retriever.index_batch(source_docs)
        logger.info("Indexed %d chunks from %d source documents", n_chunks, len(source_docs))
    elif retriever._chunks:
        logger.info("Using pre-built index (%d chunks)", len(retriever._chunks))

    # Step 3: Retrieve passages for each claim
    pairs: list[tuple[Claim, list[RetrievalResult]]] = []
    for claim in claims:
        normalized_query = extractor.normalize_claim_query(claim.claim_text)
        passages = retriever.retrieve(normalized_query, top_k=3)
        pairs.append((claim, passages))

    # Step 4: Classify distortions
    classifications: list[ClassificationResult] = classifier.classify_batch(pairs)

    # Step 5: Check for retracted sources — one item per unique citation key
    seen_keys: set[str] = set()
    retraction_items: list[dict] = []
    for claim in claims:
        for key in claim.citation_keys:
            if key not in seen_keys:
                seen_keys.add(key)
                # Try to find a real DOI by matching the citation key against loaded papers
                real_doi = _find_doi_for_citation(key, source_docs or [])
                retraction_items.append({"citation_key": key, "doi": real_doi, "title": key})
    retraction_results = checker.check_batch(retraction_items)

    # Step 6: Generate report
    report = report_generator.generate(classifications, retraction_results, paper_title)

    # Step 7: Persist outputs
    if save_outputs:
        json_path = report_generator.save_json(report)
        txt_path = report_generator.save_text_summary(report)
        logger.info("Saved audit JSON → %s", json_path)
        logger.info("Saved audit summary → %s", txt_path)

    print(report_generator.generate_text_summary(report))
    return report


if __name__ == "__main__":
    import pathlib

    text = pathlib.Path("data/sample_papers/sample_intro.txt").read_text()
    report = run_audit(text, paper_title="Sample Neuroinflammation Paper")
    print(f"\nIntegrity score: {report.integrity_score:.0f}/100")
    print(f"Distortions found: {sum(v for k, v in report.distortion_counts.items() if k != 'none')}")
