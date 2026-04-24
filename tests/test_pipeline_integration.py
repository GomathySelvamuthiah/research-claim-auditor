"""Integration test for src/pipeline.py"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.claim_extractor import Claim
from src.distortion_classifier import ClassificationResult, DistortionType
from src.pipeline import run_audit
from src.report_generator import AuditReport


def _fake_claim() -> Claim:
    return Claim(
        claim_id=1,
        claim_text="Neuroinflammation drives cognitive decline in diabetic patients [1].",
        citation_keys=["[1]"],
        claim_type="causal",
        confidence=0.9,
        source_doi=None,
        hedging_language=[],
        claim_strength="assertive",
    )


def _fake_classification(claim: Claim) -> ClassificationResult:
    return ClassificationResult(
        claim=claim,
        distortion_type=DistortionType.NONE,
        severity=0,
        severity_label="clean",
        confidence=0.9,
        explanation="Claim accurately represents the source.",
        problematic_phrase="",
        what_source_actually_says="same finding",
        adequacy_score=0.9,
    )


def test_run_audit_structure():
    mini_text = (
        "Neuroinflammation drives cognitive decline in diabetic patients [1]. "
        "Gut dysbiosis correlates with systemic endotoxemia [2]."
    )
    claim = _fake_claim()
    classification = _fake_classification(claim)

    with (
        patch("src.pipeline.ClaimExtractor") as MockExtractor,
        patch("src.pipeline.SourceRetriever") as MockRetriever,
        patch("src.pipeline.DistortionClassifier") as MockClassifier,
        patch("src.pipeline.RetractionChecker") as MockChecker,
    ):
        mock_extractor = MockExtractor.return_value
        mock_extractor.extract.return_value = [claim]
        mock_extractor.normalize_claim_query.return_value = "neuroinflammation cognitive decline"

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = []

        mock_classifier = MockClassifier.return_value
        mock_classifier.classify_batch.return_value = [classification]

        mock_checker = MockChecker.return_value
        mock_checker.check_batch.return_value = {}

        report = run_audit(
            mini_text,
            paper_title="Test Paper",
            save_outputs=False,
        )

    assert isinstance(report, AuditReport)
    assert report.total_claims == 1
    assert 0 <= report.integrity_score <= 100
    assert isinstance(report.entries, list)
    assert len(report.entries) == 1
    assert report.entries[0].distortion_type == "none"
    assert report.entries[0].is_retracted is False
