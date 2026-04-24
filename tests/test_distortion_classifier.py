"""Tests for src/distortion_classifier.py"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.claim_extractor import Claim
from src.distortion_classifier import ClassificationResult, DistortionClassifier, DistortionType
from src.source_retriever import RetrievalResult, SourceChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_claim(
    claim_id: int = 1,
    claim_text: str = "X causes Y in all human populations.",
    claim_strength: str = "assertive",
    hedging_language: list | None = None,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_text=claim_text,
        citation_keys=["[1]"],
        claim_type="causal",
        confidence=0.9,
        source_doi="10.test/001",
        hedging_language=hedging_language or [],
        claim_strength=claim_strength,
    )


def _make_passage(text: str = "X was associated with Y.", score: float = 0.75) -> RetrievalResult:
    return RetrievalResult(
        chunk=SourceChunk(
            chunk_id=0,
            doi="10.test/001",
            title="Test Source",
            year=2021,
            text=text,
            chunk_index=0,
        ),
        score=score,
    )


def _mock_api_response(payload: dict):
    """Return a mock OpenAI response whose choices[0].message.content is payload JSON."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(payload)
    return mock_response


# ---------------------------------------------------------------------------
# Tests: UNVERIFIABLE short-circuits (no API call needed)
# ---------------------------------------------------------------------------

def test_classify_unverifiable_no_passages():
    with patch("src.distortion_classifier.OpenAI"):
        classifier = DistortionClassifier()

    result = classifier.classify(_make_claim(), passages=[])
    assert result.distortion_type == DistortionType.UNVERIFIABLE
    assert result.confidence == 1.0
    assert "No sufficiently relevant" in result.explanation


def test_classify_unverifiable_low_scores():
    with patch("src.distortion_classifier.OpenAI"):
        classifier = DistortionClassifier(min_similarity_threshold=0.15)

    passages = [_make_passage(score=0.05), _make_passage(score=0.08)]
    result = classifier.classify(_make_claim(), passages=passages)
    assert result.distortion_type == DistortionType.UNVERIFIABLE
    assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# Tests: API-backed classification (mocked)
# ---------------------------------------------------------------------------

def test_classify_accurate():
    api_payload = {
        "distortion_type": "none",
        "severity": 0,
        "severity_label": "clean",
        "confidence": 0.92,
        "explanation": "Claim matches source.",
        "problematic_phrase": "",
        "what_source_actually_says": "same finding",
        "adequacy_score": 0.9,
    }
    with patch("src.distortion_classifier.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_api_response(api_payload)

        classifier = DistortionClassifier()
        result = classifier.classify(_make_claim(), passages=[_make_passage(score=0.75)])

    assert result.distortion_type == DistortionType.NONE
    assert result.severity == 0
    assert result.severity_label == "clean"
    assert result.confidence == pytest.approx(0.92)
    assert result.adequacy_score == pytest.approx(0.9)


def test_classify_certainty_inflation():
    api_payload = {
        "distortion_type": "certainty_inflation",
        "severity": 2,
        "severity_label": "moderate",
        "confidence": 0.88,
        "explanation": "Source says 'suggests'; claim says 'establishes'.",
        "problematic_phrase": "establishes a causal link",
        "what_source_actually_says": "results suggest a possible association",
        "adequacy_score": 0.7,
    }
    with patch("src.distortion_classifier.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_api_response(api_payload)

        classifier = DistortionClassifier()
        result = classifier.classify(_make_claim(), passages=[_make_passage(score=0.8)])

    assert result.distortion_type == DistortionType.CERTAINTY_INFLATION
    assert result.severity == 2
    assert result.severity_label == "moderate"
    assert result.problematic_phrase == "establishes a causal link"


def test_silent_failure_scope_inflation():
    api_payload = {
        "distortion_type": "scope_inflation",
        "severity": 3,
        "severity_label": "major",
        "confidence": 0.95,
        "explanation": "Source warns against generalization; claim ignores this.",
        "problematic_phrase": "holds broadly across all adult populations",
        "what_source_actually_says": "findings should not be generalized beyond the undergraduate sample",
        "adequacy_score": 0.05,
    }
    with patch("src.distortion_classifier.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_api_response(api_payload)

        classifier = DistortionClassifier()
        passage = _make_passage(
            text="These findings should not be generalized beyond the undergraduate sample.",
            score=0.65,
        )
        result = classifier.classify(_make_claim(), passages=[passage])

    assert result.distortion_type == DistortionType.SCOPE_INFLATION
    assert result.severity == 3
    assert result.adequacy_score == pytest.approx(0.05)
    assert "undergraduate sample" in result.what_source_actually_says


# ---------------------------------------------------------------------------
# Tests: batch
# ---------------------------------------------------------------------------

def test_classify_batch_returns_all():
    api_payload = {
        "distortion_type": "none",
        "severity": 0,
        "severity_label": "clean",
        "confidence": 0.85,
        "explanation": "Accurate.",
        "problematic_phrase": "",
        "what_source_actually_says": "same",
        "adequacy_score": 0.8,
    }
    with patch("src.distortion_classifier.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_api_response(api_payload)

        classifier = DistortionClassifier()
        pairs = [
            (_make_claim(claim_id=i), [_make_passage(score=0.7)])
            for i in range(1, 4)
        ]
        results = classifier.classify_batch(pairs)

    assert len(results) == 3
    assert all(isinstance(r, ClassificationResult) for r in results)
    assert all(r.distortion_type == DistortionType.NONE for r in results)
