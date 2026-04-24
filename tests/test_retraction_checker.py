"""Tests for src/retraction_checker.py"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retraction_checker import RetractionCheckResult, RetractionChecker


def _make_checker(rows: list[dict] | None = None) -> RetractionChecker:
    checker = RetractionChecker.__new__(RetractionChecker)
    checker.fuzzy_threshold = 0.85
    if rows is None:
        rows = [
            {
                "Record ID": "1",
                "Title": "effects of aspirin on cardiac outcomes",
                "Author": "Smith J",
                "Journal": "NEJM",
                "Year": "2018",
                "DOI": "10.1000/aspirin2018",
                "Retraction Date": "2020-03-01",
                "Reason": "Data fabrication",
            },
            {
                "Record ID": "2",
                "Title": "gut microbiome and depression",
                "Author": "Lee K",
                "Journal": "Lancet",
                "Year": "2019",
                "DOI": "10.1001/gutmicro2019",
                "Retraction Date": "2021-06-15",
                "Reason": "Image manipulation",
            },
        ]
    df = pd.DataFrame(rows)
    df["DOI"] = df["DOI"].str.lower().str.strip()
    df["Title"] = df["Title"].str.lower().str.strip()
    checker.db = df
    return checker


# ---------------------------------------------------------------------------
# DOI matching
# ---------------------------------------------------------------------------

def test_exact_doi_match():
    checker = _make_checker()
    result = checker.check("cite1", doi="10.1000/aspirin2018")
    assert result.is_retracted is True
    assert result.match_method == "doi_exact"
    assert result.confidence == 1.0
    assert result.citation_key == "cite1"
    assert result.reason == "Data fabrication"


def test_exact_doi_case_insensitive():
    checker = _make_checker()
    result = checker.check("cite1", doi="10.1000/ASPIRIN2018")
    assert result.is_retracted is True
    assert result.match_method == "doi_exact"
    assert result.confidence == 1.0


def test_doi_not_found():
    checker = _make_checker()
    result = checker.check("cite_missing", doi="10.9999/notindb")
    assert result.is_retracted is False
    assert result.match_method == "not_found"
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Title fuzzy matching
# ---------------------------------------------------------------------------

def test_fuzzy_title_match():
    checker = _make_checker()
    # Different casing; after normalization is identical → ratio 1.0
    result = checker.check("cite2", title="Effects of Aspirin on Cardiac Outcomes")
    assert result.is_retracted is True
    assert result.match_method == "title_fuzzy"
    assert result.confidence >= 0.85


def test_fuzzy_title_below_threshold():
    checker = _make_checker()
    result = checker.check("cite3", title="completely unrelated paper about nothing")
    assert result.is_retracted is False
    assert result.match_method == "not_found"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_no_doi_no_title():
    checker = _make_checker()
    result = checker.check("cite_empty")
    assert result.is_retracted is False
    assert result.match_method == "not_found"
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def test_check_batch():
    checker = _make_checker()
    items = [
        {"citation_key": "retracted_one", "doi": "10.1000/aspirin2018"},
        {"citation_key": "clean_one", "doi": "10.9999/doesnotexist"},
    ]
    results = checker.check_batch(items)
    assert len(results) == 2
    assert results["retracted_one"].is_retracted is True
    assert results["clean_one"].is_retracted is False
    assert isinstance(results["retracted_one"], RetractionCheckResult)
