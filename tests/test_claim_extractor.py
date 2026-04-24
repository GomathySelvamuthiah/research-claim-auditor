"""Tests for src/claim_extractor.py"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.claim_extractor import Claim, ClaimExtractor


def test_has_citation_numbered():
    assert ClaimExtractor._has_citation("This finding is well supported [1].") is True


def test_has_citation_author_year():
    assert ClaimExtractor._has_citation("Evidence shows this (Smith et al., 2021).") is True


def test_has_citation_no_citation():
    assert ClaimExtractor._has_citation("Plain text with no citation marker here.") is False


def test_split_paragraphs_basic():
    text = "Para one.\n\nPara two.\n\nPara three."
    parts = ClaimExtractor._split_paragraphs(text)
    assert len(parts) == 3
    assert parts[0] == "Para one."
    assert parts[1] == "Para two."
    assert parts[2] == "Para three."


def test_repair_json_valid():
    data = {"claims": [{"claim_id": 1, "claim_text": "Test claim"}]}
    result = ClaimExtractor._repair_json(json.dumps(data))
    assert result == data


def test_repair_json_embedded():
    raw = 'Here is the result: {"claims": [{"claim_id": 1}]} — end of output.'
    result = ClaimExtractor._repair_json(raw)
    assert result == {"claims": [{"claim_id": 1}]}


def test_repair_json_broken():
    result = ClaimExtractor._repair_json("totally broken input {{{")
    assert result == {"claims": []}


def test_extract_mocked():
    fake_response_json = json.dumps({
        "claims": [
            {
                "claim_id": 1,
                "claim_text": "Diabetes increases dementia risk",
                "citation_keys": ["[1]"],
                "claim_type": "causal",
                "confidence": 0.9,
                "hedging_language": [],
                "claim_strength": "assertive",
            },
            {
                "claim_id": 2,
                "claim_text": "Neuroinflammation may drive cognitive decline",
                "citation_keys": ["[2]"],
                "claim_type": "mechanistic",
                "confidence": 0.8,
                "hedging_language": ["may"],
                "claim_strength": "speculative",
            },
        ]
    })

    mock_response = MagicMock()
    mock_response.choices[0].message.content = fake_response_json

    with patch("src.claim_extractor.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        extractor = ClaimExtractor()
        text = "Diabetes increases dementia risk [1]. Neuroinflammation may drive cognitive decline [2]."
        claims = extractor.extract(text)

    assert len(claims) == 2

    assert claims[0].claim_id == 1
    assert claims[0].claim_text == "Diabetes increases dementia risk"
    assert claims[0].citation_keys == ["[1]"]
    assert claims[0].claim_type == "causal"
    assert claims[0].confidence == 0.9
    assert claims[0].hedging_language == []
    assert claims[0].claim_strength == "assertive"
    assert claims[0].source_doi is None

    assert claims[1].claim_id == 2
    assert claims[1].claim_text == "Neuroinflammation may drive cognitive decline"
    assert claims[1].citation_keys == ["[2]"]
    assert claims[1].claim_type == "mechanistic"
    assert claims[1].confidence == 0.8
    assert claims[1].hedging_language == ["may"]
    assert claims[1].claim_strength == "speculative"
