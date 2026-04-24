"""
claim_extractor.py

Extracts individual claims and their associated in-text citations from a provided
paper section (abstract, introduction, results, etc.) using Claude as the backbone LLM.

Responsibilities:
- Parse raw text and identify citation markers (e.g. [Author et al., Year])
- Decompose multi-claim sentences into atomic claim units
- Return structured Claim objects with source span, citation key, and claim text
"""

import json
import logging
import re
from dataclasses import dataclass, field

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an academic citation analyst. Extract every cited claim from the provided paragraph.

For each sentence that makes a claim backed by a citation, extract:
- claim_text: the exact claim being made (just the assertion, no citation markers)
- citation_keys: list of citation identifiers found (e.g. "[1]", "Smith et al., 2020")
- claim_type: one of causal / statistical / descriptive / mechanistic / comparative
- confidence: your confidence in the extraction (0.0-1.0)
- hedging_language: list of hedging words/phrases found ("may", "suggests", "appears to", "could", "preliminary", "possible", etc.)
- claim_strength: one of speculative / suggestive / assertive / definitive
  - speculative: uses "may", "could", "possible", "hypothesized"
  - suggestive: uses "suggests", "indicates", "appears", "seems"
  - assertive: uses "shows", "demonstrates", "found", "revealed"
  - definitive: uses "proves", "establishes", "confirms", "has been shown"

Return ONLY a JSON object with this exact structure, no other text:
{"claims": [{"claim_id": 1, "claim_text": "...", "citation_keys": [...], "claim_type": "...", "confidence": 0.0, "hedging_language": [...], "claim_strength": "..."}]}"""


@dataclass
class Claim:
    claim_id: int
    claim_text: str
    citation_keys: list[str]
    claim_type: str
    confidence: float
    source_doi: str | None
    hedging_language: list[str]
    claim_strength: str


class ClaimExtractor:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def extract(self, text: str, source_doi: str | None = None) -> list[Claim]:
        paragraphs = self._split_paragraphs(text)
        cited = [p for p in paragraphs if self._has_citation(p)]

        all_claims: list[Claim] = []
        id_offset = 0
        for paragraph in cited:
            claims = self._extract_from_paragraph(paragraph, id_offset)
            for claim in claims:
                claim.source_doi = source_doi
            all_claims.extend(claims)
            id_offset += len(claims)

        # Deduplicate by claim_text, preserving first occurrence
        seen: set[str] = set()
        unique: list[Claim] = []
        for claim in all_claims:
            if claim.claim_text not in seen:
                seen.add(claim.claim_text)
                unique.append(claim)

        # Re-assign sequential IDs starting from 1
        for i, claim in enumerate(unique, start=1):
            claim.claim_id = i

        return unique

    def _extract_from_paragraph(self, paragraph: str, id_offset: int) -> list[Claim]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": paragraph},
                ],
            )
            raw = response.choices[0].message.content.strip()
            data = self._repair_json(raw)

            claims: list[Claim] = []
            for i, item in enumerate(data.get("claims", []), start=1):
                claims.append(Claim(
                    claim_id=id_offset + i,
                    claim_text=item.get("claim_text", ""),
                    citation_keys=item.get("citation_keys", []),
                    claim_type=item.get("claim_type", "descriptive"),
                    confidence=float(item.get("confidence", 0.0)),
                    source_doi=None,
                    hedging_language=item.get("hedging_language", []),
                    claim_strength=item.get("claim_strength", "assertive"),
                ))
            return claims
        except Exception as e:
            logger.error("Claim extraction failed for paragraph: %s", e)
            return []

    def normalize_claim_query(self, claim_text: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": (
                        "Rewrite the following academic claim as a plain-language retrieval query. "
                        "Strip domain jargon and restate the core assertion simply. "
                        "Return only the rewritten query, no explanation.\n\n"
                        f"Claim: {claim_text}"
                    ),
                }],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Query normalization failed: %s", e)
            return claim_text

    @staticmethod
    def _has_citation(text: str) -> bool:
        patterns = [
            r"\[\d+\]",                               # [1], [12]
            r"\[[A-Za-z][^\]]*et al\.,?\s*\d{4}\]",  # [Author et al., Year]
            r"\([A-Za-z][^\)]*et al\.,?\s*\d{4}\)",  # (Author et al., Year)
            r"\([A-Za-z][^\)]+,\s*\d{4}\)",           # (Author, Year)
            r"\[[A-Za-z][^\]]+&[^\]]+,\s*\d{4}\]",   # [Author & Author, Year]
            r"\([A-Za-z][^\)]+&[^\)]+,\s*\d{4}\)",   # (Author & Author, Year)
        ]
        return any(re.search(p, text) for p in patterns)

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    @staticmethod
    def _repair_json(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"claims": []}


if __name__ == "__main__":
    import pathlib

    sample_path = pathlib.Path(__file__).parent.parent / "data" / "sample_papers" / "sample_intro.txt"
    text = sample_path.read_text()

    extractor = ClaimExtractor()
    claims = extractor.extract(text)

    for claim in claims:
        keys_str = ", ".join(claim.citation_keys)
        print(f"[{claim.claim_id}] ({claim.claim_strength}) {claim.claim_text[:80]}... | cites: {keys_str}")

    if claims:
        print("\nNormalized query for first claim:")
        print(extractor.normalize_claim_query(claims[0].claim_text))
