"""
distortion_classifier.py

Classifies the relationship between a citing claim and its retrieved source passage,
detecting common forms of citation distortion in academic literature.

Distortion types handled:
- accurate              : claim faithfully represents the source
- certainty_inflation   : hedged finding presented as established fact
- causal_overclaim      : correlational result stated as causal
- scope_inflation       : finding generalised beyond the studied population
- cherry_picking        : contradictory source evidence selectively omitted

Uses Claude with structured tool-use output to produce a label + rationale.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI

from src.claim_extractor import Claim
from src.source_retriever import RetrievalResult

load_dotenv()

logger = logging.getLogger(__name__)


class DistortionType(Enum):
    NONE = "none"
    CERTAINTY_INFLATION = "certainty_inflation"
    CAUSAL_OVERCLAIM = "causal_overclaim"
    SCOPE_INFLATION = "scope_inflation"
    CHERRY_PICKING = "cherry_picking"
    UNVERIFIABLE = "unverifiable"


@dataclass
class ClassificationResult:
    claim: Claim
    distortion_type: DistortionType
    severity: int               # 0=clean 1=minor 2=moderate 3=major 4=critical
    severity_label: str         # "clean" "minor" "moderate" "major" "critical"
    confidence: float           # 0.0–1.0
    explanation: str            # 1-2 sentence rationale citing specific text
    problematic_phrase: str     # specific phrase in the claim that is distorted
    what_source_actually_says: str
    adequacy_score: float       # 0.0–1.0: logical support, not just topical match


CLASSIFICATION_SYSTEM_PROMPT = """You are an expert academic citation integrity analyst. Your job is to detect citation distortion — when a paper misrepresents what its cited source actually found.

DISTORTION TAXONOMY:
1. none — The claim accurately represents what the source says. Hedging is preserved. Scope is correct.
2. certainty_inflation — The citing paper upgrades certainty. Source says "suggests/may/could/preliminary"; citing paper says "shows/proves/establishes/confirms". Example: Source: "results suggest a possible link". Citing paper: "research has established a direct link".
3. causal_overclaim — The citing paper asserts causation when the source only reports correlation or association. Example: Source: "X was associated with Y". Citing paper: "X causes Y".
4. scope_inflation — The citing paper generalizes beyond the source's actual population or conditions. Example: Source studied 40 college students. Citing paper: "humans generally show this effect". OR source says "should not be generalized"; citing paper claims broad applicability.
5. cherry_picking — The citing paper cites only a positive/favorable finding while the source reported mixed, null, or contradictory results in the same study.
6. unverifiable — The retrieved passages are not topically relevant enough to make a determination.

CRITICAL RULE — THE SILENT FAILURE TEST:
A passage being TOPICALLY RELATED is NOT the same as LOGICALLY SUPPORTIVE.
If a source passage warns against generalization, reports null results, or contradicts the claim — that is NOT support, even if the topic matches. Classify accordingly.

SEVERITY SCALE:
- 0/clean: no distortion
- 1/minor: slight overstatement, hedging dropped but core finding preserved
- 2/moderate: meaningful misrepresentation of certainty or scope
- 3/major: causal overclaim or significant scope inflation that could mislead
- 4/critical: claim directly contradicts what source says, or source is retracted

FEW-SHOT EXAMPLE (silent failure):
Claim: "Research has shown this cognitive effect holds broadly across all adult populations."
Source passage: "These findings should not be generalized beyond the specific undergraduate sample studied."
Classification: scope_inflation, severity 3, adequacy_score 0.05
Explanation: The source explicitly warns against generalization; the citing paper does exactly what the source warns against.
Problematic phrase: "holds broadly across all adult populations"
What source actually says: "findings should not be generalized beyond the specific undergraduate sample"

Return ONLY a JSON object with this exact structure, no other text:
{"distortion_type": "...", "severity": 0, "severity_label": "...", "confidence": 0.0, "explanation": "...", "problematic_phrase": "...", "what_source_actually_says": "...", "adequacy_score": 0.0}"""

_SEVERITY_LABELS = {0: "clean", 1: "minor", 2: "moderate", 3: "major", 4: "critical"}

_UNVERIFIABLE_RESULT_FIELDS = {
    "distortion_type": "unverifiable",
    "severity": 0,
    "severity_label": "clean",
    "confidence": 0.0,
    "explanation": "",
    "problematic_phrase": "",
    "what_source_actually_says": "",
    "adequacy_score": 0.0,
}


class DistortionClassifier:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        min_similarity_threshold: float = 0.15,
    ):
        self.model = model
        self.min_similarity_threshold = min_similarity_threshold
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def classify(self, claim: Claim, passages: list[RetrievalResult]) -> ClassificationResult:
        usable = [p for p in passages if p.score >= self.min_similarity_threshold]

        if not usable:
            return ClassificationResult(
                claim=claim,
                distortion_type=DistortionType.UNVERIFIABLE,
                severity=0,
                severity_label="clean",
                confidence=1.0,
                explanation="No sufficiently relevant source passages were retrieved to assess this claim.",
                problematic_phrase="",
                what_source_actually_says="",
                adequacy_score=0.0,
            )

        try:
            user_message = self._build_user_message(claim, usable)
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            raw = response.choices[0].message.content.strip()
            parsed = self._parse_response(raw)
            return self._build_result(claim, parsed)
        except Exception as e:
            logger.error("Classification API call failed for claim %d: %s", claim.claim_id, e)
            return ClassificationResult(
                claim=claim,
                distortion_type=DistortionType.UNVERIFIABLE,
                severity=0,
                severity_label="clean",
                confidence=0.0,
                explanation=f"Classification failed due to API error: {e}",
                problematic_phrase="",
                what_source_actually_says="",
                adequacy_score=0.0,
            )

    def classify_batch(
        self, pairs: list[tuple[Claim, list[RetrievalResult]]]
    ) -> list[ClassificationResult]:
        results: list[ClassificationResult] = []
        for i, (claim, passages) in enumerate(pairs):
            if i > 0 and i % 5 == 0:
                logger.info("Classified %d/%d claims", i, len(pairs))
            results.append(self.classify(claim, passages))
        return results

    def _build_user_message(self, claim: Claim, passages: list[RetrievalResult]) -> str:
        lines = [
            f"CITING CLAIM: {claim.claim_text}",
            f"CLAIM STRENGTH (as written): {claim.claim_strength}",
            f"HEDGING LANGUAGE PRESENT: {claim.hedging_language}",
            "",
            "RETRIEVED SOURCE PASSAGES:",
        ]
        for i, result in enumerate(passages[:3], start=1):
            lines.append(f"[Passage {i}, similarity: {result.score:.2f}]")
            lines.append(result.chunk.text)
            lines.append("")
        lines.append("Classify the distortion type. Apply the Silent Failure Test.")
        return "\n".join(lines)

    @staticmethod
    def _parse_response(raw: str) -> dict:
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
        return dict(_UNVERIFIABLE_RESULT_FIELDS)

    @staticmethod
    def _severity_label(severity: int) -> str:
        return _SEVERITY_LABELS.get(severity, "clean")

    def _build_result(self, claim: Claim, parsed: dict) -> ClassificationResult:
        raw_type = parsed.get("distortion_type", "unverifiable")
        try:
            distortion_type = DistortionType(raw_type)
        except ValueError:
            distortion_type = DistortionType.UNVERIFIABLE

        severity = int(parsed.get("severity", 0))
        return ClassificationResult(
            claim=claim,
            distortion_type=distortion_type,
            severity=severity,
            severity_label=parsed.get("severity_label") or self._severity_label(severity),
            confidence=float(parsed.get("confidence", 0.0)),
            explanation=parsed.get("explanation", ""),
            problematic_phrase=parsed.get("problematic_phrase", ""),
            what_source_actually_says=parsed.get("what_source_actually_says", ""),
            adequacy_score=float(parsed.get("adequacy_score", 0.0)),
        )


if __name__ == "__main__":
    import json as _json
    import pathlib

    from src.source_retriever import SourceChunk

    pairs_path = pathlib.Path("data/evaluation_set/labeled_pairs.json")
    pairs = _json.loads(pairs_path.read_text())[:3]

    classifier = DistortionClassifier()

    for i, pair in enumerate(pairs, start=1):
        claim = Claim(
            claim_id=pair["id"],
            claim_text=pair["citing_claim"],
            citation_keys=[],
            claim_type="descriptive",
            confidence=1.0,
            source_doi=None,
            hedging_language=[],
            claim_strength="assertive",
        )
        chunk = SourceChunk(
            chunk_id=0,
            doi="demo/eval",
            title="Evaluation Source",
            year=2024,
            text=pair["source_passage"],
            chunk_index=0,
        )
        passage = RetrievalResult(chunk=chunk, score=0.9)

        result = classifier.classify(claim, [passage])
        ground_truth = pair["distortion_type"]
        print(
            f"[{i}] Ground truth: {ground_truth} | "
            f"Predicted: {result.distortion_type.value} | "
            f"Confidence: {result.confidence:.0%} | "
            f"{result.explanation[:80]}"
        )
