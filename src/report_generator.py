"""
report_generator.py

Assembles a structured audit report from pipeline results and renders it as
both a human-readable PDF (via ReportLab) and a machine-readable JSON artifact.

Responsibilities:
- Accept a completed AuditResult object from pipeline.py
- Render distortion findings, retraction flags, and overall integrity score
- Produce matplotlib/seaborn summary charts embedded in the PDF
- Write outputs to the /outputs directory with a timestamped filename
"""

import dataclasses
import json
import logging
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.distortion_classifier import ClassificationResult
from src.retraction_checker import RetractionCheckResult

logger = logging.getLogger(__name__)

_HEADER = "═" * 43


@dataclass
class CitationAuditEntry:
    claim_id: int
    claim_text: str
    citation_keys: list[str]
    distortion_type: str        # e.g. "certainty_inflation"
    severity: int
    severity_label: str
    confidence: float
    explanation: str
    problematic_phrase: str
    what_source_actually_says: str
    adequacy_score: float
    is_retracted: bool
    retraction_reason: str | None


@dataclass
class AuditReport:
    paper_title: str
    audit_timestamp: str
    total_claims: int
    distortion_counts: dict     # {"certainty_inflation": 2, "none": 5, ...}
    retraction_flags: int
    integrity_score: float      # 0–100
    full_text_coverage: float   # placeholder 1.0
    entries: list[CitationAuditEntry]
    overall_risk: str           # "low" | "medium" | "high"


class ReportGenerator:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        classifications: list[ClassificationResult],
        retraction_results: dict[str, RetractionCheckResult],
        paper_title: str = "Unknown Paper",
    ) -> AuditReport:
        entries: list[CitationAuditEntry] = []

        for cr in classifications:
            # Check if any citation key for this claim is flagged as retracted
            retracted = False
            retraction_reason: str | None = None
            for key in cr.claim.citation_keys:
                rr = retraction_results.get(key)
                if rr and rr.is_retracted:
                    retracted = True
                    retraction_reason = rr.reason
                    break

            entries.append(CitationAuditEntry(
                claim_id=cr.claim.claim_id,
                claim_text=cr.claim.claim_text,
                citation_keys=cr.claim.citation_keys,
                distortion_type=cr.distortion_type.value,
                severity=cr.severity,
                severity_label=cr.severity_label,
                confidence=cr.confidence,
                explanation=cr.explanation,
                problematic_phrase=cr.problematic_phrase,
                what_source_actually_says=cr.what_source_actually_says,
                adequacy_score=cr.adequacy_score,
                is_retracted=retracted,
                retraction_reason=retraction_reason,
            ))

        distortion_counts: dict[str, int] = {}
        for entry in entries:
            distortion_counts[entry.distortion_type] = (
                distortion_counts.get(entry.distortion_type, 0) + 1
            )

        retraction_flags = sum(1 for e in entries if e.is_retracted)

        score = 100.0
        for entry in entries:
            if entry.distortion_type not in ("none", "unverifiable") and entry.confidence >= 0.7:
                score -= 15.0
            if entry.is_retracted:
                score -= 25.0
        score = max(0.0, score)

        if score >= 75:
            risk = "low"
        elif score >= 40:
            risk = "medium"
        else:
            risk = "high"

        return AuditReport(
            paper_title=paper_title,
            audit_timestamp=datetime.now(timezone.utc).isoformat(),
            total_claims=len(entries),
            distortion_counts=distortion_counts,
            retraction_flags=retraction_flags,
            integrity_score=score,
            full_text_coverage=1.0,
            entries=entries,
            overall_risk=risk,
        )

    def save_json(self, report: AuditReport, filename: str | None = None) -> pathlib.Path:
        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"audit_{ts}.json"
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(report), f, indent=2)
        return path

    def generate_text_summary(self, report: AuditReport) -> str:
        n_distorted = sum(v for k, v in report.distortion_counts.items() if k != "none")
        pct = n_distorted / report.total_claims if report.total_claims else 0.0

        lines = [
            _HEADER,
            "RESEARCH CLAIM AUDITOR — AUDIT REPORT",
            _HEADER,
            f"Paper: {report.paper_title}",
            f"Audited: {report.audit_timestamp}",
            f"Integrity Score: {report.integrity_score:.0f}/100  [{report.overall_risk} risk]",
            "",
            "SUMMARY",
            f"Total claims analyzed: {report.total_claims}",
            f"Distortions flagged:   {n_distorted} ({pct:.0%})",
            f"Retracted sources:     {report.retraction_flags}",
            "",
        ]

        sorted_types = sorted(
            ((k, v) for k, v in report.distortion_counts.items() if v > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        if sorted_types:
            lines.append("DISTORTION BREAKDOWN")
            for dtype, count in sorted_types:
                lines.append(f"  {dtype}: {count}")
            lines.append("")

        lines.append("CLAIM-LEVEL FINDINGS")

        for entry in report.entries:
            is_red = entry.is_retracted or entry.severity >= 3
            is_warn = not is_red and entry.distortion_type != "none"
            icon = "🔴" if is_red else ("⚠️ " if is_warn else "✅")

            preview = entry.claim_text[:80]
            if len(entry.claim_text) > 80:
                preview += "..."

            label = entry.distortion_type
            if entry.is_retracted:
                label += " + RETRACTED SOURCE"

            if entry.distortion_type == "none" and not entry.is_retracted:
                lines.append(f'{icon} [{entry.claim_id}] {label} — "{preview}"')
            else:
                conf_pct = int(entry.confidence * 100)
                lines.append(
                    f'{icon} [{entry.claim_id}] {label} (confidence {conf_pct}%) — "{preview}"'
                )
                if entry.explanation:
                    lines.append(f"     Issue: {entry.explanation}")
                if entry.what_source_actually_says:
                    lines.append(f"     Source says: {entry.what_source_actually_says}")

        return "\n".join(lines)

    def save_text_summary(self, report: AuditReport, filename: str | None = None) -> pathlib.Path:
        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"audit_{ts}.txt"
        path = self.output_dir / filename
        path.write_text(self.generate_text_summary(report), encoding="utf-8")
        return path
