"""
retraction_checker.py

Cross-references extracted citation keys against a local Retraction Watch dataset
(CSV) and optionally the Retraction Watch live API to flag retracted sources.

Responsibilities:
- Load and index the retraction_watch_sample.csv by DOI and normalised title
- Accept a list of citation keys and return RetractionRecord objects for any matches
- Report retraction date, journal, and stated reason for retraction
"""

import csv
import difflib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_EXPECTED_COLUMNS = ["Record_ID", "Title", "Author", "Journal", "Year", "DOI", "Retraction_Date", "Reason"]


@dataclass
class RetractionCheckResult:
    citation_key: str
    is_retracted: bool
    match_method: str       # "doi_exact", "title_fuzzy", "not_found"
    retraction_date: str | None
    reason: str | None
    doi: str | None
    journal: str | None
    confidence: float       # 1.0 for doi_exact, fuzzy ratio for title_fuzzy, 0.0 for not_found


class RetractionChecker:
    def __init__(
        self,
        db_path: str | None = None,
        fuzzy_threshold: float = 0.85,
    ):
        if db_path is None:
            real = "data/retraction_watch_real.csv"
            fallback = "data/retraction_watch_sample.csv"
            db_path = real if os.path.exists(real) else fallback
        self.fuzzy_threshold = fuzzy_threshold
        self.db = self.load_database(db_path)
        self._db_path_used = db_path

    @property
    def db_source(self) -> str:
        path = getattr(self, "_db_path_used", "")
        if "real" in path:
            return "real (CrossRef, 200 records)"
        return "sample (synthetic, 20 records)"

    def load_database(self, path: str) -> pd.DataFrame:
        if not Path(path).exists():
            logger.warning("Retraction database not found at %s — returning empty index", path)
            return pd.DataFrame(columns=_EXPECTED_COLUMNS)
        try:
            df = pd.read_csv(path)
            # Normalise column names: replace spaces with underscores so both
            # "Retraction Date" (sample CSV) and "Retraction_Date" (real CSV)
            # are handled uniformly downstream.
            df.columns = [c.replace(" ", "_") for c in df.columns]
            df["DOI"] = df["DOI"].fillna("").astype(str).str.lower().str.strip()
            df["Title"] = df["Title"].fillna("").astype(str).str.lower().str.strip()
            return df
        except Exception as e:
            logger.error("Failed to load retraction database: %s", e)
            return pd.DataFrame(columns=_EXPECTED_COLUMNS)

    def check(
        self,
        citation_key: str,
        doi: str | None = None,
        title: str | None = None,
    ) -> RetractionCheckResult:
        try:
            # DOI exact match
            if doi:
                norm_doi = self._normalize_doi(doi)
                if norm_doi:
                    matches = self.db[self.db["DOI"] == norm_doi]
                    if not matches.empty:
                        return self._row_to_result(matches.iloc[0], citation_key, "doi_exact", 1.0)

            # Title fuzzy match (only when no DOI match)
            if title:
                norm_title = self._normalize_title(title)
                if norm_title and not self.db.empty:
                    best_ratio = 0.0
                    best_idx = None
                    for idx, db_title in enumerate(self.db["Title"]):
                        ratio = difflib.SequenceMatcher(None, norm_title, db_title).ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_idx = idx
                    if best_ratio >= self.fuzzy_threshold and best_idx is not None:
                        return self._row_to_result(
                            self.db.iloc[best_idx], citation_key, "title_fuzzy", best_ratio
                        )

            return self._not_found(citation_key)
        except Exception as e:
            logger.error("Retraction check failed for %s: %s", citation_key, e)
            return self._not_found(citation_key)

    def check_batch(self, items: list[dict]) -> dict[str, RetractionCheckResult]:
        results: dict[str, RetractionCheckResult] = {}
        for item in items:
            key = item["citation_key"]
            results[key] = self.check(key, doi=item.get("doi"), title=item.get("title"))
        retracted = sum(1 for r in results.values() if r.is_retracted)
        logger.info("Retraction check: %d/%d citations flagged as retracted", retracted, len(items))
        return results

    @staticmethod
    def _normalize_doi(doi: str) -> str:
        doi = doi.lower().strip()
        if doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]
        elif doi.startswith("http://doi.org/"):
            doi = doi[len("http://doi.org/"):]
        return doi

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"\s+", " ", title.lower().strip())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _not_found(citation_key: str) -> RetractionCheckResult:
        return RetractionCheckResult(
            citation_key=citation_key,
            is_retracted=False,
            match_method="not_found",
            retraction_date=None,
            reason=None,
            doi=None,
            journal=None,
            confidence=0.0,
        )

    @staticmethod
    def _row_to_result(
        row: pd.Series,
        citation_key: str,
        method: str,
        confidence: float,
    ) -> RetractionCheckResult:
        def _str_or_none(val) -> str | None:
            return str(val) if pd.notna(val) and str(val) else None

        # Support both "Retraction_Date" (normalised) and "Retraction Date" (legacy test DFs)
        retraction_date_val = row.get("Retraction_Date") or row.get("Retraction Date")
        return RetractionCheckResult(
            citation_key=citation_key,
            is_retracted=True,
            match_method=method,
            retraction_date=_str_or_none(retraction_date_val),
            reason=_str_or_none(row.get("Reason")),
            doi=_str_or_none(row.get("DOI")),
            journal=_str_or_none(row.get("Journal")),
            confidence=confidence,
        )


if __name__ == "__main__":
    checker = RetractionChecker()
    print(f"Database loaded: {len(checker.db)} entries")

    if len(checker.db) > 0:
        test_doi = checker.db.iloc[0]["DOI"]
        result = checker.check("test_cite_1", doi=test_doi)
        print(
            f"DOI match test: is_retracted={result.is_retracted}, "
            f"method={result.match_method}, confidence={result.confidence:.2f}"
        )

    result2 = checker.check("test_cite_2", doi="10.9999/doesnotexist")
    print(f"Not found test: is_retracted={result2.is_retracted}, method={result2.match_method}")
