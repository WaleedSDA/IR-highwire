from __future__ import annotations

import logging
import time
from functools import lru_cache

import requests


NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_REQUEST_TIMEOUT = 10
_RETRY_DELAY = 1.0   # seconds to wait on 429 before one retry
_log = logging.getLogger(__name__)
_EMPTY: dict = {"synonyms": [], "related": []}


class MeSHExpander:
    """
    Expands query terms using the Medical Subject Headings (MeSH) vocabulary
    via the NCBI E-utilities API.
    Attributes: meshTree (resolved from NCBI at query time, cached per term).
    """

    def __init__(self, email: str = "user@example.com"):
        self.email = email

    @lru_cache(maxsize=512)
    def _fetch_mesh_data(self, term: str) -> dict:
        # Search with [mh] (MeSH Heading) for an exact descriptor match.
        # A bare-term search returns subheadings and supplemental concepts first,
        # which are plain-text records — not the descriptor we want.
        ids = self._esearch(f"{term}[mh]")
        if not ids:
            ids = self._esearch(term)  # fallback: accept any match
        if not ids:
            return _EMPTY

        text = self._efetch(ids[0])
        if not text:
            # Don't cache transient failures — evict this entry so the next
            # call retries against NCBI instead of returning empty forever.
            self._fetch_mesh_data.cache_clear()
            return _EMPTY

        return self._parse_text(text, term)

    def _esearch(self, term: str) -> list[str]:
        try:
            resp = requests.get(
                f"{NCBI_EUTILS}/esearch.fcgi",
                params={"db": "mesh", "term": term, "retmode": "json", "email": self.email},
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            _log.warning("MeSH esearch failed for %r: %s", term, exc)
            return []

    def _efetch(self, mesh_id: str) -> str:
        # NCBI efetch for db=mesh returns plain text regardless of retmode.
        # Retry once on 429 (rate-limit) after a short sleep.
        for attempt in range(2):
            try:
                resp = requests.get(
                    f"{NCBI_EUTILS}/efetch.fcgi",
                    params={"db": "mesh", "id": mesh_id, "email": self.email},
                    timeout=_REQUEST_TIMEOUT,
                )
                if resp.status_code == 429:
                    if attempt == 0:
                        time.sleep(_RETRY_DELAY)
                        continue
                    _log.warning("MeSH efetch rate-limited for id %r, giving up", mesh_id)
                    return ""
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                _log.warning("MeSH efetch failed for id %r: %s", mesh_id, exc)
                return ""
        return ""

    @staticmethod
    def _parse_text(text: str, term: str) -> dict:
        """
        Parse the plain-text MeSH descriptor record.
        Sections of interest:
          Entry Terms:   → synonyms (indented lines until blank / next section)
          See Also:      → related descriptors
        Lines under "All MeSH Categories" are tree navigation — ignored.
        """
        synonyms: list[str] = []
        related: list[str] = []
        section = None
        term_lower = term.lower()

        for line in text.splitlines():
            # Tree navigation — stop collecting
            if line.strip().startswith("All MeSH Categories"):
                break

            stripped = line.strip()

            if stripped.startswith("Entry Terms:"):
                section = "synonyms"
                continue
            if stripped.startswith("See Also:"):
                section = "related"
                continue
            # Any non-indented non-empty line that's a new section header
            if stripped and not line.startswith(" ") and not line.startswith("\t"):
                if stripped.endswith(":"):
                    section = None
                continue

            if section and stripped and stripped.lower() != term_lower:
                if section == "synonyms":
                    synonyms.append(stripped)
                elif section == "related":
                    related.append(stripped)

        return {"synonyms": synonyms[:8], "related": related[:4]}

    def get_synonyms(self, term: str) -> list[str]:
        return self._fetch_mesh_data(term)["synonyms"]

    def get_related(self, term: str) -> list[str]:
        return self._fetch_mesh_data(term)["related"]

    def expand(self, term: str) -> list[str]:
        data = self._fetch_mesh_data(term)
        all_terms = [term] + data["synonyms"] + data["related"]
        seen: set[str] = set()
        unique: list[str] = []
        for t in all_terms:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                unique.append(t)
        return unique
