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
        import os
        import sqlite3
        self.email = os.environ.get("MESH_EMAIL", email)
        self.api_key = os.environ.get("NCBI_API_KEY", None)
        
        # Check if local compiled SQLite database exists
        self.db_path = "mesh_synonyms.db"
        if os.path.exists(self.db_path):
            _log.info("MeSHExpander: Loading in offline mode using local database '%s'", self.db_path)
            self._db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._offline = True
        else:
            _log.info("MeSHExpander: Local database '%s' not found. Operating in online API mode.", self.db_path)
            self._offline = False
            self._db_conn = None
            
        self._delay = 0.10 if self.api_key else 0.35
        self._cache: dict[str, dict] = {}
        
        # Pre-populate cache with common English stopwords
        stopwords = {
            "what", "is", "the", "a", "of", "in", "and", "to", "for", "on", "with", 
            "at", "by", "an", "be", "this", "that", "from", "are", "which", "or", 
            "as", "but", "not", "how", "why", "who", "where", "when", "does", "do", "did"
        }
        for sw in stopwords:
            self._cache[sw] = _EMPTY

    def _fetch_mesh_data(self, term: str) -> dict:
        if term in self._cache:
            return self._cache[term]

        if self._offline and self._db_conn:
            try:
                cursor = self._db_conn.cursor()
                cursor.execute("SELECT synonyms FROM mesh_synonyms WHERE term = ?", (term.lower(),))
                row = cursor.fetchone()
                if row and row[0]:
                    synonyms = row[0].split("|")[:8]
                    res = {"synonyms": synonyms, "related": []}
                    self._cache[term] = res
                    return res
                self._cache[term] = _EMPTY
                return _EMPTY
            except Exception as e:
                _log.warning("MeSH offline lookup failed for %r: %s", term, e)
                # Fallback to online search if DB fails for some reason
                pass

        # Search with [mh] (MeSH Heading) for an exact descriptor match.
        ids = self._esearch(f"{term}[mh]")
        if not ids:
            ids = self._esearch(term)  # fallback: accept any match
        if not ids:
            self._cache[term] = _EMPTY
            return _EMPTY

        text = self._efetch(ids[0])
        if not text:
            self._cache[term] = _EMPTY
            return _EMPTY

        parsed = self._parse_text(text, term)
        self._cache[term] = parsed
        return parsed

    def _esearch(self, term: str) -> list[str]:
        # NCBI rate limit: stay below allowed limit (10 req/s with key, 3 req/s without)
        for attempt in range(3):
            try:
                time.sleep(self._delay)
                params = {"db": "mesh", "term": term, "retmode": "json", "email": self.email}
                if self.api_key:
                    params["api_key"] = self.api_key
                    
                resp = requests.get(
                    f"{NCBI_EUTILS}/esearch.fcgi",
                    params=params,
                    timeout=_REQUEST_TIMEOUT,
                )
                if resp.status_code == 429:
                    time.sleep(2.0)
                    continue
                resp.raise_for_status()
                return resp.json().get("esearchresult", {}).get("idlist", [])
            except Exception as exc:
                if attempt == 2:
                    _log.warning("MeSH esearch failed for %r: %s", term, exc)
                    return []
                time.sleep(1.0)
        return []

    def _efetch(self, mesh_id: str) -> str:
        # NCBI rate limit: stay below allowed limit (10 req/s with key, 3 req/s without)
        for attempt in range(3):
            try:
                time.sleep(self._delay)
                params = {"db": "mesh", "id": mesh_id, "email": self.email}
                if self.api_key:
                    params["api_key"] = self.api_key
                    
                resp = requests.get(
                    f"{NCBI_EUTILS}/efetch.fcgi",
                    params=params,
                    timeout=_REQUEST_TIMEOUT,
                )
                if resp.status_code == 429:
                    time.sleep(2.0)
                    continue
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                if attempt == 2:
                    _log.warning("MeSH efetch failed for id %r: %s", mesh_id, exc)
                    return ""
                time.sleep(1.0)
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
        # If the term is excessively long (longer than 100 chars) or contains weights,
        # it is not a valid single MeSH term. Skip it to avoid 414 Request-URI Too Long errors.
        if len(term) > 100 or "^" in term:
            return [term]
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
