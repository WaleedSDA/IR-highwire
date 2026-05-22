from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from functools import lru_cache

import requests


NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_REQUEST_TIMEOUT = 10
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
        # --- esearch: resolve term → MeSH ID ---
        try:
            resp = requests.get(
                f"{NCBI_EUTILS}/esearch.fcgi",
                params={"db": "mesh", "term": term, "retmode": "json", "email": self.email},
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            _log.warning("MeSH esearch failed for %r: %s", term, exc)
            return _EMPTY

        if not ids:
            return _EMPTY

        # --- efetch: retrieve descriptor XML ---
        try:
            fetch_resp = requests.get(
                f"{NCBI_EUTILS}/efetch.fcgi",
                params={"db": "mesh", "id": ids[0], "retmode": "xml", "email": self.email},
                timeout=_REQUEST_TIMEOUT,
            )
            fetch_resp.raise_for_status()
            root = ET.fromstring(fetch_resp.text)
        except Exception as exc:
            _log.warning("MeSH efetch/parse failed for %r: %s", term, exc)
            return _EMPTY

        synonyms: list[str] = []
        for string_el in root.iter("String"):
            val = (string_el.text or "").strip()
            if val and val.lower() != term.lower():
                synonyms.append(val)

        related: list[str] = []
        for see_also in root.iter("SeeRelatedDescriptor"):
            name_el = see_also.find(".//String")
            if name_el is not None and name_el.text:
                related.append(name_el.text.strip())

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
