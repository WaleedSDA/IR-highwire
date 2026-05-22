from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Iterator


@dataclass
class PositionalIndex:
    """
    Wraps PyTerrier's IterDictIndexer with positional (blocks=True) and field indexing.
    Implements the Index interface from the UML.
    """

    index_path: str
    blocks: bool = True
    _index_ref: Optional[object] = field(default=None, repr=False, init=False)
    _index: Optional[object] = field(default=None, repr=False, init=False)

    def index(self, docs: Iterator[dict]) -> None:
        import pyterrier as pt
        if not pt.started():
            pt.init()

        # fields= is omitted intentionally: PyTerrier 1.0's field-zone mode requires
        # all named fields to be non-empty or Terrier registers every document as empty
        # and skips building the lexicon. Single-stream text indexing (text_attrs default)
        # is correct for the Highwire corpus where title/journal are often absent.
        indexer = pt.IterDictIndexer(
            self.index_path,
            blocks=self.blocks,
            meta={"docno": 26, "text": 131072},
            meta_reverse=["docno"],
            overwrite=True,
        )
        self._index_ref = indexer.index(docs)
        self._index = pt.IndexFactory.of(self._index_ref)

    def load(self) -> None:
        import pyterrier as pt
        if not pt.started():
            pt.init()

        props = os.path.join(self.index_path, "data.properties")
        if not os.path.exists(props):
            raise FileNotFoundError(f"No index found at {self.index_path}")
        self._index_ref = pt.IndexRef.of(props)
        self._index = pt.IndexFactory.of(self._index_ref)

    @property
    def index_ref(self):
        return self._index_ref

    @property
    def terrier_index(self):
        return self._index

    def get_positions(self, term: str):
        self._require_loaded()
        lex = self._index.getLexicon()
        return lex.getLexiconEntry(term)

    def phrase_query(self, terms: list[str]) -> str:
        """Returns a Terrier phrase query string — handled natively by the retriever."""
        return f'"{" ".join(terms)}"'

    def proximity_search(self, term1: str, term2: str, window: int) -> str:
        """Returns a Terrier proximity query string — handled natively by the retriever."""
        return f"#{window}({term1} {term2})"

    def _require_loaded(self) -> None:
        if self._index is None:
            raise RuntimeError("Index not loaded. Call index() or load() first.")
