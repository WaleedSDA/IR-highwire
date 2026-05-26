from __future__ import annotations

import pyterrier as pt

def init_pyterrier():
    """
    Initializes PyTerrier with forced US English locale in the JVM.
    This resolves case-folding bugs on Turkish system locales (TİTLE vs TITLE).
    """
    if not pt.java.started():
        pt.init(jvm_opts=["-Duser.language=en", "-Duser.country=US"])
