"""MitraSETI — Intelligent SETI Signal Analysis.

High-performance SETI signal processing powered by a Rust core with
Python ML inference.  This package exposes the compiled Rust extension
module (`mitraseti._core`) alongside the Python analysis pipeline.
"""

__version__ = "0.1.0"

try:
    from mitraseti._core import (
        DedopplerEngine,
        FilterbankHeader,
        FilterbankReader,
        RFIFilter,
        SearchParams,
        SearchResult,
        SignalCandidate,
    )

    __all__ = [
        "DedopplerEngine",
        "FilterbankReader",
        "RFIFilter",
        "SignalCandidate",
        "FilterbankHeader",
        "SearchParams",
        "SearchResult",
    ]
except ImportError:
    pass
