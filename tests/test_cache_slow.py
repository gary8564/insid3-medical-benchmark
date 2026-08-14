"""Optional checks against a local processed cache (skipped if data/ is absent)."""

from __future__ import annotations

import pytest

from src.data.paths import PROCESSED_ROOT
from src.datasets.cardiac_loader import list_pairs as list_cardiac
from src.datasets.kidney_tumor_loader import list_pairs as list_kidney
from src.datasets.polyp_loader import list_pairs as list_polyp

EXPECTED_MIN = {
    "polyp": 900,
    "kidney_tumor": 2,
    "cardiac": 2,
}


@pytest.mark.slow
def test_processed_cache_counts():
    if not PROCESSED_ROOT.is_dir():
        pytest.skip("data/processed is not present")

    loaders = {
        "polyp": list_polyp,
        "kidney_tumor": list_kidney,
        "cardiac": list_cardiac,
    }
    for domain, loader in loaders.items():
        domain_root = PROCESSED_ROOT / domain
        if not domain_root.is_dir():
            continue
        pairs = loader(PROCESSED_ROOT)
        assert len(pairs) >= EXPECTED_MIN[domain]
