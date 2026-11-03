"""Data-side helpers: teencode normalisation and near-duplicate detection.

Both are used by the controls that the paper needs to close its remaining
limitations: a text normalisation control, a near-duplicate control that goes
beyond exact string equality, and the split builders.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

# Vietnamese teencode and chat abbreviations that actually occur in this corpus.
TEENCODE = {
    "ko": "không", "k": "không", "kh": "không", "khg": "không", "hok": "không",
    "sp": "sản phẩm", "spham": "sản phẩm", "spam": "sản phẩm",
    "dc": "được", "đc": "được", "dk": "được", "đk": "được",
    "vs": "với", "vss": "với", "z": "vậy", "zậy": "vậy",
    "mk": "mình", "mik": "mình", "mn": "mọi người", "mng": "mọi người",
    "bn": "bạn", "b": "bạn", "ns": "nói", "nch": "nói chuyện",
    "nv": "nhân viên", "ship": "giao hàng", "shipper": "người giao hàng",
    "đt": "điện thoại", "dt": "điện thoại", "đth": "điện thoại",
    "ok": "tốt", "oke": "tốt", "okela": "tốt", "okie": "tốt",
    "đẹp": "đẹp", "xịn": "tốt", "sịn": "tốt",
    "cx": "cũng", "cg": "cũng", "cũg": "cũng",
    "r": "rồi", "rùi": "rồi", "roi": "rồi",
    "j": "gì", "z": "vậy", "s": "sao", "sao": "sao",
    "ng": "người", "ngta": "người ta", "t": "tôi", "tui": "tôi", "mk": "mình",
    "bt": "bình thường", "bth": "bình thường", "bthg": "bình thường",
    "hq": "hàng quảng cáo", "tl": "trả lời", "rep": "trả lời",
    "km": "khuyến mãi", "sale": "khuyến mãi",
    "sdt": "số điện thoại", "sđt": "số điện thoại",
    "fb": "facebook", "ib": "nhắn tin", "inbox": "nhắn tin",
    "5s": "5 sao", "5*": "5 sao", "1s": "1 sao",
}

_REPEAT_CHARS = re.compile(r"(.)\1{2,}")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, expand teencode, collapse repeated characters and whitespace."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _REPEAT_CHARS.sub(r"\1\1", text)
    tokens = text.split()
    expanded = [TEENCODE.get(token, token) for token in tokens]
    text = " ".join(expanded)
    return _WHITESPACE.sub(" ", text).strip()


def _shingles(text: str, size: int = 3) -> set[str]:
    text = _WHITESPACE.sub(" ", str(text).lower()).strip()
    if len(text) <= size:
        return {text}
    return {text[i : i + size] for i in range(len(text) - size + 1)}


def near_duplicate_indices(
    train_texts: list[str], test_texts: list[str], threshold: float = 0.8, anchor: int = 5
) -> set[int]:
    """Indices of training rows that are near-duplicates of some test row.

    Candidates are found through a shared 5-gram index, then confirmed with a
    3-gram Jaccard similarity. This is O(n) in practice for this corpus size.
    """
    test_shingles = [_shingles(text) for text in test_texts]
    index: dict[str, set[int]] = defaultdict(set)
    for row, shingles in enumerate(test_shingles):
        for gram in _anchors(shingles, anchor):
            index[gram].add(row)

    flagged: set[int] = set()
    for row, text in enumerate(train_texts):
        shingles = _shingles(text)
        candidates: set[int] = set()
        for gram in _anchors(shingles, anchor):
            candidates |= index.get(gram, set())
        for candidate in candidates:
            other = test_shingles[candidate]
            inter = len(shingles & other)
            if inter == 0:
                continue
            union = len(shingles | other)
            if union and inter / union >= threshold:
                flagged.add(row)
                break
    return flagged


def _anchors(shingles: set[str], limit: int) -> list[str]:
    """A deterministic subset of shingles, used as index keys."""
    chosen = sorted(shingles)[:limit]
    return chosen
