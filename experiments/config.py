"""Where the ViSpamReviews data lives.

Every script resolves the dataset through this module, so nothing depends on one
developer's home directory. The lookup order is:

1. ``VISPAM_ZIP`` / ``VISPAM_DISJOINT_DIR`` from the environment;
2. ``<repo>/data/vispamdetectionv2_official.zip`` and
   ``<repo>/data/product_disjoint_split_seed42/``;
3. a clear error that says how to obtain the file.

To set up a fresh checkout, download the release from
https://huggingface.co/datasets/sonlam1102/vispamdetection2, save the archive as
``data/vispamdetectionv2_official.zip`` (train.csv, dev.csv, test.csv inside),
then run ``python experiments/build_product_disjoint_split.py`` once.
"""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

_ZIP_NAME = "vispamdetectionv2_official.zip"
_DISJOINT_NAME = "product_disjoint_split_seed42"

_HELP = (
    "Download the ViSpamReviews v2 release from "
    "https://huggingface.co/datasets/sonlam1102/vispamdetection2 and save the "
    "archive as {target}, or point VISPAM_ZIP at it."
)


def dataset_zip() -> str:
    """Absolute path to the release archive; raises if it is not there."""
    env = os.environ.get("VISPAM_ZIP")
    target = env or os.path.join(DATA_DIR, _ZIP_NAME)
    if not os.path.exists(target):
        raise FileNotFoundError(_HELP.format(target=target))
    return target


def disjoint_dir() -> str:
    """Directory holding the product-disjoint split (created by the build script)."""
    env = os.environ.get("VISPAM_DISJOINT_DIR")
    if env:
        return env
    return os.path.join(DATA_DIR, _DISJOINT_NAME)


def audit_dir() -> str:
    """Directory for the dataset audit output."""
    return os.path.join(DATA_DIR, "dataset_audit")
