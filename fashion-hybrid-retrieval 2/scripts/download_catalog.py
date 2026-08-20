"""Obtain the article catalog from a free source.

Two routes, both free of charge and free of subscription. They are NOT
equivalent, and the choice matters for what can be reported.

``--source kaggle`` (DEFAULT, and the one to use for reported results)
    The original competition file, exactly as H&M released it. A Kaggle
    account is free and nothing is ever charged, but the competition rules
    must be accepted once in the browser, which cannot be automated here.
    This route yields an auditable, citable artifact: you know the catalog
    size and you can explain every exclusion, because you applied them.

``--source huggingface`` (convenience only)
    A community mirror on the Hugging Face Hub. No account needed, so it is
    useful for getting started quickly or when Kaggle is unavailable. But the
    known mirrors are PROCESSED, not raw: rows with missing fields have been
    dropped by the uploader, and pre-computed embedding columns have been
    added. That means the catalog size is not the official one and the
    exclusion criteria are not yours to document -- both of which weaken the
    label-space analysis (Section 4.2) and the provenance of any reported
    number. Pre-computed embedding columns are dropped by this script, since
    the experiment must build its own text representation (Section 4.3).

Do not mix routes across experiments. Pick one and report it.

Usage:
    python scripts/download_catalog.py                    # instructions
    python scripts/download_catalog.py --source huggingface
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "raw" / "articles.csv"

#: Public mirrors of the H&M article table on the Hugging Face Hub. Mirrors are
#: community uploads and can disappear or change schema, so more than one is
#: listed and the schema is validated after download.
MIRRORS = [
    ("Qdrant/hm_ecommerce_products", None),
    ("einrafh/hnm-fashion-recommendations-data", "data/raw/articles.csv"),
]

#: Columns added by mirrors that must never reach the pipeline: the experiment
#: builds its own product text and its own embeddings (Sections 4.2 and 4.3).
DROP_COLUMNS = {"bge_embedding", "splade_embedding", "dense_embedding", "sparse_embedding"}

REQUIRED_COLUMNS = {
    "article_id",
    "prod_name",
    "product_type_name",
    "colour_group_name",
    "index_group_name",
    "detail_desc",
}

COMPETITION = "h-and-m-personalized-fashion-recommendations"
TARGET_FILE = "articles.csv"

KAGGLE_SETUP = f"""\
One-time Kaggle setup (free, nothing is ever charged)

  1. Create a free account at https://www.kaggle.com

  2. Accept the competition rules ONCE, in the browser. This is the only step
     that cannot be automated:
     https://www.kaggle.com/competitions/{COMPETITION}/rules

  3. Create an API token: https://www.kaggle.com/settings/account
     -> "Create New Token" downloads kaggle.json. Put it at:
       Linux/macOS: ~/.kaggle/kaggle.json   (then: chmod 600 ~/.kaggle/kaggle.json)
       Windows:     C:\\Users\\<you>\\.kaggle\\kaggle.json

  4. Install the free client:
       pip install kaggle

Then re-run:
    python scripts/download_catalog.py

Only {TARGET_FILE} is downloaded (a few MB). The competition's image archive
(~25 GB) is NOT needed: ranking in this project is text-to-text.
"""


def _kaggle_credentials_present() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    config = Path(os.getenv("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    return config.exists()


def from_kaggle(out: Path) -> int:
    """Download articles.csv through the free Kaggle API client."""
    # Checked before importing kaggle: the package prints its own noisy auth
    # prompt at import time when credentials are missing.
    if not _kaggle_credentials_present():
        print("No Kaggle credentials found (~/.kaggle/kaggle.json).\n", file=sys.stderr)
        print(KAGGLE_SETUP, file=sys.stderr)
        return 1

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("The free `kaggle` client is not installed: pip install kaggle\n",
              file=sys.stderr)
        print(KAGGLE_SETUP, file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Kaggle client could not start: {exc}\n", file=sys.stderr)
        print(KAGGLE_SETUP, file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {TARGET_FILE} from the {COMPETITION} competition ...")
    try:
        api.competition_download_file(COMPETITION, TARGET_FILE, path=str(out.parent))
    except Exception as exc:
        message = str(exc)
        print(f"Download failed: {message}\n", file=sys.stderr)
        if "403" in message or "Forbidden" in message:
            print(
                "A 403 almost always means the competition rules have not been "
                "accepted yet for this account.\n",
                file=sys.stderr,
            )
        print(KAGGLE_SETUP, file=sys.stderr)
        return 1

    # The API delivers either the file itself or a zip wrapping it.
    archive = out.parent / f"{TARGET_FILE}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out.parent)
        archive.unlink()

    downloaded = out.parent / TARGET_FILE
    if downloaded != out and downloaded.exists():
        downloaded.replace(out)

    if not out.exists():
        print(f"Expected {out} after download but it is not there.", file=sys.stderr)
        return 1

    if not validate(out):
        print("Downloaded file does not have the expected schema.", file=sys.stderr)
        return 1

    import pandas as pd

    n_rows = sum(1 for _ in open(out, encoding="utf-8")) - 1
    print(f"\nWrote {n_rows} articles to {out}  (official Kaggle file)")
    print("Next: python -m fashion_retrieval prepare")
    return 0


def validate(path: Path) -> bool:
    """Check that the downloaded file has the columns the pipeline needs."""
    import pandas as pd

    try:
        head = pd.read_csv(path, nrows=5)
    except Exception as exc:
        print(f"  could not read {path}: {exc}")
        return False
    missing = REQUIRED_COLUMNS - set(head.columns)
    if missing:
        print(f"  missing expected columns: {sorted(missing)}")
        return False
    return True


def from_huggingface(out: Path) -> int:
    try:
        import pandas as pd
        from huggingface_hub import list_repo_files, snapshot_download
    except ImportError:
        print(
            "This route needs the free huggingface_hub package:\n"
            "    pip install huggingface_hub pandas\n"
            "No account or token is required for public datasets.",
            file=sys.stderr,
        )
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)

    for repo_id, hint in MIRRORS:
        print(f"Trying mirror {repo_id} ...")
        try:
            files = list_repo_files(repo_id, repo_type="dataset")
        except Exception as exc:
            print(f"  unavailable: {exc}")
            continue

        candidates = [hint] if hint and hint in files else []
        candidates += [
            f for f in files
            if f.endswith((".csv", ".parquet")) and "article" in f.lower()
        ]
        candidates += [f for f in files if f.endswith(".parquet")]
        if not candidates:
            print("  no article table found in this repo")
            continue

        try:
            local_dir = snapshot_download(
                repo_id, repo_type="dataset", allow_patterns=[candidates[0]]
            )
        except Exception as exc:
            print(f"  download failed: {exc}")
            continue

        source = Path(local_dir) / candidates[0]
        frame = (
            pd.read_parquet(source)
            if source.suffix == ".parquet"
            else pd.read_csv(source)
        )

        dropped = [c for c in frame.columns if c in DROP_COLUMNS]
        if dropped:
            frame = frame.drop(columns=dropped)
            print(f"  dropped pre-computed embedding columns: {dropped}")

        frame.to_csv(out, index=False)

        if validate(out):
            print(f"\nWrote {len(frame)} articles to {out}")
            print(
                "\nWARNING: this is a community mirror, not the official file.\n"
                "  - the uploader already removed rows with missing fields, so this\n"
                "    catalog is smaller than the official 105,542 articles and the\n"
                "    exclusion criteria are not documented by you;\n"
                "  - for results that go into the monograph, use the Kaggle route\n"
                "    (python scripts/download_catalog.py --source kaggle).\n"
                "  - whichever you pick, report it and do not mix the two."
            )
            print("\nNext: python -m fashion_retrieval prepare")
            return 0
        print("  schema does not match, trying the next mirror")

    print(
        "\nNo working mirror found. Mirrors are community uploads and do go "
        "stale.\nUse the official Kaggle route instead:\n",
        file=sys.stderr,
    )
    print(KAGGLE_INSTRUCTIONS, file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=["kaggle", "huggingface"], default="kaggle",
        help="kaggle = official file (recommended); huggingface = mirror (convenience)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.source == "kaggle":
        return from_kaggle(args.out)
    return from_huggingface(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
