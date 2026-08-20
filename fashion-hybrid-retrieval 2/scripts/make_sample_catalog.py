"""Generate a small synthetic catalog with the H&M articles.csv schema.

This exists so that the pipeline can be exercised end to end without
downloading the real dataset (and so that CI can run). It is a plumbing
device: no result produced from it belongs in the monograph.

Usage:
    python scripts/make_sample_catalog.py --n 2000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PRODUCT_TYPES = [
    "Dress", "Skirt", "Trousers", "Blouse", "Sweater", "T-shirt",
    "Jacket", "Shirt", "Shorts", "Vest top", "Coat", "Hoodie",
]
COLOURS = [
    "Black", "White", "Dark Blue", "Light Pink", "Beige", "Dark Green",
    "Red", "Light Blue", "Grey", "Yellow", "Brown", "Dark Red",
]
COLOUR_MASTER = {
    "Black": "Black", "White": "White", "Dark Blue": "Blue", "Light Pink": "Pink",
    "Beige": "Beige", "Dark Green": "Green", "Red": "Red", "Light Blue": "Blue",
    "Grey": "Grey", "Yellow": "Yellow", "Brown": "Brown", "Dark Red": "Red",
}
APPEARANCE = ["Solid", "All over pattern", "Stripe", "Melange", "Lace", "Denim"]
AUDIENCE = ["Ladieswear", "Menswear", "Baby/Children", "Divided", "Sport"]
GARMENT_GROUP = ["Dresses Ladies", "Jersey Basic", "Trousers", "Knitwear", "Outdoor"]
SECTIONS = ["Womens Everyday Collection", "Mens Casual", "Kids Boy", "Divided Basics"]

FABRICS = ["cotton", "viscose", "linen blend", "soft jersey", "woven fabric", "denim"]
FITS = ["relaxed", "fitted", "oversized", "regular", "slim"]
DETAILS = [
    "a round neckline and short sleeves",
    "a V-neck and long sleeves",
    "buttons down the front",
    "an elasticated waist",
    "side pockets",
    "a drawstring hem",
]
LENGTHS = ["midi", "maxi", "knee-length", "cropped", "ankle-length"]


def make_row(i: int, rng: random.Random) -> dict:
    product_type = rng.choice(PRODUCT_TYPES)
    colour = rng.choice(COLOURS)
    fabric = rng.choice(FABRICS)
    fit = rng.choice(FITS)
    detail = rng.choice(DETAILS)
    length = rng.choice(LENGTHS)

    description = (
        f"{fit.capitalize()} {product_type.lower()} in {fabric} with {detail}. "
        f"{length.capitalize()} length. Unlined."
    )
    return {
        "article_id": f"{100000000 + i}",
        "product_code": f"{100000 + i // 3}",
        "prod_name": f"{colour} {product_type}",
        "product_type_no": PRODUCT_TYPES.index(product_type),
        "product_type_name": product_type,
        "product_group_name": "Garment Upper body",
        "graphical_appearance_no": 1010016,
        "graphical_appearance_name": rng.choice(APPEARANCE),
        "colour_group_code": COLOURS.index(colour),
        "colour_group_name": colour,
        "perceived_colour_value_name": rng.choice(["Dark", "Light", "Medium", "Dusty Light"]),
        "perceived_colour_master_name": COLOUR_MASTER[colour],
        "department_name": "Jersey",
        "index_name": "Ladieswear",
        "index_group_name": rng.choice(AUDIENCE),
        "section_name": rng.choice(SECTIONS),
        "garment_group_name": rng.choice(GARMENT_GROUP),
        "detail_desc": description,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000, help="number of articles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data" / "raw" / "articles.csv"
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [make_row(i, rng) for i in range(args.n)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} synthetic articles to {args.out}")


if __name__ == "__main__":
    main()
