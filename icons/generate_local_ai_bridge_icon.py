"""Convert an existing square image into LocalAiBridge desktop icons.

No drawing, cropping, recoloring, or background removal is performed. The PNG
keeps the source pixels; ICO and ICNS only resample them for required sizes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
PNG_PATH = ROOT / "LocalAIBridge.png"
ICO_PATH = ROOT / "windows" / "LocalAIBridge.ico"
ICNS_PATH = ROOT / "macos" / "LocalAIBridge.icns"


def convert(source_path: Path) -> None:
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source)
        image.load()

    if image.width != image.height:
        raise ValueError(f"Icon source must be square: {source_path} ({image.size})")
    if image.width < 1024:
        raise ValueError(f"Icon source must be at least 1024 px: {source_path}")

    # Pillow detects the input format from its contents, even if the filename
    # extension is incorrect. Keep the decoded color channels unchanged.
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")

    ICO_PATH.parent.mkdir(parents=True, exist_ok=True)
    ICNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(PNG_PATH, format="PNG", optimize=True)
    image.save(
        ICO_PATH,
        format="ICO",
        sizes=[(size, size) for size in (16, 20, 24, 32, 40, 48, 64, 128, 256)],
    )
    image.save(ICNS_PATH, format="ICNS")
    for path in (PNG_PATH, ICO_PATH, ICNS_PATH):
        print(f"Generated {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=PNG_PATH,
        help="source image; defaults to icons/LocalAIBridge.png",
    )
    args = parser.parse_args()
    convert(args.source)


if __name__ == "__main__":
    main()
