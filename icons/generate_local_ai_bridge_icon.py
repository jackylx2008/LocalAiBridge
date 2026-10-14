"""Crop the reference robot icon and export opaque blue desktop icon tiles.

Only the gray surround and its shadow are removed. The robot and blue tile
come from the original image, with uniform scaling and transparent corners.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = ROOT / "LocalAIBridge-reference.jpg"
PNG_PATH = ROOT / "LocalAIBridge.png"
ICO_PATH = ROOT / "windows" / "LocalAIBridge.ico"
ICNS_PATH = ROOT / "macos" / "LocalAIBridge.icns"
OUTPUT_SIZE = 1024


def _is_tile_blue(pixel: tuple[int, int, int]) -> bool:
    red, green, blue = pixel
    return blue > red + 70 and green > red + 45 and blue > green + 20


def build_source() -> Image.Image:
    """Extract the rounded cyan tile without its gray surrounding canvas."""
    with Image.open(REFERENCE_PATH) as reference:
        image = reference.convert("RGB")

    if image.size != (2048, 2048):
        raise ValueError(f"Unexpected reference size: {image.size}")

    pixels = image.load()
    mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # The tile is the largest cyan region. Fill between its visible left and
    # right edges on each row so the robot stays fully opaque inside it.
    for y in range(256, 1792):
        blue_x = [x for x in range(2048) if _is_tile_blue(pixels[x, y])]
        if len(blue_x) >= 100:
            mask_draw.line((blue_x[0], y, blue_x[-1], y), fill=255)

    # JPEG edge pixels blend blue with the former gray backdrop. Give only the
    # outer 5 px the tile's cyan color so no pale halo survives the cutout.
    interior = mask.filter(ImageFilter.MinFilter(11))
    image = Image.composite(
        image, Image.new("RGB", image.size, image.getpixel((1024, 350))), interior
    )
    image.putalpha(mask.filter(ImageFilter.GaussianBlur(1)))

    # A 56 px margin around the former tile makes it fill about 93% of the
    # canvas, matching the reference size in Windows Explorer.
    return image.crop((200, 200, 1848, 1848)).resize(
        (OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS
    )


def main() -> None:
    source = build_source()
    ICO_PATH.parent.mkdir(parents=True, exist_ok=True)
    ICNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    source.save(PNG_PATH, format="PNG", optimize=True)
    source.save(
        ICO_PATH,
        format="ICO",
        sizes=[(size, size) for size in (16, 20, 24, 32, 40, 48, 64, 128, 256)],
    )
    source.save(ICNS_PATH, format="ICNS")
    for path in (PNG_PATH, ICO_PATH, ICNS_PATH):
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
