"""Generate the LocalAiBridge source PNG and Windows multi-size icon."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
SOURCE_SIZE = 1024
DRAW_SCALE = 4


def _scale(value: int) -> int:
    return value * DRAW_SCALE


def _box(values: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return tuple(_scale(value) for value in values)


def _points(values: list[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(round(x * DRAW_SCALE), round(y * DRAW_SCALE)) for x, y in values]


def _quadratic(
    start: tuple[int, int],
    control: tuple[int, int],
    end: tuple[int, int],
    steps: int = 48,
) -> list[tuple[float, float]]:
    points = []
    for step in range(steps + 1):
        t = step / steps
        inverse = 1 - t
        x = inverse * inverse * start[0] + 2 * inverse * t * control[0] + t * t * end[0]
        y = inverse * inverse * start[1] + 2 * inverse * t * control[1] + t * t * end[1]
        points.append((x, y))
    return points


def _tile_background(size: int) -> Image.Image:
    top = (24, 101, 244, 255)
    bottom = (5, 28, 104, 255)
    gradient = Image.new("RGBA", (size, size))
    draw = ImageDraw.Draw(gradient)
    for y in range(size):
        ratio = y / (size - 1)
        color = tuple(round(top[i] * (1 - ratio) + bottom[i] * ratio) for i in range(4))
        draw.line((0, y, size, y), fill=color)

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        _box((32, 32, 992, 992)),
        radius=_scale(204),
        fill=255,
    )
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    tile.paste(gradient, mask=mask)
    return tile


def build_source() -> Image.Image:
    """Draw one concrete bridge with a small integrated AI chip."""
    size = SOURCE_SIZE * DRAW_SCALE
    image = _tile_background(size)
    cyan = "#35E8F0"
    white = "#FFFFFF"
    navy = "#08245F"

    cable_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    cable_draw = ImageDraw.Draw(cable_layer)
    cable_points = (
        _quadratic((72, 504), (168, 450), (236, 316))
        + _quadratic((236, 316), (354, 500), (512, 500))[1:]
        + _quadratic((512, 500), (670, 500), (788, 316))[1:]
        + _quadratic((788, 316), (856, 450), (952, 504))[1:]
    )
    cable_draw.line(
        _points(cable_points),
        fill=cyan,
        width=_scale(44),
        joint="curve",
    )
    image.alpha_composite(cable_layer)

    bridge_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    bridge_draw = ImageDraw.Draw(bridge_layer)

    # Two large towers make the subject recognizable before any detail is seen.
    for tower in ((184, 270, 292, 794), (732, 270, 840, 794)):
        bridge_draw.rounded_rectangle(_box(tower), radius=_scale(30), fill=white)

    # The broad upward-curving deck visually bridges the entire icon.
    deck_points = _quadratic((92, 670), (512, 548), (932, 670), steps=64)
    bridge_draw.line(
        _points(deck_points),
        fill=white,
        width=_scale(86),
        joint="curve",
    )
    image.alpha_composite(bridge_layer)

    # A compact chip sits inside the bridge rather than becoming a separate icon.
    chip_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    chip_draw = ImageDraw.Draw(chip_layer)
    chip_draw.rounded_rectangle(
        _box((332, 310, 692, 710)),
        radius=_scale(92),
        fill=white,
    )
    chip_draw.rounded_rectangle(
        _box((388, 366, 636, 654)),
        radius=_scale(60),
        fill=navy,
    )
    chip_draw.rounded_rectangle(
        _box((448, 438, 576, 582)),
        radius=_scale(32),
        fill=cyan,
    )
    image.alpha_composite(chip_layer)

    return image.resize((SOURCE_SIZE, SOURCE_SIZE), Image.Resampling.LANCZOS)


def main() -> None:
    source = build_source()
    png_path = ROOT / "LocalAIBridge.png"
    ico_path = ROOT / "windows" / "LocalAIBridge.ico"
    ico_path.parent.mkdir(parents=True, exist_ok=True)

    source.save(png_path, optimize=True)
    source.save(
        ico_path,
        format="ICO",
        sizes=[
            (16, 16),
            (20, 20),
            (24, 24),
            (32, 32),
            (40, 40),
            (48, 48),
            (64, 64),
            (128, 128),
            (256, 256),
        ],
    )
    print(f"Generated {png_path}")
    print(f"Generated {ico_path}")


if __name__ == "__main__":
    main()
