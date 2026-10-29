"""Download and SHA256-check the ComfyUI FP8 files for Qwen-Image-2512."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from download_sdxl import download_ranged

ROOT = Path(os.environ.get("COMFYUI_ROOT", str(Path.home() / "LocalAiBridge/ComfyUI")))
BASE = "https://hf-mirror.com/Comfy-Org/Qwen-Image/resolve/main/split_files/"
FILES = (
    ("diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors", 20_430_679_144,
     "5dc80554d5d83390046a2f4a94ece06afb7700bf7b0aaf8bde9769793875876b"),
    ("text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", 9_384_670_680,
     "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4"),
    ("vae/qwen_image_vae.safetensors", 253_806_246,
     "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"),
)


def main() -> None:
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download_ranged, BASE + name, ROOT / "models" / name, size, digest, 8)
                   for name, size, digest in FILES]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
