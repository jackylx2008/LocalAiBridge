"""Resumable, hash-checked SDXL checkpoint download for restricted networks."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
SIZE = 6_938_078_334
SHA256 = "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"
CHUNK = 16 * 1024 * 1024
TARGET = Path(os.environ.get("COMFYUI_ROOT", str(Path.home() / "LocalAiBridge/ComfyUI"))) / "models/checkpoints/sd_xl_base_1.0.safetensors"
def download_ranged(url: str, target: Path, size: int, sha256: str, workers: int = 8) -> None:
    state = target.with_suffix(".parts.json")
    lock = threading.Lock()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == size and not state.exists():
        with target.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() == sha256:
                print(f"Already verified {target}", flush=True)
                return
    completed = set(json.loads(state.read_text()) if state.exists() else [])
    with target.open("ab") as output:
        output.truncate(size)
    count = (size + CHUNK - 1) // CHUNK

    def fetch(index: int) -> None:
        start = index * CHUNK
        end = min(size - 1, start + CHUNK - 1)
        for attempt in range(5):
            try:
                request = Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range", "").split("/")[0] != f"bytes {start}-{end}":
                        raise RuntimeError("服务器未按请求返回区间")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise RuntimeError("区间下载不完整")
                with target.open("r+b") as output:
                    output.seek(start)
                    output.write(data)
                with lock:
                    completed.add(index)
                    state.write_text(json.dumps(sorted(completed)))
                    if len(completed) % 10 == 0 or len(completed) == count:
                        print(f"{target.name}: {len(completed)}/{count} blocks", flush=True)
                return
            except Exception:
                if attempt == 4:
                    raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, index) for index in range(count) if index not in completed]
        for future in as_completed(futures):
            future.result()
    with target.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != sha256:
        state.unlink(missing_ok=True)
        raise RuntimeError(f"SHA256 不匹配：{digest}")
    state.unlink(missing_ok=True)
    print(f"Verified {target}")


def main() -> None:
    download_ranged(URL, TARGET, SIZE, SHA256)


if __name__ == "__main__":
    main()
