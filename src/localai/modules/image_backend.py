"""Run local ComfyUI workflows for SDXL and Qwen-Image-2512."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode


class ComfyBackend:
    def __init__(self, settings: dict, output_dir: Path) -> None:
        self.python = Path(settings["python_path"])
        self.root = Path(settings["root"])
        self.checkpoint = str(settings["checkpoint"])
        self.qwen = settings.get("qwen", {})
        self.port = int(settings.get("port", 8188))
        self.url = f"http://127.0.0.1:{self.port}"
        self.output_dir = output_dir
        self.process: subprocess.Popen | None = None

    def _json(self, path: str, payload: dict | None = None, timeout: int = 10) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.url + path, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def start(self) -> None:
        if not self.python.is_file() or not (self.root / "main.py").is_file():
            raise FileNotFoundError(f"ComfyUI 程序不存在：{self.python} / {self.root / 'main.py'}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logs = self.output_dir.parent / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / "comfyui.out.log").open("ab") as out, (logs / "comfyui.err.log").open("ab") as err:
            self.process = subprocess.Popen(
                [str(self.python), str(self.root / "main.py"), "--listen", "127.0.0.1",
                 "--port", str(self.port), "--disable-auto-launch"],
                cwd=self.root, stdout=out, stderr=err,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"ComfyUI 启动失败；查看 {logs / 'comfyui.err.log'}")
            try:
                self._json("/system_stats", timeout=3)
                return
            except (OSError, ValueError):
                time.sleep(1)
        self.stop()
        raise TimeoutError("等待 ComfyUI 启动超时")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.process = None

    def _model_file(self, folder: str, name: str) -> None:
        path = self.root / "models" / folder / name
        if not path.is_file():
            raise FileNotFoundError(f"生图模型文件不存在：{path}")
        if path.with_suffix(".parts.json").exists():
            raise RuntimeError(f"生图模型仍在下载或校验：{path}")

    def generate(self, prompt: str, negative_prompt: str, width: int, height: int, steps: int, seed: int,
                 model: str = "sdxl") -> Path:
        if model == "qwen-image-2512":
            diffusion = str(self.qwen.get("diffusion", "qwen_image_2512_fp8_e4m3fn.safetensors"))
            encoder = str(self.qwen.get("text_encoder", "qwen_2.5_vl_7b_fp8_scaled.safetensors"))
            vae = str(self.qwen.get("vae", "qwen_image_vae.safetensors"))
            for folder, name in (("diffusion_models", diffusion), ("text_encoders", encoder), ("vae", vae)):
                self._model_file(folder, name)
            workflow = {
                "1": {"class_type": "UNETLoader", "inputs": {"unet_name": diffusion, "weight_dtype": "default"}},
                "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": encoder, "type": "qwen_image", "device": "default"}},
                "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
                "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
                "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["2", 0]}},
                "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
                "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.1}},
                "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": 4,
                        "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
                        "model": ["7", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
                "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
                "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "LocalAiBridge-Qwen2512"}},
            }
            output_node = "10"
        elif model == "sdxl":
            self._model_file("checkpoints", self.checkpoint)
            workflow = self._sdxl_workflow(prompt, negative_prompt, width, height, steps, seed)
            output_node = "7"
        else:
            raise ValueError(f"不支持的生图模型：{model}")
        result = self._json("/prompt", {"prompt": workflow})
        prompt_id = result["prompt_id"]
        deadline = time.monotonic() + (1200 if model == "qwen-image-2512" else 600)
        while time.monotonic() < deadline:
            history = self._json(f"/history/{prompt_id}")
            if prompt_id in history:
                record = history[prompt_id]
                if record.get("status", {}).get("status_str") == "error":
                    raise RuntimeError("ComfyUI 生成失败；查看 comfyui.err.log")
                images = record.get("outputs", {}).get(output_node, {}).get("images", [])
                if not images:
                    raise RuntimeError("ComfyUI 未输出图片")
                image = images[0]
                query = urlencode({key: image[key] for key in ("filename", "subfolder", "type")})
                with urlopen(self.url + "/view?" + query, timeout=30) as response:
                    data = response.read()
                self.output_dir.mkdir(parents=True, exist_ok=True)
                target = self.output_dir / f"{uuid.uuid4()}.png"
                target.write_bytes(data)
                return target
            time.sleep(1)
        raise TimeoutError("ComfyUI 生图超时")

    def _sdxl_workflow(self, prompt: str, negative_prompt: str, width: int, height: int, steps: int, seed: int) -> dict:
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.checkpoint}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
            "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "5": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": 7.0,
                    "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                    "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "LocalAiBridge"}},
        }
        return workflow
