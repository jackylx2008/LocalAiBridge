# LocalAiBridge

在 Windows 主机上运行 Qwen3.8 和 ComfyUI 文生图，通过固定的局域网 API 提供给开发设备。LocalAiBridge 独占对外 8090 端口，内部 `llama-server` 使用本机 8081，ComfyUI 使用本机 8188。

## 文生图与模式切换

使用默认 SDXL 时，须将基础模型放到 ComfyUI 的 `models/checkpoints/sd_xl_base_1.0.safetensors`，并在 `common.env` 中设置 `COMFYUI_ROOT`、`COMFYUI_PYTHON` 和 `COMFYUI_CHECKPOINT`。选择 Qwen-Image-2512 时则需要下文列出的三个模型文件。生图默认 1024×1024，限制为一次一张、512–1024 像素；SDXL 最多 40 步，Qwen-Image-2512 最多 50 步。程序每次生图完成或失败后自动恢复文字模式。GUI 中有手动切换按钮；外部程序也可使用下列接口。所有接口使用现有 `LLAMACPP_API_KEY` Bearer Key。

```text
GET  /admin/status                         当前模式及任务状态
POST /admin/mode                           {"mode":"text"} 或 {"mode":"image"}
POST /v1/images/generations                {"model":"sdxl","prompt":"a red fox","width":768,"height":768,"steps":20}
POST /v1/images/generations                {"model":"qwen-image-2512","prompt":"秋日书店","width":1024,"height":1024,"steps":30}
GET  /v1/images/jobs/{id}                  查询生图任务
GET  /v1/images/jobs/{id}/content          下载生成的 PNG
```

生图提交返回 HTTP 202 和任务 ID。任务状态有 `queued`、`running`、`restoring`、`completed`、`failed`。生成图像保存在项目 `outputs/`，返回的 `image_url` 可以通过同一 Bearer Key 获取。生图切换期间，新文字请求会等待当前任务释放全局锁；如果文字模式不可用，返回 HTTP 503。现有 `/v1/chat/completions` 和 `/v1/models` 仍由固定 8090 地址转发。

Windows 上可运行 `python scripts/windows/download_sdxl.py` 分块下载并校验官方 SDXL 模型 SHA256；脚本使用镜像传输，只有校验通过才视为安装完成。模型文件约 6.94 GB。项目只管理本程序启动的后端进程。

Qwen-Image-2512 可通过 `"model":"qwen-image-2512"` 选择。它需要 ComfyUI 的 `models/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors`、`models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` 和 `models/vae/qwen_image_vae.safetensors`。`python scripts/windows/download_qwen_image_2512.py` 可分块下载并校验三个文件。人工下载时，分别从 [Comfy-Org 扩散模型](https://huggingface.co/Comfy-Org/Qwen-Image/blob/main/split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors)、[文本编码器](https://huggingface.co/Comfy-Org/Qwen-Image/blob/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) 和 [VAE](https://huggingface.co/Comfy-Org/Qwen-Image/blob/main/split_files/vae/qwen_image_vae.safetensors) 下载，放入上述目录。Qwen 默认 50 步；两个 FP8 主文件合计约 30 GB，24 GB 显卡需依靠 ComfyUI 内存调度。

本机 RTX 5090 D v2（总显存 24455 MiB）实测 Qwen：512×512/12 步耗时 81 秒，768×768/20 步耗时 73 秒，1024×1024/30 步耗时 86 秒，均包含模型切换和文字服务恢复；整卡显存采样峰值分别为 23478、23510、23350 MiB。三次均成功出图并恢复文字请求。显存余量约 1 GiB，因此当前只开放单张、最大 1024×1024；默认 50 步尚未单独计时。详情见 [2026-10-29 更新记录](docs/UPDATE_2026-10-29.md)。

## 本机准备

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item common.env.example common.env
```

在 `common.env` 中配置 `llama-server.exe`、GGUF、mmproj 路径及随机 API Key。局域网访问时将
`BRIDGE_HOST` 设为 `0.0.0.0`；`LLAMACPP_HOST` 保持 `127.0.0.1`，本机后端地址是 `http://127.0.0.1:8081/v1`。

CUDA 12 运行库统一放在项目上级的共享目录 `../vendor/cuda12`，即本机当前目录结构中的
`D:\CloudStation\Python\Project\vendor\cuda12`。`LLAMACPP_EXTRA_DLL_DIRS` 默认指向该共享目录，
不要再向各项目的 `vendor` 目录复制相同 DLL。共享目录至少应包含 `cudart64_12.dll`、
`cublas64_12.dll` 和 `cublasLt64_12.dll`。

## 启动图形界面

项目只有一个应用入口 `main.py`。Windows 日常使用推荐无终端启动：

```powershell
.\.venv\Scripts\pythonw.exe main.py
```

开发调试、希望同时在终端查看日志时使用：

```powershell
.\.venv\Scripts\python.exe main.py
```

图形界面在 Windows 运行时使用 `icons/windows/LocalAIBridge.ico` 作为窗口和任务栏图标。
`icons/macos/LocalAIBridge.icns` 用作 macOS `.app` 应用包的图标；打包时应将该文件传给所用
打包工具的应用图标选项。图标沿用参考图的机器人、无线信号和蓝色圆角底板；生成脚本裁掉
外部灰色背景与阴影，让底板占据约 94% 的画布宽度。蓝色底板不透明，圆角外透明。
原始参考文件保存在 `icons/LocalAIBridge-reference.jpg`。运行
`python icons/generate_local_ai_bridge_icon.py` 可重新生成 PNG、包含 16–256 像素帧的
Windows ICO 和 macOS ICNS。

### 构建无控制台 Windows EXE

首次构建先安装构建依赖，然后执行构建脚本：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\scripts\windows\build_windows_exe.ps1
```

生成的单文件程序位于 `build/delivery/LocalAiBridge.exe`，双击运行不会显示 CMD 窗口。程序优先读取
EXE 同目录的 `common.env` 和 `config.yaml`；从本项目的 `build/delivery` 目录直接运行时，也会自动
读取项目根目录中的配置。默认 `config.yaml` 和窗口图标同时内置在 EXE 中，但模型、llama.cpp 和本机
密钥仍使用 `common.env` 指向的外部资源。

该程序使用 Windows GUI 子系统构建；它启动的 `llama-server` 以及停止服务时调用的 `taskkill`
也使用无窗口进程标志，因此正常运行和退出均不会弹出黑色控制台。日常使用可直接运行 EXE。

`build/`、`dist/`、PyInstaller 自动生成的 `*.spec`、`.tmp*/` 和 `.pytest_tmp*/` 都是可重新生成的
本地产物，已在 `.gitignore` 中排除，不会提交到 Git。

本次图标转换和打包调整的记录见 [2026-10-14 更新说明](docs/UPDATE_2026-10-14.md)。

窗口启动后会自动在后台加载模型，显眼的进度条显示当前阶段；“服务运行信息”区域显示模型、局域网
IP、端口、Mac 访问地址和 PID。点击“健康检查”可重新检查 `/health` 与模型列表；点击“停止模型”
或关闭窗口会停止 `llama-server` 并释放内存/显存。窗口日志使用线程安全队列更新，模型加载期间界面
仍可正常移动、缩放和切换选项卡。

在“配置概览”中可设置“上下文长度（token）”。输入正整数并点击“保存并重新加载”，
程序会将其写入 `common.env` 的 `LLAMACPP_CTX_SIZE`，再以该值作为 `llama-server -c` 的参数
重新启动模型。旁边的“一键设为 16K”按钮会保存 `16384` 并重新加载。

新配置默认 `32768`（32K）；若进程没有设置同名环境变量，已有 `common.env` 中的值会覆盖
这个默认值。旧配置若仍是 `8192`，可在 GUI 中修改。数值不必是 2 的幂，例如 `24576` 或
`40960` 也可以；建议以 256 的倍数逐档测试，减少 llama.cpp 内部对齐带来的差异。
模型标称支持的上下文长度不等于当前显卡能稳定运行的长度。32K 是否可用需要实际加载和
推理验证；此前约 3.0 GiB 的显存余量不足以
保证 64K 可用。详见 [2026-09-30 更新说明](docs/UPDATE_2026-09-30.md)。

`llama-server` 意外退出时，窗口会从本次启动产生的底层日志中提炼问题本质，不再只显示退出码和
大段 Python traceback。目前可直接识别端口占用、内存或显存不足、模型加载失败；其他错误会显示
最后一条关键底层信息，并提示查看 `logs/llama_server.err.log`。

如果提示端口占用，检查对外 8090、文字后端 8081 或 ComfyUI 8188 是否已有服务。可用以下命令查看占用进程：

```powershell
Get-NetTCPConnection -LocalPort 8090 -State Listen |
  Select-Object LocalAddress, LocalPort, OwningProcess
Get-Process -Id <OwningProcess>
```

主窗口关闭时会自动停止服务并卸载模型。应用日志写入 `logs/main.log`，底层服务日志写入
`logs/llama_server.out.log` 和 `logs/llama_server.err.log`。

代码结构中的 `flows/` 保存场景工作流；`src/localai/modules/` 保存可复用的 llama.cpp 客户端等
基础能力；根目录不再保留多个 Python 工作流入口。

## Mac Mini 通过局域网访问

Windows 上运行 `main.py` 后，GUI 的“服务运行信息”会显示当前联网网卡的 IP 和 Mac 访问地址：

```text
本机局域网 IP ：192.168.x.x
Mac 访问地址  ：http://192.168.x.x:8090/v1
```

Mac Mini 必须和 Windows 主机位于同一局域网。客户端使用终端显示的“Mac 访问地址”作为 Base URL，
模型名为 `Qwen3.8-27B-Q4_K_M`，API Key 使用 Windows 主机 `common.env` 中的
`LLAMACPP_API_KEY`。不要使用 `127.0.0.1`，因为在 Mac 上该地址代表 Mac 自己。

Windows 主机需要在管理员 PowerShell 中执行一次：

```powershell
.\scripts\windows\configure_windows_firewall.ps1
```

该脚本只允许专用网络中本地子网访问 TCP 8090。不要把该端口暴露到公网，也不要把
`common.env` 或 API Key 提交到版本库。

### 1. 在 Mac Mini 配置连接信息

```bash
export LOCAL_AI_BASE_URL="<GUI 显示的 Mac 访问地址>"
export LOCAL_AI_API_KEY="<common.env 中的 LLAMACPP_API_KEY>"
export LOCAL_AI_MODEL="Qwen3.8-27B-Q4_K_M"
```

例如 GUI 显示 `http://192.168.1.100:8090/v1` 时：

```bash
export LOCAL_AI_BASE_URL="http://192.168.1.100:8090/v1"
```

### 2. 验证模型列表

```bash
curl -sS "$LOCAL_AI_BASE_URL/models" \
  -H "Authorization: Bearer $LOCAL_AI_API_KEY"
```

返回内容中应包含 `Qwen3.8-27B-Q4_K_M`。

### 3. 发送“你好”验证推理

```bash
curl -sS "$LOCAL_AI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $LOCAL_AI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-Q4_K_M",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

返回 JSON 的 `choices[0].message.content` 中有模型回答，即表示 Mac Mini 已成功访问 Windows 上的本地模型。

### 4. 配置代码编辑器或代理工具

工具支持 OpenAI 兼容接口时填写：

```text
Provider / API 类型：OpenAI Compatible
Base URL：GUI 显示的 Mac 访问地址
API Key：common.env 中的 LLAMACPP_API_KEY
Model：Qwen3.8-27B-Q4_K_M
```

如果 Mac 无法连接，请依次确认两台电脑在同一子网、Windows 网络配置文件为“专用”、防火墙脚本已用
管理员权限执行，并确认没有启用会隔离局域网设备的访客 Wi-Fi 或 VPN。

## 测试

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m pytest --basetemp .pytest_tmp
```

详细运行环境和故障排查见 [本地 AI 运行文档](docs/LOCAL_AI_RUNTIME_SETUP.md)。
本次图标与错误提示改进见 [2026-09-25 更新说明](docs/UPDATE_2026-09-25.md)。
