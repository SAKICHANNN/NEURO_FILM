# Windows Remote GPU Setup — K-MCFM

> M5 Mac 开发 + Win RTX 5070 Ti 远程训练。SSH + Git 同步。

---

## 第 1 步：Windows — 装 OpenSSH 服务器

打开 **PowerShell（管理员）**：

```powershell
# 安装 OpenSSH Server
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# 启动服务
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'

# 确认防火墙放行端口 22
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22

# 确认服务在运行
Get-Service sshd
```

## 第 2 步：Mac — 配置免密登录

```bash
# Mac 上生成密钥（如果没有）
ssh-keygen -t ed25519 -C "m5-mac"

# 复制公钥到 Windows
# 先在 Windows PowerShell（管理员）里创建 .ssh 目录：
# New-Item -Path $env:USERPROFILE\.ssh -ItemType Directory -Force

# Mac 上执行（替换 YOURUSER 和 WIN_IP）：
scp ~/.ssh/id_ed25519.pub YOURUSER@WIN_IP:"C:\Users\YOURUSER\.ssh\authorized_keys"

# Windows PowerShell 里设置权限：
# icacls $env:USERPROFILE\.ssh\authorized_keys /inheritance:r /grant "$env:USERNAME:(R,W)"

# Mac 上测试：
ssh YOURUSER@WIN_IP
```

### 获取 Windows 的 IP

Windows PowerShell：
```powershell
ipconfig | findstr IPv4
# 找 192.168.x.x 或 10.x.x.x
```

## 第 3 步：Windows — 装 Python + CUDA

### 3.1 Python

https://www.python.org/downloads/ → 下载 Python 3.12.x **64-bit**

安装时**必须勾选** `Add Python to PATH`。安装完成后打开 PowerShell 验证：

```powershell
python --version    # 应显示 Python 3.12.x
pip --version
```

### 3.2 CUDA Toolkit 12.8

https://developer.nvidia.com/cuda-12-8-download-archive

选择：Windows → x86_64 → 10 → exe (local)

安装**只选 CUDA（不要 Driver，不要 GeForce Experience）**，因为 5070 Ti 的驱动已经自带。

验证：
```powershell
nvidia-smi
# 应显示 RTX 5070 Ti, CUDA Version: 12.8, 12GB VRAM
```

### 3.3 cuDNN（可选但推荐）

https://developer.nvidia.com/cudnn-downloads → 下载 cuDNN for CUDA 12.x → Windows

解压，把 `bin/`、`include/`、`lib/` 三个文件夹复制到 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\`

## 第 4 步：Windows — 装 Python 依赖

在 PowerShell 里（不要用 cmd）：

```powershell
# 创建项目目录
mkdir C:\projects
cd C:\projects

# 克隆仓库
git clone https://github.com/SAKICHANNN/NEURO_FILM.git
cd NEURO_FILM
git checkout codex/experiment-log

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 升级 pip
python -m pip install --upgrade pip

# 安装 PyTorch（CUDA 12.8 版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 安装核心依赖
pip install diffusers transformers accelerate peft safetensors huggingface_hub xformers

# 安装训练工具
pip install datasets bitsandbytes wandb tensorboard

# 安装可选依赖
pip install lpips piq kornia filmgrainer

# 验证 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')"
# 应显示：CUDA: True, GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU, VRAM: 12.0GB
```

## 第 5 步：Windows — 下载模型权重

```powershell
# 在 .venv 激活状态下
cd C:\projects\NEURO_FILM

# 下载 InstructPix2Pix（~3GB，用于 baseline）
python -c "
from diffusers import StableDiffusionInstructPix2PixPipeline
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    'timbrooks/instruct-pix2pix',
    torch_dtype=torch.float16, safety_checker=None
)
print('Downloaded IP2P')
"

# 下载 SDXL（~7GB，用于 LoRA 训练）
python -c "
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16, use_safetensors=True
)
print('Downloaded SDXL')
"
```

## 第 6 步：Windows — 同步数据

Git clone 只得到代码。以下数据需要手动放入（推荐从 Mac SCP 传输，已有全部数据）。

### 最终目录结构（Windows 端应长这样）

```
C:\projects\NEURO_FILM\
├── .venv/                  # Python 虚拟环境（由 setup_win.ps1 创建）
├── scripts/                # 所有训练/推理/下载脚本
├── configs/                # 配置文件
├── outputs/                # 输出图片
├── loras/                  # ⚠️ 需要放入
│   ├── velvia_50_sd15.safetensors   (1.5MB)
│   ├── hp5_sd15.safetensors         (1.5MB)
│   ├── portra_400_sd15.safetensors  (1.5MB)
│   ├── portra_800_sd15.safetensors  (1.5MB)
│   ├── vision3_500t_sd15.safetensors(1.5MB)
│   ├── vision3_250d_sd15.safetensors(1.5MB)
│   ├── ektar_100_sd15.safetensors   (1.5MB)
│   ├── tri_x_400_sd15.safetensors   (1.5MB)
│   ├── film_photography_style.safetensors (870MB, HF)
│   └── film_grain.safetensors             (163MB, HF)
├── data/
│   ├── film_domain/         # ⚠️ 需要放入（~2.6GB 总计）
│   │   ├── portra_400/      # 500 张 JPEG
│   │   ├── portra_800/      # 500 张
│   │   ├── vision3_500t/    # 500 张
│   │   ├── vision3_250d/    # 403 张
│   │   ├── ektar_100/       # 499 张
│   │   ├── tri_x_400/       # 500 张
│   │   ├── velvia_50/       # 384 张
│   │   └── hp5/             # 500 张
│   └── physics/             # 可选，参考用（~50MB）
│       ├── kodak_vision3_500t/technical_data.pdf
│       └── ...（共 12 份 PDF）
└── .env                     # ⚠️ 需要创建
    FLICKR_API_KEY=xxx        # Flickr API Key
    FLICKR_API_SECRET=xxx     # Flickr API Secret
```

### 三类数据：必须 / 建议 / 可选

| 优先级 | 目录 | 大小 | 用途 | 不装的后果 |
|:---:|------|:---:|------|------|
| **必须** | `data/film_domain/` | 2.6GB | SDXL LoRA 训练 | 无法训练任何 LoRA |
| **必须** | `.env` | 1KB | Flickr 下载、CivitAI 下载 | 无法重新下载数据 |
| 建议 | `loras/` | ~1GB | SD 1.5 推理 | 只能跑 IP2P 推理 |
| 可选 | `data/physics/` | 50MB | H&D 曲线参考 | 不影响训练 |

### 从 Mac 传输（最快）

```bash
# Mac 终端执行
cd ~/neuro_film

# 必须：胶片扫描数据（~2.6GB）
rsync -avz --progress data/film_domain/ YOURUSER@WIN_IP:"/c/projects/NEURO_FILM/data/film_domain/"

# 必须：Flickr API 密钥
scp .env YOURUSER@WIN_IP:"/c/projects/NEURO_FILM/.env"

# 建议：现有 LoRA 权重（~1GB）
rsync -avz --progress loras/ YOURUSER@WIN_IP:"/c/projects/NEURO_FILM/loras/"
```

> 替换 `YOURUSER` 和 `WIN_IP`（Windows 上 `ipconfig` 找 IPv4）
> Windows 需要先装 rsync：`winget install rsync`

## 第 7 步：日常工作流

### 7.1 Mac 开发 → Push

```bash
# Mac 上
cd ~/neuro_film
git checkout -b feature/xxx
# 写代码...
git add -A && git commit -m "xxx"
git push origin feature/xxx
```

### 7.2 Windows 拉取 → 训练

```powershell
# SSH 到 Windows
ssh YOURUSER@WIN_IP
cd C:\projects\NEURO_FILM
.venv\Scripts\activate
git pull origin feature/xxx

# 运行训练
python scripts/train_ip2p.py --film-dir data/film_domain/portra_400 --style portra_400 --steps 15000 --device cuda
```

### 7.3 Windows 把权重推送回 Mac

```powershell
# Windows 上
# 训练完成后，把权重加入 LFS 或直接 scp

# 方法1：用 git 传输小文件（LoRA < 10MB）
git add loras/portra_400_lora.safetensors
git commit -m "trained portra 400 lora"
git push

# 方法2：大文件直接 SCP 到 Mac
scp loras\portra_400_lora.safetensors YOURUSER@MAC_IP:~/neuro_film/loras/
```

### 7.4 Mac 拉取 → 验证

```bash
# Mac 上
cd ~/neuro_film
git pull
python scripts/translate.py test.jpg --style portra_400
```

## 快速命令速查

```bash
# ===== Mac 端 =====
ssh YOURUSER@WIN_IP                                    # 连 Windows
scp file.txt YOURUSER@WIN_IP:"C:\projects\file.txt"    # 传文件到 Windows
scp YOURUSER@WIN_IP:"C:\projects\result.txt" .         # 从 Windows 拉文件

# ===== Windows 端 (SSH 进去后) =====
cd C:\projects\NEURO_FILM && .venv\Scripts\activate    # 进入项目
python scripts/translate.py test.jpg --style portra_400  # 推理
python scripts/train_ip2p.py --film-dir data/film_domain/portra_400 --style portra_400 --steps 15000 --device cuda
nvidia-smi                                              # 看 GPU 状态
```

## 故障排查

| 问题 | 解决 |
|------|------|
| SSH 连接被拒绝 | Windows 防火墙：`New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22` |
| `torch.cuda.is_available()` = False | 确认装了 CUDA 12.8 + PyTorch cu128 版。不要用默认的 `pip install torch` |
| CUDA OOM（显存不足） | 减小 `--resolution 256`，加 `--train_batch_size 1`，开 `--gradient_checkpointing` |
| SCP 很慢 | 用 `rsync -avz` 代替 `scp`。Windows 上需要先装 rsync（`winget install rsync`） |
| Git push/pull 慢 | 确认两边都用 SSH key 连接 GitHub |
| xformers 报错 | 5070 Ti 是 Blackwell（sm_120），xformers 可能不支持。回退到 `F.scaled_dot_product_attention` |

---

*编写日期：2026-05-24 | K-MCFM Windows Remote GPU Setup*
