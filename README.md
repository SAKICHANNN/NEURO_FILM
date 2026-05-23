# K-MCFM: KAN-Mamba Conditional Flow Matching for Film Simulation

基于深度学习的数字到胶片（Digital-to-Film）图像转换系统。通过条件流匹配（CFM）、Mamba 状态空间模型、KAN 网络，精确模拟胶片的物理成像特性。

**目标硬件**: NVIDIA RTX 5070 Ti Laptop GPU 12GB VRAM
**输入**: 数字 RAW / 16-bit TIFF
**输出**: 胶片模拟图像（Kodak Vision3 500T / 250D、Portra 400 / 800、Ektar 100、Fujifilm Velvia 50、Ilford HP5 / Kodak Tri-X 等）

---

## 当前状态

- 当前仓库处于骨架、数据资源和环境验证阶段：`1.1` 已完成，下一步是 `1.2` 数据管线。
- 当前已落地的脚本入口是 `scripts/download_data.py`；训练、推理和评估入口仍是目标实现。
- `src/` 当前主要包含包结构和跨模块接口 stub；`configs/` 已有目标配置骨架。
- Git 基线仍待提交：2026-05-22 的 `master` 还没有 commit，项目文件仍在未跟踪工作区。

---

## 技术架构

| 组件 | 作用 |
|------|------|
| **Conditional Flow Matching (CFM)** | 模拟连续物理过程的 ODE 轨迹，OT 耦合提供直线路径 |
| **Mamba (SSMs)** | 替代 Transformer/U-Net 的视觉骨干，线性复杂度 |
| **KAN (Kolmogorov-Arnold Networks)** | B-spline 激活替代 MLP，用于物理信息层 |
| **Instance-Disentangled Attention** | 分离内容/风格语义，保持结构一致性 |
| **Latent-space Emulation** | VAE 压缩到潜空间以减少显存（目标 <2.5GB） |
| **PyTorch SDPA / xformers** | SDPA 为当前主路线，xformers 保留作后续对照优化 |
| **LoRA** | 低秩适配，快速风格切换（Portra 400 → Vision3 500T，包大小 <10MB） |

### 模拟的胶片物理特性

1. **Color Response / Tone Mapping** — H&D 曲线（Toe/Shoulder/Roll-off），Neural ODEs 或 KAN 建模
2. **Spectral Sensitivity / Color Cross-talk** — 光谱重叠、耦合器吸收剖面
3. **Micro-contrast (DIR)** — Developer Inhibitor Release，反应-扩散方程
4. **Halation** — 光散射/Bloom，多尺度高斯模拟
5. **Film Grain** — NPS（噪声功率谱）频域合成，曝光依赖性
6. **Film Non-uniformity** — Vignetting、显影条纹

---

## 目标项目结构

下面是目标实现结构。当前工作区已经建立目录、接口 stub、配置骨架和数据下载脚本；训练/推理模块会按实施阶段补齐。

```
neuro_film/
├── requirements.txt            # Python 依赖
├── IMPL_PLAN.md                # 超详细实施计划
├── GAP_ANALYSIS.md             # 资源缺口分析
├── guidelines.pdf              # 原始研究指南
│
├── data/                       # 数据集
│   ├── raw/                    # 原始数据 (RAW/TIFF + 胶片扫描)
│   ├── processed/              # 预处理后的配对数据
│   └── synthetic/              # PBR 合成数据
│
├── src/
│   ├── models/
│   │   ├── vae/                # 潜空间编解码器
│   │   ├── cfm/                # 条件流匹配模块
│   │   ├── mamba/              # Mamba 视觉骨干
│   │   ├── kan/                # KAN 物理层
│   │   ├── attention/          # 实例解耦注意力
│   │   └── film/               # 物理特性子模块
│   │       ├── tone.py         # H&D 曲线 / Tone Mapping
│   │       ├── crosstalk.py    # 光谱交叉
│   │       ├── dir.py          # 微对比度 DIR
│   │       ├── halation.py     # 光散射
│   │       ├── grain.py        # 颗粒生成
│   │       └── uniformity.py   # 非均匀性
│   ├── losses/                 # 损失函数 (感知/物理/结构)
│   ├── data/                   # 数据管线
│   ├── training/               # 训练循环 & 优化器
│   ├── eval/                   # 评估指标 & 用户研究
│   └── inference/              # 推理管线 & 导出
│
├── configs/                    # YAML 配置文件
│   ├── model/                  # 模型配置
│   ├── data/                   # 数据配置
│   ├── training/               # 训练配置
│   └── styles/                 # 胶片风格预设
│
├── scripts/                    # 训练/推理/评估脚本
└── tests/                      # 单元测试
```

---

## 快速开始

### 环境要求

- **GPU**: NVIDIA RTX 5070 Ti Laptop GPU 12GB
- **CUDA**: 12.8
- **Python**: 3.12.10
- **OS**: Windows 11 / Linux

### 安装

```powershell
# 1. 使用项目内虚拟环境
.\.venv\Scripts\activate

# 2. 当前环境为 PyTorch 2.11.0 + CUDA 12.8
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# 3. 安装/核对其余依赖
pip install -r requirements.txt

# 4. 注意当前缺口
# mamba-ssm 未安装；scikit-learn/tensorboard 已补齐

# 5. 验证安装
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('All OK')
"
```

### 当前可运行入口

```powershell
# 数据资源下载/补齐入口
python scripts/download_data.py --help
```

### ARM Mac / GitHub 迁移

Apple Silicon Mac 使用 `requirements-macos-arm.txt`，不要使用 CUDA 版 `requirements.txt`。迁移和数据重建步骤见 [docs/MAC_ARM_MIGRATION.md](docs/MAC_ARM_MIGRATION.md)。
被 Git 忽略的数据集/校准资源复现清单见 [docs/DATA_REPRODUCTION_MANIFEST.json](docs/DATA_REPRODUCTION_MANIFEST.json)。

### 目标推理入口

以下命令描述目标 CLI，当前 `scripts/inference.py` 尚未实现。

```bash
# 基础推理
python scripts/inference.py input.tiff --style vision3_500t

# 可用风格
python scripts/inference.py input.tiff --style portra_400
python scripts/inference.py input.tiff --style ektar_100
python scripts/inference.py input.tiff --style fuji_velvia_50
python scripts/inference.py input.tiff --style ilford_hp5

# 混合风格 (插值)
python scripts/inference.py input.tiff --style portra_400 --mix vision3_250d --mix-weight 0.5

# 调整参数
python scripts/inference.py input.tiff --style vision3_500t \
    --grain 0.8 \
    --halation 0.6 \
    --contrast 1.0
```

### 目标训练入口

以下命令描述目标训练入口，当前训练脚本尚未实现。

```bash
# 阶段 1: VAE 训练
python scripts/train_vae.py --config configs/training/phase1_vae.yaml

# 阶段 2: CFM + Mamba 骨干
python scripts/train_cfm.py --config configs/training/phase2_cfm.yaml

# 阶段 3: 物理特性集成
python scripts/train_physics.py --config configs/training/phase3_physics.yaml

# LoRA 风格微调
python scripts/train_lora.py --config configs/styles/kodak_portra_400.yaml
```

### 目标评估入口

以下命令描述目标评估入口，当前评估脚本和 `src.eval.user_study` 尚未实现。

```bash
# 自动指标 (FID, LPIPS, H&D RMSE, NPS 相关性)
python scripts/eval.py --checkpoint checkpoints/cfm_epoch_100.pt --data data/processed/test

# 用户研究工具
python -m src.eval.user_study --model checkpoints/cfm_epoch_100.pt --output results/2afc/
```

---

## 实现阶段

| 阶段 | 内容 | 预计时间 | 状态 |
|------|------|---------|------|
| P0 | 基线验证 (CFM+UNet) | 1 周 | 依赖 1.2 + 1.3，未开始 |
| 1 | 基础架构 (环境/数据/VAE/训练框架) | 3-4 周 | 1.1 已完成，1.2 待开始 |
| 2 | 核心模型 (CFM/Mamba/KAN/Attention) | 5-6 周 | 待开始 |
| 3 | 物理模拟集成 (6 个物理模块) | 5-6 周 | 待开始 |
| 4 | RTX 5070 Ti / 12GB 显存优化 (SDPA/xformers/FP8) | 3-4 周 | 待开始 |
| 5 | 训练与评估 (联合损失/LoRA/指标) | 4-5 周 | 待开始 |
| 6 | 部署 (ONNX/TensorRT/CLI) | 2 周 | 待开始 |

详见 [IMPL_PLAN.md](IMPL_PLAN.md)

---

## 核心参考文献

| 论文 | 会议/期刊 | 关联模块 |
|------|----------|---------|
| Lipman et al. — Flow Matching for Generative Modeling | ICLR 2023 | CFM |
| Tong et al. — Conditional Flow Matching | TMLR 2024 | OT-CFM |
| Gu & Dao — Mamba: Selective State Spaces | 2023 | Mamba |
| Dao & Gu — Mamba-2: State Space Duality | ICML 2024 | Mamba-2 加速 |
| Hatamizadeh & Kautz — MambaVision | CVPR 2025 | 视觉骨干 |
| Liu et al. — KAN: Kolmogorov-Arnold Networks | ICLR 2025 | KAN 物理层 |
| DiMSUM — Diffusion Mamba | NeurIPS 2024 | 扩散+Mamba 参考 |
| Rombach et al. — Latent Diffusion Models | CVPR 2022 | VAE 骨干 |
| Hu et al. — LoRA | ICLR 2022 | 风格适配 |

完整参考文献（30 篇）见 [IMPL_PLAN.md](IMPL_PLAN.md)

---

## 许可证

本项目代码: MIT License

预训练模型权重:
- MambaVision: [NVIDIA Source Code License-NC](https://github.com/NVlabs/MambaVision)
- Stable Diffusion VAE: [OpenRAIL++-M](https://huggingface.co/stabilityai/sd-vae-ft-mse)

第三方数据集:
- MIT-Adobe FiveK: Adobe + MIT 双许可（研究用途）
- FilmSet / FilmGrainStyle740k / DPED / Cinestill800T / 自建扫描数据: 见 [DATA_LICENSE_BOUNDARIES.md](DATA_LICENSE_BOUNDARIES.md)

---

## 致谢

本项目参考了以下开源工作:
- [torchcfm](https://github.com/atong01/conditional-flow-matching) (MIT License) — CFM 参考实现
- [MambaVision](https://github.com/NVlabs/MambaVision) (NVIDIA License) — 视觉骨干
- [pykan](https://github.com/KindXiaoming/pykan) — KAN 参考实现
- [DiMSUM](https://github.com/VinAIResearch/DiMSUM) (CC BY 4.0) — 扩散+Mamba 参考
- [diffusers](https://github.com/huggingface/diffusers) (Apache 2.0) — 扩散模型框架
- [lora](https://github.com/cloneofsimo/lora) (Apache 2.0) — LoRA 实现
