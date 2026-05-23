# K-MCFM: 超详细项目实施计划

---

## 状态边界

- 本文档描述目标实现、模块接口和原始实施草案，不等同于当前仓库文件清单。
- 当前仓库状态见 `AGENTS.md`、`TASK_BOARD.md` 和 `README.md`：`1.1` 环境已验证，下一步是 `1.2` 数据管线。
- 当前环境事实以 `requirements.txt` 为准：Python 3.12.10、CUDA 12.8、PyTorch 2.11.0+cu128、RTX 5070 Ti Laptop GPU；本计划中保留的早期 RTX 3060 / cu121 片段只作为原始方案参考。

---

## 文献综述与研究背景

### 1. 核心方法论文献

| # | 论文 | 作者/会议 | 关键贡献 | 与本项目关系 |
|---|------|----------|---------|------------|
| 1 | Flow Matching for Generative Modeling | Lipman et al., ICLR 2023 | 提出 CFM，仿真自由训练 CNF；OT-CFM 提供更简单、更稳定的训练 | **核心训练框架** |
| 2 | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu & Dao, 2023 | 选择性 SSM，输入依赖的参数化，硬件感知并行算法 | **视觉骨干基础** |
| 3 | Mamba-2: State Space Duality | Dao & Gu, 2024 | SSD 框架，比 Mamba-1 快 2-8×，理论连接 SSM 与 Attention | **加速骨干** |
| 4 | Vision Mamba (Vim) | Zhu et al., ICML 2024 | 双向 Mamba 用于视觉，比 DeiT 快 2.8×，内存节省显著 | **视觉 Mamba 验证** |
| 5 | MambaVision: A Hybrid Mamba-Transformer Vision Backbone | Hatamizadeh & Kautz, CVPR 2025 | 前 2 阶段 Conv + 后 2 阶段 Mamba/Attention 混合，SOTA 分类 | **直接参考架构** |
| 6 | KAN: Kolmogorov-Arnold Networks | Liu et al., ICLR 2025 | 边上学得的 B-spline 激活替代 MLP，更快缩放律，可解释 | **物理层基础** |
| 7 | LESA: Learnable Stage-Aware Predictors for Diffusion Acceleration | CVPR 2026 | KAN 用于扩散模型加速，5-6.25× 加速，质量损失极小 | **KAN 在扩散中应用** |
| 8 | DiMSUM: Diffusion Mamba | NeurIPS 2024 | 小波+Mamba+Transformer 混合扩散骨干，开源 | **直接参考架构** |

### 2. 胶片模拟/风格迁移文献

| # | 论文 | 会议/期刊 | 方法 | 结果与局限 |
|---|------|---------|------|-----------|
| 9 | Film-GAN: Towards Realistic Analog Film Photo Generation | Neural Computing & Applications, 2024 | GAN + Gram 矩阵风格迁移 | 首个用 Gram 矩阵做胶片风格；局限于单风格，未建模物理 |
| 10 | A Large-Scale Film Style Dataset for Learning Multi-frequency Driven Film Enhancement | Li et al., IJCAI 2023 | 多频段增强 + 专用数据集 | 43 引用；提供了数据集但方法仍是传统增强，非物理模拟 |
| 11 | CNNs for Style Transfer of Digital to Film Photography | Mackenzie et al., 2024 | CNN 风格迁移 (Cinestill-800T) | 验证了 CNN 可模拟胶片；实验规模小，仅单一胶片 |
| 12 | The Automation of Style: Seeing Photographically in Generative AI | Media Theory, 2024 | 理论分析 | 分析了相机厂商内置胶片模拟的历史和 AI 风格迁移的关系 |

### 3. 扩散+Mamba 前沿文献

| # | 论文 | 关键点 |
|---|------|--------|
| 13 | DiM: Diffusion Mamba (Mo & Tian, 2024) | 双向 SSM 替代 Attention，线性复杂度 |
| 14 | Dimba: Transformer-Mamba Diffusion (2024) | 交替堆叠 Transformer 和 Mamba 层做文生图 |
| 15 | U-Shape Mamba (USM, CVPR 2025 Workshop) | U-Net+Mamba，FLOPs 仅 1/3 |
| 16 | LinGen (CVPR 2025) | Mamba-2 做文生视频，单 GPU 生成分钟级高清视频 |
| 17 | AiM (2024) | 自回归 Mamba 图像生成，FID 2.21 on ImageNet 256 |
| 18 | StyMam (ICASSP 2026) | Mamba 生成器做艺术风格迁移 |

### 4. 关键工具/基础设施

| # | 工具 | 作用 |
|---|------|------|
| 19 | FlashAttention-2 (Dao, 2023) | GPU 注意力加速背景；当前 RTX 5070 Ti 路线优先使用 PyTorch SDPA |
| 20 | LoRA (Hu et al., 2021) | 低秩适配，1-6MB 风格包 |
| 21 | Stable Diffusion VAE (Rombach et al., CVPR 2022) | 8× 下采样，4 通道潜空间 |
| 22 | AV1 Film Grain Synthesis | 参数化颗粒模型：强度/相关/长宽比 |

---

## 从已有工作中提炼的关键教训

### 成功经验

1. **潜空间压缩是必须的**: SD VAE 的 8× 潜空间下采样使得 12GB VRAM 上可以训练生成模型。几乎所有成功的扩散模型（SD, DiT, DiMSUM）都在潜空间操作。

2. **混合架构优于纯架构**: MambaVision 的 Conv+Mamba+Attention 混合设计优于纯 Mamba 或纯 Transformer。DiMSUM 用小波低频增强 Mamba 证明了多路径融合的价值。

3. **Mamba 替代 Attention 在生成任务中已验证**: DiMSUM (NeurIPS 2024), DiM, Dimba, LinGen (CVPR 2025) 等多篇顶会论文明确定了 Mamba 可以替代扩散模型中的 Attention。

4. **LoRA 极高效用于风格切换**: LoRA 在 SD 社区的广泛应用证明，1-6MB 的秩分解矩阵可以实现完整的风格切换。PTI (Pivotal Tuning Inversion) 工作流效果最好。

5. **CFM 训练比扩散更稳定**: OT-CFM 的 Straight ODE 路径在训练稳定性和推理步数上均优于传统扩散路径。

### 失败教训与陷阱

1. **KAN 不是即插即用的**: KAN 作者明确警告 KAN "尚不能作为开箱即用的插件"。在复杂 CV 任务中直接替换 MLP 可能无法收敛。**正确用法**: 限制在物理损失函数和低维特征映射。

2. **胶片模拟纯深度学习方法尚未商业化**: 所有商用产品（Dehancer, DxO FilmPack, RNI Films）仍使用传统 LUT + 图像处理。Film-GAN 和 CNN 风格迁移论文存在但影响有限。这说明**仅靠端到端学习不足以物理精确地模拟胶片**。

3. **数据集是最大瓶颈**: 配对数据（同一场景数字+胶片）极其稀缺。FiveK 负责 RAW→专业后期预训练，FilmSet 负责胶片风格分布，Kodak 技术图表负责物理曲线，PBR 负责可控配对，自建数据负责最终验证。任何单一数据源都不够。

4. **12GB 本机训练需要精心优化**: 当前 RTX 5070 Ti Laptop GPU 仍按 12GB 预算设计，优先用 BF16、8-bit Adam、Gradient Checkpointing 和 PyTorch SDPA。不能假设可训练大体量附加编码器。批量大小仍受限。

5. **过于复杂的组合会导致调试噩梦**: 三个新组件（CFM + Mamba + KAN）同时集成时，任何一个出问题都无法定位。必须分层验证。

---

## 目标项目目录结构

以下目录树是目标实现结构。当前仓库只完成目录骨架、接口 stub、配置骨架、数据资源和 `scripts/download_data.py`。

```
neuro_film/
├── data/
│   ├── raw/                     # 原始数据 (RAW/TIFF + 扫描胶片)
│   │   ├── digital/             # 数字拍摄
│   │   ├── film/                # 胶片扫描 (配对)
│   │   └── metadata.csv         # ISO, 光圈, 快门, 胶片类型, 光照条件
│   ├── processed/               # 预处理后的配对数据
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── synthetic/               # PBR 合成数据 (物理渲染器)
│       ├── kodak_vision3_500t/
│       ├── fuji_velvia_50/
│       ├── kodak_ektar_100/
│       └── generic/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vae/
│   │   │   ├── __init__.py
│   │   │   ├── encoder.py       # SD-style VAE encoder (8× latent downsample)
│   │   │   ├── decoder.py       # VAE decoder
│   │   │   ├── quantizer.py     # DiagonalGaussian + KL loss
│   │   │   └── discriminator.py # PatchGAN 判别器 (可选 VAE-GAN)
│   │   ├── cfm/
│   │   │   ├── __init__.py
│   │   │   ├── flow_matching.py # CFM 5 个类 + OT 耦合
│   │   │   ├── velocity_net.py  # 速度场预测网络
│   │   │   ├── ode_solver.py    # Dormand-Prince / Euler / RK4 求解器
│   │   │   └── scheduler.py     # 时间步调度 (均匀/余弦/指数)
│   │   ├── mamba/
│   │   │   ├── __init__.py
│   │   │   ├── ssm_block.py     # 选择性扫描 SSM 核心
│   │   │   ├── mamba_layer.py   # Mamba 层 (含 Conv1d + SiLU)
│   │   │   ├── vision_backbone.py # MambaVision-style 4阶段骨干
│   │   │   ├── scan.py          # 双向/四向空间扫描策略
│   │   │   └── mamba2_block.py  # Mamba-2 SSD (可选加速)
│   │   ├── kan/
│   │   │   ├── __init__.py
│   │   │   ├── kan_layer.py     # KANLayer: B-spline + 残差 base_fun(SiLU)
│   │   │   ├── kan_network.py   # KAN 网络封装 (多层 + 剪枝 + 符号拟合)
│   │   │   ├── bspline.py       # B-spline 基函数、网格扩展、coef2curve
│   │   │   └── physics_kan.py   # 物理学专用 KAN 层 (H&D, DIR 等)
│   │   ├── attention/
│   │   │   ├── __init__.py
│   │   │   ├── instance_disentangled.py  # 实例解耦注意力
│   │   │   └── cross_attention.py        # 跨模态交叉注意力
│   │   └── film/
│   │       ├── __init__.py
│   │       ├── tone.py          # H&D 曲线 / Tone Mapping (Neural ODE)
│   │       ├── crosstalk.py     # 光谱交叉 / Coupler 模拟
│   │       ├── dir.py           # 微对比度 DIR (反应扩散 + PINNs)
│   │       ├── halation.py      # 光散射 Halation (Helmholtz + 反向传播)
│   │       ├── grain.py         # 颗粒 (Poisson-Binomial + NPS)
│   │       └── uniformity.py    # 非均匀性 (Vignetting/条纹)
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── perceptual.py        # VGG-19 LPIPS, AlexNet LPIPS
│   │   ├── physical.py          # H&D RMSE, DIR Loss, NPS Loss
│   │   ├── structural.py        # SSIM, MS-SSIM, HaarWavelet
│   │   ├── adversarial.py       # LSGAN / Hinge GAN loss
│   │   └── identity.py          # 一致性损失 (循环一致性等)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # PairedFilmDataset (h5/memmap)
│   │   ├── augment.py           # RAW 处理增强 (曝光/白平衡/噪声)
│   │   ├── loader.py            # DataLoader (预取/内存固定)
│   │   ├── raw_io.py            # RAW 解码 (LibRaw/rawpy wrapper)
│   │   └── synthetic_gen.py     # Mitsuba/Blender PBR 合成数据生成
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # 主训练循环 (混合精度 + GC)
│   │   ├── lora.py              # LoRA 注入/训练/合并/插值
│   │   ├── optimizer.py         # 8-bit Adam, GC 配置, EMA
│   │   ├── memory.py            # 显存预算管理 & 分析
│   │   └── distributed.py       # DDP (可选多卡)
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py           # FID, KID, IS, LPIPS
│   │   ├── physical_metrics.py  # H&D RMSE, NPS 对比, ΔE2000
│   │   ├── alignment.py         # SIFT/SURF 特征点对齐评估
│   │   └── user_study.py        # 2AFC 用户研究工具
│   └── inference/
│       ├── __init__.py
│       ├── pipeline.py          # 两阶段推理: 潜空间快速流 + 像素精炼
│       ├── export.py            # ONNX / TensorRT 导出
│       └── cli.py               # CLI: python -m src.inference.cli input.tiff --style vision3
├── configs/
│   ├── base.yaml                # 基础配置
│   ├── model/
│   │   ├── vae_tiny.yaml        # 轻量 VAE (<2.5GB)
│   │   ├── cfm_base.yaml        # CFM 配置
│   │   ├── mamba_tiny.yaml      # Mamba-T (31.8M)
│   │   └── kan_physics.yaml     # KAN 物理层配置
│   ├── data/
│   │   ├── paired.yaml          # 配对数据配置
│   │   └── synthetic.yaml       # 合成数据配置
│   ├── training/
│   │   ├── phase1_vae.yaml      # 第一阶段: VAE 训练
│   │   ├── phase2_cfm.yaml      # 第二阶段: CFM + Mamba
│   │   ├── phase3_physics.yaml  # 第三阶段: 物理特性
│   │   └── phase4_optimize.yaml # 第四阶段: RTX 5070 Ti / 12GB 优化
│   └── styles/
│       ├── kodak_vision3_500t.yaml
│       ├── fuji_velvia_50.yaml
│       ├── kodak_ektar_100.yaml
│       ├── kodak_portra_800.yaml
│       ├── kodak_vision3_250d.yaml
│       ├── kodak_tri_x_400.yaml
│       ├── kodak_portra_400.yaml
│       └── ilford_hp5.yaml
├── scripts/
│   ├── train_vae.py
│   ├── train_cfm.py
│   ├── train_physics.py
│   ├── train_lora.py
│   ├── inference.py
│   └── eval.py
├── tests/
│   ├── test_cfm.py
│   ├── test_mamba.py
│   ├── test_kan.py
│   ├── test_vae.py
│   ├── test_grain.py
│   └── test_pipeline.py
├── requirements.txt
├── AGENTS.md                    # Agent 知识库与当前状态
├── TASK_BOARD.md                # 多 Agent 协调板
├── README.md                    # 对外项目说明
├── GAP_ANALYSIS.md              # 资源缺口分析
├── IMPL_PLAN.md                 # 本文件
└── guidelines.pdf               # 原始研究指南
```

---

## 第一阶段: 基础架构搭建 (预计 3-4 周)

### 1.1 项目环境搭建 (2 天)

#### 当前已验证环境覆盖

当前环境锁定在仓库根目录的 `requirements.txt`，与下面保留的早期草案快照不同：

| 项目 | 当前状态 |
|------|---------|
| Python | 3.12.10 |
| CUDA / PyTorch | CUDA 12.8 / PyTorch 2.11.0+cu128 |
| GPU | RTX 5070 Ti Laptop GPU 11.94GB |
| Attention | PyTorch SDPA 为主；`xformers` 已安装供后续对照 |
| Mamba | `causal-conv1d==1.6.1` 已安装；`mamba-ssm` 仍未安装 |
| KAN / 日志 | `pykan==0.2.8`、`scikit-learn==1.8.0`、`pandas==3.0.2`、`tensorboard==2.20.0` 已验证 |

#### 初始计划 requirements 快照

下面版本块保留早期 RTX 3060 / cu121 草案，不能替代当前 `requirements.txt`。
```
torch==2.4.0+cu121
torchvision==0.19.0+cu121
causal-conv1d==1.4.0          # Mamba SSM CUDA 核心
mamba-ssm==2.2.2               # Mamba-2 Python 接口
flash-attn==2.6.3              # FlashAttention-2 (Ampere)
xformers==0.0.27.post2         # 内存高效注意力
diffusers==0.30.0              # 参考架构 (VAE 等)
timm==1.0.8                    # 预训练 backbone
kornia==0.7.2                  # 图像处理
einops==0.8.0                  # 张量操作
rawpy==0.22.0                  # RAW 解码
opencv-python-headless==4.10.0 # 图像 I/O
scikit-image==0.24.0           # SSIM 等指标
omegaconf==2.3.0               # 配置管理
hydra-core==1.3.2              # 高级配置
wandb==0.17.5                  # 实验跟踪
tensorboard==2.17.0
pytorch-lightning==2.3.3       # 可选训练框架
bitsandbytes==0.43.3           # 8-bit 优化器
accelerate==0.33.0             # 训练加速
pykan==0.2.2                   # KAN 官方实现
pytorch-wavelets==1.3.0        # 小波变换 (DiMSUM 参考)
ninja                          # JIT 编译 CUDA kernel
triton==3.0.0                  # 自定义 kernel (可选)
lpips==0.1.4                   # 感知损失
piq==0.8.0                     # 图像质量指标集合
numpy==1.26.4
scipy==1.13.1
pillow==10.4.0
matplotlib==3.9.0
tqdm==4.66.4
```

#### 初始计划环境初始化脚本 `scripts/setup.sh`

下面脚本同样是早期草案。当前 Windows `.venv` 环境以根目录 `requirements.txt` 和 `README.md` 的当前安装说明为准。
```bash
conda create -n kmcfm python=3.10 -y
conda activate kmcfm
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install causal-conv1d mamba-ssm --no-build-isolation
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
# 验证 GPU
python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
```

#### 配置文件系统 (configs/base.yaml)
```yaml
# 基础配置 - 被所有其他配置继承
project: "K-MCFM"
seed: 42
device: "cuda"
precision: "bf16-mixed"  # 当前 12GB 主路线默认 BF16；FP8 留给后续验证

# 显存预算 (12GB)
memory:
  vram_total_mb: 11264    # 12GB - 系统开销
  vram_target_inference_mb: 3800
  vram_target_training_mb: 10240
  gradient_checkpointing: true
  use_8bit_adam: true

# 数据
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  synthetic_dir: "data/synthetic"
  resolution: 512          # 训练分辨率
  crop: "random"           # random | center | none
  batch_size: 2            # 12GB 兼容 (潜空间后可调到4-8)

# 日志
logging:
  wandb_project: "kmcfm"
  log_every_n_steps: 50
  save_every_n_epochs: 5
  checkpoint_dir: "checkpoints"
```

### 1.2 数据管线 (5 天)

#### 数据格式规范
- **配对数据**: 每组包含 `{id}_digital.{tiff,raw}` + `{id}_film.{tiff}` + `{id}_meta.json`
- **元数据必需字段**: iso, aperture, shutter_speed, focal_length, film_stock, lighting_condition, white_balance_k
- **处理后的存储格式**: HDF5 或 numpy memmap (随机访问，内存友好)

#### 数据策略 V2 — 多源 manifest

所有数据源统一索引到 `data/processed/manifest.jsonl`，每行一条样本，避免把 FiveK、FilmSet、PBR、真实胶片强行塞进同一种目录命名。

```json
{"id":"fivek_a0001","source":"fivek","task":"raw_to_expert","input":"data/processed/fivek/input/a0001.tiff","target":"data/processed/fivek/expert_c/a0001.tiff","style":"expert_c","license":"fivek_research","weight":1.0}
{"id":"filmset_000001","source":"filmset","task":"digital_to_film_style","input":"...","target":"...","style":"filmset_style_0","license":"check_kaggle","weight":0.7}
{"id":"pbr_vision3_500t_000001","source":"pbr","task":"digital_to_physical_film","input":"...","target":"...","style":"kodak_vision3_500t","license":"generated","weight":1.0}
{"id":"cinestill800t_0001","source":"real_pair","task":"digital_to_real_film","input":"...","target":"...","style":"cinestill_800t","license":"paper_dataset","weight":0.3}
```

数据源职责:
- **FiveK DNG + Expert C TIFF16**: 已下载；用于 VAE 重建和 RAW→专家后期 CFM 预训练。
- **FilmSet**: 已下载并解压；用于胶片风格分布、multi-frequency 风格监督、LoRA 预热。
- **物理/色彩校准资料**: 技术 PDF 已放在 `data/physics/`；CIE/相机光谱数据已放在 `data/calibration/`；数字化后输出 `data/physics/{stock}/hd_curve.csv`, `spectral_sensitivity.csv`, `mtf.csv`, `grain.csv`。
- **PBR 合成**: 用 Mitsuba 3 spectral variant 生成完美对齐数字/胶片模拟对。
- **FilmGrainStyle740k / 自建灰卡扫描**: 只用于颗粒先验和 NPS 校准，按许可隔离，不进入可再分发数据包。
- **真实小配对**: Cinestill800T 公开小数据或自建 200-500 对，用于最终验证和过拟合 sanity check。

执行顺序:
1. 写 `scripts/prepare_fivek.py`: 解包/索引 DNG，检查并下载/导入 Expert TIFF。
2. 写 `src/data/manifest_dataset.py`: 读取 manifest，按 task/source 输出统一 batch。
3. 写 `scripts/build_manifest.py`: 生成 train/val/test split，记录 license/source/style/weight。
4. 写 `scripts/digitize_film_curves.py`: 先落 CSV schema，手动数字化结果可直接放入。
5. 再启动 VAE 和 CFM baseline。

#### `src/data/raw_io.py` — RAW 解码器
```python
import rawpy
import numpy as np
import torch

class RAWPipeline:
    """
    RAW 文件标准化处理管线
    
    步骤:
    1. 黑电平减法
    2. 白平衡乘法
    3. 去马赛克 (Demosaicing)
    4. 色彩空间转换 (Camera RGB → XYZ → Linear sRGB)
    5. 色调映射 (log/linear)
    6. 归一化到 [0, 1]
    """
    def __init__(self, 
                 demosaic_method: str = "AMaZE",  # rawpy 最佳质量
                 output_color_space: str = "srgb",
                 bit_depth: int = 16):
        ...
    
    def process(self, raw_path: str) -> torch.Tensor:
        """返回 (3, H, W) float32 tensor [0, 1]"""
        with rawpy.imread(raw_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=self.bit_depth,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AMAZE,
                no_auto_bright=True,  # 不做自动亮度
                output_color=getattr(rawpy.ColorSpace, self.output_color_space)
            )
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / (2**self.bit_depth - 1)
        return tensor
```

#### `src/data/augment.py` — 数据增强
```python
class FilmAugmentation:
    """
    针对胶片模拟的特定增强:
    - 曝光偏移 (模拟不同曝光)
    - 白平衡扰动 (不同色温/色调)
    - 传感器噪声 (模拟数字噪声)
    - 镜头模糊/锐化 (模拟不同镜头)
    - 透视畸变 (PBR 合成数据不太需要的增强)
    - 颜色抖动 (HSV 空间)
    """
    def __init__(self, p=0.5):
        self.transforms = [
            ExposureShift(range_ev=(-2, 2)),
            WhiteBalancePerturb(kelvin_range=(2500, 10000)),
            PoissonGaussianNoise(),
            RandomCrop(size=512),
            RandomHorizontalFlip(),
        ]
```

#### `src/data/synthetic_gen.py` — PBR 合成数据
```python
class PBRFilmRenderer:
    """
    使用物理渲染器 (Mitsuba 3) 生成合成配对数据
    
    场景模板:
    - Macbeth ColorChecker (24 色标准参考)
    - HDR 环境贴图 (各种光照条件)
    - 标准测试图 (ISO 12233 等)
    
    胶片模拟通过以下方式实现:
    1. 光谱渲染 (波长采样而非 RGB)
    2. Kodak 已发布的光谱灵敏度曲线
    3. 化学过程的简化 ODE 模型
    """
    def render_pair(self, scene: Scene, film_stock: FilmStock) -> tuple[Tensor, Tensor]:
        """
        返回 (digital_render, film_simulation) 精确配对
        """
```

### 1.3 VAE 潜空间编解码器 (7-10 天)

#### 架构设计

基于 Stable Diffusion AutoencoderKL 但做以下修改：

```
Encoder:
  Input (B, 3, H, W)
  → Conv2d(3→64, 3×3, p=1) + SiLU
  → ResBlock(64→128, stride=2)  → (B, 128, H/2, W/2)   [Stage 1]
  → ResBlock(128→256, stride=2) → (B, 256, H/4, W/4)   [Stage 2]
  → ResBlock(256→512, stride=2) → (B, 512, H/8, W/8)   [Stage 3]
  → ResBlock(512→512, stride=1) → (B, 512, H/8, W/8)   [Stage 4]
  → MidBlock(512 → 512, w/ Attention)
  → Conv2d(512→8, 3×3, p=1)    → (B, 8, H/8, W/8)
  → DiagonalGaussian: split mean(4ch) + logvar(4ch)
  
Decoder:
  逆向，上采样用 nearest+conv (不用转置卷积，减少棋盘伪影)
  latent: 4 channels × (H/8) × (W/8)
  总压缩: 3 × 8 × 8 / 4 = 48× 张量元素压缩
```

#### 关键超参数
```yaml
vae:
  latent_channels: 4
  block_out_channels: [64, 128, 256, 512]  # 标准 SD 的 1/2 (节省显存)
  layers_per_block: 2
  attention_resolutions: [16]  # 仅在 16×16 及以下加 attention
  dropout: 0.0
  kl_weight: 0.000001  # 极小的 KL 权重，保真度优先
  perceptual_loss_weight: 1.0  # LPIPS
  adversarial_loss_weight: 0.5
  discriminator_start_step: 50000  # 先训重建，后加 GAN
```

#### VAE 训练策略
```
阶段 A (0-50k steps): 纯重建 (L1 + LPIPS)
  学习率: 1e-4, batch=4, 分辨率=256
  目标: 确保 VAE 能可靠编解码

阶段 B (50k-150k steps): 加入 KL + GAN
  学习率: 1e-4→5e-5 (cosine decay)
  加入 PatchGAN 判别器
  KL weight 从 0→1e-6 线性 warmup (前 20k steps)
  目标: PSNR > 32dB, SSIM > 0.92, LPIPS < 0.08

阶段 C (150k-250k steps): 高分辨率微调
  分辨率: 512, batch=1 (gradient accumulation=4)
  目标: 最终 PSNR > 34dB, 潜空间可用
```

#### VAE 训练损失
```python
def vae_loss(recon, target, mu, logvar, 
             discriminator=None, global_step=0) -> dict:
    """
    返回: {
        'total': total_loss,
        'recon_l1': L1_loss,
        'perceptual': LPIPS_loss,
        'kl': KL_divergence,
        'gan': GAN_generator_loss (if discriminator is not None)
    }
    """
    # 重建损失
    l1 = F.l1_loss(recon, target)
    lpips_loss = lpips_fn(recon, target)
    
    # KL 散度 (Diagonal Gaussian)
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()
    
    # 总损失
    total = l1 + 1.0 * lpips_loss + kl_weight * kl
    
    if discriminator is not None:
        # LSGAN generator loss
        fake_pred = discriminator(recon)
        gan_loss = ((fake_pred - 1) ** 2).mean()
        total += 0.5 * gan_loss
    
    return {'total': total, 'recon_l1': l1, 'perceptual': lpips_loss, 'kl': kl}
```

### 1.4 训练/验证框架 (3 天)

#### `src/training/trainer.py` 核心设计
```python
class KMCfmtrainer:
    """
    统一的训练循环, 支持:
    - 混合精度 (BF16)
    - 梯度累积
    - 梯度裁剪
    - Gradient Checkpointing
    - EMA (指数移动平均)
    - 多阶段学习率调度
    - WandB/TensorBoard 日志
    - 定期保存检查点 + 验证
    """
    def __init__(self, model, optimizer, config):
        self.scaler = torch.cuda.amp.GradScaler('cuda')  # BF16 不需要 scaler，但 FP16 保留
        self.ema = EMA(model, decay=0.9999)
    
    def train_step(self, batch):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss_dict = self.model(batch)
        
        total_loss = loss_dict['total'] / self.gradient_accumulation_steps
        total_loss.backward()
        
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.ema.update()
    
    def validate(self, dataloader):
        """每 N epoch 验证: FID, LPIPS, PSNR, 物理指标"""
```

---

## 第二阶段: 核心模型 (预计 5-6 周)

### 2.1 CFM 模块 (5 天)

#### `src/models/cfm/flow_matching.py` — 完整 CFM 实现
```python
"""
基于 torchcfm (MIT License) 重写，适配图像到图像翻译。

5 个 CFM 变体:
1. ConditionalFlowMatcher — 独立耦合 (x0,x1)
2. ExactOptimalTransportConditionalFlowMatcher — OT 耦合 ★ 主用
3. TargetConditionalFlowMatcher — Lipman 2023, 高斯→数据
4. SchrodingerBridgeConditionalFlowMatcher — 熵正则化 OT
5. VariancePreservingConditionalFlowMatcher — 三角插值保方差
"""

import torch
import torch.nn.functional as F
from torchcfm.utils import OTPlanSampler  # 复用 OT 采样器

def pad_t_like_x(t, x):
    """时间标量广播到与 x 相同的 spatial 维度"""
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

class ConditionalFlowMatcher:
    """基础 CFM: 独立耦合 (x0, x1) ~ q(x0)q(x1)"""
    def __init__(self, sigma=1e-5):
        self.sigma = sigma  # 小 sigma 使路径接近直线
    
    def compute_mu_t(self, x0, x1, t):
        """条件均值: mu_t = t * x1 + (1-t) * x0"""
        t_pad = pad_t_like_x(t, x0)
        return t_pad * x1 + (1 - t_pad) * x0
    
    def compute_sigma_t(self, t):
        """条件标准差: 常数 sigma (接近确定的 ODE)"""
        return self.sigma
    
    def sample_xt(self, x0, x1, t, eps):
        """从条件概率路径采样: xt ~ N(mu_t, sigma_t^2 I)"""
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        return mu_t + sigma_t * eps
    
    def compute_conditional_flow(self, x0, x1, t, xt):
        """条件向量场: u_t(xt|x0,x1) = x1 - x0"""
        return x1 - x0
    
    def training_sample(self, x0, x1):
        """
        训练用采样
        
        Args:
            x0: 数字图像 (源)  (B, C, H, W)
            x1: 胶片图像 (目标) (B, C, H, W)
        
        Returns:
            t:   时间 (B,)
            xt:  中间状态 (B, C, H, W)
            ut:  目标向量场 (B, C, H, W)
        """
        B = x0.shape[0]
        t = torch.rand(B, device=x0.device)
        eps = torch.randn_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        return t, xt, ut


class ExactOptimalTransportCFM(ConditionalFlowMatcher):
    """
    OT-CFM ★ 主训练类
    
    用 minibatch OT 耦合替代独立采样，产生更直的轨迹
    
    算法:
    1. 对每个 minibatch，计算 (x0, x1) 的成对距离矩阵
    2. 求解 Earth Mover's Distance (EMD) / 线性分配
    3. 按 OT 计划重新配对 (x0, x1)
    4. 用重配对后的 (x0, x1) 计算条件向量场
    """
    def __init__(self, sigma=1e-5, ot_method="exact"):
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(method=ot_method)  # "exact" = EMD
    
    def training_sample(self, x0, x1):
        """带 OT 耦合的训练采样"""
        B = x0.shape[0]
        # 展平空间维度用于距离计算
        x0_flat = x0.reshape(B, -1)
        x1_flat = x1.reshape(B, -1)
        
        # OT 重配对
        x0_ot, x1_ot = self.ot_sampler.sample_plan(x0_flat, x1_flat)
        x0_ot = x0_ot.reshape_as(x0)
        x1_ot = x1_ot.reshape_as(x1)
        
        # 后续与基础 CFM 相同，但用 OT 后的配对
        t = torch.rand(B, device=x0.device)
        eps = torch.randn_like(x0)
        xt = self.sample_xt(x0_ot, x1_ot, t, eps)
        ut = self.compute_conditional_flow(x0_ot, x1_ot, t, xt)
        return t, xt, ut
```

#### `src/models/cfm/velocity_net.py` — 速度场网络
```python
class VelocityFieldPredictor(nn.Module):
    """
    预测向量场 v_theta(xt, t)
    
    输入: 潜空间中的噪声图像 xt + 时间 t
    输出: 与 xt 相同 shape 的速度向量
    
    架构选择（渐进式）:
    1. Phase 2a: 简单 U-Net (验证 CFM 流程)
    2. Phase 2b: MambaVision backbone (替换 U-Net)
    3. Phase 2c: 加入 KAN 物理调节
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        # 时间嵌入
        self.time_embed = SinusoidalTimeEmbedding(config.embed_dim)
        # 图像骨干
        self.backbone = self._build_backbone(config)
        # 输出头
        self.output_head = nn.Sequential(
            nn.GroupNorm(32, config.latent_channels),
            nn.SiLU(),
            nn.Conv2d(config.latent_channels, config.latent_channels, 3, padding=1)
        )
    
    def forward(self, xt, t):
        """
        xt: (B, 4, H/8, W/8)    潜空间表示
        t:  (B,)                 时间步
        
        返回: v_theta(xt, t) same shape as xt
        """
        t_emb = self.time_embed(t)  # (B, embed_dim)
        features = self.backbone(xt, t_emb)
        return self.output_head(features)
```

#### `src/models/cfm/ode_solver.py` — ODE 求解器
```python
class ODESolver:
    """
    常微分方程求解器用于推理
    
    方法 (效率递增):
    - dopri5: Dormand-Prince 5(4), 自适应步长，最高精度，慢
    - rk4:   经典 4 阶 Runge-Kutta，固定步长
    - euler: 显式 Euler，最快，精度最低
    - midpoint: 中点法，Euler 和 RK4 的折中
    
    对于 Straight ODE (OT-CFM): 用 Euler 或中点法 30-50 步即可
    """
    def __init__(self, method="dopri5", atol=1e-5, rtol=1e-5):
        self.method = method
        self.atol = atol
        self.rtol = rtol
    
    @torch.no_grad()
    def solve(self, velocity_fn, x0, t_span=(0, 1), n_steps=50):
        """
        求解 dx/dt = v_theta(x, t) from t=0 to t=1
        
        Args:
            velocity_fn: callable (x, t) -> v
            x0: 初始状态 (源图像潜变量)
            n_steps: 推理步数
        
        Returns:
            x1: 生成结果 (胶片图像潜变量)
        """
        if self.method == "euler":
            return self._euler(velocity_fn, x0, t_span, n_steps)
        elif self.method == "midpoint":
            return self._midpoint(velocity_fn, x0, t_span, n_steps)
        elif self.method == "rk4":
            return self._rk4(velocity_fn, x0, t_span, n_steps)
        else:
            return self._dopri5(velocity_fn, x0, t_span)
```

### 2.2 Mamba 视觉骨干 (10-12 天)

#### 架构设计 — 基于 MambaVision + DiMSUM 经验

设计选择理由:
- **前半段卷积**: 高分辨率特征提取不需要长距离依赖
- **中间 Mamba SSM**: 中等分辨率时建立空间关系，线性复杂度
- **后半段 Attention**: 低分辨率时已有全局上下文，标准的 self-attention 最优
- **加入小波分支** (DiMSUM 启发): 增强局部结构感知

```python
class KMFM_MambaBackbone(nn.Module):
    """
    K-MCFM 专用 Mamba 视觉骨干
    
    架构:
    Input (B, 4+embed_dim, H/8, W/8)  [潜空间 + 时间嵌入]
    
    Stage 1: Conv × 2  (latent grid→latent grid,     dim=96)
    Stage 2: Conv × 3  (latent grid→latent grid/2,   dim=192)
    Stage 3: Mamba SSM × 6 + Attention × 2  (latent grid/2→latent grid/4, dim=384)
    Stage 4: Mamba SSM × 3 + Attention × 2  (latent grid/4, dim=768)
    Output head restores the latent grid for velocity prediction.
    
    关键创新: 双向 + 四向空间扫描
    - 前 3 个 Mamba 块: 行优先扫描 (left→right)
    - 后 3 个 Mamba 块: 四向扫描 (L→R, R→L, T→B, B→T)
    - Attention 块: 仅在最后 2 个位置，作为全局关系增强器
    """
    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList([
            self._build_conv_stage(dim=96, depth=2, input_dim=config.latent_channels),
            self._build_conv_stage(dim=192, depth=3, stride=2),
            self._build_mamba_stage(dim=384, mamba_depth=6, attn_depth=2, stride=2),
            self._build_mamba_stage(dim=768, mamba_depth=3, attn_depth=2, stride=1),
        ])
```

#### Mamba SSM 块核心实现 (`src/models/mamba/ssm_block.py`)
```python
class MambaSSMBlock(nn.Module):
    """
    选择性状态空间模型块
    
    参数:
        d_model: 输入/输出通道数
        d_state: SSM 状态维度 (默认 16，增强记忆)
        d_conv: 卷积核大小 (默认 4)
        expand: 扩展因子 (默认 2)
        dt_rank: Δ投影秩 (auto = ceil(d_model/16))
        scan_direction: "unidirectional" | "bidirectional" | "quad-directional"
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, 
                 dt_rank="auto", scan="quad-directional"):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand)
        
        if dt_rank == "auto":
            dt_rank = math.ceil(d_model / 16)
        
        # 输入投影: x → (x_branch, z_branch)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # 1D 卷积预处理器 (深度可分离)
        self.conv1d_x = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, 
            groups=self.d_inner, padding=d_conv - 1
        )
        self.conv1d_z = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            groups=self.d_inner, padding=d_conv - 1
        )
        
        # SSM 参数
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner)
        
        # 初始化 A, D
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        self.scan = scan
    
    def forward(self, x):
        """
        x: (B, L, d_model) — 序列格式 [L = H×W]
        
        处理流程:
        1. x, z = split(in_proj(x))         # 门控双分支
        2. x = SiLU(conv1d_x(x))            # 局部特征
        3. dt, B, C = x_proj(x)             # 选择性参数
        4. y = selective_scan(x, dt, A, B, C, D)  # SSM 核心
        5. output = out_proj(y * SiLU(conv1d_z(z)))  # 门控 + 投影
        """
        residual = x
        B, L, D = x.shape
        
        # 多方向扫描
        if self.scan == "bidirectional":
            x_fwd = self._ssm_forward(x)
            x_rev = self._ssm_forward(torch.flip(x, dims=[1]))
            x_out = (x_fwd + torch.flip(x_rev, dims=[1])) / 2
        elif self.scan == "quad-directional":
            # 4 方向: L→R, R→L, T→B, B→T
            x_out = self._quad_scan(x)
        else:
            x_out = self._ssm_forward(x)
        
        return x_out + residual  # 残差连接
```

#### MambaVision 配置 — Tiny 变体 (适配当前 12GB 预算)
```python
MAMBA_TINY_CONFIG = {
    'depths': [2, 3, 8, 3],       # 每 stage 块数 (原 T 为 [1,3,8,4])
    'dim': 80,                      # 基础通道 (原 80)
    'in_dim': 32,                   # Patch Embed 中间维度
    'mlp_ratio': 4,
    'd_state': 8,                   # SSM 状态维度 (减小以节省显存)
    'd_conv': 3,                    # Conv1d 核大小
    'expand': 1,                    # Mamba 扩展 (1 = 无扩展，省显存)
    'conv_stages': [0, 1],          # 前 2 阶段用 Conv
    'mamba_stages': [2, 3],         # 后 2 阶段用 Mamba+Attention
    'attn_in_last_blocks': True,    # 最后几块用 Attention
    'window_size': [8, 8, 14, 7],  # 各阶段窗口大小
    'drop_path_rate': 0.1,
}
# 参数量估算: ~28M (略小于 MambaVision-T 的 31.8M)
```

### 2.3 KAN 物理层 (7-8 天)

#### 设计原则
基于 KAN 作者的建议和 LESA 的经验:
1. **不要在主干网络中替换所有 MLP** — 这会导致训练极慢且可能不收敛
2. **限制在小维度物理映射中** — KAN 在小维度 (< 64) 物理函数拟合上表现最好
3. **使用 KAN 作为物理损失函数的可微模块** — H&D 曲线, DIR 反应扩散等
4. **混合使用 MLP + KAN** — MLP 处理通用特征，KAN 处理已知物理关系的映射

#### `src/models/kan/physics_kan.py` — 物理 KAN 层
```python
class PhysicsKANWrapper(nn.Module):
    """
    将 KAN 用于物理建模的封装
    
    设计: 对于每个物理特性，用一个小的 KAN [in_dim, 8, 4, out_dim] 
    来表示已知的物理映射关系。
    
    使用场景:
    1. H&D 曲线:    密度 = f(log_exposure)，1D→1D 映射
    2. 光谱交叉:     3×3 耦合矩阵 → 9 参数
    3. DIR 微对比度: 局部对比度 → 抑制量，局部映射
    4. 曝光 → 颗粒强度: 1D→1D 标量映射
    """
    def __init__(self, in_dim, hidden_dims, out_dim, grid=5, k=3):
        super().__init__()
        from pykan.kan.KANLayer import KANLayer
        
        self.layers = nn.ModuleList()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        
        for i in range(len(dims) - 1):
            self.layers.append(KANLayer(
                in_dim=dims[i],
                out_dim=dims[i+1],
                num=grid,  # B-spline 网格数
                k=k,       # spline 阶数
                grid_range=[-1, 1]
            ))
    
    def forward(self, x):
        for layer in self.layers:
            x, _, _, _ = layer(x)
        return x


class HnDCurveKAN(nn.Module):
    """
    H&D 特征曲线的 KAN 表示
    
    物理背景:
    - D = f(log H) 其中 D 是光学密度，H 是曝光量
    - 曲线形状: Toe → 线性区域 (gamma) → Shoulder
    - gamma = dD/d(log H) 在线性区域
    
    KAN 架构: [1, 8, 4, 1] — 1D 输入(log E)，1D 输出(密度)
    
    训练:
    - 使用 Kodak 已发布的特性曲线数据作为监督
    - 损失: MSE(D_pred, D_gt) + 平滑正则化 (二阶导数)
    """
    def __init__(self):
        super().__init__()
        self.kan = PhysicsKANWrapper(1, [8, 4], 1, grid=8, k=3)
        
        # 物理约束: gamma 必须在合理范围
        self.gamma_min = 0.3
        self.gamma_max = 1.5
    
    def forward(self, log_exposure):
        """
        Args:
            log_exposure: (B,) 或 (B, 1) — log10(曝光)
        Returns:
            density: (B,) — 光学密度
        """
        if log_exposure.dim() == 1:
            log_exposure = log_exposure.unsqueeze(-1)
        density = self.kan(log_exposure).squeeze(-1)
        return density
    
    def gamma_loss(self):
        """保证 gamma 在物理合理范围内的正则化"""
        # 从 B-spline 系数计算带符号的导数
        # (利用 KAN 的可解释性)
        ...
```

### 2.4 Instance-Disentangled Attention (4-5 天)

```python
class InstanceDisentangledAttention(nn.Module):
    """
    实例解耦注意力机制
    
    目的: 在图像翻译中，分离"内容"和"风格"的语义表示，
    防止风格特征污染内容结构。
    
    与标准交叉注意力的区别:
    - 标准: Q(content), K(styleref), V(styleref) → 可能混淆
    - 解耦: Q_split → Q_content + Q_style，分别处理
    
    实现:
    1. Content Branch: Q_K = content_proj(Q) → 关注空间结构相似性
    2. Style Branch:  Q_S = style_proj(Q)  → 关注色彩/纹理相似性
    3. Gating:        gate = sigmoid(gate_proj(Q)) → 自适应融合
    4. Output:        gate * AttnOut_content + (1-gate) * AttnOut_style
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Content branch
        self.qkv_content = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Style branch  
        self.qkv_style = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Feature gate (adaptive fusion)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.proj_content = nn.Linear(dim, dim)
        self.proj_style = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim * 2, dim)
    
    def forward(self, x, context=None):
        """
        x:       (B, N, C) — 待增强特征
        context: (B, M, C) — 上下文(可选, 默认 self-attention)
        """
        if context is None:
            context = x
        
        B, N, C = x.shape
        
        # Content attention
        q_c, k_c, v_c = self.qkv_content(x).chunk(3, dim=-1)
        q_c = q_c.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_c = k_c.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_c = v_c.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_c = F.scaled_dot_product_attention(q_c, k_c, v_c)
        attn_c = attn_c.transpose(1, 2).reshape(B, N, C)
        out_c = self.proj_content(attn_c)
        
        # Style attention
        q_s, k_s, v_s = self.qkv_style(x).chunk(3, dim=-1)
        q_s = q_s.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_s = k_s.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_s = v_s.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_s = F.scaled_dot_product_attention(q_s, k_s, v_s)
        attn_s = attn_s.transpose(1, 2).reshape(B, N, C)
        out_s = self.proj_style(attn_s)
        
        # Adaptive fusion
        gate = self.gate(x)  # (B, N, 1)
        out_fused = gate * out_c + (1 - gate) * out_s
        
        return self.proj_out(torch.cat([out_c, out_s], dim=-1))
```

---

## 第三阶段: 物理模拟集成 (预计 5-6 周)

### 3.1 Tone Mapping — H&D 曲线模块 (5 天)

```python
class HnDToneMapper(nn.Module):
    """
    基于 Neural ODE 的 H&D 特性曲线模拟
    
    物理学:
    1. 数字传感器响应是线性的 (raw linear)
    2. 胶片响应是非线性的，遵循 D = f(log H)
    3. RGB 三个通道有各自独立的 H&D 曲线
    4. 曲线受曝光、显影时间、温度影响
    
    方法: Neural ODE
    - 将 H&D 曲线建模为 ODE: dD/d(logH) = f(D, logH; θ)
    - f 是一个小的 MLP (或 KAN)
    - 从 logH_min 到 logH_max 积分得到完整曲线
    - 训练时用已知胶片的曲线数据做监督
    
    也可以直接用 KAN 拟合已知曲线 (见 HnDCurveKAN)
    """
    def __init__(self, use_kan=True, n_rgb_channels=3):
        super().__init__()
        self.n_channels = n_rgb_channels
        
        if use_kan:
            # 每通道独立的 KAN 曲线
            self.curves = nn.ModuleList([
                HnDCurveKAN() for _ in range(n_rgb_channels)
            ])
        else:
            # MLP 替代: 小 ODE 函数
            self.ode_funcs = nn.ModuleList([
                self._build_ode_func() for _ in range(n_rgb_channels)
            ])
    
    def forward(self, linear_rgb):
        """
        Args:
            linear_rgb: (B, 3, H, W) — 线性场景辐射度
        
        Returns:
            density_rgb: (B, 3, H, W) — 模拟的胶片密度
        """
        log_exposure = torch.log10(linear_rgb.clamp(min=1e-6))
        
        outputs = []
        for c in range(self.n_channels):
            # 逐通道应用 H&D 曲线
            density = self.curves[c](log_exposure[:, c:c+1].flatten())
            density = density.reshape_as(log_exposure[:, c:c+1])
            outputs.append(density)
        
        density = torch.cat(outputs, dim=1)
        return density
```

### 3.2 Halation / Bloom 散射 (5 天)

```python
class HalationSimulator(nn.Module):
    """
    光散射 / Bloom 模拟
    
    物理机制:
    - 光穿过乳剂层到达片基后反射回来
    - 反射光在相邻区域产生二次曝光
    - 表现为亮区周围的红晕 (特别是红色通道，因为红光穿透最深)
    
    数学模型: Helmholtz 方程 + 散射反向传播
    
    简化实现: 多尺度高斯卷积模拟光扩散
    - 红色通道: 大 sigma (红光散射最严重)
    - 绿色通道: 中 sigma
    - 蓝色通道: 小 sigma (蓝光主要在表面)
    
    可训练参数: 每通道的散射强度 + 扩散半径
    """
    def __init__(self, max_radius=31):
        super().__init__()
        # 每通道的散射强度 (初始值基于物理观测)
        self.scatter_intensity = nn.Parameter(torch.tensor([0.15, 0.08, 0.03]))
        self.scatter_radius = nn.Parameter(torch.tensor([21.0, 13.0, 7.0]))
    
    def forward(self, x):
        """
        x: (B, 3, H, W)
        
        方法:
        1. 提取高亮区域 (threshold)
        2. 逐通道高斯模糊 (sigma = scatter_radius[c])
        3. 乘以散射强度加到原图
        """
        B, C, H, W = x.shape
        halation = torch.zeros_like(x)
        
        for c in range(C):
            # 提取超过阈值的高亮像素
            threshold = x[:, c].mean() + 1.5 * x[:, c].std()
            highlights = torch.where(x[:, c:c+1] > threshold, x[:, c:c+1], 
                                     torch.zeros_like(x[:, c:c+1]))
            
            # 高斯散射核
            sigma = self.scatter_radius[c].abs()
            kernel_size = int(6 * sigma + 1) // 2 * 2 + 1  # 奇数
            kernel = self._gaussian_kernel(kernel_size, sigma, device=x.device)
            kernel = kernel.view(1, 1, kernel_size, kernel_size)
            
            # 散射
            scattered = F.conv2d(
                highlights, kernel, padding=kernel_size // 2
            )
            halation[:, c:c+1] = scattered * self.scatter_intensity[c].abs()
        
        return x + halation
```

### 3.3 Film Grain 生成 (5 天)

#### 完整实现基于前面的调研

```python
class FilmGrainSynthesizer(nn.Module):
    """
    曝光依赖的 Poisson-Binomial 胶片颗粒模型
    
    两种模式:
    A. NPS 模式 (推理): 用频率域噪声功率谱快速生成
    B. 物理模式 (训练): Poisson-Binomial 颗粒级模拟
    
    关键物理参数 (可训练):
    - grain_intensity: 颗粒总强度 (ISO 相关)
    - grain_coarseness: 颗粒粗度 (尺寸分布)
    - exposure_dependency: 曝光依赖性的强度
    - color_correlation: RGB 通道间颗粒相关性
    """
    def __init__(self, mode="nps", resolution=512):
        super().__init__()
        self.mode = mode
        
        # 可训练物理参数
        self.grain_intensity = nn.Parameter(torch.tensor(0.015))
        self.grain_coarseness = nn.Parameter(torch.tensor(0.5))
        self.exposure_dependency = nn.Parameter(torch.tensor(1.0))
        self.color_correlation = nn.Parameter(torch.tensor(0.3))
        
        if mode == "nps":
            # NPS 参数化 — AV1 风格
            self.nps_magnitude = nn.Parameter(torch.tensor(0.8))
            self.nps_correlation_freq = nn.Parameter(torch.tensor(0.12))
            self.nps_exponent = nn.Parameter(torch.tensor(1.2))
    
    def forward(self, image, exposure_map=None):
        """
        Args:
            image: (B, 3, H, W) — 干净图像
            exposure_map: (B, 1, H, W) — 可选局部曝光
        
        Returns:
            grainy: (B, 3, H, W) — 加颗粒图像
        """
        if self.mode == "nps":
            return self._nps_synthesis(image, exposure_map)
        else:
            return self._physical_synthesis(image, exposure_map)
    
    def _nps_synthesis(self, image, exposure_map):
        """
        频率域 NPS 合成 (快速路径)
        
        步骤:
        1. 生成白噪声
        2. 用 NPS 滤波器在频域着色
        3. 根据曝光调整颗粒强度
        4. 加入颜色相关性
        """
        B, C, H, W = image.shape
        
        # 1. 白噪声
        noise = torch.randn(B, 1, H, W, device=image.device)
        
        # 2. 频域 NPS 滤波器
        freqs_y = torch.fft.fftfreq(H, device=image.device).abs()
        freqs_x = torch.fft.fftfreq(W, device=image.device).abs()
        f_grid = torch.sqrt(freqs_y[:, None]**2 + freqs_x[None, :]**2)
        
        fc = self.nps_correlation_freq.abs()
        p = self.nps_exponent.abs().clamp(0.5, 2.0)
        nps_filter = 1.0 / (1.0 + (f_grid / fc)**2) ** (p / 2)
        nps_filter = torch.fft.fftshift(nps_filter)
        
        # 3. 频域滤波
        noise_f = torch.fft.fft2(noise)
        grain_f = noise_f * nps_filter[None, None, :, :]
        grain = torch.fft.ifft2(grain_f).real
        
        # 4. 暴露依赖
        if exposure_map is not None:
            # 中间调区域颗粒最明显 (已有理论依据)
            # 暗部: 量子噪声主导 (已在前端处理)
            # 亮部: 颗粒被高密度掩盖
            exposure_weight = 1.0 - 2.0 * (exposure_map - 0.5).abs()
        else:
            exposure_weight = torch.ones(B, 1, H, W, device=image.device)
        
        grain = grain * self.grain_intensity.abs() * exposure_weight
        
        # 5. 颜色相关性 (亮度颗粒影响所有通道)
        grain_r = grain * self.grain_coarseness.abs()
        grain_g = grain
        grain_b = grain / self.grain_coarseness.abs().clamp(min=0.3)
        grain_rgb = torch.cat([grain_r, grain_g, grain_b], dim=1)
        
        return image + grain_rgb
    
    def compute_nps_loss(self, gt_film, pred_film):
        """
        NPS 损失: 确保生成图像的噪声功率谱匹配真实胶片
        
        这是训练颗粒合成器的核心损失函数
        """
        gt_grain = self._extract_grain(gt_film)
        pred_grain = self._extract_grain(pred_film)
        
        # 计算两者的 NPS
        gt_nps = self._compute_nps(gt_grain)
        pred_nps = self._compute_nps(pred_grain)
        
        return F.mse_loss(pred_nps, gt_nps)
```

### 3.4 MICRO-CONTRAST DIR (4 天)

```python
class DIRMicroContrast(nn.Module):
    """
    显影抑制释放 (Developer Inhibitor Release) 微对比度
    
    物理机制:
    1. 显影过程中，显影剂氧化产物在银颗粒周围积累
    2. 抑制物扩散到相邻区域，抑制相邻颗粒的显影
    3. 结果: 边缘处产生"边缘效应"——亮边更亮，暗边更暗
    4. 这增强了视觉上的清晰度和微对比度
    
    建模: 反应-扩散方程
    ∂C/∂t = D·∇²C + R(C) - k·C
    
    其中 C 是抑制物浓度，D 是扩散系数，
    R(C) 是生成率 (与局部显影速率成比例)，
    k 是衰减常数。
    
    实现: 用 PINNs (Physics-Informed Neural Networks) 
    或近似为自适应反锐化掩模 (unsharp mask with edge-aware kernel)
    """
    def __init__(self, diffusion_radius=5):
        super().__init__()
        self.diffusion = nn.Parameter(torch.tensor(1.5))
        self.inhibition_strength = nn.Parameter(torch.tensor(0.3))
        self.edge_threshold = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, image):
        """
        简化实现: 边缘感知的局部对比度增强
        
        1. 检测边缘 (Laplacian)
        2. 在边缘周围做有向扩散
        3. 增强边缘两侧对比度
        """
        # 拉普拉斯算子检测边缘
        laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32, device=image.device)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        edges = F.conv2d(image, laplacian_kernel, padding=1, groups=3)
        edge_mask = (edges.abs() > self.edge_threshold).float()
        
        # 高斯扩散核 (模拟抑制物扩散)
        sigma = self.diffusion.abs()
        diff_kernel = self._gaussian_kernel_2d(
            int(sigma * 4 + 1), sigma, device=image.device
        )
        diff_kernel = diff_kernel.view(1, 1, -1, -1).repeat(3, 1, 1, 1)
        
        # 扩散边缘 → 边缘效应
        inhibited = F.conv2d(image, diff_kernel, padding='same', groups=3)
        edge_enhanced = image + self.inhibition_strength.abs() * edge_mask * (image - inhibited)
        
        return edge_enhanced
```

---

## 第四阶段: RTX 5070 Ti / 12GB 优化 (预计 3-4 周)

### 4.1 显存预算详细计算

```
总 VRAM: 12GB = 12,288 MB
系统保留: ~1,000 MB (驱动 + 桌面)
可用: ~11,200 MB

训练时显存分配:
  ├── 模型参数:         ~150 MB (Mamba-Tiny ~28M × 4 bytes bf16 + optimizer states)
  ├── Optimizer States:   ~450 MB (Adam 8-bit: 1 byte/param, 而非 4 bytes)
  ├── 激活值:           ~3,000 MB (含 gradient checkpointing 重计算)
  ├── 梯度:              ~120 MB (bf16, gradient_checkpointing 减少)
  ├── 批量数据:          ~400 MB (batch=2, 512×512×3×4 bytes × 2)
  ├── 潜空间缓冲区:       <1 MB   (batch=2, 4×64×64 bf16/fp32 latent tensors)
  ├── CUDA 上下文:       ~500 MB
  ├── 工作区 (cuBLAS):   ~1,000 MB
  └── 余量:             ~5,500 MB
```

### 4.2 关键优化技术实现

#### `src/training/optimizer.py`
```python
class MemoryOptimizer:
    """12GB 本机训练优化器配置"""
    
    @staticmethod
    def configure_optimizer(model, config):
        """
        8-bit Adam 配置
        
        内存对比:
        - FP32 Adam: param × 4 bytes × 2 (momentum + variance) = 8 bytes/param
        - 8-bit Adam:  param × 1 byte × 2 = 2 bytes/param
        - 节省: 75%
        """
        import bitsandbytes as bnb
        
        return bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
            eps=1e-8
        )
    
    @staticmethod
    def enable_gradient_checkpointing(model):
        """在关键模块启用梯度检查点"""
        # 对 Mamba block 启用 (重计算 scan，节省激活内存)
        from torch.utils.checkpoint import checkpoint
        
        for name, module in model.named_modules():
            if 'ssm' in name.lower() or 'mamba' in name.lower():
                module._gradient_checkpointing = True
    
    @staticmethod
    def memory_profile(model, sample_input):
        """显存使用分析"""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        print(f"Before forward: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model(sample_input)
            loss = output.sum()
            loss.backward()
        
        print(f"Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        print(f"Current memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

#### `src/training/memory.py` — 显存预算管理器
```python
class VRAMBudget:
    """
    显存预算管理器，追踪和预测显存使用
    
    策略:
    1. 动态调整 batch_size (如果 OOM)
    2. 动态调整 gradient_accumulation_steps
    3. 监控峰值显存，预警 OOM
    4. 自动选择是否使用 GC (gradient checkpointing)
    """
    def __init__(self, target_vram_mb=10240, safety_margin_mb=512):
        self.target = target_vram_mb
        self.margin = safety_margin_mb
        self.peak_history = []
    
    def check_oom_risk(self, model, batch):
        """粗略预测当前 batch 的显存需求"""
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        grad_mem = param_mem  # 梯度与参数同大小
        optimizer_mem = param_mem * 0.25  # 8-bit: ~1 byte/param avg
        batch_mem = batch[0].numel() * batch[0].element_size() * len(batch)
        
        estimated = (param_mem + grad_mem + optimizer_mem + batch_mem) * 3.5  # 激活近似 2× 开销
        estimated_mb = estimated / (1024**2)
        
        if estimated_mb > self.target - self.margin:
            return False, f"Risk: {estimated_mb:.0f} MB > {self.target - self.margin:.0f} MB"
        return True, f"Safe: {estimated_mb:.0f} MB"
    
    def adjust_batch_size(self, current_bs):
        """OOM 时自动减半批量大小"""
        return max(1, current_bs // 2)
```

#### FlashAttention-2 集成
```python
# 在 Attention 模块中自动使用 FA-2 (PyTorch 2.0+)
# 条件: Ampere (RTX 30xx) 或更新 + CUDA 11.6+

def attention_forward(q, k, v, use_flash=True):
    if use_flash and hasattr(F, 'scaled_dot_product_attention'):
        # PyTorch 2.0+ 自动调用 FA-2 (如果可用)
        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=False
        )
    else:
        # 回退到手动实现
        attn = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        attn = F.softmax(attn, dim=-1)
        return attn @ v
```

### 4.3 两阶段推理管线

```python
class TwoStageInferencePipeline:
    """
    两阶段推理: 速度与质量的平衡
    
    阶段 1 — 潜空间快速流 (1-2 NFE):
    - 在最粗糙的潜空间中求解 ODE
    - 使用 Euler 大步长
    - 得到粗略的胶片效果
    
    阶段 2 — 像素空间精炼 (2-3 NFE):
    - VAE 解码到像素空间
    - 小步长 Euler 或中点法微调
    - 主要修正颗粒、微对比度等局部细节
    """
    
    def __init__(self, cfm_model, vae, config):
        self.cfm = cfm_model
        self.vae = vae
        self.config = config
    
    @torch.no_grad()
    def __call__(self, digital_image, film_style="vision3_500t", 
                 n_steps_stage1=3, n_steps_stage2=4):
        """
        完整推理管线
        
        Args:
            digital_image: (1, 3, H, W) — 数字图像
            film_style: 胶片风格标识符
            n_steps_stage1: 潜空间 ODE 步数
            n_steps_stage2: 像素精炼步数
        """
        # Stage 1: 潜空间
        z0 = self.vae.encode(digital_image).latent_dist.mode()
        
        # 加载风格特定的 LoRA 权重
        if film_style is not None:
            self.cfm.load_lora(film_style)
        
        # ODE 积分: z0 → z1_film
        z1 = self._solve_ode_latent(z0, n_steps_stage1)
        
        # VAE 解码到像素空间
        film_latent_preview = self.vae.decode(z1).sample
        
        # Stage 2: 像素空间精炼 (颗粒 + 微对比度)
        if n_steps_stage2 > 0:
            film_final = self._refine_pixel(film_latent_preview, n_steps_stage2)
        else:
            film_final = film_latent_preview
        
        return film_final
```

---

## 第五阶段: 训练与评估 (预计 4-5 周)

### 5.1 多维度损失函数

```python
class CombinedLoss(nn.Module):
    """
    联合损失函数
    
    权重分配策略:
    1. 内容保持:   w_recon = 1.0     (L1 + SSIM)
    2. 感知质量:   w_percep = 0.5    (LPIPS)
    3. 物理精度:   w_phys = 0.3      (H&D, DIR, NPS)
    4. 对抗训练:   w_gan = 0.1       (可选)
    5. 色彩精度:   w_color = 1.0     (ΔE + 直方图)
    6. 身份保持:   w_id = 0.2        (内容一致)
    """
    
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lpips_fn = lpips.LPIPS(net='vgg').eval()
        self.ssim_fn = kornia.metrics.SSIM(11)
    
    def forward(self, pred, target, pred_phys=None, target_phys=None):
        losses = {}
        
        # 像素级
        losses['l1'] = self.l1(pred, target)
        losses['ssim'] = 1 - self.ssim_fn(pred, target)
        
        # 感知级
        losses['lpips'] = self.lpips_fn(pred, target).mean()
        
        # 物理级 (如果提供了物理特征)
        if pred_phys is not None:
            losses['hd_rmse'] = F.mse_loss(pred_phys['tone'], target_phys['tone'])
            losses['dir_corr'] = 1 - F.cosine_similarity(
                pred_phys['dir'].flatten(1), target_phys['dir'].flatten(1)
            ).mean()
            losses['nps_mse'] = F.mse_loss(pred_phys['nps'], target_phys['nps'])
        
        # 色彩级
        losses['delta_e'] = self.delta_e2000(pred, target)
        losses['histogram_emd'] = self.histogram_emd(pred, target)
        
        return losses
    
    def total_loss(self, losses):
        weights = {
            'l1': 1.0, 'ssim': 2.0, 'lpips': 0.5,
            'hd_rmse': 1.0, 'dir_corr': 0.5, 'nps_mse': 0.5,
            'delta_e': 1.0, 'histogram_emd': 0.3
        }
        return sum(weights.get(k, 0) * v for k, v in losses.items())
```

### 5.2 LoRA 微调管线

```python
class FilmLoRAPipeline:
    """
    胶片风格 LoRA 微调管线
    
    工作流 (基于 PTI 方法论):
    1. 基础训练: 在通用数据上训练基础 K-MCFM
    2. 冻结基础模型
    3. 对每种胶片风格:
       a. 在注意力层的 Q, K, V, O 注入 LoRA 层
       b. 只用该风格的配对数据微调 (100-500 steps)
       c. 保存 LoRA 权重 (~2-5 MB/styl e)
    4. 推理时动态加载 LoRA 权重
    
    LoRA 参数:
    - rank=4: 极小但效果不错 (2MB)
    - rank=16: 标准设置 (5MB)
    - rank=64: 高质量但更大 (8MB)
    - alpha=rank: LoRA 缩放因子
    """
    
    def __init__(self, base_model, rank=16, alpha=16):
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        
    def inject_lora(self, target_modules=['qkv', 'proj', 'fc']):
        """在指定模块注入 LoRA 层"""
        for name, module in self.base_model.named_modules():
            if any(t in name.lower() for t in target_modules):
                if isinstance(module, nn.Linear):
                    self._add_lora_to_linear(module, name)
    
    def merge_lora(self, lora_weight_path, scale=1.0):
        """
        合并 LoRA 权重: W' = W + (alpha/r) * BA * scale
        用于永久性风格合并
        """
        ...
    
    def interpolate_loras(self, lora_A, lora_B, weight=0.5):
        """
        两种风格的插值: W' = W + (1-w)*ΔW_A + w*ΔW_B
        用于创造新的混合风格
        """
        ...
```

### 5.3 评估指标体系

```python
class FilmEvaluationSuite:
    """
    完整评估套件
    
    自动指标:
    1. FID (Frechet Inception Distance): 分布级质量
    2. KID (Kernel Inception Distance): 无偏分布距离
    3. LPIPS: 感知级逐对相似度
    4. PSNR / SSIM: 传统保真度
    5. ΔE2000: 色彩精确度
    6. H&D RMSE: 特性曲线匹配精度
    7. NPS 相关性: 颗粒/噪声结构匹配
    8. SIFT 匹配率: 内容保持程度
    
    人工评估:
    9. 2AFC (Two-Alternative Forced Choice): 偏好测试
    10. Mean Opinion Score (MOS): 5 级质量评分
    """
    
    def evaluate_auto(self, model, test_loader):
        """自动指标评估"""
        results = {}
        
        # FID (需要 InceptionV3)
        results['fid'] = self._compute_fid(model, test_loader)
        
        # 物理指标
        results['hd_rmse'] = self._compute_hd_rmse(model, test_loader)
        results['nps_corr'] = self._compute_nps_correlation(model, test_loader)
        
        # 内容保持
        results['sift_match'] = self._compute_sift_match_rate(model, test_loader)
        
        return results
    
    def run_user_study(self, model, output_dir, n_participants=20):
        """2AFC 用户研究"""
        ...
```

---

## 第六阶段: 部署 (预计 2 周)

### 6.1 ONNX/TensorRT 导出
### 6.2 CLI 推理工具

---

## 详细时间表

| 阶段 | 任务 | 工作日 | 关键里程碑 |
|------|------|--------|-----------|
| **P0** | 基线验证 | 5d | CFM+UNet 在配对数据上收敛 |
| **1.1** | 项目环境 | 2d | GPU 训练可用 |
| **1.2** | 数据管线 | 5d | 配对数据流打通 |
| **1.3** | VAE 训练 | 10d | PSNR > 34dB, 潜空间可用 |
| **1.4** | 训练框架 | 3d | 训练循环 + WandB |
| **2.1** | CFM 模块 | 5d | OT-CFM 训练收敛 |
| **2.2** | Mamba 骨干 | 12d | Mamba backbone 性能 ≥ U-Net |
| **2.3** | KAN 物理层 | 8d | KAN H&D 曲线拟合误差 < 0.02 |
| **2.4** | IDA 注意力 | 5d | 内容一致性 ≥ 基线 |
| **3.1** | Tone Mapping | 5d | 各胶片 H&D 曲线可模拟 |
| **3.2** | 光谱交叉 | 5d | 色彩偏差 ΔE < 3 |
| **3.3** | DIR 微对比度 | 4d | 边缘对比度增强可见 |
| **3.4** | Halation | 5d | 高光红晕效果可见 |
| **3.5** | Film Grain | 5d | NPS 相关性 > 0.85 |
| **3.6** | 非均匀性 | 3d | Vignetting 校正有效 |
| **4.1** | 显存优化 | 5d | 推理 < 3.8 GB |
| **4.2** | FP8/Flash | 5d | 训练加速 1.5× |
| **4.3** | 两阶段推理 | 5d | 推理时间 < 5s (512×512) |
| **5.1** | 多维度损失 | 3d | 所有损失正常收敛 |
| **5.2** | LoRA 管线 | 5d | 风格切换 < 1s, 包 < 10MB |
| **5.3** | 评估套件 | 7d | FID < 15, H&D RMSE < 0.05 |
| **6.1** | ONNX/TRT | 5d | TensorRT 推理 < 2s |
| **6.2** | CLI 工具 | 3d | 命令行可用 |
| **总计** | | **~110d** | **约 5.5 个月** |

---

## 风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 配对数据不足 | 高 | 严重 | 优先 PBR 合成数据; 用未配对方法作为后备 |
| Mamba 在生成任务中不收敛 | 中 | 高 | P0 阶段验证; 保留 U-Net 后备 |
| KAN 训练极慢 | 高 | 中 | 限制 KAN 到小维度物理层; 不用在骨干 |
| 12GB VRAM OOM | 中 | 中 | 逐阶段监控显存; 自动降级 batch/分辨率 |
| CFM ODE 推理不稳定 | 低 | 中 | 用 OT-CFM (最稳定变体); 加引导 |
| 三个新组件交互 bug | 高 | 高 | 分层验证: 逐一加入，单独测试 |

---

## 参考文献 (完整列表)

1. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. *ICLR 2023*. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

2. Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., Wolf, G., & Bengio, Y. (2024). Conditional Flow Matching: Simulation-Free Dynamic Optimal Transport. *TMLR*. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482). 代码: [torchcfm](https://github.com/atong01/conditional-flow-matching)

3. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

4. Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. *ICML 2024*. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) — Mamba-2

5. Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., & Wang, X. (2024). Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model. *ICML 2024*. [arXiv:2401.09417](https://arxiv.org/abs/2401.09417)

6. Hatamizadeh, A., & Kautz, J. (2025). MambaVision: A Hybrid Mamba-Transformer Vision Backbone. *CVPR 2025*. [arXiv:2407.08083](https://arxiv.org/abs/2407.08083). 代码: [NVlabs/MambaVision](https://github.com/NVlabs/MambaVision)

7. Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2024). KAN: Kolmogorov-Arnold Networks. *ICLR 2025*. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756). 代码: [pykan](https://github.com/KindXiaoming/pykan)

8. LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration. (2026). *CVPR 2026*. [arXiv:2602.20497](https://arxiv.org/abs/2602.20497)

9. DiMSUM: Diffusion Mamba — A Scalable and Unified Spatial-Frequency Method. (2024). *NeurIPS 2024*. [arXiv:2411.04168](https://arxiv.org/abs/2411.04168). 代码: [VinAIResearch/DiMSUM](https://github.com/VinAIResearch/DiMSUM)

10. DiM: Diffusion Mamba for Efficient High-Resolution Image Synthesis. (2024). [arXiv:2405.14224](https://arxiv.org/abs/2405.14224)

11. DiM: Scaling Diffusion Mamba with Bidirectional SSMs. (2024). [arXiv:2405.15881](https://arxiv.org/abs/2405.15881)

12. Dimba: Transformer-Mamba Diffusion Models. (2024). [arXiv:2406.01159](https://arxiv.org/abs/2406.01159)

13. U-Shape Mamba (USM): State Space Model for Faster Diffusion. (2025). *CVPR 2025 Workshop*. [arXiv:2504.13499](https://arxiv.org/abs/2504.13499)

14. Gong, Y., et al. (2024). Film-GAN: Towards Realistic Analog Film Photo Generation. *Neural Computing and Applications*.

15. Li, Z., et al. (2023). A Large-Scale Film Style Dataset for Learning Multi-frequency Driven Film Enhancement. *IJCAI 2023*.

16. Mackenzie, S., et al. (2024). CNNs for Style Transfer of Digital to Film Photography.

17. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). 代码: [cloneofsimo/lora](https://github.com/cloneofsimo/lora)

18. Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) — FA-1

19. Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) — FA-2 (Ampere 支持)

20. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) — Stable Diffusion VAE

21. LinGen: LinGen: Toward High-Resolution Minute-Length Text-to-Video Generation with Linear Computational Complexity. (2025). *CVPR 2025*. [arXiv:2412.09856](https://arxiv.org/abs/2412.09856)

22. StyMam: A Mamba-Based Generator for Artistic Style Transfer. (2026). *ICASSP 2026*. [arXiv:2601.12954](https://arxiv.org/abs/2601.12954)

23. AV1 Film Grain Synthesis. Alliance for Open Media. [技术文档](https://aomediacodec.github.io/av1-spec/av1-spec.pdf) — Appendix: Film Grain Synthesis

24. PlainMamba: A Non-Hierarchical State Space Model for General Visual Recognition. (2024). *BMVC 2024*. [arXiv:2403.17695](https://arxiv.org/abs/2403.17695)

25. Chen, Y., Dai, X., Liu, M., Chen, D., Yuan, L., & Liu, Z. (2020). Dynamic Convolution: Attention over Convolution Kernels. *CVPR 2020*.

26. EDM2: Analyzing and Improving the Training Dynamics of Diffusion Models. (2024). *CVPR 2024*. [GitHub: NVlabs/edm2](https://github.com/NVlabs/edm2)

27. Dainty, J. C., & Shaw, R. (1974). *Image Science: Principles, Analysis and Evaluation of Photographic-Type Imaging Processes*. Academic Press.

28. *The Theory of the Photographic Process* (4th ed.). (1977). T. H. James (Ed.). Macmillan.

29. Kodak. *Basic Photographic Sensitometry Workbook*. Kodak Publication.

30. Selwyn, E. W. H. (1935). A Theory of Granularity. *Photographic Journal*, 75, 571-580.

---

*计划版本: v1.0 | 生成日期: 2026-04-27 | 基于截至 2026 年 4 月的公开文献和代码库*
