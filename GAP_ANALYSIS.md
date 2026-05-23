# K-MCFM 项目缺口分析

## 状态边界

- 本文档保留 2026-04-27 调研时的缺口判断、风险表和保守估算。
- 当前环境事实以 `requirements.txt` 与 `AGENTS.md` 为准；当前数据落盘状态以 `data/raw/dataset_status.json` 为准。
- 当前硬件已从原始 RTX 3060 规划切换到 RTX 5070 Ti Laptop GPU 11.94GB，但显存预算仍按 12GB 本机路线控制。

## 概述

本文档在 2026 年 4 月 27 日的线上调研基础上，系统识别 K-MCFM 项目实施所需的所有资源、数据、软件、外部依赖，评估每一项的获取状态，并对不可获取项提供替代方案。

调研范围涵盖：学术文献（30+ 篇）、代码仓库（15+ 个）、公开数据集平台（GitHub、HuggingFace、Kaggle、PapersWithCode、Internet Archive）、商业产品（Dehancer、DxO FilmPack、FilmConvert）、制造商技术文档（Kodak）。

---

## 1. 数据集缺口 — 核心瓶颈

### 1.1 理想需求定义

K-MCFM 训练的理想数据集应满足以下全部条件：

| 要求 | 说明 | 重要性 |
|------|------|--------|
| 配对性 | 同一场景、同一光照、同一构图，同时用数字相机 (RAW) 和胶片相机拍摄，胶片经专业扫描数字化 | **必需** |
| 多样性 | 覆盖多种场景类型（室内/室外、人像/风景/静物）、多种光照条件（日光/钨丝灯/混合光）、不同曝光水平 | **必需** |
| 分辨率 | 至少 512×512，理想 2048×2048 以上，以保留颗粒纹理和微对比度细节 | **高** |
| 胶片类型 | 标注胶片型号（Kodak Vision3 500T / 250D、Portra 400 / 800、Ektar 100、Velvia 50、HP5、Tri-X 等），每类型至少 200 组 | **高** |
| 元数据 | ISO、光圈、快门、色温、扫描仪型号、扫描设置 | **高** |
| 色彩精度 | 包含 Macbeth ColorChecker 或类似标准色卡的场景，用于色彩校准 | **中** |
| 数量 | 每胶片类型 ≥500 组配对；总量 ≥2000 组 | **中** |
| 许可 | 可自由用于学术研究和/或商业用途 | **必需** |

### 1.2 调研结论：理想数据集不存在

经过对 GitHub、HuggingFace Datasets、Kaggle、PapersWithCode、Google Scholar、Internet Archive 的系统搜索，结论如下：

> **截至 2026 年 4 月，全球范围内不存在满足上述任一子集的公开配对数据集。** "同一场景 → 数字 RAW + 胶片扫描"的配对数据从未被系统性收集和公开发布。

搜索结果：
- GitHub `film simulation dataset paired digital`：**0 结果**
- GitHub `film simulation deep learning pytorch`：**0 结果**
- HuggingFace Datasets `film photography style transfer`：**0 结果**
- Kaggle `film simulation dataset analog photography`：**0 结果**
- PapersWithCode / GitHub 指向 IJCAI 2023 FilmSet；当前 GitHub README 提供 Kaggle 入口，需实际下载验证许可和文件结构
- Internet Archive `film negative scan dataset`：无相关结构化数据集

### 1.3 现有可用的替代数据集

#### 1.3.1 MIT-Adobe FiveK（★★★★★ 推荐用于预训练）

| 属性 | 值 |
|------|-----|
| **全称** | MIT-Adobe FiveK Dataset |
| **规模** | 5,000 张 RAW 照片 |
| **输出** | 每张 5 位专家的 Lightroom 后期版本（共 25,000 张 TIFF） |
| **格式** | 输入：DNG (RAW)；输出：16-bit TIFF，ProPhoto RGB，无损压缩 |
| **分辨率** | 6MP – 16.6MP（Nikon D70 到 Canon EOS-1Ds Mark II） |
| **元数据** | EXIF + 语义标签（室内/室外、时间、光源类型、主题类别） |
| **体积** | DNG 单包约 50 GB；专家 TIFF 输出需单独下载或按文件拉取 |
| **发布方** | MIT CSAIL + Adobe (Bychkovsky et al., CVPR 2011) |
| **许可** | Adobe + MIT 双许可（研究用途免费） |
| **下载** | https://data.csail.mit.edu/graphics/fivek/ |

**与 K-MCFM 的相关性：**
- ✓ RAW → 专业后期，是"摄影风格转换"最接近的大规模配对数据
- ✓ 可作为 VAE + CFM 骨干的预训练语料，让模型先学会"摄影增强"的通用能力
- ✓ 5 位专家的不同风格可类比"5 种胶片风格"，适合 LoRA 解耦训练
- ✗ **不是胶片模拟**：专家后期追求的是美观，而非特定胶片的物理特性
- ✗ 无颗粒、无光晕、无 H&D 曲线匹配——这些物理特性需要其他数据源

**推荐用法：**
```
阶段 A0：用本地 DNG 做 VAE 重建/RAW 解码验证
阶段 A1：补齐 Expert C/A 的 TIFF16 输出，训练 RAW→专家后期的通用摄影增强
阶段 B：用 FilmSet / PBR / 少量真实胶片配对数据微调 + LoRA 注入物理特性
```

#### 1.3.2 DPED（★★★☆☆ 辅助预训练）

| 属性 | 值 |
|------|-----|
| **全称** | DSLR Photo Enhancement Dataset |
| **规模** | iPhone 3GS/7、BlackBerry Passport、Sony Xperia Z1 → Canon 70D (DSLR) |
| **输出** | 配对 patches（CNN 训练用） |
| **格式** | PNG patches |
| **发布方** | Ignatov et al. (ICCV 2017) |
| **许可** | 学术研究免费 |
| **下载** | https://aiff22.github.io/#dataset（需邮件申请） |

**与 K-MCFM 的相关性：**
- ✓ 手机照片 → DSLR 质量，验证了跨设备图像质量翻译的可行性
- ✓ 适合预训练 I2I 翻译的基础能力
- ✗ 手机画质远低于 RAW，且目标仅是"像 DSLR"而非"像胶片"
- ✗ patches 格式，不支持全图训练

**推荐用法：** 辅助预训练（可选），非必需。

#### 1.3.3 FilmSet（★★★★☆ — 可申请/下载，优先验证）

| 属性 | 值 |
|------|-----|
| **全称** | FilmSet: A Large-Scale Film Style Dataset for Learning Multi-frequency Driven Film Enhancement |
| **论文** | Li et al., IJCAI 2023（43 引用） |
| **声称规模** | 3 种胶片风格，5,285 张高分辨率图像 |
| **状态** | 代码仓库已归档但 README 指向 Kaggle FilmSet 数据集 |
| **下载** | https://github.com/CXH-Research/FilmNet → FilmSet (Kaggle) |

**评价：** 这是公开文献中最接近 K-MCFM 的胶片风格数据集，应列为 L1.5 风格监督数据。它仍不是严格物理配对数据：论文目标是视觉风格增强，未提供可校准的 H&D/NPS/扫描流程，因此不能替代 L2/L3/L4。

#### 1.3.4 Cinestill800T 小规模真实配对（★★★☆☆ — 验证集候选）

| 属性 | 值 |
|------|-----|
| **全称** | CNNs for Style Transfer of Digital to Film Photography dataset |
| **规模** | 41 组 raw pair，预处理后 38 组 |
| **目标胶片** | Cinestill 800T |
| **状态** | 论文声称公开 raw + processed dataset，需进一步定位下载入口 |

**评价：** 规模太小，不适合主训练；但它是少见的真实数字相机 + 胶片相机对齐数据，适合做 sanity check、配准流程参考和小型过拟合实验。

#### 1.3.5 FilmGrainStyle740k（★★★☆☆ — 颗粒模块辅助）

| 属性 | 值 |
|------|-----|
| **全称** | FilmGrainStyle740k Dataset |
| **规模** | 148,694 张 clean 图 + 740k grainy 图 |
| **用途** | film grain 分析、去除、合成、强度预测、质量评估 |
| **许可** | 免费研究/评估，禁止商业用途和再分发 |

**评价：** 适合训练/校准 `FilmGrainSynthesizer` 的 NPS 先验和 grain intensity predictor；因许可限制，不能作为项目可再分发数据，也不能作为最终商业训练依据。

#### 1.3.6 通用图像增强数据集（★★☆☆☆ — 不直接适用）

| 数据集 | 规模 | 任务 | 相关性 |
|--------|------|------|--------|
| **LoL** (Low-Light) | 500 对 | 低光增强 | 仅低光场景 |
| **SID** (See-in-Dark) | 5094 张 | 极端低光 RAW 增强 | 特定 domain |
| **SIDD** (Smartphone Denoising) | ~30K 张 | 手机去噪 | 噪声 domain |
| **DIV2K** | 1000 张 | 超分辨率 | 无配对翻译 |

**评价：** 这些数据集的 domain（低光、去噪、超分）与胶片模拟的目标 domain（色彩、色调、颗粒）不重叠。**不推荐用作训练数据。**

### 1.4 推荐数据策略（六层架构）

```
                         ┌──────────────────────────┐
                         │  L5: 真实配对验证集       │
                         │  200-500 组自建           │
                         │  (自己拍摄 + 扫描)        │
                         ├──────────────────────────┤
                         │  L4: 真实小配对/验证      │
                         │  Cinestill800T 等          │
                         │  对齐流程 + sanity check   │
                         ├──────────────────────────┤
                         │  L3: 胶片物理参数         │
                         │  Kodak 技术文档数字化     │
                         │  (H&D 曲线 + 光谱 + NPS)  │
                         ├──────────────────────────┤
                         │  L2: PBR 合成配对数据     │
                         │  Mitsuba 3 光谱渲染       │
                         │  无限量，完美对齐          │
                         ├──────────────────────────┤
                         │  L1.5: 胶片风格监督       │
                         │  FilmSet + grain dataset   │
                         │  风格/颗粒分布先验         │
                         ├──────────────────────────┤
                         │  L1: 真实摄影预训练       │
                         │  MIT-Adobe FiveK           │
                         │  (RAW → 专业后期)         │
                         └──────────────────────────┘
```

**L1 — FiveK 预训练（立即可开始）：**
- 本地已有 DNG tar 和 Expert C TIFF16 输出
- 预处理：DNG → 线性 16-bit TIFF；Expert TIFF ProPhoto RGB → 统一 linear RGB
- 训练 VAE + CFM 骨干（通用图像增强）
- 预期效果：模型学会"高质量摄影增强"的通用先验

**L1.5 — 胶片风格/颗粒监督（新增，立即验证可得性）：**
- FilmSet：已下载并解压，用于胶片风格分布、multi-frequency 风格监督、非物理 LoRA 预热
- FilmGrainStyle740k：用于颗粒 NPS 先验和颗粒强度/纹理估计器
- 约束：这些数据只提供视觉风格/颗粒目标，不提供物理曝光链路；只作为辅助监督

**L2 — PBR 合成（立即可开始，1-2 周实现）：**
- 安装 Mitsuba 3 光谱渲染器
- 搭建场景：Macbeth ColorChecker + HDR 环境贴图
- 输入参数：Kodak Vision3 光谱灵敏度 + H&D 曲线
- 双通道输出：(1) 线性 RGB 数字版 (2) 经胶片物理模拟版
- 优势：完美对齐、无限量、全参数可控
- 开源工具：Mitsuba 3 (BSD 许可), Blender (GPL), OpenColorIO

**L3 — 胶片物理参数数字化（1-2 天）：**
- 从 Kodak 技术出版物提取 Vision3 500T, Portra 400 等胶片的 H&D 曲线数据
- 来源：Kodak / Kodak Alaris 技术数据 PDF，优先 Vision3 50D/250D/500T 与 Portra 400
- 替代源：学术文献《The Theory of the Photographic Process》(James, 1977) 中的通用模型参数
- 提取方法：PDF 图表 → 手动/半自动数字化 → CSV 表格

**L4 — 真实小配对验证（短期，新增）：**
- 验证 Cinestill800T 公开小数据集是否可下载
- 用它测试配准、亮度匹配、patch 训练、真实 halation/grain 的可学习性
- 不作为主训练，只作为真实世界 sanity check

**L5 — 自建真实配对（长期，取决于资源）：**
- 设备需求：1 台数字相机 + 1 台胶片相机（同卡口）+ 三脚架 + 测光表
- 扫描需求：Noritsu HS-1800 或 Fuji Frontier SP-3000（专业扫描），或 DSLR 翻拍 + Negative Lab Pro 转换
- 每次拍摄：同一场景用两台相机各拍一张（测光表确保曝光一致）
- 目标：200 组配对 / 胶片类型，覆盖多种光照和场景
- 体积：每对约 100-200MB（16-bit TIFF + RAW）
- 当前决策：L5 要做；磁盘策略允许继续使用当前项目磁盘，但每次大批量扫描入库前仍需记录容量。

---

## 2. 胶片物理表征数据缺口

### 2.1 H&D 特性曲线数据

| 胶片 | H&D 曲线 | 光谱灵敏度 | RMS 颗粒度 | MTF |
|------|---------|-----------|-----------|-----|
| Kodak Vision3 500T (5219) | Kodak 技术文档有图表 | 有 | 已公开 | 有 |
| Kodak Vision3 250D (5207) | Kodak 技术文档有图表 | 有 | 已公开 | 有 |
| Kodak Vision3 50D (5203) | Kodak 技术文档有图表 | 有 | 已公开 | 有 |
| Kodak Portra 400 | Kodak 技术文档（静态摄影线） | 部分 | 有 | 有限 |
| Kodak Portra 160 | 同上 | 部分 | 有 | 有限 |
| Fujifilm Velvia 50 | Fujifilm 数据指南/宣传资料可辅助 | 有限 | 有限 | 有限 |
| Kodak Ektar 100 | Kodak 技术资料可用 | 部分 | 有 | 有限 |
| Kodak Portra 800 | Kodak 技术资料可用 | 部分 | 有 | 有限 |
| Kodak Tri-X 400 | Kodak 技术资料可用 | B&W 光谱响应 | 有 | 有限 |
| Fuji Reala | 已停产，独立公开技术数据不足 | 不作为默认目标 | 不作为默认目标 | 不作为默认目标 |

**关键来源：**
1. Kodak Motion Picture Film Technical Data — https://www.kodak.com/en/motion/page/technical-data/ — Vision3 系列的 H&D 曲线、光谱灵敏度、RMS 颗粒度、MTF 均在此发布（可能需要 VPN 或 Internet Archive 获取历史版本）
2. Kodak Professional Film Data Sheets — 静态摄影线（Portra, Ektar, Tri-X）
3. 《The Theory of the Photographic Process》4th ed. (James, 1977) — 通用物理模型，含反应动力学、光谱敏化机制
4. 《Image Science》(Dainty & Shaw, 1974) — 颗粒度、NPS、MTF 的数学基础

**数字化工作量估算：**
- 单条 H&D 曲线（RGB 三通道 + 视觉密度）：~30 个数据点 × 3 通道 = 90 个点
- 提取工具：WebPlotDigitizer（开源）或手动读取
- 单胶片全参数提取：约 2-3 小时
- 目标胶片数：5 种 → 约 15 小时

### 2.2 颗粒 / NPS 数据

- Kodak 公布的是 **RMS Granularity**（单点值，48μm 孔径，D=1.0），如下：
  - Vision3 500T: RMS ~5（典型中速电影负片）
  - Vision3 50D: RMS ~3（极细颗粒）
- **完整的 NPS 曲线**（二维 Wiener Spectrum）通常不公开，需要从学术文献或独立测试中获取
- 替代方案：使用 AV1 Film Grain Synthesis 的标准参数化模型（见 IMPL_PLAN.md 3.3 节），再用 FilmGrainStyle740k 或自建均匀灰卡胶片扫描校准视觉颗粒特征

---

## 3. 预训练模型权重 — 状态良好

### 3.1 MambaVision 预训练权重

| 模型变体 | 参数量 | 分辨率 | ImageNet-1K Acc | ImageNet-21K Acc | HuggingFace |
|---------|--------|--------|-----------------|------------------|-------------|
| MambaVision-T | 31.8M | 224 | 82.3% | — | ✓ nvidia/MambaVision-T-1K |
| MambaVision-T2 | 35.1M | 224 | 82.7% | — | ✓ |
| MambaVision-S | 50.1M | 224 | 83.3% | — | ✓ |
| MambaVision-B | 97.7M | 224 | 84.2% | 84.9% | ✓ |
| MambaVision-L | 227.9M | 224 | 85.0% | 86.1% | ✓ |
| MambaVision-L2 | 241.5M | 224 | 85.3% | — | ✓ |
| MambaVision-L3 | 739.6M | 256 | — | 88.1% | ✓ |
| **共 12 个检查点** | | | | | [HuggingFace Collection](https://huggingface.co/collections/nvidia/mambavision) |

**安装：**
```bash
pip install mambavision
```

**分类用法：**
```python
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained(
    "nvidia/MambaVision-T-1K", trust_remote_code=True
)
```

**特征提取（用于 K-MCFM 骨干初始化）：**
```python
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "nvidia/MambaVision-T-1K", trust_remote_code=True
)
# 提取分层特征 → 去掉分类头 → 接入速度场预测头
```

**推荐选择：** 对于当前 RTX 5070 Ti Laptop GPU 的 12GB 显存预算，仍先使用 **MambaVision-T (31.8M)** 作为初始骨干，后续视显存余量考虑升级到 S (50.1M)。

### 3.2 VAE 预训练权重

SD AutoencoderKL 在 HuggingFace diffusers 中可直接加载：
```python
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
```

- Mojitoo 变体（`sd-vae-ft-mse`）：用 MSE 微调，重建质量更好，适合我们的用途
- 原始变体（`sd-vae-ft-ema`）：EMA 权重
- 潜空间：4 通道 × 8× 下采样（f=8, 64× 像素压缩）
- **注意：** SD VAE 针对 512×512 训练，若我们的目标分辨率高很多，可能需要在 FiveK 上微调 VAE 或采用 tiled encoding

### 3.3 其他可复用权重

| 模块 | 预训练来源 | 用途 |
|------|-----------|------|
| LPIPS (VGG-19) | `lpips` pip 包 | 感知损失 |
| InceptionV3 | torchvision | FID 计算 |
| SSIM / MS-SSIM | `piq` pip 包 | 结构损失 |
| DIET (LoRA) | `peft` 库 / diffusers | LoRA 注入参考 |
| OT Sampler | `torchcfm` pip 包 | OT 耦合采样 |

**结论：预训练权重方面不存在瓶颈。** 所有关键模块都有公开的、可下载的、许可兼容的预训练权重。

---

## 4. 软件环境缺口

### 4.1 编译依赖（高风险安装项）

以下包的安装需要 C++ 编译器和 CUDA Toolkit，可能在 Windows 上遇到问题：

当前已验证覆盖：`torch==2.11.0+cu128`、`causal-conv1d==1.6.1`、`xformers==0.0.35`、`bitsandbytes==0.49.2` 已落入当前环境；`mamba-ssm` 仍未安装；Attention 主路线改为 PyTorch SDPA。下表保留初始调研风险版本，便于追踪风险来源。

| 包 | 版本 | 用途 | Windows 兼容性 |
|----|------|------|---------------|
| `causal-conv1d` | 1.4.0 | Mamba SSM 核心 CUDA kernel | ⚠️ 需 MSVC + CUDA |
| `mamba-ssm` | 2.2.2 | Mamba-2 Python 接口 | ⚠️ 依赖 causal-conv1d |
| `flash-attn` | 2.6.3 | FlashAttention-2 | ⚠️ 需 CUDA 11.6+ + Ampere |
| `triton` | 3.0.0 | 自定义 CUDA kernel DSL | ⚠️ Windows 支持有限 |
| `xformers` | 0.0.27 | 内存高效注意力 | ✅ Windows 预编译 wheel |
| `bitsandbytes` | 0.43.3 | 8-bit 优化器 | ✅ Windows 预编译 wheel |
| `torch` | 2.4.0+cu121 | 深度学习框架 | ✅ Windows 预编译 wheel |

**缓解措施：**
1. `causal-conv1d` + `mamba-ssm` 优先用预编译 wheel（`pip install causal-conv1d mamba-ssm --no-build-isolation --index-url https://download.pytorch.org/whl/cu121`）
2. 若无预编译 wheel，需要 Visual Studio Build Tools 2022 + CUDA 12.1 安装在系统
3. `flash-attn` 若无 Windows wheel，可用 PyTorch 2.0+ 自带的 `F.scaled_dot_product_attention` 作为回退（性能稍低但功能等价）
4. `triton` 在 Windows 上可能不可用——可跳过，不影响核心功能

### 4.2 验证检查清单

```bash
# 1. CUDA 可用
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 2. Mamba SSM 可用
python -c "from mamba_ssm import Mamba; print('OK')"

# 3. FlashAttention 可用
python -c "from flash_attn import flash_attn_func; print('OK')"

# 4. 8-bit Adam 可用
python -c "import bitsandbytes as bnb; print('OK')"

# 5. VRAM 确认 (应 ≥ 11GB)
python -c "import torch; print(torch.cuda.get_device_properties(0).total_mem / 1e9, 'GB')"
```

---

## 5. 硬件与计算资源缺口

### 5.1 训练时间估算

以下训练时间表沿用原始 RTX 3060 12GB 的保守估算，以 MambaVision-T 骨干 + 潜空间 CFM 为基准；当前 RTX 5070 Ti Laptop GPU 的实际时间需要在训练框架落地后重测：

其中 `32×32 latent` 行按 256×256 patch 经 8× VAE 下采样估算，不改变项目接口中的 `H/8` 潜空间约定。

| 阶段 | 数据量 | 分辨率 | Batch | Estimated time | 迭代数 |
|------|--------|--------|-------|---------------|--------|
| VAE 训练 (FiveK) | ~4,000 img | 256→512 | 4(grad_acc=2) | ~40 hours | 250k steps |
| CFM 预训练 (FiveK 潜空间) | ~4,000 img | 32×32 latent | 8 | ~30 hours | 200k steps |
| CFM 微调 (PBR 合成) | ~10,000 对 | 32×32 latent | 8 | ~20 hours | 100k steps |
| 物理模块集成 | ~2,000 对 | 512 | 2(grad_acc=4) | ~15 hours | 50k steps |
| LoRA 每风格 | ~200 对/风格 | 512 | 2 | ~1 hour/风格 | 500 steps |
| **总计** | | | | **~110 hours** | |

这是可管理的训练时间，但需要注意的是：
- 上述估算是乐观的（假设无 OOM、无 NaN、无调试时间）
- 实际调试时间可能是训练时间的 2-5×
- 建议使用 `wandb` 全程监控 loss 曲线和系统资源

### 5.2 存储需求

| 项目 | 体积 |
|------|------|
| FiveK 原始数据集 | ~50 GB |
| FiveK 预处理 (HDF5) | ~30 GB |
| PBR 合成数据 | ~20 GB |
| 模型检查点 (10 epochs × 5 阶段) | ~10 GB |
| 训练日志 + WandB | ~2 GB |
| **总计** | **~120 GB** |

当前决策是继续使用项目所在磁盘存放数据和 checkpoints；实际余量继续按数据落盘和 checkpoint 增长复核。

---

## 6. 知识缺口

### 6.1 需要深入理解的领域

| 领域 | 现状 | 行动 |
|------|------|------|
| Mamba-2 SSD 机制的数学细节 | 论文已读，但实现层面需更多研究 | 读 DiMSUM/USM 代码 |
| KAN B-spline 梯度流 | pykan 代码已分析，但训练稳定性未知 | 先在小实验上验证 |
| OT 耦合在高维潜空间的稳定性 | torchcfm 提供了实现 | 先在 256 patch 对应的 4×32×32 潜空间上验证 |
| 胶片化学过程的精确建模 | 物理模型已有，但参数校准需真实数据 | 从 Kodak 技术文档提取 |
| CFM ODE 求解在 30-50 步的精度 | 文献报告可行 | 在自己的数据上验证 |

### 6.2 需要的外部专业知识

- 胶片扫描仪的运作方式和工作原理（如果自建数据）
- 色彩管理（RAW 处理、色彩空间转换、白平衡）
- PBR 渲染中的光谱渲染参数调整

---

## 7. 缺口优先级与行动计划

### P0（阻塞项 — 必须立即解决）

| # | 缺口 | 解决方案 | 预计时间 |
|---|------|---------|---------|
| 1 | 训练数据目标不完整 | FiveK Expert C 与 FilmSet 已补齐；下一步建立 dataset manifest | 0.5-1 天 |
| 2 | `mamba-ssm` 原生路径未打通 | 当前环境已验证；2.2 前继续验证 `mamba-ssm` 安装路径或明确 SDPA/替代实现边界 | 0.5-1 天 |

### P1（重要 — 第一周内解决）

| # | 缺口 | 解决方案 | 预计时间 |
|---|------|---------|---------|
| 3 | 胶片 H&D 曲线数据 | 数字化 Kodak Vision3 50D/500T + Portra 400 技术图表 | 1-2 天 |
| 4 | 模块测试基线缺失 | 对已实现模块补 forward/接口测试；数据管线先补 manifest 与 loader 测试 | 1 天 |
| 5 | FiveK 数据集可训练化 | DNG/Expert C 已下载；下一步解包/索引 + 预处理脚本 | 1-2 天 |

### P2（启动第二周内解决）

| # | 缺口 | 解决方案 | 预计时间 |
|---|------|---------|---------|
| 6 | FilmSet / grain 辅助数据 | FilmSet 已下载验证；FilmGrainStyle740k 需许可申请 | 0.5-1 天 |
| 7 | PBR 合成管线搭建 | Mitsuba 3 安装 + 场景搭建 + 渲染脚本 | 1-2 周 |
| 8 | 预训练权重集成 | 下载并验证 MambaVision + VAE 权重 | 0.5 天 |
| 9 | 基线验证跑通 | CFM+UNet 在 FiveK/FilmSet 上首次训练 | 3-5 天 |

### P3（长期 — 持续进行）

| # | 缺口 | 解决方案 | 预计时间 |
|---|------|---------|---------|
| 10 | Cinestill800T 小配对 | 定位下载入口并纳入验证集 | 0.5-1 天 |
| 11 | 自建真实配对数据 | 拍摄 + 扫描 | 持续，取决于设备和资源 |
| 12 | 更多胶片类型的物理数据 | 扩大胶片覆盖范围 | 按需 |
| 13 | 用户研究 / 主观评估 | 2AFC 测试 | 后期 |

---

## 8. 参考文献

1. Bychkovsky, V., Paris, S., Chan, E., & Durand, F. (2011). Learning Photographic Global Tonal Adjustment with a Database of Input/Output Image Pairs. *CVPR 2011*. — MIT-Adobe FiveK 数据集论文。

2. Ignatov, A., Kobyshev, N., Timofte, R., Vanhoey, K., & Van Gool, L. (2017). DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks. *ICCV 2017*. — DPED 数据集论文。

3. Li, Z., Chen, X., Wang, S., & Pun, C. M. (2023). A Large-Scale Film Style Dataset for Learning Multi-frequency Driven Film Enhancement. *IJCAI 2023*. — FilmSet 数据集论文；代码仓库 README 指向 Kaggle 数据集，需下载验证。

4. Kodak. Technical Data for KODAK VISION3 Films. — Kodak 官方技术文档，含 H&D 曲线、光谱灵敏度、MTF、RMS Granularity。https://www.kodak.com/en/motion/page/technical-data/

5. Hatamizadeh, A., & Kautz, J. (2025). MambaVision: A Hybrid Mamba-Transformer Vision Backbone. *CVPR 2025*. — MambaVision 预训练权重来源。HuggingFace: https://huggingface.co/collections/nvidia/mambavision（12 个检查点）

6. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*. — SD VAE 预训练权重来源。

7. James, T. H. (Ed.). (1977). *The Theory of the Photographic Process* (4th ed.). Macmillan. — 胶片物理化学过程的权威参考书，含反应动力学、光谱敏化、显影机制的数学描述。

8. Dainty, J. C., & Shaw, R. (1974). *Image Science: Principles, Analysis and Evaluation of Photographic-Type Imaging Processes*. Academic Press. — 颗粒度、NPS、MTF 的数学基础。

9. Alliance for Open Media. AV1 Film Grain Synthesis. — AV1 编解码器附录中的参数化颗粒模型，可直接用于颗粒合成的参考实现。

10. Mitsuba 3 Renderer. https://mitsuba.readthedocs.io/ — 开源光谱渲染器 (BSD 许可)，用于 PBR 合成配对训练数据。

---

*文档版本: v1.0 | 生成日期: 2026-04-27 | 基于截至 2026 年 4 月的公开资源调研*
