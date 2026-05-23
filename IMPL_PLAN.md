# K-MCFM: 修订实施计划

> 基于 `docs/ARCH_REDESIGN.md` 的架构重设计。原始 CFM+Mamba+KAN 方案已放弃。
> 
> **工期**: 4-6 周 | **VRAM**: ≤12 GB per task | **许可**: MIT-compatible 依赖

---

## Phase 1: 手动基线管线（1 周）

**目标**: 在不需要任何模型训练的情况下，产出一个完整的胶片模拟输出。

### 1.1 H&D 曲线数字化（1 天）

**输入**: Kodak/Fujifilm/Ilford 技术 PDF（`data/physics/`）
**输出**: 每胶片每通道 (R/G/B) 的 1D LUT + 原始数据点 CSV

**步骤**:
1. 用 PDF 阅读器打开技术文档，定位 H&D 曲线图表页
2. 用 WebPlotDigitizer (https://apps.automeris.io/wpd/) 数字化每条 R/G/B 曲线
3. 输出为 CSV 格式：`log_exposure, density_r, density_g, density_b`
4. 存储到 `data/physics/<film>/hd_curve.csv`

**数学处理**:
- 密度 D 与曝光 H 的关系：D = f(log₁₀ H)
- 映射到线性 RGB [0,1]：需要反函数 + 归一化
- 3 个通道独立处理，插值生成 256 点的 1D LUT

**胶片清单**（按优先级）:
```
kodak_vision3_500t    (高)
kodak_portra_400      (高)
fuji_velvia_50        (中)
kodak_ektar_100       (中)
kodak_portra_800      (中)
kodak_vision3_250d    (中)
ilford_hp5            (低, BW)
kodak_tri_x_400       (低, BW)
```

**接口**:
```python
class HnDToneMapper:
    def __init__(self, csv_path: str):
        """从 CSV 加载曲线并构建 1D LUT"""
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B,3,H,W) linear RGB [0,1] → 输出 (B,3,H,W)"""
```

**文件**: `src/models/film/tone.py`

---

### 1.2 光晕实现（1 天）

**物理原理**: 光穿过胶片片基后被反射回乳剂层，造成高光周围的红晕。
**实现**: 对 highlight 区域做各通道不同半径的 Gaussian blur

**步骤**:
1. 从输入图像提取 highlight 区域（亮度 > threshold）
2. 对 R 通道用大 σ (0.5-1.0 px), G 通道用中 σ (0.3-0.5), B 通道用小 σ (0.1-0.2)
3. 按强度混合回原图

**接口**:
```python
class HalationSimulator:
    def __init__(self, r_sigma: float, g_sigma: float, b_sigma: float, threshold: float):
        """σ in pixels normalized to image size, threshold for highlight detection"""
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) → (B,3,H,W)"""
```

**文件**: `src/models/film/halation.py`

---

### 1.3 颗粒模块集成（1 天）

**方案**: filmgrainer (MIT) 或自实现 Newson Boolean 模型

**filmgrainer 接口**:
```python
import filmgrainer

class FilmGrainRenderer:
    def __init__(self, r_ms: int, g_ms: int, b_ms: int, shadow_power, midtone_power, highlight_power):
        """
        r_ms/g_ms/b_ms: 每通道毫秒级颗粒大小参数
        shadow_power/midtone_power/highlight_power: 不同亮度区域的颗粒强度
        """
    def apply(self, x: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8 → (H,W,3) uint8"""
```

**RMS granularity → filmgrainer 参数映射**:
```
RMS 5-8   (Ektar 100):     small grains, low power
RMS 9-12  (Portra 400):    medium grains, medium power
RMS 13-17 (Portra 800, Tri-X): larger grains, high power
RMS 18-25 (Vision3 500T, HP5): large grains, very high power
```

**自实现 Newson 算法**（如果 filmgrainer 效果不够）:
- 参考 IPOL 2017 论文 + Joseph Wardle Rust 实现 (MIT)
- Pixel-wise 算法 + PyTorch GPU 加速
- 参数: grain_radius, grain_sigma, filter_sigma, n_monte_carlo
- 开发时间: 3-5 天

**文件**: `src/models/film/grain.py`

---

### 1.4 管线 CLI（2 天）

**接口**:
```bash
python scripts/pipeline.py input.tiff --style kodak_portra_400 --output output.tiff
python scripts/pipeline.py input.tiff --style all --output_dir ./outputs/
```

**流程**:
```
1. 加载输入 → linear RGB tensor
2. (可选) 应用基础颜色 LUT
3. 应用 H&D 色调映射
4. 应用光晕
5. 应用颗粒
6. 转 sRGB → 保存 TIFF/PNG
```

**文件**: `scripts/pipeline.py`

---

## Phase 2: 颜色风格转移训练（2-3 周）

**目标**: 学习数字→胶片域的颜色分布映射。

### 2.1 胶片域数据收集（2 天）

**数据来源**:
1. FilmSet 数据集（已下载）: 3 种风格，每类 5,285 张
2. Web 爬取: Flickr/r/analog 等胶片摄影社区
3. 自拍摄: 如果可能，数字+胶片配对场景

**目标**: 每胶片 ≥500 张高质量图片

**目录结构**:
```
data/film_domain/
    kodak_portra_400/
    kodak_vision3_500t/
    fuji_velvia_50/
    ...
data/digital_domain/
    fivek_inputs/      ← FiveK DNG 处理后的线性 TIFF
    coco/              ← COCO 数据集（可选）
    ...
```

**预处理**:
- 统一缩放到 256×256
- 线性 RGB 颜色空间
- float32, [0,1]

---

### 2.2 CUT 颜色迁移训练（7 天）

**方案**: CUT (Contrastive Unpaired Translation), Park et al., ECCV 2020

**为什么选 CUT**:
- 单边一致 (half the models of CycleGAN)
- 31% 更少 VRAM (3.3GB vs 4.8GB at 256²)
- PatchNCE 损失在多层 (0,4,8,12,16) 上保留结构细节
- FastCUT 模式更轻量 (2.3GB)

**训练参数**:
```yaml
model: cut                    # or fastcut
input_size: 256               # training resolution
batch_size: 1
lr: 0.0002
n_epochs: 400                 # 200 + 200 decay
nce_layers: [0, 4, 8, 12, 16]
nce_idt: True                 # identity loss for color preservation
lambda_NCE: 1.0
lambda_identity: 1.0
```

**VRAM 预算** (256²):
| 模式 | 训练 | 推理 (512²) |
|------|:---:|:---:|
| CUT | ~3.3 GB | ~3 GB |
| FastCUT | ~2.3 GB | ~3 GB |

**PyTorch 2.x 兼容性修复**（预期）:
- `torch.tensor()` → `torch.tensor()` API 更新
- `nn.Module.module` → 新 API
- 5-10 行代码修改

**训练命令**:
```bash
python scripts/train_cut.py \
    --dataroot ./data \
    --name portra400_cut \
    --model cut \
    --input_nc 3 --output_nc 3 \
    --load_size 286 --crop_size 256 \
    --batch_size 1 --lr 0.0002 \
    --n_epochs 200 --n_epochs_decay 200
```

**文件**: `scripts/train_cut.py`, `src/models/color/cut_model.py`

---

### 2.3 3D LUT 颜色（备选，5 天）

**如果 CUT 颜色精度不够**，使用 3D LUT 预测方案。

**方案 A: Image-Adaptive-3DLUT**
- 使用 FiveK 预训练权重
- 在胶片域微调权重预测器 CNN
- 优点：有预训练权重，推理 <1ms
- 缺点：需要成对数据（可以使用自监督 perturbation）

**方案 B: Neural Preset**
- 自监督：在任意数据集上应用随机胶片 LUT 做 perturbation
- 训练 CNN → DNCM (Deterministic Neural Color Mapping) 预测器
- 优点：完全不需要配对数据
- 缺点：依赖 .cube LUT 文件质量

**文件**: `src/models/color/lut_predictor.py`

---

## Phase 3: 集成与微调（1-2 周）

### 3.1 全管线集成（3 天）

**串联流程**:
```
input → [ColorTransfer] → [HnDToneMapper] → [Halation] → [Grain] → output
```

**颜色空间转换**:
- H&D 映射在 log 空间计算，内部转换
- 光晕在 RGB 空间
- 颗粒在 sRGB 空间（filmgrainer 期望 sRGB 输入）

**文件**: `scripts/pipeline_full.py`

---

### 3.2 逐胶片参数调优（3 天）

**每种胶片一个配置 YAML**:

```yaml
# configs/styles/kodak_portra_400.yaml
film:
  name: Kodak Portra 400
  type: negative_color_still
  iso: 400

hd_curve:
  csv_path: data/physics/kodak_portra_400/hd_curve.csv
  toe_strength: 0.15
  shoulder_strength: 0.1

halation:
  r_sigma: 0.6
  g_sigma: 0.3
  b_sigma: 0.1
  threshold: 0.85

grain:
  method: filmgrainer
  r_ms: ...     # 根据 RMS granularity 映射
  g_ms: ...
  b_ms: ...
  shadow_power: 0.7
  midtone_power: 0.5
  highlight_power: 0.3

color:
  model_path: checkpoints/cut_portra400/latest_net_G.pth
  # 或
  lut_path: checkpoints/lut/portra400.lut
```

---

### 3.3 批量推理 CLI（2 天）

```bash
python scripts/pipeline_full.py input.tiff --style kodak_portra_400
python scripts/pipeline_full.py input.tiff --style all --output_dir ./outputs/
python scripts/pipeline_full.py ./inputs/ --style kodak_vision3_500t -o ./outputs/
```

---

## 评估方案

| 指标 | 方法 | 目标 |
|------|------|------|
| 颜色精度 | 每通道 histogram 与目标胶片域图片对比 | 分布匹配 |
| 颜色色差 | ΔE2000 对 ColorChecker | <3.0 平均 |
| 颗粒逼真度 | 主观 A/B 测试 | 优于纯噪声 |
| 光晕自然度 | 主观评估高光区域的红晕 | 自然 |
| 色调曲线 | H&D 曲线拟合 RMSE | <0.05 density |

---

## 项目依赖（新增）

```text
# 颜色模块依赖
torch>=2.11.0          # 已有
torchvision>=0.26.0    # 已有
pillow>=12.1.1         # 已有
lpips>=0.1.4           # 已有 (for perceptual loss)
piq>=0.8.0             # 已有 (for FID)

# 颗粒模块
filmgrainer             # MIT, pip install filmgrainer

# 手动管线
opencv-python-headless>=4.13.0  # 已有 (Gaussian blur)
scikit-image>=0.26.0   # 已有
rawpy>=0.26.1          # 已有 (RAW input)

# 曲线数字化 (开发工具，非运行时)
# WebPlotDigitizer (网页应用, 不需要 pip)
```

**不再需要的依赖**（可保留在 requirements.txt 但不再使用）:
- `mamba-ssm`, `causal-conv1d` — Mamba 骨干已放弃
- `diffusers` — VAE 潜空间不再使用
- `torchcfm` — CFM 方案已放弃
- `pykan` — KAN 方案已放弃
- `pytorch-wavelets` — DiMSUM 小波不再需要
- `xformers` — SDPA 已内置

---

## 对比：原始计划 vs 修订计划

| 维度 | 原始 | 修订 |
|------|------|------|
| 工期 | 5.5 个月 | 4-6 周 |
| 核心架构 | MambaVision + KAN + CFM + VAE | CUT/CycleGAN + LUT + 后处理 |
| 配对数据需求 | 必需 (L1-L5 间接拼凑) | 不需要 (CUT 无配对, H&D 从 PDF) |
| 训练阶段 | 4 阶段 (VAE → CFM → 物理 → 优化) | 2 阶段 (手动管线 → 颜色训练) |
| VRAM 训练 | ≤10.2 GB (潜空间) | ≤5 GB (256² CUT) |
| 可解释性 | 低 (潜空间 + CFM + ODE) | 高 (每模块独立观察) |
| 技术风险 | 高 (3 个新技术同时集成) | 低 (成熟方案组合) |
| 成功率估计 | 15-25% | 60-70% |

---

*修订版本: v2.0 | 2026-05-23*
