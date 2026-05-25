# Experiment Log — K-MCFM Film Translation

> 记录所有技术路线尝试、成功与失败的完整实验日志。
> 2026-05-23 ~ 2026-05-24，M5 32GB 统一内存。

---

## 总体结论

**在 M5 上能达到的最佳效果**：预训练 `timbrooks/instruct-pix2pix`（SD 1.5），参数 `image_guidance_scale=1.5, guidance_scale=7.5, 40 steps`，搭配优化过的胶片指令。色彩可见变化，内容有一定保真，但仍有轻微涂抹感。

**核心瓶颈**：没有真正的 digital↔film 配对数据。所有训练尝试（SD 1.5 LoRA、3D LUT GAN、Tiny UNet、IP2P fine-tune）都因为训练数据质量太差而失败。

**M5 训练限制**：MPS 推理可用，但**训练严重不稳定**：
- SDXL LoRA 训练：MPS 崩溃
- SD 1.5 全 UNet 训练：浮动 → swap
- 小规模 FP32 训练可行（如 IP2P fine-tune 200 步），但数据质量决定了上限

**可行的下一步**：CUDA 训练（RTX 5070 Ti）+ M5 推理。或者接受 M5 上限：InstructPix2Pix 预训练 + 参数调优。

---

## 第一轮：原始架构设计

**时间**：2026-05-23 之前

**方案**：CFM（Conditional Flow Matching）+ MambaVision 骨干 + KAN 物理层 + VAE 潜空间 + 6 个物理模拟模块

**放弃原因**：
- 无配对数据：L1-L5 五层间接数据拼凑，没有任何 pixel-level 监督
- MambaVision 不适合图像生成（MambaOut CVPR 2025 证明 Mamba 对视觉任务非必要）
- KAN 训练极慢，B-spline GPU 不能并行
- 过度工程化：19 个 YAML 配置，4 阶段训练，5 种 CFM 变体
- 综合成功率估计 15-25%

**文件**：`docs/ARCH_REDESIGN.md` V1 部分

---

## 第二轮：CUT + 3D LUT 管线

**时间**：2026-05-23（第一版重设计）

**方案**：
1. CUT/CycleGAN 做颜色域迁移（无配对 GAN）
2. 3D LUT 预测器做颜色映射
3. H&D 曲线 1D LUT 做色调映射
4. 光晕后处理（Gaussian scatter）
5. 颗粒后处理（filmgrainer/Newson 模型）

**放弃原因**：
- 用户反馈："不是另一个胶片 LUT 和加很假的颗粒"
- CUT 颜色精度弱，会有伪影和内容渗漏
- 整体是传统图像处理管线，缺少 AI 对胶片美学的"深层理解"

**文件**：`docs/ARCH_REDESIGN.md` V2 部分

---

## 第三轮：扩散模型（SDEdit + LoRA）

**时间**：2026-05-23（第二版重设计）

**方案**：SDXL img2img（SDEdit）+ 胶片 LoRA + IP-Adapter/ControlNet 内容锚定

**VRAM 分析**：
- SDXL + LoRA 推理 1024²：~8GB（M5 可行）
- SDXL LoRA 训练 512²：~8-10GB（MPS 崩溃）
- SD 3.5 Medium：~6GB
- Flux GGUF Q4：~12GB

**结果**：SDXL img2img + 社区 LoRA（CivitAI Portra 400）内容严重扭曲、颜色变化不足。

**根本原因**：SDEdit 本质是通过噪声→降噪重建内容。任何非零强度都会引入"生成"成分。扩散模型无法做确定性的色彩变换。

**文件**：`docs/ARCH_REDESIGN.md` V3 部分

---

## 第四轮：SD 1.5 交叉注意力 LoRA 自训练

**时间**：2026-05-23

**数据**：Flickr 真实胶片扫描，8 种胶片 × 384-500 张（共 3,896 张）

**训练方案**：
- 冻结 SD 1.5 UNet，只解冻交叉注意力 `to_k`/`to_v`（19M/860M 参数）
- 256²，500 步，1e-4 LR
- 后训练 SVD 压缩（rank=16）→ 1.5MB LoRA

**结果**：
- **颜色**：无可见变化
- **内容**：严重扭曲
- **Loss**：收敛到 ~0.003

**失败分析**（参考学术文献）：
1. **训练目标错误**：标准噪声预测 loss 只能教模型"如何降噪胶片图"，不能教"如何把数字图转成胶片"。需要 x0-prediction 或 reconstruction loss（ConsisLoRA, UltraStyle 论文）。
2. **训练层选错**：只解冻交叉注意力相当于只在 CLIP 文本通道做修改。胶片颜色是逐像素全局映射，不是语义概念。需要训练全部自注意力+交叉注意力层。
3. **训练量不足**：500 步 × 19M 参数 = 平均每个参数被更新 <5 次，没有收敛。
4. **后训练 SVD 压缩**：从没收敛的权重提取主成分 = 提取噪声。
5. **SD 1.5 容量不足**：LAION-5B 几乎没有胶片扫描，基座模型缺乏胶片美学表征。

**文件**：`scripts/train_lora.py`, `scripts/train_lora_sd.py`

---

## 第五轮：3D LUT CNN 预测器（回到确定性方案）

**时间**：2026-05-23

**方案**：ResNet-18 预测 3D LUT 融合权重，PatchGAN 判别器做无配对训练

**训练**：
- 源域：1 张数字图
- 目标域：500 张 Portra 400 扫描
- 30 个 epoch，1e-4 LR

**结果**：
- **颜色**：无可见变化
- **内容**：完全保留（LUT 是逐像素确定性映射，不会改结构）

**失败分析**：
1. 只有 1 张源图 → GAN 无法学到分布
2. Identity loss (L1) 权重 5.0 太高 → 直接惩罚颜色变化
3. 20 epoch 不够 → 没有足够训练信号

**文件**：`scripts/train_lut.py`

---

## 第六轮：SDXL 极低强度 img2img

**时间**：2026-05-23

**方案**：SDXL img2img，不加载 LoRA，strength 0.10-0.30，用 prompt "Kodak Portra 400 film..."

**结果**：
- strength 0.10-0.15：颜色无变化
- strength 0.20-0.30：内容开始扭曲，出现轻微颜色变化
- 无法在内容保真和颜色变化之间找到平衡

**根本原因**：SDEdit 的强度参数同时控制内容保真和风格强度，两者是互斥的。

---

## 第七轮：IP-Adapter + ControlNet-depth（内容锚定尝试）

**时间**：2026-05-23

**IP-Adapter SDXL**：
- 将输入图通过 CLIP 编码后注入 UNet 交叉注意力
- `ip-adapter_sdxl.safetensors` 加载成功
- `ip-adapter-plus_sdxl_vit-h.safetensors` → shape mismatch（ViT-H vs ViT-bigG 维度不同）
- 各种 IP scale（0.3-0.6）+ strength（0.25-0.35）组合
- **结果**：色彩失败，事实失真

**ControlNet-depth SDXL**：
- 提取深度图 → ControlNet 锁结构 + img2img 改颜色
- 多次 size mismatch（含需 1024² 正方形输入）
- 768²、1024² 都尝试过
- **结果**：色彩无变化，内容有轻微扭曲

**InstructPix2Pix + ControlNet**：由于 IP2P 的 conv_in 是 8 通道（标准 SD 是 4 通道），无法直接组合。需要从头训练 ControlNet for IP2P。

---

## 第八轮：InstructPix2Pix（突破——首次色彩成功）

**时间**：2026-05-23 ~ 2026-05-24

### 8.1 初始测试（SD 1.5 版）

**模型**：`timbrooks/instruct-pix2pix`（Berkeley 2023，SD 1.5 基础）
**原理**：输入图直接拼在潜空间（8 通道 conv_in），不是 SDEdit 式的噪声残留。双 CFG 机制：
```
noise_pred = uncond + guidance_scale × (text - image) + image_guidance_scale × (image - uncond)
```

**参数调优**：

| 轮次 | ig | tg | 结果 |
|------|:---:|:---:|------|
| 初始 | 1.2-1.5 | 7.5-9.0 | 色彩可见，轻微内容失真 |
| 极端 | 2.5-8.0 | 3.0-7.5 | 油画感严重 |
| 优化 | 1.5 | 7.5 | **最佳平衡** |
| 再调试 | 1.6-2.0 | 8.0-9.0 | ig>1.8 开始出伪影 |

**关键发现**：
- ig 越高**反而失真越严重**——CFG 公式中 (image - uncond) 项乘上大系数后进入不稳定区
- IP2P 论文说 sI 范围 1.0-1.5，sT 范围 5-10，我们验证了这一点
- tg=7.5 色彩改动最明显

### 8.2 SDXL 版 IP2P

**模型**：`diffusers/sdxl-instructpix2pix-768`

**结果**：
- 推理极慢：M5 上 ~2 分钟/张（vs 24s SD 1.5）
- 文件更大：143-171KB（vs 57-94KB SD 1.5）→ 更多细节
- 色彩改动更弱，内容扭曲更强
- 出现严重油画涂抹感

**放弃原因**：
- 蒸馏论文明确说："InstructPix2Pix-SD1.5 works significantly better than the SDXL counterpart"
- 速度不可接受

### 8.3 指令词优化

| 指令关键词 | 效果 |
|-----------|------|
| "convert to film" | 改动过大，内容漂移 |
| "apply film color grading" | 中等改动 |
| "preserving all details, composition, and structure exactly" | 加这半句有助内容保真 |

### 8.4 IP2P Fine-tune 尝试（失败）

**数据**：200 对伪配对（Portra 400 扫描 + WB/Gamma 扰动版本）
**训练**：200 步，256² fp32，M5 ~10 分钟
**loss**：0.38 → 0.0003（过拟合）

**结果**：
- 推理出现**极其严重的涂抹感**
- 颜色退化，内容全部扭曲

**失败原因**：伪配对数据是垃圾——"数字版"只是胶片扫描做 WB+gamma 扰动，不是真的数码照片。模型学会了撤销这些特定扰动，没学会通用色彩映射。预训练在自己 45 万对上的先验被覆盖了。

---

## 第九轮：Tiny UNet 颜色自动编码器

**时间**：2026-05-24

**方案**：162K 参数 UNet，3 级 encoder-decoder + skip connections。瓶颈 4×4 空间 → 太小无法编码结构，只能学全局颜色。训练：对胶片图加颜色扰动，让模型重建原图。

**结果**：
- MPS 通道维度 bug（skip connection concat 后通道数不匹配）
- 修了几次仍有问题
- 没完成完整训练

---

## 数据收集

> 完整下载指令见文末 [附录 A：数据下载完整指南](#附录-a数据下载完整指南)

### 已下载并使用的数据集

| 数据集 | 数量 | 大小 | 来源 | 用途 | 需要密钥 |
|--------|:---:|:---:|------|------|:---:|
| **Flickr 胶片扫描** | 3,896 张 | ~2.6GB | Flickr API | SD 1.5 LoRA 自训练 + IP2P fine-tune | ✅ API Key + Secret |
| **FiveK DNG** | 5,396 文件 | 50.83GB | MIT CSAIL | 未使用（备用） | ❌ 公开 |
| **物理 PDF** | 12 份 | ~50MB | Kodak/Fuji/Ilford 官网 | 参考 H&D 曲线数据 | ❌ 公开 |
| **CIE 校准** | 8 文件 | <1MB | CIE 官网 | 色彩空间校准（未使用） | ❌ 公开 |
| **相机光谱** | 18 文件 | ~5MB | RIT + Tokyo Open Vision | 传感器光谱参考（未使用） | ❌ 公开 |

### 下载但未使用的数据集

| 数据集 | 原因 |
|--------|------|
| **FilmSet**（10GB, 21,140 文件） | Capture One 软件模拟的胶片风格，不是真实胶片扫描。提取后全部删除。 |

### 尝试下载但失败的数据集

| 数据集 | 原因 |
|--------|------|
| **HuggingFace vintage-photography-450k** | URL 超时，流式下载极端缓慢（~0 有效下载） |
| **CivitAI 胶片 LoRA**（Vision3 500T/250D, Ektar, Tri-X） | 大部分需要登录。仅 Portra 400（model 723250）下载成功。其余 401 Unauthorized |

### 胶片扫描明细（Flickr）

每种胶片的 Flickr 搜索标签和最终下载量：

| 胶片 | 标签 | 收集量 | 训练 LoRA | IP2P 训练 |
|------|------|:---:|:---:|:---:|
| Kodak Portra 400 | `kodak portra 400`, `portra400`, `portra 400 film` | 500 | ✅ | ✅ |
| Kodak Portra 800 | `kodak portra 800`, `portra800`, `portra 800 film` | 500 | ✅ | ❌ |
| Kodak Vision3 500T | `kodak vision3 500t`, `vision3 500t`, `5219 film` | 500 | ✅ | ❌ |
| Kodak Vision3 250D | `kodak vision3 250d`, `vision3 250d`, `5207 film` | 403 | ✅ | ❌ |
| Kodak Ektar 100 | `kodak ektar 100`, `ektar100`, `ektar 100 film` | 499 | ✅ | ❌ |
| Kodak Tri-X 400 | `kodak tri-x 400`, `tri-x 400`, `tri-x400`, `tri x 400 film` | 500 | ✅ | ❌ |
| Fujifilm Velvia 50 | `fujifilm velvia 50`, `velvia 50`, `velvia50`, `fuji velvia film` | 384 | ✅ | ❌ |
| Ilford HP5 Plus | `ilford hp5`, `hp5+`, `ilford hp5 plus`, `hp5 plus film` | 500 | ✅ | ❌ |

> **数据质量说明**：图片来自 Flickr 公开搜索，原始分辨率各异。可能存在水印、压缩伪影、过度后期。在 LoRA 训练中使用 `CenterCrop(resolution)` + `RandomFlip` 做增强。所有图片归原始上传者所有，不随项目分发。

### 数据目录结构

```
data/
├── film_domain/           # 胶片扫描（从 Flickr 下载，8 个子目录）
│   ├── portra_400/        # 500 张 → 961MB
│   ├── portra_800/        # 500 张
│   ├── vision3_500t/      # 500 张
│   ├── vision3_250d/      # 403 张
│   ├── ektar_100/         # 499 张
│   ├── tri_x_400/         # 500 张
│   ├── velvia_50/         # 384 张 → 961MB
│   └── hp5/               # 500 张 → 585MB
├── physics/               # 胶片物理 PDF（12 份）
│   ├── kodak_vision3_500t/technical_data.pdf
│   ├── kodak_vision3_250d/technical_data.pdf
│   ├── kodak_vision3_50d/technical_data.pdf
│   ├── kodak_portra_400/technical_data.pdf
│   ├── kodak_portra_800/technical_data.pdf
│   ├── kodak_portra_160/technical_data.pdf
│   ├── kodak_ektar_100/technical_data.pdf
│   ├── kodak_trix/technical_data.pdf
│   ├── ilford_hp5/technical_data.pdf
│   ├── fujifilm_velvia_50/product_information_bulletin.pdf
│   └── fujifilm_professional_guide/professional_film_data_guide.pdf
├── calibration/           # 色彩科学数据
│   ├── cie/               # CIE 标准（XYZ 1931, D50/D65/Illuminant A）
│   └── camera_spectral/   # 相机光谱灵敏度（RIT + Tokyo）
├── raw/                   # 原始数据集
│   ├── fivek/             # FiveK DNG (50.83GB)
│   └── filmset/           # FilmSet zip (10GB) — 未使用
├── ip2p_train/            # IP2P 伪配对训练数据（200 对）
└── processed/             # 预处理数据（空）
```

> **注意**：`data/raw/` 和 `data/film_domain/` 在 `.gitignore` 中——不在 Git 仓库里。新机器需要重新执行下载步骤。

---

## 最终状态

### ✅ 可用方案

| 方案 | 色彩 | 内容保真 | 速度 | 模型 |
|------|:---:|:---:|:---:|------|
| **IP2P ig=1.5 tg=7.5** | ⭐⭐⭐ | ⭐⭐⭐ | 24s/张 | `timbrooks/instruct-pix2pix` |

### ❌ 失败的方案（共 9 个）

1. SDEdit img2img（SD 1.5 / SDXL）
2. 社区 LoRA + SDXL（CivitAI/HuggingFace）
3. 交叉注意力 SD 1.5 LoRA 自训练
4. 全 UNet IP2P Fine-tune
5. IP-Adapter + SDXL
6. ControlNet-depth + SDXL
7. 3D LUT CNN 预测器
8. Tiny UNet 颜色自动编码器
9. SDXL InstructPix2Pix

### 已知可行的提升路径

1. **CUDA 训练 SDXL 全 UNet LoRA**（kohya-ss, rank 64, 3000步, 512²）——解决 M5 训练不稳定
2. **IP2P fine-tune on 5070 Ti** with original IP2P dataset + our film instructions
3. **InstructPix2Pix SDXL fine-tune** with real paired data（同一场景数码+胶片）
4. **物理模拟管线**（agx-emulsion / filmr / vkdt filmsim）作为后处理验证

## 附录 A：数据下载完整指南

### A.1 前置准备

```bash
python3 -m venv .venv
source .venv/bin/activate         # Mac
pip install flickrapi requests pillow safetensors datasets
```

### A.2 Flickr 胶片扫描下载 — 需要密钥

**获取密钥**：
1. https://www.flickr.com/services/apps/create/
2. "Apply for a Non-Commercial Key"
3. App name: `K-MCFM Film Research`
4. Description: `Academic research project studying film photography aesthetics using machine learning.`

在项目根目录创建 `.env`（已在 `.gitignore` 中）：

```bash
FLICKR_API_KEY=你的API_KEY
FLICKR_API_SECRET=你的API_SECRET
```

运行下载（每种胶片约 5-10 分钟，支持断点续传和去重）：

```bash
python scripts/scrape_films.py --stock portra_400 --count 500 --dedup
python scripts/scrape_films.py --stock vision3_500t --count 500 --dedup
python scripts/scrape_films.py --stock vision3_250d --count 500 --dedup
python scripts/scrape_films.py --stock portra_800 --count 500 --dedup
python scripts/scrape_films.py --stock ektar_100 --count 500 --dedup
python scripts/scrape_films.py --stock tri_x_400 --count 500 --dedup
python scripts/scrape_films.py --stock velvia_50 --count 500 --dedup
python scripts/scrape_films.py --stock hp5 --count 500 --dedup
```

### A.3 物理 PDF（公开，无需密钥）

```bash
python scripts/download_data.py physics
```

### A.4 CIE 色彩校准（公开，无需密钥）

```bash
python scripts/download_data.py cie
```

### A.5 相机光谱（公开，无需密钥）

```bash
python scripts/download_data.py camera-spectral
```

### A.6 FiveK DNG（公开，50GB，可选）

```bash
python scripts/download_data.py fivek-dng
```

### A.7 HuggingFace 模型权重（自动下载，无需密钥）

首次使用时自动从 HuggingFace 下载到 `~/.cache/huggingface/`：

| 模型 | 大小 | 用途 |
|------|:---:|------|
| `timbrooks/instruct-pix2pix` | ~3GB | IP2P 推理 |
| `stabilityai/stable-diffusion-xl-base-1.0` | ~12GB | SDXL 基座 |
| `runwayml/stable-diffusion-v1-5` | ~5GB | SD 1.5 基座 |

### A.8 Windows 端数据同步

```bash
# Mac 端执行（替换 WIN_IP）
rsync -avz --progress data/film_domain/ YOURUSER@WIN_IP:"/c/projects/NEURO_FILM/data/film_domain/"
rsync -avz --progress data/physics/ YOURUSER@WIN_IP:"/c/projects/NEURO_FILM/data/physics/"
rsync -avz --progress loras/ YOURUSER@WIN_IP:"/c/projects/NEURO_FILM/loras/"
```

### A.9 所需密钥汇总

| 服务 | 用途 | 获取地址 |
|------|------|---------|
| Flickr API | 下载胶片扫描 | https://www.flickr.com/services/apps/create/ |
| GitHub Token | Git push/pull | https://github.com/settings/tokens |
| HuggingFace | 模型下载 | 不需要密钥 |
| CivitAI | 社区 LoRA | https://civitai.com/user/account（可选） |

---

*文档生成：2026-05-24 | 实验总计：9 轮，~16 小时*
