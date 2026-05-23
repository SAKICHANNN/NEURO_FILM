# Architecture Redesign — Feasibility Critique & Revised Plan

> 基于 2026 年 5 月对原始 K-MCFM 方案的深度可行性分析和学术搜索，系统性地推翻原始架构，给出更务实、更可行的替代方案。

---

## 1. 执行摘要

原始方案（CFM + MambaVision + KAN + VAE 潜空间 + 6 物理模块）存在三个致命缺陷，综合成功率估计 **15-25%**。修订方案（CycleGAN/CUT + 3D LUT + H&D 曲线 LUT + 外挂颗粒 + 光晕后处理）将成功率提升到 **60-70%**，开发周期从 5.5 个月缩短到 6-8 周。

---

## 2. 原始方案致命缺陷

### 2.1 数据问题：没有 Ground Truth

**核心矛盾**：目标是"数字→胶片"转换，但不存在公开的 digital↔film 配对数据集可以监督训练。

原始方案自知这个缺陷，设计了 6 层数据策略试图绕开：

| 层级 | 方案 | 根本问题 |
|------|------|---------|
| L1 FiveK | 用 RAW→专家修图做预训练 | 专家修图 ≠ 胶片模拟，分布完全不同 |
| L1.5 FilmSet | 3 种胶片风格图片 | 是终态图片，不是配对数据，无法做像素级监督 |
| L2 PBR 合成 | Mitsuba 3 光谱渲染 | 需要完整搭建光谱渲染管线，工作量 = 单独一个项目 |
| L3 物理 PDF | 手工数字化 H&D 曲线 | 只提供宏观参数，不提供像素级监督信号 |
| L4 Cinestill800T | 41 对真实配对 | 论文声称已公开发布（arxiv 2411.15967, 2024），原方案说"不可用"是信息错误，但 41 对确实太小 |
| L5 自建 | 200-500 对 | 需要拍摄+扫描流程，成本高，周期长 |

**结论**：5 个非直接数据源拼凑一个"隐式"胶片模拟器，中间没有任何一步可以直接验证输出是否真的像 Portra 400。这是监督学习的根本问题——你无法训练一个模型输出你没有 ground truth 的东西。

### 2.2 MambaVision：不适合图像生成任务

多篇 2024-2025 顶会论文直接质疑 Mamba 在视觉任务中的必要性：

| 来源 | 关键发现 |
|------|---------|
| **MambaOut (CVPR 2025)** | 去掉 SSM 模块后视觉 Mamba 性能**反而提升**。Mamba 对图像分类/生成不必要，仅对长序列任务有用。 |
| **Mamba® (arxiv 2025)** | Vision Mamba 存在严重 artifact——背景 token norm 值可达 4000+（正常 <100），从 tiny 模型开始就有，且随模型增大加剧。训练极不稳定。 |
| **Non-deterministic (GH Issue #117)** | CUDA kernel 使用 `atomicAdd` 导致训练不可重复。同一 seed 不同 run 结果不一致。 |
| **Achilles' Heel (2025)** | SSM 非线性卷积引入不对称偏差，Mamba 本质上难以学习对称模式。 |
| **ONNX/TensorRT** | MambaVision 官方确认无法导出到 ONNX/TensorRT（自定义 CUDA kernel 不兼容）。 |

**结论**：MambaVision 作为图像生成骨干的选择理由不成立。UNet 或标准 CNN 更成熟可靠，且 PyTorch 2.x 内置的 SDPA 已足够高效。

### 2.3 KAN：不是即插即用的

- KAN 作者自己明确警告"不能即插即用"
- B-spline 计算在 GPU 上极慢，无法并行化
- 用于拟合 1D H&D 曲线（1→1 映射）是杀鸡用牛刀——2 层 MLP 或解析 sigmoid 函数即可
- 4 个 KAN 子网络（H&D、crosstalk、DIR、exposure_to_grain）都是 ≤64 维的小映射，不需要 KAN

**结论**：全部替换为更简单的替代方案（MLP / 解析函数 / LUT）。

### 2.4 次要问题

| 问题 | 说明 |
|------|------|
| **复杂度膨胀** | 19 个 YAML 配置、4 阶段训练、5 种 CFM 变体、8 种胶片、6 个物理模块——过度工程化 |
| **VAE 潜空间** | 增加一层抽象，引入 KL 坍塌风险，增加调试难度。像素空间直接操作更简单可解释 |
| **CFM + ODE** | O-DE 求解器在推理时需要 30-50 步数值积分。GAN 单次前向即可推理 |
| **mamba-ssm 未安装** | 原始计划依赖 Mamba 骨干，但 `mamba-ssm` 至今无法安装 |

---

## 3. 搜索发现：更可行的替代方案

### 3.1 无配对数据的风格迁移（成熟方案）

**CycleGAN (ICCV 2017, 10K+ 引用)**: 无需配对数据，只需两个域的图像集合即可训练双向映射。12GB VRAM 上训练标准 CycleGAN (ResNet9Blocks + PatchGAN, batch_size=1, 256×256) 仅需 ~5GB VRAM。

**CUT (ECCV 2020)**: 单边对比学习，比 CycleGAN 少 31% VRAM（3.3GB vs 4.8GB at 256²），40% 更快，且保留结构细节更好。

| 方法 | 训练 VRAM (256²) | 推理 VRAM (512²) | 收敛时间 |
|------|:---:|:---:|:---:|
| CycleGAN | ~4.8 GB | ~3 GB | 1-2 天 |
| CUT | ~3.3 GB | ~3 GB | 1-2 天 |
| FastCUT | ~2.3 GB | ~3 GB | 0.5-1 天 |
| CycleGAN-Turbo | ~26 GB | 不可行 | 需云 GPU |

**推荐**：CUT/FastCUT 作为主训练方案。CUT 的 PatchNCE 损失在多层（layer 0,4,8,12,16）上做对比学习，对纹理和颜色都有良好的匹配能力。标准 CycleGAN 作为备选。

### 3.2 3D LUT 颜色映射（轻量高精度）

3D LUT 预测是颜色风格的成熟方案，需要极少的 VRAM 和参数：

| 方法 | 训练 VRAM | 参数量 | 推理速度 | 无配对支持 |
|------|:---:|:---:|:---:|:---:|
| **Image-Adaptive-3DLUT (TPAMI 2021)** | <2 GB | 1536 (LUT) + CNN | <1ms | 需配对 |
| **Neural Preset (CVPR 2023)** | ~6-10 GB | 256 (DNCM) + CNN | 52fps@4K | **是**（自监督） |
| **StarGAN + 3D LUT (ns144)** | <2 GB | 1536 + CNN | <1ms | **是**（GAN） |
| **NILUT (AAAI 2024)** | <1 GB | ~100K (MLP) | 可烘焙 | 否（LUT 拟合工具） |
| **SA-LUT (ICCV 2025)** | >12 GB | 208MB | 16fps | **是** |

**推荐**：
- **A 方案**：Image-Adaptive-3DLUT（使用 FiveK 预训练权重，在胶片数据上微调）——最快速，有预训练模型
- **B 方案**：Neural Preset 自监督训练（在任意图片集上应用随机胶片 LUT 做 perturbation）——无需配对数据

### 3.3 颗粒模拟：SilverGrain 不可用，filmgrainer 可行

**SilverGrain**（PyPI, 2026.02）：
- 实现 Newson et al. (2017) Boolean 模型 + Monte Carlo 渲染
- CUDA 加速（~750× vs CPU），pip install silvergrain[gpu]
- **致命问题**：
  - **AGPL-3.0 许可**：不能用于 MIT 项目，除非整个项目改为 AGPL
  - **RTX 5070 Ti (Blackwell, sm_120) 兼容性问题**：依赖 numba-cuda==0.20.1（预编译 kernel 不包含 sm_120 目标），大概率在 Blackwell GPU 上失败
  - 单一作者、0 fork，维护风险高

**filmgrainer**（PyPI, MIT licensed, 48 stars）：
- MIT 许可，pip install filmgrainer
- 多尺度颗粒 + gamma 感知 + 每通道 (R/G/B) 强度控制
- 支持 shadows/midtones/highlights 分别调节功率
- 非 Newson 级别物理精度，但视觉效果合理
- **100% 兼容 12GB VRAM + RTX 5070 Ti**

**自定义 Newson 实现**：
- 参考算法：Newson et al. (IPOL 2017)，Joseph Wardle Rust 重新实现 (MIT)
- 自己用 PyTorch/Triton 实现 pixel-wise 算法，MIT 许可
- 开发时间：3-5 天

**推荐**：先用 filmgrainer 快速验证管线，如需更高物理精度再自行实现 Newson 算法。

### 3.4 H&D 曲线数字化（已解决的问题）

- H&D 曲线在 PDF 中以图表形式存在，数字化是成熟流程
- 工具：WebPlotDigitizer、engauge-digitizer（开源）
- 数学形式：toe（低曝光缓起）→ linear region（直线段，斜率 = gamma）→ shoulder（高曝光饱和）
- 实现形式：每个通道（R/G/B）的 1D LUT，共 3×256 个 float32 值
- 已有开源实现参考：filmeon (2026), RafalLukawiecki/film_tests (2017)

**推荐**：手工数字化每个胶片的 H&D 曲线 → 存储为 CSV → 运行时加载为 1D LUT

### 3.5 光晕（Halation）

- 物理原理：光穿过片基后被反射回乳剂层，造成"红晕"
- 实现极简单：对高亮区域做各通道不同半径的 Gaussian blur，红通道半径最大
- 开源参考：Ansel 暗房软件, Filmulator
- 不需要训练或神经网络

---

## 4. 修订架构

```
Digital Input (B, 3, H, W) linear RGB, float32, [0,1]
        │
        ▼
   ┌───────────────────────────────────┐
   │ Module 1: Color Style Transfer    │
   │ ├─ Option A: CUT (unpaired GAN)   │
   │ ├─ Option B: 3D LUT Predictor     │
   │ └─ Option C: Neural Preset        │
   │ Output: (B,3,H,W) film-colored    │
   └───────────────┬───────────────────┘
                   │
                   ▼
   ┌───────────────────────────────────┐
   │ Module 2: H&D Tone Mapping        │
   │ └─ Per-channel 1D curve LUT       │
   │    (digitized from film PDFs)     │
   │ Output: (B,3,H,W) film-toned      │
   └───────────────┬───────────────────┘
                   │
                   ▼
   ┌───────────────────────────────────┐
   │ Module 3: Halation                │
   │ └─ Per-channel Gaussian scatter   │
   │    R > G > B radius, highlight-thresholded
   │ Output: (B,3,H,W) + red glow      │
   └───────────────┬───────────────────┘
                   │
                   ▼
   ┌───────────────────────────────────┐
   │ Module 4: Film Grain              │
   │ └─ filmgrainer (MIT) or           │
   │    Custom Newson Boolean model    │
   │ Output: FINAL (B,3,H,W) film      │
   └───────────────────────────────────┘
```

### 设计原则

1. **像素空间直接操作** — 不通过 VAE 潜空间，训练和推理都在像素空间
2. **模块解耦** — 每个模块独立开发、独立测试、独立验证
3. **后期可微调** — 颜色模块用 GAN 学习，物理模块用解析函数，互不依赖
4. **渐进式复杂度** — 先跑通全手动管线（模块 2+3+4），再加上颜色学习（模块 1）

---

## 5. 修订实施计划

### Phase 1: 手动基线管线（1 周）

| 任务 | 内容 | 产出 |
|------|------|------|
| 1.1 H&D 数字化 | 从 PDF 提取每个胶片的 H&D 曲线数据点，拟合为 1D LUT | `data/physics/<film>/hd_curve.csv` |
| 1.2 光晕实现 | 实现 highlight-thresholded + 多通道 Gaussian blur 光晕 | `src/models/film/halation.py` |
| 1.3 颗粒集成 | 集成 filmgrainer，映射 RMS granularity → filmgrainer 参数 | `src/models/film/grain.py` |
| 1.4 管线组装 | Python CLI: input.tiff → (可选 LUT 颜色) → H&D 映射 → 光晕 → 颗粒 → output.tiff | `scripts/pipeline.py` |
| 1.5 主观评估 | 在 1-2 个胶片上对比输出与真实胶片扫描 | 评估截图 |

### Phase 2: 颜色风格转移训练（2-3 周）

| 任务 | 内容 | 产出 |
|------|------|------|
| 2.1 胶片域数据收集 | 从 FilmSet + web 收集每种胶片的图片（≥500/类） | `data/film_domain/<film>/` |
| 2.2 CUT 训练 | 在 digital 域 (FiveK 输入) ↔ film 域 上训练 CUT/FastCUT | 颜色映射 generator |
| 2.3 备选：3D LUT 预测器 | 训练 CNN→LUT 预测器（如果 CUT 颜色精度不够） | LUT predictor |
| 2.4 颜色精度评估 | color histogram matching, ΔE2000 对比等 | 量化指标 |

### Phase 3: 集成与微调（1-2 周）

| 任务 | 内容 | 产出 |
|------|------|------|
| 3.1 全管线集成 | 颜色模块 + H&D + 光晕 + 颗粒串联 | `scripts/pipeline_full.py` |
| 3.2 逐胶片参数调优 | 每种胶片调整 H&D LUT、颗粒参数、光晕半径、颜色风格权重 | 每胶片 1 个 YAML |
| 3.3 批量推理 | 支持批量处理、多胶片切换 | CLI |
| 3.4 最终评估 | 对每种胶片做主观 + 定量评估 | 评估报告 |

**总工期：4-6 周**（vs 原计划 5.5 个月）

---

## 6. 放弃的项目

| 原计划模块 | 放弃原因 | 替代方案 |
|-----------|---------|---------|
| MambaVision 骨干 | 不适合图像生成，训练不稳定，无法导出 | CUT 的 ResNet/UNet 生成器 |
| KAN 物理层 | 杀鸡用牛刀，B-spline GPU 不能并行 | 解析函数 / MLP / LUT |
| CFM + ODE 求解器 | 推理需要 30-50 步积分 | GAN 单次前向 |
| VAE 潜空间 | 增加抽象层，引入 KL 坍塌风险 | 像素空间直接操作 |
| DIR 微对比度 (反应扩散) | 过于前沿，无验证手段 | 延期到 v2 |
| 光谱交叉 talk | 需要完整光谱数据，当前不可得 | 延期 |
| 非均匀性校正 | 次要美化功能 | 延期 |
| 实例解耦注意力 | 依赖 Mamba 骨干 | 不再需要 |

---

## 7. 成功率评估

| 方案 | 成功率 | 说明 |
|------|:---:|------|
| **原始 K-MCFM** | **15-25%** | 数据问题无法绕过 + Mamba+KAN 双重新技术风险 |
| **修订方案（全管线）** | **60-70%** | 每个模块独立可行，有成熟方案参考，可渐进验证 |
| **修订方案（仅手动管线）** | **85-90%** | H&D LUT + 光晕 + 颗粒 = 纯确定性操作 |
| 颜色模块用 CUT | **65-75%** | GAN 训练有不确定性，但 CUT 在多个域上被验证 |
| 颜色模块用 3D LUT | **70-80%** | 有预训练权重可用，微调比从零训练更可靠 |

---

## 8. 关键依赖与风险

| 风险 | 概率 | 影响 | 缓解 |
|------|:---:|------|------|
| filmgrainer 在 RTX 5070 Ti 上不工作 | 低 | 中 | 回退到 CPU 模式或自实现 Newson 算法 |
| CUT 训练不稳定（GAN mode collapse） | 中 | 高 | 先用 3D LUT 方案做 baseline |
| PDF H&D 曲线数字化精度不足 | 低 | 中 | 多个数据点取平均值，必要时用 MLP 平滑 |
| 收集的胶片域图片质量参差 | 中 | 中 | 人工筛选 + FilmSet 高质量子集 |

---

## 9. 参考资料

| 领域 | 来源 |
|------|------|
| CUT | Park et al., ECCV 2020, github.com/taesungp/contrastive-unpaired-translation |
| CycleGAN | Zhu et al., ICCV 2017, github.com/junyanz/pytorch-CycleGAN-and-pix2pix |
| 3D LUT 增强 | Zeng et al., TPAMI 2021, github.com/HuiZeng/Image-Adaptive-3DLUT |
| Neural Preset | Ke et al., CVPR 2023, github.com/ZHKKKe/NeuralPreset |
| NILUT | Conde et al., AAAI 2024, github.com/mv-lab/nilut |
| StarGAN + 3D LUT | ns144, github.com/ns144/3D-LUT |
| 颗粒: filmgrainer | PyPI filmgrainer, MIT |
| 颗粒: Newson et al. | IPOL 2017, github.com/alasdairnewson/film_grain_rendering |
| 光晕: Ansel | github.com/aurelienpierre/ansel |
| H&D 曲线: filmeon | github.com/helios1138/filmeon |
| H&D 曲线工具 | RafalLukawiecki/film_tests, github.com |
| SilentStill (41 对 Cinestill) | arxiv 2411.15967, github.com/mikasenghaas/sillystill |
| MambaOut | Yu et al., CVPR 2025 |
| Mamba® | arxiv 2405.14858 |
| 胶片社区 LoRA | civitai.com 已有 Kodak Vision3/Portra/Ektar LoRA |

---

*编写日期: 2026-05-23 | 修订版本: v1.0*
