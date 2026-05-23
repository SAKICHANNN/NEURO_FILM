# AGENTS.md — K-MCFM Project Knowledge Base

> **这个文件是给 Coding Agent 看的。** 当你换用任何 AI 编程助手时，让 Agent 先读这个文件即可获得完整项目上下文。每当项目发生实质性变化，Agent 应主动更新本文件。

---

## 1. 项目身份卡片

| 字段 | 值 |
|------|-----|
| **项目名** | K-MCFM: KAN-Mamba Conditional Flow Matching |
| **一句话** | 基于深度学习的数字→胶片图像转换：用 CFM+Mamba+KAN 物理精确地模拟胶片成像特性 |
| **目标硬件** | NVIDIA RTX 5070 Ti Laptop GPU 12GB VRAM |
| **输入** | 数字 RAW / 16-bit TIFF |
| **输出** | 胶片模拟图像（Kodak Vision3 500T / 250D、Portra 400 / 800、Ektar 100、Fujifilm Velvia 50、Ilford HP5、Kodak Tri-X） |
| **Python** | 3.12.10 (.venv) |
| **CUDA** | 12.8 |
| **PyTorch** | 2.11.0+cu128 |
| **项目类型** | 学术研究 + 工程实现 |
| **当前阶段** | 规划完成；1.1 环境完成；下一步启动 1.2 数据管线 |
| **许可** | MIT |

---

## 2. 核心文档地图

| 文件 | 内容 | Agent 使用场景 |
|------|------|---------------|
| `AGENTS.md` | 本文件 — 项目知识全集 | **每次新会话先读这个** |
| `TASK_BOARD.md` | 多 Agent 协调板：任务分配、文件锁、心跳 | **每次会话第二个读这个** |
| `IMPL_PLAN.md` | ~800 行实施计划：每个模块的代码骨架、超参数、接口定义 | 开始编码任何模块前查阅 |
| `GAP_ANALYSIS.md` | 资源缺口：数据集、物理数据、软件环境、行动计划 | 需要了解"还缺什么"时查阅 |
| `DATA_LICENSE_BOUNDARIES.md` | 数据许可、发表、权重发布边界 | 发布模型/论文/demo 前必须查阅 |
| `docs/MAC_ARM_MIGRATION.md` | 通过 GitHub 迁移到 Apple Silicon Mac 的环境/数据重建 runbook | 换机、Mac ARM 初始化、避免提交数据集时查阅 |
| `docs/DATA_REPRODUCTION_MANIFEST.json` | 被 Git 忽略的数据资源复现清单：路径、校验、恢复命令、缺口 | 在新机器恢复 `data/` 或核对数据状态时查阅 |
| `FILMGRAINSTYLE740K_REQUEST_EMAIL.md` | FilmGrainStyle740k 许可申请邮件草稿 | 用户申请受限颗粒数据时使用 |
| `guidelines.pdf` | 原始研究指南 (13 页，图片型 PDF) | 理解项目原始动机和学术背景 |
| `README.md` | 面向人类的项目介绍 | 了解项目对外呈现 |
| `requirements.txt` | Python 依赖（精确版本） | 搭建环境 |
| `requirements-macos-arm.txt` | Apple Silicon Mac 依赖（MPS，无 CUDA-only 包） | ARM Mac 开发环境搭建 |

---

## 3. 技术架构全景

```
数字 RAW/TIFF (B, 3, H, W)
        │
        ▼
   [VAE Encoder]  ← SD AutoencoderKL, 4ch latent, 8× down
        │
  (B, 4, H/8, W/8)  ← 潜空间 (目标 <2.5GB VRAM)
        │
        ▼
   [CFM Velocity Field Predictor]
     ├── MambaVision Backbone (4-stage: Conv → Conv → SSM+Attn → SSM+Attn)
     ├── KAN Physics Layers (B-spline, 仅在小维度物理映射中使用)
     ├── Instance-Disentangled Attention (内容/风格分离)
     └── Time Embedding (sinusoidal)
        │
   [ODE Solver]  ← Euler/RK4/dopri5, 30-50 NFE
        │
        ▼
   [VAE Decoder]
        │
        ▼
   [Physical Refinement]  ← 颗粒, 光晕, 微对比度 (像素空间)
        │
        ▼
   胶片模拟输出 (B, 3, H, W)
```

**关键架构决策（Agent 需要知道 WHY）：**

1. **潜空间操作** — 不在像素空间跑 CFM。像素空间 512×512×3 = 786K dims，8× VAE 潜空间 64×64×4 = 16K dims。这是 12GB VRAM 本机环境下可行的路线。
2. **MambaVision 混合架构** — 前两阶段用 Conv（高分辨率不需要长距离依赖），后两阶段用 SSM+Attention（中等分辨率建立全局关系）。纯 Transformer 在 12GB 上跑不动高分辨率。
3. **KAN 只在物理层使用** — KAN 作者自己警告"不能即插即用"，其优势在小维度 (<64) 物理函数拟合。**不要**在骨干网络中替换 MLP。
4. **OT-CFM 是主训练变体** — 5 种 CFM 变体中，OT-CFM（ExactOptimalTransport）提供最直的 ODE 轨迹，训练最稳定，推理最快。其他变体作为对照实验。
5. **LoRA 做风格切换** — 全模型 30M+ 参数，但 LoRA 包只需 2-5MB。每种胶片一个 LoRA，推理时动态加载/插值。

---

## 4. 项目规则（Agent 必须遵守）

### 硬性约束

1. **不能修改项目目录外的任何文件**。Write/Edit 操作限制在 `C:\Users\hhvrf\Documents\neuro_film\` 下（`.claude/settings.local.json` 已配置）。

2. **不要在没有明确确认的情况下给用户创建 PR、提交代码、推送**。Agent 可以自由创建本地 commit，但 push/PR 需要用户确认。

3. **不要创建无价值的文档**。不要生成 README、CHANGELOG、或任何 `.md` 除非明确要求。

4. **在开始大的实现任务前，先进入 Plan Mode**。参考 `IMPL_PLAN.md` 中对应模块的设计。

5. **不要引入不必要的依赖**。坚持 `requirements.txt` 中的版本。新依赖必须与 MIT/BSD/Apache 2.0 许可兼容。

### 编码约定

6. **默认不写注释**。只在 WHY 不明显时写一行注释。不要多行 docstring——命名应自说明。

7. **三个相似行 > 一个过早抽象**。不要引入不需要的辅助函数、工具函数、基类。直接写。

8. **不要在不能发生的场景上加错误处理**。信任内部代码和框架保证。只在系统边界验证。

9. **Type hints 在公共接口上**。`def process(raw_path: str) -> torch.Tensor:` 是好的。内部 helper 不需要。

10. **配置 > 硬编码**。所有超参数放在 `configs/` YAML 文件中。代码只读配置。

### 显存约束（Agent 需要持续关注）

11. **训练模式显存预算**: ≤10.2 GB / 12 GB。自动启用 gradient checkpointing, 8-bit Adam, BF16。
12. **推理模式显存预算**: ≤3.8 GB / 12 GB。使用 tiled VAE encoding/decoding。
13. **批量大小**: 潜空间训练 = 8, 像素空间训练 = 1-2 (gradient accumulation = 4)。
14. **如果 OOM**: 先减 batch → 再减分辨率 → 最后才减模型大小。

### 文件组织

15. **新模型代码** → `src/models/<模块>/`，配 `__init__.py`
16. **新损失函数** → `src/losses/`
17. **新数据管线** → `src/data/`
18. **配置** → `configs/` YAML 文件
19. **脚本入口** → `scripts/`
20. **测试** → `tests/`

---

## 5. 当前实现状态

### 已完成

| 项目 | 状态 |
|------|------|
| 项目结构搭建 | ✅ 目录 + `__init__.py` 全部就位 |
| 文档体系 | ✅ IMPL_PLAN, GAP_ANALYSIS, README, AGENTS, TASK_BOARD |
| 配置文件框架 | ✅ `.gitignore`, `requirements.txt`, `.claude/settings.local.json` |
| 依赖清单 | ✅ 版本锁定 |
| **1.1 软件环境** | ✅ **PyTorch 2.11+cu128, diffusers, timm, kornia, bitsandbytes 等 30+ 包已验证** |
| 多 Agent 协调 | ✅ TASK_BOARD.md + 接口 stubs + AGENTS.md §11 |
| 数据资源补齐 | ✅ FiveK DNG + Expert C TIFF16, FilmSet, Kodak/Ilford PDF 已落盘；见 `data/raw/dataset_status.json` |
| 数据下载脚本 | ✅ `scripts/download_data.py` 支持 FiveK DNG/Expert、FilmSet、DPED、CIE、physics PDF、camera spectral 断点补齐 |
| GitHub → ARM Mac 迁移档案 | ✅ `docs/MAC_ARM_MIGRATION.md` + `docs/DATA_REPRODUCTION_MANIFEST.json` + `requirements-macos-arm.txt`；目标 Mac = Apple Silicon / macOS Tahoe 26；数据目录仅提交 `.gitkeep` |

### 待开始

| 任务 | 优先级 | 预计工时 | 备注 |
|------|--------|---------|------|
| P0: 基线验证 (CFM+UNet) | **P0** | 5d | 依赖 1.2 + 1.3，不能直接开工 |
| Git 基线提交 | P0 | <1h | 当前 `master` 尚无 commit；项目文件仍在未跟踪工作区 |
| 1.2: 数据管线 | P0 | 5d | 数据已补齐到可开工；下一步建立 manifest |
| 1.3: VAE | P1 | 10d | |
| 1.4: 训练框架 | P1 | 3d | |

### 硬件实况 (与计划差异)

| 项目 | 计划 | 实际 | 影响 |
|------|------|------|------|
| GPU | RTX 3060 12GB | **RTX 5070 Ti Laptop GPU 11.94GB** (Blackwell) | 计算强很多；FA-3/FP8 可用 |
| CUDA | 12.1 | **12.8** (PyTorch 2.11.0+cu128) | 需 CUDA Toolkit 12.8 |
| Python | 3.10 | **3.12.10** (.venv) | 项目目录内 venv |
| FlashAttention | FA-2 | **PyTorch SDPA 内置** | 无需单独安装 FA-2 |
| xformers | 需要 | **已安装但当前路线不依赖** | PyTorch 2.11 内置 SDPA 为主；保留依赖供后续对照 |
| mamba-ssm | pip install | **未安装** | `causal-conv1d 1.6.1` 可用；`mamba_ssm` 导入失败 |
| pykan | pip install | **0.2.8 已安装；scikit-learn 1.8.0 + pandas 3.0.2 已补齐；`kan` 导入 OK** | 训练时仍需控制 KAN 维度 |
| tensorboard | pip install | **2.20.0 已安装** | 可用于训练日志 |
| 数据磁盘策略 | 待确认 | **继续使用当前项目磁盘** | 后续 checkpoints/data 可继续落在本项目目录 |
| 真实扫描验证集 | 待确认 | **用户决定要做 L5** | 后续需要拍摄/扫描流程与元数据模板 |

| 1.1: 软件环境搭建 & 验证 | P0 | 2d |
| 1.2: FiveK 数据下载 & 预处理 | P0 | 2d |
| 1.3: VAE 训练 | P1 | 10d |
| 2.1: CFM 模块实现 | P1 | 5d |
| 2.2: Mamba 骨干实现 | P1 | 12d |
| 2.3: KAN 物理层实现 | P2 | 8d |
| 3.x: 6 个物理模块实现 | P2 | 27d |
| 4.x: RTX 5070 Ti / 12GB 显存优化 | P2 | 15d |
| 5.x: 训练 & 评估 | P2 | 15d |
| 6.x: 部署 | P3 | 8d |

---

## 6. 关键外部资源位置

| 资源 | 位置 |
|------|------|
| CFM 参考实现 | `pip install torchcfm` 或 https://github.com/atong01/conditional-flow-matching |
| Mamba SSM kernel | `pip install causal-conv1d mamba-ssm` |
| MambaVision 预训练权重 | https://huggingface.co/collections/nvidia/mambavision (12 checkpoints) |
| KAN 参考实现 | `pip install pykan` 或 https://github.com/KindXiaoming/pykan |
| DiMSUM 代码 | https://github.com/VinAIResearch/DiMSUM |
| SD VAE 权重 | `diffusers.AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")` |
| MIT-Adobe FiveK 数据集 | https://data.csail.mit.edu/graphics/fivek/ (DNG + Expert TIFF/Lightroom catalog) |
| FilmSet 数据集 | https://github.com/CXH-Research/FilmNet → Kaggle FilmSet |
| FilmGrainStyle740k | https://www.interdigital.com/data_sets/filmgrainstyle740k-dataset (研究许可，需申请) |
| Kodak Portra 400 技术数据 | https://imaging.kodakalaris.com/sites/default/files/files/resources/e4050_portra_400.pdf |
| Kodak/Fujifilm/Ilford 技术 PDF | `data/physics/`（Vision3, Portra, Ektar, Tri-X, HP5, Velvia, Fujifilm guide） |
| CIE 色彩匹配/照明体数据 | `data/calibration/cie/`（XYZ 1931 2°, D50, D65, Illuminant A） |
| 相机光谱灵敏度 | `data/calibration/camera_spectral/`（RIT cam spec + Tokyo Open Vision spectra） |
| 数据集状态清单 | `data/raw/dataset_status.json` |
| DPED 数据集 | `data/raw/dped/`（sample + CNN training patches archives）或 https://aiff22.github.io/#dataset |
| 数据许可边界 | `DATA_LICENSE_BOUNDARIES.md` |
| FilmGrainStyle740k 申请邮件 | `FILMGRAINSTYLE740K_REQUEST_EMAIL.md` |
| LoRA 参考实现 | https://github.com/cloneofsimo/lora |

---

## 7. 模块间接口约定

### 数据流规范

```
RawLoader → (B, 3, H, W) float32, [0, 1], linear space
VAE.encode → (B, 4, H/8, W/8) float32, ~N(0,1) after scaling_factor
CFM.train_step → dict(loss=..., cfm_loss=..., phys_losses={...})
CFM.inference → (B, 4, H/8, W/8) → VAE.decode → (B, 3, H, W)
Film modules → (B, 3, H, W) → (B, 3, H, W)  (像素空间操作)
```

### 张量形状约定

- 所有图像: **(B, C, H, W)**, float32
- 潜空间: **(B, 4, H//8, W//8)**, float32
- 时间: **(B,)**, float32, [0, 1]
- 时间嵌入: **(B, embed_dim)**, float32
- Loss: scalar `torch.Tensor`

### 颜色空间

- 输入/输出: **线性 RGB** (不做 gamma 校正)
- 胶片物理计算: **log10(曝光)** 空间
- 评估: **线性 RGB** 输入指标，**sRGB** 展示

---

## 8. 常见 Agent 任务模式

### 实现一个新模块

1. 查 `IMPL_PLAN.md` 中该模块的代码骨架和超参数
2. 在 `src/models/<模块>/` 创建 `.py` 文件
3. 在对应 `configs/` YAML 加配置
4. 在 `tests/` 写基础测试（至少验证 forward pass shape）
5. 更新本文档第 5 节的状态

### 调试 OOM

1. 用 `src/training/memory.py` 的 `VRAMBudget` 测量峰值
2. 检查 gradient checkpointing 是否在 Mamba blocks 上启用
3. 减小 `batch_size` → 减小 `resolution` → 减小 `model_dim`
4. 确认 BF16（非 FP32）在使用

### 添加新胶片风格

1. 在 `configs/styles/` 创建 YAML（包含目标 H&D 曲线参数、颗粒参数、光晕参数）
2. 训练 LoRA: `python scripts/train_lora.py --config configs/styles/<style>.yaml`
3. 验证: `python scripts/inference.py data/test/img.tiff --style <style>`

> 上述训练/推理命令是目标入口约定。当前已落地的脚本入口只有 `scripts/download_data.py`，其余入口随对应实现任务补齐。

### 搜索 & 研究

本项目允许自由上网搜索，无需授权。搜索域包括 arxiv.org、github.com、HuggingFace、paperswithcode.com 等。

---

## 9. 已知陷阱（Agent 注意）

| 陷阱 | 说明 |
|------|------|
| **`causal-conv1d` 安装失败** | Windows 上需 MSVC + CUDA。若预编译 wheel 不可用，用 `F.scaled_dot_product_attention` 回退 |
| **KAN 训练极慢** | B-spline 计算无法 GPU 并行化。严格控制 KAN 用在小维度（<64） |
| **VAE 潜空间坍塌** | KL weight 太小会导致 posterior collapse。从 1e-6 开始慢慢增加 |
| **CFM ODE 不收敛** | OT-CFM 的 sigma 设太大（>1e-3）会导致噪声路径。保持 sigma=1e-5 |
| **Mamba scan 方向不对** | 2D 图像需要四向扫描（L→R, R→L, T→B, B→T），仅单向会导致方向伪影 |
| **颗粒 vs 数字噪声** | 纯随机 N(0,1) 看起来是数字噪声，不是胶片颗粒。必须用 NPS 滤波器着色 |
| **色彩空间混乱** | H&D 曲线在 log 空间定义。不要在线性 RGB 上直接应用 sigmoid |

---

## 10. 更新协议

**Agent 应在以下情况更新本文件：**

1. 某个实现状态从"待开始"变为"已完成"（更新第 5 节）
2. 添加/删除/重命名文件或模块（更新第 2、4 节）
3. 发现并修复了一个长期存在的 bug（添加到第 9 节）
4. 架构决策发生变更（更新第 3 节）
5. 添加/修改了模块间接口（更新第 7 节）
6. 添加了新的外部依赖或预训练权重（更新第 6 节）

**更新原则：保持简洁。** 不写描述性叙述，只写 Agent 下次需要知道的事实。如果一节变得太长（>30 行），考虑把细节移到对应子文档。

---

*最后更新: 2026-05-22 | 更新者: Codex | 项目阶段: 数据策略 V2 → 启动 1.2；Git 基线待提交*

---

## 11. 多 Agent 协调 (Claude Code + Codex)

### 关键约束

本项目有两个 coding agent（Claude Code 和 Codex）同时在同一文件系统工作。
它们通过共享文件 `TASK_BOARD.md` 协调，没有直接通信渠道。

**硬性规则：**

1. **每次会话开始时第二个读** `TASK_BOARD.md`（第一个读本文件）。不读就开始编码会导致文件冲突。
2. **每次状态变化时重写** `TASK_BOARD.md`（认领任务、锁定/解锁文件、完成任务、心跳）。
3. **绝不编辑其他 agent 锁定的文件**。编辑任何文件前，检查 `TASK_BOARD.md` 的 Locked Files 列。
4. **绝不同时编辑同一个文件**。即使文件被另一个 agent 锁定但看起来闲置，也不要碰。
5. **每 30 分钟更新一次心跳**（更新 `TASK_BOARD.md` Active Tasks 中自己的 `Last Active` 列）。
6. **不认领有未完成依赖的任务**。检查 Active Tasks 和 Available Tasks 的 Deps 列。
7. **完成任务时同时更新** `AGENTS.md` §5 的实现状态。
8. **如果你的依赖模块被另一个 agent 完成**，等用户合并到共享基线分支后，从该分支获取新接口。

### 操作速查表

| 操作 | 动作序列 |
|------|---------|
| **认领任务** | Read board → verify deps done + file locks clear → `git checkout -b <agent>/<id>-<name>` → Update board (in_progress, locked files) → `git commit` board |
| **编辑文件前** | Read board → verify file not in anyone's Locked Files → Add file to your Locked Files → Edit |
| **释放文件** | Remove file from your Locked Files → Write board |
| **完成任务** | `git commit` code → Update board (done, clear locks) → Update AGENTS.md §5 → `git commit` docs |
| **心跳** | Update your row's Last Active → Write board |
| **检测到冲突** | Mark task blocked in board → notify user |

### Agent 标识

本项目中的两个 agent 在 `TASK_BOARD.md` 的 `Agent` 列使用以下标识：
- **claude**: Claude Code
- **codex**: Codex

### 接口 Stub 文件

为了支持两个 agent 在依赖模块间的并行开发，以下接口 stub 文件已在当前工作区创建，定义了跨模块的契约；初始 Git 基线提交仍待完成：

| 文件 | 接口 | 消费方 → 实现方 |
|------|------|----------------|
| `src/models/mamba/interfaces.py` | `MambaBackboneInterface` | CFM (2.1) → Mamba (2.2) |
| `src/models/kan/interfaces.py` | `KANPhysicsInterface` | Film 模块 (3.x) → KAN (2.3) |
| `src/models/vae/interfaces.py` | `VAEInterface` | CFM/Inference → VAE (1.3) |
| `src/data/interfaces.py` | `FilmDatasetInterface` | Everyone → Data (1.2) |

**使用接口开发：** 依赖方 `import` 接口类并编码，实现方继承接口类并填充方法体。
**不要修改接口签名** 除非两个 agent 同意（通过 TASK_BOARD.md 上的标记）。

### 分支命名规范

`<agent>/<phase>.<task-id>-<short-desc>`

示例: `claude/1.3-vae-encoder`, `codex/2.2-mamba-backbone`

### Stale 任务回收

如果 Active Tasks 中有任务 `status=in_progress` 且 `Last Active > 2h ago`：
1. 可以要求用户检查该 agent 的 git 分支状态
2. 用户决定 `abandon`（回收）还是保留
3. 回收后，任务回到 `available`，文件锁自动释放
