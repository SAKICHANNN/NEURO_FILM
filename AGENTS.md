# AGENTS.md — K-MCFM Project Knowledge Base (Revised)

> **每次新会话先读这个。** 项目架构已于 2026-05-23 重设计。原始 CFM+Mamba+KAN 方案已放弃，改用模块化像素空间管线。

---

## 1. 项目身份卡片

| 字段 | 值 |
|------|-----|
| **项目名** | K-MCFM: Modular Film Simulation Pipeline |
| **一句话** | 模块化数字→胶片图像转换：颜色风格迁移 + H&D 色调映射 + 光晕 + 颗粒 |
| **目标硬件** | NVIDIA RTX 5070 Ti Laptop GPU 12GB VRAM |
| **输入** | 数字 RAW / 16-bit TIFF |
| **输出** | 胶片模拟图像（Kodak Vision3 500T / 250D、Portra 400 / 800、Ektar 100、Fujifilm Velvia 50、Ilford HP5、Kodak Tri-X） |
| **Python** | 3.12.10 (.venv) |
| **CUDA** | 12.8 |
| **PyTorch** | 2.11.0+cu128 |
| **项目类型** | 学术研究 + 工程实现 |
| **当前阶段** | 架构重设计完成；启动 Phase 1（手动基线管线） |
| **许可** | MIT |

---

## 2. 核心文档地图

| 文件 | 内容 | 使用场景 |
|------|------|---------|
| `AGENTS.md` | 本文件 — 项目知识全集 | **每次新会话先读这个** |
| `docs/ARCH_REDESIGN.md` | 架构重设计文档：为什么放弃原方案、新方案依据、替代方案来源 | 理解项目方向变更原因 |
| `IMPL_PLAN.md` | 修订实施计划：3 阶段，4-6 周总工期 | 开始编码前查阅 |
| `TASK_BOARD.md` | 任务分配、文件锁、心跳 | 每次会话第二个读这个 |
| `GAP_ANALYSIS.md` | 资源缺口：数据集、物理数据、软件环境 | 需要了解"还缺什么"时查阅 |
| `DATA_LICENSE_BOUNDARIES.md` | 数据许可、发表、权重发布边界 | 发布前必读 |
| `README.md` | 面向人类的项目介绍 | 对外呈现 |
| `requirements.txt` | Python 依赖（精确版本） | 搭建环境 |
| `requirements-macos-arm.txt` | Apple Silicon Mac 依赖 | ARM Mac 开发环境 |

---

## 3. 技术架构全景

```
Digital Input (B, 3, H, W) linear RGB, float32, [0,1]
        │
        ▼
   [Module 1] Color Style Transfer  ← CUT/CycleGAN 或 3D LUT Predictor
        │                               (像素空间, 256²–512² 训练, ≤5GB VRAM)
        ▼
   [Module 2] H&D Tone Mapping      ← 每通道 1D 曲线 LUT
        │                               (从 PDF 数字化, 3×256 float32)
        ▼
   [Module 3] Halation              ← 多通道 Gaussian scatter
        │                               (R > G > B 半径, highlight 阈值)
        ▼
   [Module 4] Film Grain            ← filmgrainer (MIT) 或自实现 Newson 模型
        │                               (RMS granularity → 颗粒参数)
        ▼
   Film Output (B, 3, H, W)
```

### 关键架构决策

1. **像素空间直接操作** — 不使用 VAE 潜空间。像素空间更简单、可解释、无 KL 坍塌风险。
2. **模块解耦** — 4 个独立模块，各自开发和验证。颜色模块用 GAN/CNN 学习，物理模块用解析函数。
3. **渐进式开发** — Phase 1 先跑通全手动管线（H&D + 光晕 + 颗粒），Phase 2 再加颜色学习。
4. **无配对数据友好** — 颜色模块使用 CUT（无配对 GAN）或 3D LUT 预测（自监督），不依赖 digital↔film 配对。
5. **MIT 许可兼容** — 所有依赖均为 MIT/BSD/Apache 2.0 许可。不使用 AGPL 库（silvergrain）。

### 放弃的原始方案

| 原计划 | 放弃原因 |
|--------|---------|
| MambaVision 骨干 | 不适合图像生成（MambaOut CVPR 2025），训练不稳定（Mamba®），无法导出 ONNX |
| KAN 物理层 | 杀鸡用牛刀，B-spline GPU 不能并行 |
| CFM + ODE 求解器 | 推理需 30-50 步积分 vs GAN 单次前向 |
| VAE 潜空间 | 增加抽象层，KL 坍塌风险 |
| DIR 微对比度 / 光谱交叉 | 延期到 v2 |

详见 `docs/ARCH_REDESIGN.md`。

---

## 4. 项目规则

### 硬性约束

1. **不能修改项目目录外的任何文件**。
2. **不要在没有明确确认的情况下创建 PR、提交代码、推送**。Agent 可以自由创建本地 commit，但 push/PR 需要用户确认。
3. **不要创建无价值的文档**。不生成 README、CHANGELOG、或任何 `.md` 除非明确要求。
4. **在开始大的实现任务前，先读 IMPL_PLAN.md** 中对应模块的设计。
5. **不要引入不必要的依赖**。坚持 `requirements.txt` 中的版本。新依赖必须与 MIT/BSD/Apache 2.0 许可兼容。

### 编码约定

6. **默认不写注释**。只在 WHY 不明显时写一行注释。
7. **三个相似行 > 一个过早抽象**。不引入不需要的辅助函数。
8. **不要在不能发生的场景上加错误处理**。信任内部代码和框架保证。
9. **Type hints 在公共接口上**。`def process(raw_path: str) -> torch.Tensor:` 是好的。
10. **配置 > 硬编码**。所有超参数放在 `configs/` YAML 文件中。

### 显存约束

11. **训练模式显存预算**: ≤10.2 GB / 12 GB。CUT 训练 256² ≈3.3GB，512² ≈8-10GB。
12. **推理模式显存预算**: ≤3.8 GB / 12 GB。全管线推理 512² 轻松满足。
13. **批量大小**: CUT = 1, 3D LUT = 16+。
14. **如果 OOM**: 先减分辨率 → 再减 batch → 检查是否有内存泄漏。

### 文件组织

15. **新模型代码** → `src/models/<模块>/`，配 `__init__.py`
16. **物理模块代码** → `src/models/film/`（halation.py, grain.py, tone.py, color.py）
17. **新数据管线** → `src/data/`
18. **配置** → `configs/` YAML 文件
19. **脚本入口** → `scripts/`
20. **测试** → `tests/`

---

## 5. 当前实现状态

### 已完成

| 项目 | 状态 |
|------|------|
| 1.1 软件环境 | ✅ Python 3.12.10, PyTorch 2.11+cu128, 30+ 包已验证 |
| 数据下载脚本 | ✅ `scripts/download_data.py` |
| 配置文件框架 | ✅ `.gitignore`, `requirements.txt` |
| 文档体系 | ✅ AGENTS, README, GAP_ANALYSIS, DATA_LICENSE_BOUNDARIES |
| 架构重设计 | ✅ `docs/ARCH_REDESIGN.md` |
| 数据资源 (需重建) | ✅ FiveK DNG + Expert TIFF16, FilmSet, PDF |

### 待开始

| 任务 | 优先级 | 预计 | 备注 |
|------|--------|------|------|
| **Phase 1.1**: H&D 曲线数字化 | P0 | 1d | 从 PDF 提取曲线数据点 |
| **Phase 1.2**: 光晕实现 | P0 | 1d | 多通道 Gaussian scatter |
| **Phase 1.3**: 颗粒模块集成 | P0 | 1d | filmgrainer 集成 + 参数映射 |
| **Phase 1.4**: 手动管线 CLI | P0 | 2d | input.tiff → output.tiff |
| **Phase 2.1**: 胶片域数据收集 | P1 | 2d | 从 FilmSet + web 收集 |
| **Phase 2.2**: CUT 颜色迁移训练 | P1 | 7d | 256² 训练, 512² 推理 |
| **Phase 2.3**: 3D LUT 颜色（备选） | P2 | 5d | 如果 CUT 颜色精度不够 |
| **Phase 3.1**: 全管线集成 | P1 | 3d | 4 模块串联 |
| **Phase 3.2**: 逐胶片参数调优 | P1 | 3d | 每胶片 1 个 YAML |
| **Phase 3.3**: 批量推理 CLI | P2 | 2d | 多胶片切换 |

### 硬件实况

| 项目 | 值 |
|------|-----|
| GPU | RTX 5070 Ti Laptop GPU 11.94GB (Blackwell) |
| CUDA | 12.8 (PyTorch 2.11.0+cu128) |
| Python | 3.12.10 (.venv) |
| 注意力 | PyTorch SDPA 内置（无需 FA-2/xformers） |

---

## 6. 关键外部资源

| 资源 | 位置 |
|------|------|
| CUT 实现 | github.com/taesungp/contrastive-unpaired-translation |
| CycleGAN 实现 | github.com/junyanz/pytorch-CycleGAN-and-pix2pix |
| Image-Adaptive-3DLUT | github.com/HuiZeng/Image-Adaptive-3DLUT |
| Neural Preset | github.com/ZHKKKe/NeuralPreset |
| filmgrainer (颗粒, MIT) | PyPI filmgrainer |
| Newson 颗粒参考 (IPOL 2017) | github.com/alasdairnewson/film_grain_rendering |
| Newson Rust 重实现 (MIT) | github.com/josephwardle/film_grain |
| H&D 曲线/胶片反转: filmeon | github.com/helios1138/filmeon |
| 光晕参考: Ansel | github.com/aurelienpierre/ansel |
| FiveK 数据集 | https://data.csail.mit.edu/graphics/fivek/ |
| FilmSet 数据集 | https://github.com/CXH-Research/FilmNet → Kaggle |
| Kodak 技术 PDF | `data/physics/` |
| 架构重设计文档 | `docs/ARCH_REDESIGN.md` |

---

## 7. 模块间接口约定

### 数据流规范

```
Loader → (B, 3, H, W) float32, linear RGB, [0,1]
ColorTransfer.forward → (B, 3, H, W) float32
ToneMap.apply → (B, 3, H, W) float32
Halation.apply → (B, 3, H, W) float32
Grain.apply → (B, 3, H, W) float32
```

### 张量形状约定

- 所有图像: **(B, 3, H, W)**, float32
- 颜色空间: **线性 RGB** (输入/输出均不做 gamma 校正)
- 1D LUT: **(3, 256)** float32 (3 通道 × 256 采样点)
- 颗粒参数: `dict`，各胶片独立

### 颜色空间

- 全管线操作在 **线性 RGB** 空间
- H&D 曲线在 **log10(曝光)** 空间定义，内部转换
- 最终输出可转 sRGB 展示

---

## 8. 添加新胶片风格

1. 数字化 H&D 曲线 → `data/physics/<film>/hd_curve.csv`
2. 提取颗粒参数 (RMS granularity) → `configs/styles/<film>.yaml`
3. 设定光晕参数 (scatter radius per channel)
4. 收集胶片域图片 → `data/film_domain/<film>/`
5. (可选) 微调颜色模块权重
6. 验证: `python scripts/pipeline.py input.tiff --style <film>`

---

## 9. 已知陷阱

| 陷阱 | 说明 |
|------|------|
| **颜色模块输出超出 [0,1]** | CUT 生成器可能输出负值或 >1，需要 clamp |
| **H&D 曲线未归一化** | PDF 曲线在密度空间 (0-3+)，需映射到线性 RGB 的 [0,1] |
| **颗粒与分辨率关系** | 颗粒参数是像素级别的，不同分辨率需等比缩放 |
| **光晕过度** | 红通道 Gaussian σ 太大 (>0.5) 会导致不自然的红晕 |
| **许可冲突** | 绝对不 import silvergrain (AGPL-3.0)，使用 filmgrainer (MIT) |
| **CUT PyTorch 2.x 兼容** | CUT 官方仓库基于 PyTorch 1.x，需要 minor API 更新 |

---

## 10. 更新协议

Agent 应在以下情况更新本文件：
1. 实现状态变更（更新 §5）
2. 文件/模块添加删除（更新 §2, §3, §7）
3. 发现并修复 bug（添加到 §9）
4. 架构决策变更（更新 §3, §4）
5. 添加新依赖/权重（更新 §6）

---

*最后更新: 2026-05-23 | 更新者: Codex | 项目阶段: 架构重设计 → 启动 Phase 1*
