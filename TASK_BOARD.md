# TASK_BOARD.md — V3 Diffusion-Based Film Translation

> 架构 V3 (2026-05-23)：SDEdit + 胶片 LoRA + IP-Adapter

---

## Phase 1: SDXL 基线 + 社区 LoRA — 1 周

| ID | Task | Status | Priority | Estimated |
|----|------|--------|----------|-----------|
| 1.1 | 环境搭建 (diffusers + SDXL + peft) | documented | P0 | 0.5d |
| 1.2 | SDXL img2img pipeline 搭建 (`scripts/pipeline.py`) | implemented-unverified | P0 | 0.5d |
| 1.3 | 下载已验证社区 SDXL LoRA (Portra 400, Vision3 500T, Vision3 250D, Ektar 100) | ready | P0 | 0.5d |
| 1.4 | strength 参数调优 (grid search per film) | available | P0 | 1d |
| 1.5 | CLI 实现 (pipeline.py) | implemented-unverified | P0 | 1d |

## Phase 2: 自训练胶片 LoRA — 2 周

| ID | Task | Status | Priority | Estimated |
|----|------|--------|----------|-----------|
| 2.1 | 胶片域数据收集 (Flickr/API 恢复 3,896 张记录数据, ≥200/film) | blocked-local-data-missing | P1 | 2d |
| 2.2 | kohya-ss/diffusers LoRA 训练环境 | available | P1 | 0.5d |
| 2.3 | 训练 Portra 800 LoRA (无精确 SDXL 社区 LoRA) | available | P1 | 1d |
| 2.4 | 训练 Tri-X 400 LoRA (无精确 SDXL 社区 LoRA) | available | P1 | 1d |
| 2.5 | 训练 Velvia 50 LoRA (正片特化) | available | P1 | 1d |
| 2.6 | 训练 HP5 LoRA (黑白特化) | available | P2 | 1d |
| 2.7 | 训练/复训其余胶片 LoRA（社区质量不足时替代） | available | P2 | 2d |
| 2.8 | 验收测试 (A/B 对比 + 量化) | available | P1 | 1d |

## Phase 3: 增强管线 + 多平台 — 2 周

| ID | Task | Status | Priority | Estimated |
|----|------|--------|----------|-----------|
| 3.1 | IP-Adapter 集成 (h94/IP-Adapter SDXL) | available | P1 | 2d |
| 3.2 | ControlNet-depth 可选集成 | available | P2 | 1d |
| 3.3 | 后处理颗粒+光晕 (filmgrainer + Gaussian) | available | P2 | 1d |
| 3.4 | Mac MLX 适配 | available | P2 | 2d |
| 3.5 | 全管线 CLI | available | P1 | 1d |
| 3.6 | 定量 + 主观评估 | available | P1 | 1d |

---

## 锁文件

| Agent | Files |
|-------|-------|
| codex | AGENTS.md, IMPL_PLAN.md, TASK_BOARD.md, docs/ARCH_REDESIGN.md |

---

*V3 基线: 2026-05-23 | 数据核实: 2026-05-25 | 下一步: 运行 Phase 1.3 下载与 Phase 1.4 调参*
