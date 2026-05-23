# TASK_BOARD.md — V3 Diffusion-Based Film Translation

> 架构 V3 (2026-05-23)：SDEdit + 胶片 LoRA + IP-Adapter

---

## Phase 1: SDXL 基线 + 社区 LoRA — 1 周

| ID | Task | Status | Priority | Estimated |
|----|------|--------|----------|-----------|
| 1.1 | 环境搭建 (diffusers + SDXL + peft) | available | P0 | 0.5d |
| 1.2 | SDXL img2img pipeline 搭建 | available | P0 | 0.5d |
| 1.3 | 下载社区胶片 LoRA (Portra 400, Vision3 500T, Ektar 100, Tri-X) | available | P0 | 0.5d |
| 1.4 | strength 参数调优 (grid search per film) | available | P0 | 1d |
| 1.5 | CLI 实现 (pipeline.py) | available | P0 | 1d |

## Phase 2: 自训练胶片 LoRA — 2 周

| ID | Task | Status | Priority | Estimated |
|----|------|--------|----------|-----------|
| 2.1 | 胶片域数据收集 (FilmSet + web, ≥200/film) | available | P1 | 2d |
| 2.2 | kohya-ss/diffusers LoRA 训练环境 | available | P1 | 0.5d |
| 2.3 | 训练 Portra 400 LoRA | available | P1 | 1d |
| 2.4 | 训练 Vision3 500T LoRA | available | P1 | 1d |
| 2.5 | 训练 Velvia 50 LoRA (正片特化) | available | P1 | 1d |
| 2.6 | 训练 HP5 LoRA (黑白特化) | available | P2 | 1d |
| 2.7 | 训练其余胶片 LoRA | available | P2 | 2d |
| 2.8 | 验收测试 (A/B 对比 + 量化) | available | P1 | 1d |

## Phase 3: 增强管线 + 多平台 — 2 周

| ID | Task | Status | Priority | Estimated |
|----|------|--------|----------|-----------|
| 3.1 | IP-Adapter 集成 (h94/IP-Adapter SDXL) | available | P1 | 2d |
| 3.2 | ControlNet-depth 可选集成 | available | P2 | 1d |
| 3.3 | 后处理颗粒+光晕 (filmgrader + Gaussian) | available | P2 | 1d |
| 3.4 | Mac MLX 适配 | available | P2 | 2d |
| 3.5 | 全管线 CLI | available | P1 | 1d |
| 3.6 | 定量 + 主观评估 | available | P1 | 1d |

---

## 锁文件

| Agent | Files |
|-------|-------|
| codex | AGENTS.md, IMPL_PLAN.md, TASK_BOARD.md, docs/ARCH_REDESIGN.md |

---

*V3 基线: 2026-05-23 | 下一步: Phase 1.1 环境搭建*
