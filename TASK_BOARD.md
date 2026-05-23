# TASK_BOARD.md — 多 Agent 协调板

> **这是 K-MCFM 项目任务分配和文件锁的单一事实来源。**
>
> 规则:
> - 每次会话开始必须读此文件
> - 每次状态变更必须重写此文件
> - 绝不删除其他 agent 的行。你的最新一行是你任务的最新状态。
> - 编辑文件前，检查 Locked Files 列——绝不碰已被锁定的文件。
> - 完成一个步骤后同时更新 AGENTS.md §5 和本文件。

---

## Active Tasks

> 当前活跃任务。行按时间倒序（最新的在最上面）。
> Agent 认领任务时在此表末尾新增一行，status=`in_progress`。
> 完成任务时更新该行，status=`done`，填写 Completed 时间。

| ID | Task | Deps | Status | Agent | Branch | Locked Files | Started | Last Active | Completed |
|----|------|------|--------|-------|--------|-------------|---------|-------------|-----------|
| MIGRATE-MAC-ARM-20260523 | Prepare GitHub migration docs and ARM Mac setup files | - | done | codex | codex/migrate-mac-arm | - | 2026-05-23T13:49 | 2026-05-23T13:55 | 2026-05-23T13:55 |
| DOC-STATE-ALIGN-20260522 | Align current and target project state docs | - | done | codex | master (uncommitted) | - | 2026-05-22T22:09 | 2026-05-22T22:16 | 2026-05-22T22:16 |
| DOC-FILMGRAIN-IDENTITY-20260522 | Fill FilmGrainStyle740k requester identity | - | done | codex | master (uncommitted) | - | 2026-05-22T21:28 | 2026-05-22T21:28 | 2026-05-22T21:28 |
| DOC-FILMGRAIN-EMAIL-20260522 | Align FilmGrainStyle740k request email with access page | - | done | codex | master (uncommitted) | - | 2026-05-22T21:24 | 2026-05-22T21:24 | 2026-05-22T21:24 |
| DOC-FILMGRAIN-20260522 | Update FilmGrainStyle740k request scope | - | done | codex | master (uncommitted) | - | 2026-05-22T16:29 | 2026-05-22T16:29 | 2026-05-22T16:29 |
| DECISIONS-20260506 | Apply human dataset/style/license decisions | 1.1 | done | codex | master (uncommitted) | - | 2026-05-06T19:32 | 2026-05-06T19:39 | 2026-05-06T19:39 |
| SEED | Project seed workspace; git baseline still pending | - | done | claude | master (uncommitted) | - | 2026-04-27T22:30 | 2026-04-27T22:30 | 2026-04-27T22:30 |
| 1.1 | Environment setup & verify | - | done | claude | master (uncommitted) | - | 2026-04-28T01:30 | 2026-05-06T10:18 | 2026-04-28T02:15 | Python 3.12.10, PyTorch 2.11.0+cu128, RTX 5070 Ti Laptop GPU 11.94GB; mamba-ssm missing; pykan/scikit-learn/pandas/tensorboard verified |

---

## Repository Baseline

- Current workspace state on 2026-05-22: branch `master` has no Git commits yet and project files are still untracked.
- Active-task rows describe work recorded in the shared workspace. After the initial baseline commit exists, use real branch and commit state for new task rows.

## Available Tasks

> 从 IMPL_PLAN.md 预填充。Agent 认领时从此表选一个状态为 `available` 的任务，将其移入 Active Tasks 表。
> 如果一个任务被阻塞（依赖未完成），标记为 `blocked`。
> 如果一个 agent 认领的任务超 2h 无心跳，标记为 `abandoned`，可供其他 agent 回收。

### Phase P0 — 基线验证

| ID | Task | Status | Priority | Deps | Suggested Agent | Estimated |
|----|------|--------|----------|------|-----------------|-----------|
| GIT-BASELINE | Create initial Git baseline from current workspace | available | P0 | - | either | <1h |
| P0.1 | Baseline: CFM + UNet 验证 | available | P0 | 1.1, 1.2, 1.3 | either | 5d |

### Phase 1 — 基础架构

| ID | Task | Status | Priority | Deps | Suggested Agent | Estimated |
|----|------|--------|----------|------|-----------------|-----------|
| 1.1 | 软件环境搭建 & 验证 | done | P0 | - | claude | 2d |
| 1.2 | 数据管线 (RAW/TIFF 加载/预处理/增强) | available | P0 | 1.1 | either | 5d |
| 1.3 | VAE 潜空间编解码器 | available | P1 | 1.1, 1.2 | either | 10d |
| 1.4 | 训练/验证循环框架 | available | P1 | 1.1, 1.3 | either | 3d |

### Phase 2 — 核心模型

| ID | Task | Status | Priority | Deps | Suggested Agent | Estimated |
|----|------|--------|----------|------|-----------------|-----------|
| 2.1 | CFM 模块 (velocity field / ODE solver / OT coupling) | available | P1 | 1.3, 2.2 | either | 5d |
| 2.2 | Mamba 视觉骨干 | available | P1 | 1.1, 2.4 | either (need mamba-ssm) | 12d |
| 2.3 | KAN 物理层 (B-spline / 物理映射) | available | P2 | 1.1 | either | 8d |
| 2.4 | Instance-Disentangled Attention | available | P1 | 1.1 | either | 5d |

### Phase 3 — 物理模拟

| ID | Task | Status | Priority | Deps | Suggested Agent | Estimated |
|----|------|--------|----------|------|-----------------|-----------|
| 3.1 | Tone Mapping (H&D 曲线) | available | P2 | 2.3 | either | 5d |
| 3.2 | 光谱交叉 / Coupler 模拟 | available | P2 | 2.3 | either | 5d |
| 3.3 | DIR 微对比度 | available | P2 | 2.3 | either | 4d |
| 3.4 | Halation / Bloom 散射 | available | P2 | 2.3 | either | 5d |
| 3.5 | Film Grain (NPS 合成) | available | P2 | - | either | 5d |
| 3.6 | Non-uniformity 校正 | available | P2 | - | either | 3d |

### Phase 4 — RTX 5070 Ti / 12GB 显存优化

| ID | Task | Status | Priority | Deps | Suggested Agent | Estimated |
|----|------|--------|----------|------|-----------------|-----------|
| 4.1 | PyTorch SDPA / xformers attention optimization | available | P2 | 2.4 | either (Blackwell GPU, PyTorch 2.11) | 5d |
| 4.2 | 显存预算管理 & GC | available | P2 | 1.4 | either | 5d |
| 4.3 | 两阶段推理管线 | available | P2 | 1.3, 2.1 | either | 5d |

### Phase 5 — 训练与评估

| ID | Task | Status | Priority | Deps | Suggested Agent | Estimated |
|----|------|--------|----------|------|-----------------|-----------|
| 5.1 | 多维度损失 (感知/物理/结构) | available | P2 | 1.3 | either | 3d |
| 5.2 | LoRA 微调管线 | available | P2 | 2.1, 5.1 | either | 5d |
| 5.3 | 评估套件 (FID/LPIPS/H&D RMSE) | available | P2 | 2.1 | either | 7d |

### Phase 6 — 部署

| ID | Task | Status | Priority | Deps | Suggested Agent | Estimated |
|----|------|--------|----------|------|-----------------|-----------|
| 6.1 | ONNX/TensorRT 导出 | available | P3 | 2.1, 1.3 | either | 5d |
| 6.2 | CLI 推理工具 | available | P3 | 6.1 | either | 3d |

---

## Coordination Protocol

### 认领任务

```
1. Read TASK_BOARD.md
2. Verify: 任务 available + 所有 Deps done + 无文件锁冲突
3. Create branch: git checkout -b <agent>/<task-id>-<short-name>
4. Write TASK_BOARD.md: 任务移入 Active Tasks, status=in_progress, 填写 Locked Files
5. Commit: git add TASK_BOARD.md && git commit -m "claim <task-id>"
```

### 编辑文件前

```
1. Read TASK_BOARD.md
2. Check: 目标文件不在任何其他 agent 的 Locked Files 列
3. Update: 添加文件到自己的 Locked Files
```

### 完成任务

```
1. Commit code changes
2. Update TASK_BOARD.md: status=done, clear Locked Files, fill Completed
3. Update AGENTS.md §5: 更新实现状态
4. git add -A && git commit -m "complete <task-id>"
5. Notify user: "Task <id> done on branch <branch>"
```

### 心跳

```
每 30 分钟或任何状态变更时:
1. 更新 Active Tasks 中自己任务的 Last Active 为当前时间
```

### Stale 任务回收

```
条件: status=in_progress AND Last Active > 2h ago
用户检查 git log 后决定: abandon (回收) 或保持
```

---

*Seed workspace: 2026-04-27 | Git baseline pending | 项目处于规划完成、待启动 P0 阶段*
