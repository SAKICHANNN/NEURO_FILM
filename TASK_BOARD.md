# TASK_BOARD.md — 任务协调板

> **架构重设计完成 (2026-05-23)**。原始 CFM+Mamba+KAN 方案已放弃。
> 新方案采用模块化像素空间管线：颜色迁移 + H&D 色调映射 + 光晕 + 颗粒。

---

## Active Tasks

| ID | Task | Status | Agent | Branch | Started | Last Active |
|----|------|--------|-------|--------|---------|-------------|
| ARCH-REDESIGN | AI 可行性深度分析 + 架构重设计 | done | codex | codex/arch-redesign | 2026-05-23T15:00 | 2026-05-23T16:30 |

---

## Phase 1: 手动基线管线 — 1 周

| ID | Task | Status | Priority | Deps | Estimated |
|----|------|--------|----------|------|-----------|
| 1.1 | H&D 曲线数字化 (从 PDF 提取曲线) | available | P0 | - | 1d |
| 1.2 | 光晕实现 (Gaussian scatter) | available | P0 | - | 1d |
| 1.3 | 颗粒模块集成 (filmgrainer) | available | P0 | - | 1d |
| 1.4 | 手动管线 CLI (input→output) | available | P0 | 1.1, 1.2, 1.3 | 2d |

## Phase 2: 颜色风格转移 — 2-3 周

| ID | Task | Status | Priority | Deps | Estimated |
|----|------|--------|----------|------|-----------|
| 2.1 | 胶片域数据收集 (FilmSet + web) | available | P1 | - | 2d |
| 2.2 | CUT 颜色迁移训练 | available | P1 | 2.1 | 7d |
| 2.3 | 3D LUT 颜色 (备选方案) | available | P2 | 2.1 | 5d |

## Phase 3: 集成 — 1-2 周

| ID | Task | Status | Priority | Deps | Estimated |
|----|------|--------|----------|------|-----------|
| 3.1 | 全管线集成 (4 模块串联) | available | P1 | 1.4, 2.2 | 3d |
| 3.2 | 逐胶片参数调优 | available | P1 | 3.1 | 3d |
| 3.3 | 批量推理 CLI | available | P2 | 3.2 | 2d |

---

## Locked Files

| Agent | Files |
|-------|-------|
| codex | AGENTS.md, IMPL_PLAN.md, TASK_BOARD.md, docs/ARCH_REDESIGN.md, GAP_ANALYSIS.md, README.md |

---

## 协调协议

### 认领任务
```
1. 读 TASK_BOARD.md → 验证 deps done + 无文件锁冲突
2. git checkout -b codex/<task-id>-<name>
3. 更新本文件 (status=in_progress, 添加 Locked Files)
4. git commit 本文件更新
```

### 完成任务
```
1. git commit 代码
2. 更新本文件 (status=done, 释放 Locked Files)
3. 更新 AGENTS.md §5
4. git commit 文档更新
```

### Stale 回收
- 条件: status=in_progress + Last Active > 2h
- 用户决定 abandon/keep

---

*架构重设计完成: 2026-05-23 | 下一步: Phase 1.1 H&D 曲线数字化*
