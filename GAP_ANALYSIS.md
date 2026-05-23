# K-MCFM 项目缺口分析（修订版）

> 基于 `docs/ARCH_REDESIGN.md` 的架构重设计，重新评估资源缺口。
> 原 6 层数据策略和 CFM+Mamba+KAN 方案已放弃。

---

## 1. 已解决的缺口

| 原缺口 | 原方案 | 修订方案 | 状态 |
|--------|--------|---------|:---:|
| 无配对数据 | 6 层间接数据拼凑 | CUT 无配对 GAN + 3D LUT 自监督 | ✅ 架构级解决 |
| VAE 潜空间坍塌风险 | 从 1e-6 逐增 KL weight | 像素空间直接操作 | ✅ 去掉 |
| ODE 求解器慢 (30-50 NFE) | dopri5 自适应步长 | GAN 单次前向 | ✅ 去掉 |
| Mamba 骨干不稳定 | Mamba® 寄存器修复 | 使用 ResNet/UNet 生成器 | ✅ 替换 |
| KAN B-spline GPU 慢 | 限制维度 <64 | 解析函数 / MLP / LUT | ✅ 替换 |
| mamba-ssm 无法安装 | 待修复 | 不再需要 | ✅ 去掉 |

---

## 2. 仍存在的缺口

### 2.1 胶片域图片（颜色训练用）

| 需求 | 当前状态 | 行动 |
|------|---------|------|
| 每胶片 ≥500 张高质量图片 | FilmSet 有 3 种风格 × 5,285 张，但质量参差 | 使用 FilmSet + 补充 web 爬取 |
| 图片需反映真实胶片特征 | FilmSet 是 Capture One 胶片模拟，非真实扫描 | 优先使用 FilmSet（已有），后期补充真实扫描 |

**风险**: FilmSet 只有 3 种风格 (Cinema, ClassicNeg, Velvia)，与目标 8 种胶片不完全对应。
**缓解**: 用相近的替代对应（Velvia → Velvia 50, ClassicNeg → Portra 系列），其他胶片从 web 收集。

### 2.2 H&D 曲线 PDF

| 需求 | 当前状态 | 行动 |
|------|---------|------|
| 8 种胶片的完整 H&D 曲线 | PDF 已在 `data/physics/` (部分已下载，需重建) | `python scripts/download_data.py physics` 重新下载 |
| 数字化工具 | WebPlotDigitizer (免费网页应用) | Phase 1.1 手动操作 |

**风险**: 部分 PDF 可能只有 R/G/B 合并曲线，无分通道曲线。
**缓解**: 使用合并曲线作为所有通道的基线 + 从胶片文献手动调整通道差异。

### 2.3 颗粒参数

| 需求 | 当前状态 | 行动 |
|------|---------|------|
| 每种胶片的 RMS granularity | Kodak/Ilford 技术文档中有公布 | 从 PDF 提取 |
| RMS granularity → filmgrainer 参数映射 | 无现成映射 | 手动校准：生成不同参数下的颗粒图像，对比真实胶片扫描 |

**风险**: filmgrainer 的物理精度有限（非 Newson Boolean 模型级别）。
**缓解**: 先用 filmgrainer 快速验证管线，如需更高精度，自实现 Newson Pixel-wise 算法（3-5 天）。

### 2.4 客观评估标准

| 需求 | 当前状态 | 行动 |
|------|---------|------|
| 颜色精度量化 | 无 ground truth 配对 | 对比目标胶片域 histograms, ΔE2000 on ColorChecker |
| 颗粒逼真度 | 无 ground truth 颗粒 | 主观评估为主，NPS (Noise Power Spectrum) 对比为辅 |
| 光晕自然度 | 无 ground truth 光晕 | 主观评估，与真实胶片高光区域对比 |

### 2.5 GPU 兼容性风险

| 需求 | 风险 | 缓解 |
|------|------|------|
| filmgrainer on RTX 5070 Ti | 未知，需测试 | 先测试，如不行回退 CPU 或自实现 |
| SilverGrain | AGPL-3.0 + sm_120 不兼容 | **不使用** |

### 2.6 CUT PyTorch 2.x 兼容

| 需求 | 风险 | 缓解 |
|------|------|------|
| CUT 代码在 PyTorch 2.11 上运行 | 官方代码基于 PyTorch 1.1 | 手动修复 5-10 行 API 变更 (torch.tensor, nn.Module.module 等) |

---

## 3. 不再需要的资源

以下原计划所需资源不再需要：

| 资源 | 原因 |
|------|------|
| MambaVision 预训练权重 (HuggingFace) | Mamba 骨干已放弃 |
| KAN 相关依赖 (pykan, scikit-learn) | KAN 物理层已放弃 |
| SD VAE 预训练权重 | VAE 潜空间已放弃 |
| CFM 参考实现 (torchcfm) | CFM 方案已放弃 |
| mamba-ssm causal-conv1d | 不再需要 |
| Mitsuba 3 PBR 合成管线 | 不再需要（CUT 无配对，不需要合成配对数据） |
| 光谱灵敏度数据 | 光谱交叉模块已延期 |
| CIE 色度数据 | 链路简化，不再需要 |

---

## 4. 风险矩阵

| 风险 | 概率 | 影响 | 缓解策略 |
|------|:---:|:---:|------|
| 胶片域图片质量不足 | 中 | 高 | FilmSet + web 补充 + 最终可能需要用户自拍 |
| filmgrainer 在 RTX 5070 Ti 上不兼容 | 低 | 中 | CPU 回退 + 自实现 Newson 算法 |
| CUT 训练出现 mode collapse | 中 | 高 | 先试 3D LUT 方案 (有预训练权重) |
| H&D 曲线数字化精度不足 | 低 | 中 | 多数据点 + 插值平滑 + MLP 拟合 |
| CUT 颜色转移产生伪影 | 中 | 中 | Identity loss + lighter texture matching (减少 PatchNCE layers) |

---

## 5. 下一步行动

按优先级排序：

1. **P0**: 重新下载物理 PDF (`scripts/download_data.py physics`)
2. **P0**: 安装 filmgrainer，在 RTX 5070 Ti 上测试兼容性
3. **P0**: 数字化 2 个胶片的 H&D 曲线（Vision3 500T + Portra 400）
4. **P0**: 实现光晕 + 颗粒模块
5. **P0**: 跑通手动管线 CLI
6. **P1**: 收集胶片域图片数据
7. **P1**: CUT 颜色迁移训练

---

*修订版本: v2.0 | 2026-05-23 | 基于 ARCH_REDESIGN.md*
