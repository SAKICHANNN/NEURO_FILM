# K-MCFM: Modular Film Simulation Pipeline

模块化数字到胶片图像转换管线 — 像素空间直接操作，4 模块解耦。

**核心方法**: 颜色风格迁移 (CUT/CycleGAN or 3D LUT) → H&D 色调映射 → 光晕 → 颗粒

**目标硬件**: NVIDIA RTX 5070 Ti Laptop GPU 12GB VRAM
**输入**: 数字 RAW / 16-bit TIFF
**输出**: 胶片模拟图像（Kodak Vision3 500T/250D、Portra 400/800、Ektar 100、Fujifilm Velvia 50、Ilford HP5、Kodak Tri-X）

---

## 架构

```
Digital Input (3, H, W) linear RGB
    │
    ├─ [1] Color Style Transfer   ← CUT/CycleGAN 或 3D LUT Predictor
    ├─ [2] H&D Tone Mapping        ← 每通道 1D LUT (从 PDF 数字化)
    ├─ [3] Halation                ← 多通道 Gaussian scatter
    └─ [4] Film Grain              ← filmgrainer (MIT) 或 Newson 模型
    │
Film Output (3, H, W)
```

### 关键设计原则

- **像素空间操作**: 不使用 VAE 潜空间，更简单、可解释
- **模块解耦**: 每模块独立开发、测试、验证
- **无配对数据**: 颜色模块用 CUT（无配对 GAN），物理模块用解析函数
- **MIT 许可兼容**: 不使用 AGPL 库

---

## 实施计划（4-6 周）

| 阶段 | 内容 | 工期 |
|------|------|:---:|
| Phase 1 | 手动基线管线 (H&D + 光晕 + 颗粒 + CLI) | 1 周 |
| Phase 2 | 颜色风格转移训练 (CUT 或 3D LUT) | 2-3 周 |
| Phase 3 | 全管线集成 + 参数调优 + 批量推理 | 1-2 周 |

详细计划见 `IMPL_PLAN.md`，架构分析见 `docs/ARCH_REDESIGN.md`。

---

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt

# 运行手动管线 (Phase 1)
python scripts/pipeline.py input.tiff --style kodak_portra_400

# 训练颜色迁移 (Phase 2)
python scripts/train_cut.py \
    --dataroot ./data \
    --name portra400_cut \
    --model cut \
    --crop_size 256

# 全管线推理 (Phase 3)
python scripts/pipeline_full.py input.tiff --style kodak_portra_400
```

---

## 胶片清单

| 胶片 | 类型 | ISO |
|------|------|:---:|
| Kodak Vision3 500T | 彩色负片 (电影) | 500 |
| Kodak Vision3 250D | 彩色负片 (电影) | 250 |
| Kodak Portra 400 | 彩色负片 (静态) | 400 |
| Kodak Portra 800 | 彩色负片 (静态) | 800 |
| Kodak Ektar 100 | 彩色负片 (静态) | 100 |
| Fujifilm Velvia 50 | 彩色正片 | 50 |
| Ilford HP5 Plus | 黑白负片 | 400 |
| Kodak Tri-X 400 | 黑白负片 | 400 |

---

## 许可

MIT License. 详见 `LICENSE`。

---

*最后更新: 2026-05-23 | 架构重设计 v2.0*
