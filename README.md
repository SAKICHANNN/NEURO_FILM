# K-MCFM: Diffusion-Based Film Translation

用扩散模型（SDXL/SD3.5/Flux）将数码照片转化为内容保真的胶片摄影作品。

**核心方法**: SDEdit (noise-denoise) + 胶片 LoRA + IP-Adapter 内容锚定

**目标硬件**: RTX 5070 Ti 12GB (CUDA) / M5 32GB (MLX) / M1 Max 64GB (MLX/MPS)
**输入**: 任意数字照片 (JPEG/PNG/TIFF)
**输出**: 内容保真的胶片风格图像

---

## 架构

```
Input Image → VAE Encode → Forward Diffuse (控量噪声)
    → Denoise with Film LoRA + Prompt + IP-Adapter
    → VAE Decode → Optional: Grain + Halation → Film Output
```

### 为什么这样做

扩散模型（SD/Flux）在数十亿张图片上预训练，已学会"照片应该长什么样"。LoRA 在预训练模型上注入胶片美学，只需 2MB。通过噪声级别（strength）精确控制"改多少"——不是 GAN 的黑盒映射，不是 LUT 的全局调色。

社区有大量胶片 LoRA 可用，但 SDXL 精确胶片库存需要逐个核实；本仓库已在 `docs/ONLINE_DATA_AUDIT.md` 记录 2026-05-25 的核实结果。

---

## 快速启动

```bash
# 安装依赖
pip install diffusers>=0.38.0 transformers accelerate peft safetensors

# 单张推理
python scripts/pipeline.py input.jpg \
    --style portra_400 \
    --strength 0.45 \
    --output result.jpg

# 批量处理
python scripts/pipeline.py ./photos/ \
    --style vision3_500t \
    --strength 0.4 \
    --output_dir ./outputs/
```

---

## 实施阶段

| 阶段 | 内容 | 工期 |
|------|------|:---:|
| Phase 1 | SDXL 基线 + 社区 LoRA 推理 | 1 周 |
| Phase 2 | 自训练胶片 LoRA | 2 周 |
| Phase 3 | IP-Adapter + 后处理 + Mac 适配 | 2 周 |

---

## 胶片清单

| 胶片 | 类型 | LoRA |
|------|------|:---:|
| Kodak Vision3 500T | 彩色负片 | Civitai SDXL verified |
| Kodak Vision3 250D | 彩色负片 | Civitai SDXL verified |
| Kodak Portra 400 | 彩色负片 | Civitai SDXL verified |
| Kodak Portra 800 | 彩色负片 | 自训练 |
| Kodak Ektar 100 | 彩色负片 | Civitai SDXL verified |
| Fujifilm Velvia 50 | 彩色正片 | 自训练 |
| Ilford HP5 Plus | 黑白负片 | 自训练 |
| Kodak Tri-X 400 | 黑白负片 | 自训练 |

---

## VRAM 需求

| 配置 | 推理 1024² | 训练 512² |
|------|:---:|:---:|
| SDXL + LoRA | 8 GB | 8-10 GB |
| SDXL + LoRA + IP-Adapter | 8.5 GB | — |
| SD 3.5 Medium | 6 GB | 7-9 GB |
| Flux.1 Schnell GGUF Q4 | 12 GB | — |

---

## 许可

MIT License. 详见 `LICENSE`。

---

*最后更新: 2026-05-23 | V3 — Diffusion-Based Film Translation*
