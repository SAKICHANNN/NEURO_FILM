# Active stock-first pointer — 2026-07-15

The executable authority is `docs/ULTIMATE_EXECUTION_TRACKER.md`, governed by
`docs/planning/REAL_FILM_ULTIMATE_REOPEN_2026.md` and
`docs/data/REAL_FILM_STOCK_EVIDENCE_REGISTRY.md`.

Ultimate learns multiple specific, evidence-backed `film_stock_id` experts
from real film scans. Roll/process/scanner/source/content are nested controls.
`historical-film/unknown-stock` is a valid auxiliary expert and stress lane but
never substitutes for, or counts toward, named-stock coverage. FilmSet,
Capture One recipes, camera Film Simulations, LUTs and the diffusion-first V3
content below are historical/control material, not the active success path.

`RF0.4/SF0.1`, `SF0.2` and `SF0.3` are complete: 189 BlueNeg objects /
227,287,697 bytes are hash-verified and decode-clean under
`data/raw/blueneg_stock_pilots_v1`. Gold 100-5 is the only display-operator
research candidate; NPH400/Konica/GA remain post-negation-preview label-audit candidates, not density or colour-operator candidates. The
current leaf is `RF1.4` per-stock identifiability. LOC Phase C remains a sealed
historical/unknown auxiliary lane. No colour expert fits before RF1.4 passes.

---

# K-MCFM 实施计划（历史 V3 + 2026-07-11 现行指针）

> **现行计划已迁移。** 后续执行以 `docs/ULTIMATE_EXECUTION_TRACKER.md` 为 active authority，以 `docs/planning/ULTIMATE_ROADMAP_2026.md` 为研究和架构依据。
> 下方 V3 diffusion-first 计划保留为历史记录；SD1.5 IP2P、SDXL full-UNet IP2P 与 SDXL LoRA/SDEdit 已被后续本地实验否决为默认生产路径，不得按下方 Phase 1–3 直接重启。

## 2026-07-11 当前关键路径

```text
U0 truth/rights/repro reset
  -> U1 WorkingImage + high-precision color I/O
  -> U2 deterministic profile/reference renderer
  -> U4 autonomous severe-artifact/style evaluator
  -> U5 FilmCase identifiability -> Oracle -> simplest retrieval/ranker
  -> U6 artifact-safe effects
  -> U7 productization -> U8 release/stock expansion

U3 paired calibration is a deferred optional side lane, not a dependency.
```

`U0.1` 文档真相对齐和 `U5.FC0` 自主无配对 FilmCase 计划已完成。最近可执行叶任务：`U0.3` 数据谱系/分组/泄漏与 FilmCase eligibility 审计、`U0.4/U4` CI 与自主评测、随后 `U1.1` 把 `WorkingImage` 接入主 renderer。详细科研 DAG 见 `docs/planning/FILMCASE_AUTONOMOUS_RESEARCH_PLAN.md`。用户不会提供新图、配对或标签；这些不是依赖。许可证、大下载、付费 GPU、外部联络、push/release 仍保留人工批准门。

---

## 以下为历史 V3 计划（不再是 active authority）

> SDEdit + 胶片 LoRA + IP-Adapter 内容保真方案
> **工期**: 5-6 周 | **VRAM**: ≤12 GB | **平台**: CUDA + MLX/MPS

---

## 核心原理

```
输入图 → VAE编码 → 加噪(控量) → 带胶片LoRA降噪 → VAE解码 → 胶片输出
                ↑ t=0.4-0.5 控内容保真
                ↑ LoRA + Prompt 控胶片风格
```

**strength 参数 = 内容保真度控制器**：
- 0.3 = 几乎不动，轻胶色调
- 0.45 = 内容保留 + 明显胶片感 ← 推荐
- 0.6 = 强胶片，小细节可能变
- 1.0 = 纯文生图，内容全丢

---

## Phase 1: SDXL 基线 + 社区 LoRA（1 周）

### 1.1 环境搭建（0.5 天）

```bash
pip install diffusers>=0.38.0 transformers accelerate safetensors
pip install peft  # LoRA 加载

# 验证
python -c "from diffusers import StableDiffusionXLImg2ImgPipeline; print('OK')"
```

### 1.2 加载 SDXL + img2img pipeline（0.5 天）

```python
from diffusers import StableDiffusionXLImg2ImgPipeline
import torch

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

pipe.enable_model_cpu_offload()  # 节省 VRAM，对推理速度影响小
```

### 1.3 加载社区胶片 LoRA（0.5 天）

下载 Civitai 上的 SDXL 胶片 LoRA（safetensors 格式）：

| 胶片 | Civitai 链接 |
|------|-------------|
| Kodak Portra 400 | `https://civitai.com/models/723250` (SDXL version `808680`) |
| Kodak Vision3 500T | `https://civitai.com/models/725625` (SDXL version `820808`) |
| Kodak Vision3 250D | `https://civitai.com/models/725620` (SDXL version `820761`) |
| Kodak Ektar 100 | `https://civitai.com/models/779013` (SDXL version `1167852`) |

2026-05-25 未验证到精确 SDXL 条目：Kodak Portra 800、Kodak Tri-X 400、Fujifilm Velvia 50、Ilford HP5 Plus。它们走自训练 LoRA，避免使用泛胶片或错误模型。

```python
pipe.load_lora_weights("path/to/lora/portra_400.safetensors")
```

### 1.4 调优 strength 参数（1 天）

对每种胶片，grid search strength ∈ [0.3, 0.4, 0.45, 0.5, 0.6, 0.7]：

```python
def film_translate(image, lora_path, style_prompt, strength=0.45):
    pipe.load_lora_weights(lora_path)
    result = pipe(
        prompt=style_prompt,
        image=image,
        strength=strength,
        guidance_scale=7.5,
        num_inference_steps=30,
    ).images[0]
    return result

# Portra 400
result = film_translate(
    input_image,
    "loras/portra_400.safetensors",
    "a cinematic photograph shot on Kodak Portra 400, natural skin tones, fine grain, film photography",
    strength=0.45
)
```

评测标准：内容保真（CLIP image similarity）+ 胶片感（主观 A/B）+ 色差（ΔE to 参考胶片域）

### 1.5 CLI 实现（1 天）

```bash
python scripts/pipeline.py photo.jpg \
    --style portra_400 \
    --strength 0.45 \
    --steps 30 \
    --output result.jpg

# 批量
python scripts/pipeline.py ./photos/ --style vision3_500t --strength 0.4
```

**文件**: `scripts/pipeline.py`

### 1.6 预期产出

- [ ] img2img pipeline 运行在 12GB 5070 Ti 上
- [ ] 4 种已验证 SDXL 社区 LoRA 可以直接翻译（Portra 400, Vision3 500T, Vision3 250D, Ektar 100）
- [ ] strength 最佳值已针对每种胶片调优
- [ ] CLI 可单张/批量处理

---

## Phase 2: 自训练胶片 LoRA（2 周）

### 2.1 数据收集（2 天）

**数据源**：
1. **Flickr 胶片扫描**: git 历史记录 8 类共 3,896 张；当前 checkout 未包含，需用 API 重新抓取或从已授权机器同步
2. **FilmSet**: 3 风格 × 5285 张，已有
3. **Flickr / r/analog**: 爬取指定胶片标签的高质量扫描
4. **Civitai 数据集**: 部分 LoRA 作者公开训练数据

**目标**：每胶片 ≥200 张高质量图片

**目录结构**：
```
data/film_domain/
    kodak_portra_400/
        img_001.jpg   # 1024² 或更大
        img_002.jpg
        ...
    kodak_vision3_500t/
    fuji_velvia_50/
    ilford_hp5/
    ...
```

**预处理**：
- 去重（phash/CLIP embedding similarity）
- 缩放到 1024²（SDXL 标准分辨率）
- 裁剪到合理构图
- 标注为 `<film_name> photograph, film grain, analog photography`

### 2.2 LoRA 训练（5 天）

使用 **diffusers 官方训练脚本** 或 **kohya-ss sd-scripts**：

```bash
accelerate launch train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --dataset_name="./data/film_domain/kodak_portra_400" \
    --caption_column="text" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=2000 \
    --learning_rate=1e-4 \
    --rank=16 \
    --lr_scheduler="cosine" \
    --output_dir="./loras/portra_400"
```

**训练参数**：
```yaml
model: SDXL
resolution: 1024
batch_size: 1
gradient_accumulation: 4
lr: 1e-4
max_steps: 2000
rank: 16
alpha: 16
mixed_precision: fp16
gradient_checkpointing: true
```

**VRAM 预算**：512² ≈ 6GB, 768² ≈ 9GB, 1024² ≈ 12GB（需 gradient checkpointing）

### 2.3 验收测试（1 天）

对每个自训练 LoRA：
- 与社区 LoRA（如果有）做 A/B 对比
- 在相同 strength 下评估内容保真
- 在 10 张测试图上做主观评分
- 发布最佳 checkpoint

### 2.4 Portra 800 + Tri-X 400 + Velvia 50 + HP5（2 天）

这四种胶片截至 2026-05-25 未验证到精确 SDXL Civitai LoRA，需从零训练：
- **Portra 800（高感彩负）**：可用 Flickr `kodak portra 800` / `portra800` 数据，注意弱光题材偏多
- **Tri-X 400（黑白负片）**：需黑白扫描数据，禁用颜色提示词
- **Velvia 50（正片）**：高饱和、高对比、极细颗粒 → 需要特定数据策略
- **HP5 Plus（黑白）**：需收集黑白底片扫描，禁用颜色提示词

**文件**: `scripts/train_lora.py`

---

## Phase 3: 增强管线 + 多平台（2 周）

### 3.1 IP-Adapter 内容锚定（2 天）

用 h94/IP-Adapter 的 SDXL 版本（22M params, ±1GB VRAM）：

```python
from diffusers import StableDiffusionXLImg2ImgPipeline

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
)

# 用输入图做 IP-Adapter 的 reference image
result = pipe(
    prompt="cinematic Kodak Portra 400 film photo, fine grain",
    ip_adapter_image=input_image,
    image=input_image,
    strength=0.45,
    guidance_scale=7.5,
    ip_adapter_scale=0.5,   # IP-Adapter 强度
).images[0]
```

**IP-Adapter 参数调优**：
- `ip_adapter_scale=0.3` → 弱内容引导，更多风格自由
- `ip_adapter_scale=0.5` → 均衡 ← 推荐
- `ip_adapter_scale=0.8` → 强内容锁，风格被压制

**文件**: `src/models/diffusion/ip_adapter_wrapper.py`

### 3.2 可选：ControlNet 结构锁（1 天，如果需要）

对结构敏感的图片（建筑、文字），追加 ControlNet-depth：

```python
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

depth_map = depth_estimator(input_image)
result = pipe(
    prompt="...",
    image=input_image,
    control_image=depth_map,
    strength=0.45,
    controlnet_conditioning_scale=0.7,
)
```

**注意**：ControlNet 增加 ~1.5GB VRAM。对一般照片非必需，但对建筑/文字场景可大幅提升保真。

### 3.3 后处理：颗粒 + 光晕（1 天，可选）

如果扩散模型输出的颗粒不够真实，追加：

```python
def post_process(image, style):
    # 可选颗粒
    if style.grain_enabled:
        image = apply_filmgrainer(image, style.grain_params)
    # 可选光晕
    if style.halation_enabled:
        image = apply_halation(image, style.halation_params)
    return image
```

### 3.4 Mac MLX 适配（2 天）

**SDXL on MLX**：
```python
import mlx.core as mx
from mlx_vlm import generate  # 或 mlx 的 diffusers port

# Apple 的 mlx-examples 有 SDXL 实现
# github.com/ml-explore/mlx-examples/tree/main/stable_diffusion
```

**CoreML 转换**（更快但需要转换 SDXL 为 CoreML）：
```bash
python -m python_coreml_stable_diffusion.torch2coreml \
    --model-version sdxl-base-1.0 \
    --convert-unet --convert-vae-decoder --convert-text-encoder \
    --bundle-resources-for-swift-cli \
    --attention-implementation SPLIT_EINSUM_V2
```

**Mac 推理速度预估**：
| 设备 | SDXL 1024² (30 steps) |
|------|:---:|
| M1 Max 64GB (MLX) | 15-20s |
| M1 Max 64GB (CoreML) | 10-15s |
| M5 32GB (MLX) | 8-12s |
| M5 32GB (CoreML) | 5-8s |

### 3.5 全管线 CLI（1 天）

```bash
python scripts/pipeline.py input.jpg \
    --style portra_400 \
    --strength 0.45 \
    --steps 30 \
    --ip-adapter 0.5 \
    --grain \
    --halation \
    --device cuda \    # cuda | mps | mlx
    --output result.jpg
```

**文件**: `scripts/pipeline.py`

### 3.6 评估 + 文档（1 天）

- 每种胶片 20 张测试图
- 定量：CLIP image similarity（内容保真）、FID to film domain（风格相似）
- 主观：3 人 A/B 对比 vs 真实胶片扫描 vs 数字原图

---

## 依赖清单

```text
# 核心（新增）
diffusers>=0.38.0
transformers>=5.9.0
accelerate>=1.13.0
safetensors>=0.8.0rc0
peft>=0.19.1

# 可选
controlnet_aux          # ControlNet 预处理器
git+https://github.com/larspontoppidan/filmgrainer  # 后处理颗粒 (MIT; 未发布 PyPI)
mlx                     # Mac Apple Silicon
python_coreml_stable_diffusion  # CoreML 转换
```

---

## VRAM 预算总表

| 阶段 | 配置 | 分辨率 | VRAM |
|------|------|:---:|:---:|
| 推理 | SDXL + LoRA | 1024² | 8 GB |
| 推理 | SDXL + LoRA + IP-Adapter | 1024² | 8.5 GB |
| 推理 | SDXL + LoRA + ControlNet | 1024² | 10 GB |
| 推理 | SD 3.5 Medium + LoRA | 1024² | 7 GB |
| 推理 | Flux.1 Schnell GGUF Q4 | 1024² | 12 GB |
| 训练 | SDXL LoRA | 512² | 6 GB |
| 训练 | SDXL LoRA | 768² | 9 GB |
| 训练 | SDXL LoRA | 1024² | 12 GB |

---

## 对比：V2 vs V3

| 维度 | V2 (CUT + 3D LUT) | V3 (SDEdit + LoRA) |
|------|:---:|:---:|
| 颜色质量 | 统计匹配 | **AI 深度理解** |
| 纹理质量 | 噪声叠加 | **扩散模型生成** |
| 内容保真 | GAN 不稳定 | **噪声级别精确控制** |
| 研发周期 | 4-6 周 | 5-6 周 |
| VRAM | 3-5 GB | 7-10 GB |
| 社区生态 | 无 | **30+ 现成 LoRA** |
| Mac 支持 | 弱 | **MLX/CoreML** |
| 成功率 | 60-70% | **75-85%** |

---

*修订版本: V3.0 | 2026-05-23 | Diffusion-Based Film Translation*
