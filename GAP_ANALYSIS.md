# K-MCFM 项目缺口分析（V3 — Diffusion-Based Edition）

> 基于 V3 扩散模型方案（SDEdit + LoRA + IP-Adapter）重新评估。

---

## 1. 已解决的缺口（V1→V3）

| V1 缺口 | V3 方案 | 状态 |
|---------|---------|:---:|
| 无配对数据 | 社区 LoRA 已有 30+ 胶片风格，自训练只需目标域图片 | ✅ |
| 内容保真 | SDEdit 噪声级别控制 + IP-Adapter 内容锚定 | ✅ |
| VRAM 限制 | SDXL 8GB@1024² 推理，LoRA 训练 8-10GB@512² | ✅ |
| Mac 兼容 | MLX/MPS/CoreML 三选一 | ✅ |
| 研发周期 | 5-6 周（vs V1 的 5.5 月）| ✅ |

---

## 2. 当前缺口

### 2.1 高质量胶片域图片（LoRA 训练用）

| 胶片 | 状态 | 行动 |
|------|------|------|
| Portra 400, Vision3 500T, Ektar 100, Tri-X | Civitai 已有 LoRA | 下载 + 验证 |
| Vision3 250D, Portra 800 | Civitai 可能有 | 搜索 + 验证 |
| Velvia 50, HP5 | 无现成 | **需自训练**（每胶片需 ≥200 张） |

**数据来源优先级**：
1. FilmSet（3 风格 × 5285 张，已有，但非真实扫描）
2. Flickr 标签搜索 `kodak portra 400`, `shot on portra` 等
3. r/analog 子版（Reddit 社区，高质量）
4. 500px / Lomography 标签搜索

**风险**：Flickr/Reddit 图片可能有压缩、水印、过度后期。需人工筛选。

### 2.2 LoRA 训练环境

| 需求 | 状态 | 行动 |
|------|------|------|
| kohya-ss sd-scripts | 需安装 | `git clone` + 配置 |
| diffusers 训练脚本 | 需安装 | pip install diffusers[training] |
| 12GB 训练 1024² | 可能 OOM（需 gradient checkpointing + 8-bit Adam） | 降级到 768² 或减少 batch |

### 2.3 IP-Adapter 兼容性

| 需求 | 状态 | 行动 |
|------|------|------|
| SDXL IP-Adapter Plus | h94/IP-Adapter 提供 | 下载 safetensors 权重 |
| 与胶片 LoRA 共存 | 理论上兼容（LoRA 改 UNet，IP-Adapter 加 cross-attn） | 需要实测验证 |

### 2.4 Civitai LoRA 许可

| 风险 | 说明 |
|------|------|
| 商用限制 | 部分 LoRA 标注"不能商用"。学术/个人 OK，发布前需检查。 |
| 下载失效 | Civitai 链接可能失效。建议备份本地。 |

---

## 3. VRAM OOM 降级策略

| 优先级 | 操作 | 省 VRAM |
|:---:|------|:---:|
| 1 | 降到 768² 推理/训练 | ~2-3 GB |
| 2 | 换 SD 3.5 Medium（2.5B vs SDXL 2.6B） | ~1.5 GB |
| 3 | 关掉 IP-Adapter | ~1 GB |
| 4 | 关掉 ControlNet | ~1.5 GB |
| 5 | FP16→GGUF Q8（仅 Flux） | ~4 GB |

---

## 4. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|:---:|:---:|------|
| 社区 LoRA 质量差 | 中 | 中 | 自训练替代 |
| LoRA + IP-Adapter 冲突 | 低 | 中 | 独立测试，降低 ip_adapter_scale |
| Velvia 50 正片数据不足 | 中 | 中 | 从 Flickr/500px 正片标签收集 |
| strength 无法同时满足内容+风格 | 低 | 高 | IP-Adapter 做内容补偿 |
| Mac 推理太慢（>30s） | 中 | 低 | CoreML 转换提速 2-3× |
| Civitai LoRA 链接失效 | 低 | 低 | 本地备份所有 LoRA 文件 |

---

## 5. 不再需要的资源

| V1/V2 需求 | 原因 |
|-----------|------|
| MambaVision 权重 | 架构已放弃 |
| pykan / KAN | 架构已放弃 |
| causal-conv1d / mamba-ssm | 不再使用 |
| SD VAE 预训练（VAE 训练用） | diffusers 内置 SDXL VAE |
| CFM torchcfm | 不再使用 |
| Mitsuba 3 PBR | 不再需要合成配对数据 |
| 3D LUT 训练 | 扩散模型替代 |
| CUT 训练 | 扩散模型替代 |

---

*修订版本: V3.0 | 2026-05-23 | Diffusion-Based Edition*
