# AGENTS.md — K-MCFM Project Knowledge Base (V3)

> **架构 V3 (2026-05-23)**。采用扩散模型 SDEdit + LoRA + IP-Adapter 做内容保真的胶片翻译。
> V2 的 CUT/LUT 方案已升级为深度扩散模型方案。

---

## 1. 项目身份卡片

| 字段 | 值 |
|------|-----|
| **项目名** | K-MCFM: Film Translation via InstructPix2Pix |
| **一句话** | 用 InstructPix2Pix 指令编辑模型做内容保真的胶片色彩转换 |
| **核心方法** | SDXL SDEdit/img2img + 胶片 LoRA；IP2P (ig=1.5, tg=7.5) 保留为实测最佳 fallback |
| **目标硬件** | M5 32GB (MPS推理) / RTX 5070 Ti 12GB (CUDA训练+推理) |
| **当前阶段** | 9 轮实验完成；V3 脚本已入库，社区 SDXL LoRA/数据缺口已在 2026-05-25 复核 |
| **许可** | MIT |

---

## 2. 核心文档地图

| 文件 | 内容 | 使用场景 |
|------|------|---------|
| `AGENTS.md` | 本文件 | 每次新会话先读 |
| `docs/EXPERIMENT_LOG.md` | 9 轮完整实验日志 | 理解项目历史 |
| `docs/ARCH_REDESIGN.md` | 架构演进：V1(CFM+Mamba) → V2(CUT+LUT) → V3(SDEdit+LoRA) | 理解方向 |
| `docs/ONLINE_DATA_AUDIT.md` | 2026-05-25 联网核实的 LoRA/依赖/数据缺口 | 查最新补齐记录 |
| `docs/PROJECT_STRUCTURE.md` | 仓库目录结构和安全整理规则 | 移动/整理文件前查阅 |
| `IMPL_PLAN.md` | 当前实施计划 | 编码前查阅 |
| `TASK_BOARD.md` | 任务分配 | 第二个读 |
| `README.md` | 面向人类 | 对外 |

---

## 3. 技术架构

```
Input Image (H, W, 3)
    │
    ▼
[Preprocess] → 1024², normalize
    │
    ▼
[VAE Encode] → z_0 (4, H/8, W/8)
    │
    ▼
[Forward Diffuse] → z_t = noise(z_0, t_start=0.4-0.5)
    │
    ▼
[Denoise with Conditioning]:
  ├─ Film LoRA            ← 每胶片一个，2-5MB
  ├─ Text Prompt          ← "cinematic Kodak Portra 400 photo, film grain"
  ├─ IP-Adapter (可选)     ← 输入图为内容锚点
  └─ ControlNet (可选)    ← 深度/边缘结构锁
    │
    ▼
[VAE Decode] → (H, W, 3)
    │
    ▼
[Optional: Grain + Halation]
    │
    ▼
Output
```

### 关键原理

**SDEdit 内容保真机制**：
- `strength` 参数控制加噪量 = 控制内容保留度
- strength=0.0: 完全不变
- strength=0.45: 内容基本保留，色调胶片化 ← 推荐
- strength=0.6: 明显风格，小细节可能变
- strength=1.0: 纯文生图，内容全丢失

**为什么扩散模型比 CUT/LUT 好**：
- 扩散模型在数十亿张图片上预训练，已学会"照片应该长什么样"
- LoRA 在预训练模型上注入胶片美学，只需 2MB
- 通过噪声级别精确控制"改多少"，而非 GAN 的黑盒映射
- 社区已有 30+ 胶片 LoRA 可直接使用

---

## 4. VRAM 预算

| 配置 | 推理 (1024²) | LoRA 训练 (512²) |
|------|:---:|:---:|
| SDXL FP16 | 7.5 GB | — |
| SDXL + LoRA + IP-Adapter | 8.5 GB | — |
| SDXL LoRA 训练 | — | 8-10 GB |
| SD 3.5 Medium FP16 | 6 GB | 7-9 GB |
| Flux.1 Schnell GGUF Q4 | 12 GB | >12 GB (云) |
| **可用余量 (12GB)** | **3.5-6 GB** | **2-4 GB** |

### 如果 OOM
1. 降到 768² 推理（~5 GB）
2. 用 SD 3.5 Medium 代替 SDXL（~6 GB）
3. 关掉 ControlNet（省 1.5 GB）
4. 关掉 IP-Adapter（省 1 GB）

---

## 5. 胶片清单

| 胶片 | LoRA 状态 | 优先级 |
|------|:---:|:---:|
| Kodak Portra 400 | Civitai SDXL 已验证：model `723250`, version `808680` | P0 |
| Kodak Vision3 500T | Civitai SDXL 已验证：model `725625`, version `820808` | P0 |
| Kodak Vision3 250D | Civitai SDXL 已验证：model `725620`, version `820761` | P0 |
| Kodak Ektar 100 | Civitai SDXL 已验证：model `779013`, version `1167852` | P1 |
| Kodak Portra 800 | 未验证到精确 SDXL LoRA；需自训练或重新搜索 | P1 |
| Fujifilm Velvia 50 | 未验证到精确 SDXL LoRA；需自训练 | P1 |
| Ilford HP5 Plus | 仅找到 SD1.5 泛 Ilford LoRA；SDXL 需自训练 | P2 |
| Kodak Tri-X 400 | 未验证到精确 SDXL LoRA；旧 id `521049` 为无关模型 | P2 |

---

## 6. 实施计划

### Phase 1: SDXL 基线 + 社区 LoRA（1 周）

| 任务 | 内容 |
|------|------|
| 1.1 | 安装 diffusers + SDXL，搭建 img2img pipeline |
| 1.2 | 从 Civitai 下载胶片 LoRA (Portra 400, Vision3 500T) |
| 1.3 | 调优 strength 参数，找到内容/风格最佳平衡 |
| 1.4 | 实现 CLI: `python scripts/pipeline.py img.jpg --style portra_400` |

### Phase 2: 自训练胶片 LoRA（2 周）

| 任务 | 内容 |
|------|------|
| 2.1 | 收集每胶片 200-500 张高质量扫描图 |
| 2.2 | 用 kohya-ss/diffusers 训练 SDXL LoRA |
| 2.3 | 验证训练效果，调优 rank/alpha |
| 2.4 | 训练 Velvia 50, HP5 (社区无现成) |

### Phase 3: 增强 + 管线 + 多平台（2 周）

| 任务 | 内容 |
|------|------|
| 3.1 | IP-Adapter 集成 (h94/IP-Adapter SDXL) |
| 3.2 | 后处理颗粒+光晕 (可选) |
| 3.3 | Mac MLX 适配 |
| 3.4 | 全管线 CLI + 批量处理 |
| 3.5 | 主观 + 定量评估 |

---

## 7. 关键依赖

| 包 | 用途 | 版本 |
|----|------|------|
| diffusers | SDXL/SD3.5 pipeline | 项目锁定/PyPI 0.38.0 |
| torch | 核心框架 | 项目锁定 2.11.0(+cu128)；PyPI 2.12.0 |
| safetensors | LoRA 权重加载 | 项目锁定 0.8.0rc0 |
| accelerate | 推理加速 | 项目锁定/PyPI 1.13.0 |
| transformers | CLIP 文本编码器 | PyPI 5.9.0 |
| peft | LoRA adapter 支持 | PyPI 0.19.1 |
| filmgrainer | 可选后处理颗粒 | GitHub MIT；未发布 PyPI |
| mlx (Mac) | Apple Silicon 加速 | 需按 Mac 环境单独验证 |

关键依赖下限（requirements 已锁定具体版本）：
```
diffusers>=0.38.0
peft>=0.19.1
safetensors>=0.8.0rc0
kornia                  # 已有
```

---

## 8. 已知陷阱

| 陷阱 | 说明 |
|------|------|
| **LoRA 与 base model 不匹配** | SD 1.5 LoRA 不能用于 SDXL。必须同架构。 |
| **strength 过大内容崩塌** | >0.7 时人脸/文字会变形。从 0.4 开始，逐步增加。 |
| **IP-Adapter 与提示词冲突** | 设置 `ip_adapter_scale=0.3-0.6`，过高会让提示词失效。 |
| **Mac 上第一次推理极慢** | MPS/MLX 首次加载需编译内核。加载后复用 pipeline。 |
| **Flux GGUF LoRA 不成熟** | Flux LoRA 生态 2026 年仍不如 SDXL。优先用 SDXL。 |
| **Civitai LoRA 许可** | 部分 LoRA 标注"不能商用"。学术研究 OK，发布前检查。 |

---

## 9. 更新协议

Agent 应在以下情况更新本文件：
1. 实现状态变更（更新 §6）
2. 新增胶片 LoRA（更新 §5）
3. 发现新陷阱（追加 §8）
4. 架构变更（更新 §3）

---

*最后更新: 2026-05-23 | V3: Diffusion-Based Film Translation*
