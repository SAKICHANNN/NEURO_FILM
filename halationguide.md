可以，这个版本要更“工程化/科研化”地说。你的限制是**没有逐场景 film/digital paired data**，所以最强方案应该是：

> **物理先验主导 + 公开胶片资料标定 + 无配对真实胶片 patch 统计校准 + 可调强度的 halation layer renderer。**
> 不要让 AI 直接改整张图；AI 最多只预测 halation 参数或 residual layer。

---

# 1. 真实胶片 halation 的核心物理

真正的红色 halation 不是简单红色 glow。Kodak 对 anti-halation backing 的定义是：胶片中有一层暗色 coating，用来吸收本来会从片基反射回乳剂的光。也就是说，halation 本质上来自强光穿过乳剂后，在 film base / 背面结构 / camera pressure plate 相关位置反射或散射，再回到乳剂层形成二次曝光。([Kodak][1])

彩色胶片里，红色 halation 特别明显，是因为反射回来的光主要重新曝光更靠近片基侧的 red-sensitive layer；当反射/散射光特别强时，也可能影响 green layer，于是红 halo 中心会变成橙黄。Dehancer 的技术文章也这样解释：蓝、绿高频成分被前面的层过滤，返回光主要 backlight 红层，极强时也进入绿层，所以出现红色或橙色 halo。([Dehancer Blog][2])

---

# 2. 强度和半径对光强的真实关系

最关键结论：

> **光强主要线性/近线性抬高 halation exposure；物理半径主要由胶片结构决定；你看到的半径变大，是因为更强的散射尾巴超过了可见阈值。**

也就是说，光强不是把 kernel “拉宽”，而是把 kernel “抬高”。

## 2.1 胶片曝光层中的关系

E_{h,R}(r)=A,\alpha_R,E_s,K_R(r)

这里：

| 符号           | 意义                                                      |
| ------------ | ------------------------------------------------------- |
| (E_s)        | 光源在胶片平面上的 scene exposure / focal-plane exposure         |
| (K_R(r))     | 红层 halation scattering kernel                           |
| (\alpha_R)   | 胶片结构耦合强度，受 remjet / AHU / film base / pressure plate 影响 |
| (A)          | 你提供给用户的 **Halation Strength / Amplify** 可调参数            |
| (E_{h,R}(r)) | 距离高光边缘 (r) 处红层获得的二次曝光                                   |

如果胶片 stock、格式、扫描比例固定，(K_R(r)) 应该基本固定。**强度 slider (A)** 应该主要改变散射光对乳剂的有效曝光量，而不是直接粗暴改变 blur radius。

---

## 2.2 可见半径为什么会随光强增加？

假设只有当 halo 超过可见阈值 (T) 才看得见：

A\alpha_R E_s K_R(r)>T

如果 kernel 近似 Gaussian：

K_R(r)=e^{-\frac{r^2}{2\sigma^2}}\quad\Rightarrow\quad r_{vis}\approx\sigma\sqrt{2\ln\left(\frac{A\alpha_R E_s}{T}\right)}

如果 kernel 更像 exponential tail：

K_R(r)=e^{-r/\lambda}\quad\Rightarrow\quad r_{vis}\approx\lambda\ln\left(\frac{A\alpha_R E_s}{T}\right)

所以真实感应该是：

| 光强增加     | Halation 强度          | 可见半径          |
| -------- | -------------------- | ------------- |
| +1 stop  | 明显增强                 | 小幅变大          |
| +2 stops | 红边明显                 | 中等变大          |
| +4 stops | 红/橙红 halo 很明显        | 明显变大，但不是线性翻倍  |
| 极端过曝     | 进入 glare / bloom 混合区 | 继续扩张但边缘更软、更雾化 |

**结论：可见半径随光强是慢增长，类似 (\sqrt{\log E}) 或 (\log E)，不应该是 `radius = k * brightness`。**

---

# 3. 最强算法总览

你的最终 pipeline 应该是：

```text
Input RAW / HDR / linear image
→ scene-linear exposure reconstruction
→ film stock exposure model
→ highlight source map
→ red/green-layer halation exposure synthesis
→ add halation in film exposure / density domain
→ negative density / dye-density / scan transform
→ optional bloom / grain / output tone mapping
```

Kodak Vision3 500T 5219 的技术资料给出了 rem-jet backing、sensitometric curves、spectral sensitivity curves、spectral dye-density curves 等信息；这些资料说明胶片模拟应该基于曝光、密度、光谱敏感度和染料吸收，而不只是 RGB LUT。([Kodak][3])

---

# 4. Source map：不要用 sRGB 白色，必须用 scene exposure

假 halation 常用：

```text
source = max(sRGB_luma - threshold, 0)
```

这不够真。真正应该用：

```text
source = f(scene-linear exposure, specular confidence, local contrast)
```

建议：

```text
logE = log2(linear_luminance + eps)

S = softplus((logE - source_threshold) / softness)^gamma
S = S * specular_mask * highlight_confidence
```

其中：

| 参数                     | 作用                                               |
| ---------------------- | ------------------------------------------------ |
| `source_threshold`     | 多亮才开始产生 halation，对应 Dehancer 的 Source Limiter 思路 |
| `softness`             | 阈值过渡是否硬                                          |
| `gamma`                | 高光越强，halation 增长越快                               |
| `specular_mask`        | 灯、霓虹、金属反光、车灯权重更高                                 |
| `highlight_confidence` | 避免白墙、天空大面积乱出红边                                   |

Dehancer 的 Source Limiter 也是用来定义“最低多亮的光源才产生 halation”，Background Gain 则控制 halation 在什么背景色调上可见。([Dehancer][4])

---

# 5. Kernel：红层大半径，绿层小核心，蓝层几乎不要

最强结构：

```text
K_R = w1 * Gaussian(σ1)
    + w2 * Gaussian(σ2)
    + w3 * ExponentialTail(λ)
    + w4 * VeryLowFreqGlare

K_G = v1 * Gaussian(0.35σ1)
    + v2 * Gaussian(0.35σ2)

K_B ≈ 0
```

为什么这样：

| 层           | 作用                    |
| ----------- | --------------------- |
| Red layer   | 主体 halo，半径最大，负责红边     |
| Green layer | 只在强光核心附近参与，负责橙红/黄橙核心  |
| Blue layer  | 基本不参与；参与太多会变成普通 bloom |

Dehancer 的 Local Diffusion 参数本质上也是控制 emulsion 内光扩散距离，数值越大 halo 几何半径越大；Global Diffusion 则控制更大范围的 secondary glare。([Dehancer][4])

---

# 6. 背景可见性：必须 edge-aware / dark-side-aware

真实 halation 在黑背景、高反差边界最明显，在亮背景上会弱很多。所以不要全方向均匀红光。

建议：

```text
V = dark_background_mask
  * local_contrast_mask
  * edge_side_mask
  * not_skin_overprotect_mask
```

其中：

```text
dark_background_mask = sigmoid((bg_threshold - local_luminance) / k)
local_contrast_mask = abs(highlight_luma - surrounding_luma)
```

最终：

```text
H_R = A * beta_R * V * convolution(S, K_R)
H_G = A * beta_G * V * convolution(S_high, K_G)
H_B = 0
```

这里 `S_high` 是更高阈值的 source map，只允许超强高光触发绿层：

```text
S_high = softplus((logE - source_threshold_high) / softness)^gamma_high
```

这样会得到：

| 情况     | 结果        |
| ------ | --------- |
| 弱光     | 没有或很轻的红边  |
| 普通灯    | 红边小而自然    |
| 强霓虹/车灯 | 红色半径变大    |
| 极强白灯   | 中心橙黄，外圈红  |
| 亮背景    | halo 被抑制  |
| 暗背景    | halo 明显展开 |

---

# 7. Halation 强度怎么做成可调，而且仍然真实？

这里要分两个 slider，不能只做一个 opacity。

## Slider 1：**Amplify / Physical Strength**

这个控制物理耦合：

```text
A_physical = user_strength
H_R = A_physical * beta_R * V * conv(S, K_R)
```

它会造成：

* halo 更亮；
* 可见半径稍微变大；
* 极强时红变橙，因为绿层也开始参与；
* 更像“换了更弱 anti-halation / no-remjet / 更敏感乳剂”。

Dehancer 也明确区分 Amplify 和 Impact：Amplify 影响 emulsion 对 scattered light 的敏感度，不是 opacity；增大 Amplify 会让效果更明显并偏向黄色。([Dehancer][4])

## Slider 2：**Impact / Display Mix**

这个控制最终混合透明度：

```text
H_final = mix(original_film_render, halated_film_render, impact)
```

它会造成：

* 视觉上淡一点或浓一点；
* 尽量不改变“光强—半径”的物理关系；
* 适合用户最后微调。

Dehancer 也把 Impact 类比为整体 transparency / opacity 控制。([Dehancer][4])

## 最推荐 UI 参数

| 参数名                           | 技术含义                | 用户感知           |
| ----------------------------- | ------------------- | -------------- |
| `Halation Strength / Amplify` | 改变散射光曝光耦合 (A)       | 真实地变强，半径也略变大   |
| `Impact`                      | 最终透明度               | 效果淡/浓          |
| `Source Limiter`              | 最低触发亮度              | 只让强光出 halo     |
| `Local Diffusion`             | (σ_1, σ_2, λ) scale | halo 半径        |
| `Global Diffusion`            | 低频红/橙 glare         | 中间调暖雾          |
| `Hue / Green Coupling`        | (\beta_G / \beta_R) | 红到橙黄           |
| `Background Gain`             | 背景可见范围              | 黑背景明显/亮背景抑制    |
| `No-Remjet Amount`            | anti-halation 吸收弱化  | CineStill 风格增强 |

**最重要的设计：**
`Strength` 改物理曝光；`Impact` 改最终显示透明度。这样既真实，又可控。

---

# 8. 没有逐场景 paired data，如何做到最接近真胶片？

你的限制其实可以绕开，因为 halation 不一定需要整图 paired supervision。你应该只校准 **halation patch statistics**。

## 8.1 用官方 datasheet 建 stock prior

对 Kodak Vision3 500T 5219：

* 记录它有 rem-jet backing；
* digitize sensitometric curves；
* digitize spectral sensitivity curves；
* digitize spectral dye-density curves；
* 建立 negative density / scan transform。([Kodak][3])

如果目标是 CineStill 800T-like，则使用：

```text
base_stock ≈ Vision3 500T / tungsten-balanced motion-picture negative
anti_halation_strength = low
remjet = removed / no-remjet profile
red_layer_backscatter = high
green_layer_backscatter = medium only for extreme highlights
```

CineStill 800T 的 deep-learning 复现项目也指出，CineStill 800T 没有 anti-halation filter，因此光源周围会有明显红 halo；该项目关注 grain、halo 和 color profile 三个方面。([GitHub][5])

---

## 8.2 用公开真实胶片图像做 unpaired patch calibration

流程：

```text
公开真实胶片扫描 / CineStill 样张 / Vision3 剧照
→ 自动找高光点、灯牌、车灯、霓虹、白字黑底
→ 切 halo patch
→ 估计 source center 和背景
→ 拟合 radial falloff、hue-radius curve、dark-side ratio
```

拟合目标：

| 统计量                                  | 你要拟合什么         |
| ------------------------------------ | -------------- |
| `radial falloff`                     | halo 从边缘往外如何衰减 |
| `visible radius vs source intensity` | 半径随光强慢速扩张      |
| `hue vs radius`                      | 中心橙红，外圈红       |
| `background dependency`              | 黑背景强，亮背景弱      |
| `point vs area source`               | 点光源和大灯牌不同      |
| `red/green ratio`                    | 目标 stock 的红橙比例 |
| `global glare`                       | 中间调红/暖雾强度      |

这样不需要数码原图。你只需要真实胶片图像中的 halo 局部统计。

---

## 8.3 用公共 RAW/HDR 数据合成训练对

Google HDR+ Burst Photography Dataset 包含 3,640 组 full-resolution raw bursts，共 28,461 张 raw images，可以作为高光和 scene-linear reconstruction 的公开训练素材。([Google Research][6])

你可以这样生成 synthetic pairs：

```text
public RAW/HDR image
→ physical halation simulator
→ clean image + halation layer pair
```

训练模型时目标不是：

```text
digital image → film image
```

而是：

```text
image → halation parameter maps / residual halation layer
```

---

# 9. AI 应该怎么用？

不要让 diffusion / U-Net 直接重绘整张图。2024 年一个用 CNN 复现 CineStill800T 的工作发现，MSE/VGG loss 可以学到较好的颜色，也能产生一些 grain，但质量不高，**没有产生 halation**；作者也指出模型缺少足够包含 halation 的 patch，并建议未来结合非 ML 方法。([arXiv][7])

所以最强架构是：

```text
physical simulator → H0
neural residual model → ΔH
final halation = H0 + ΔH
```

模型输入：

```text
linear RGB
log exposure map
highlight source map
edge map
background luminance map
local contrast map
estimated light color / CCT
stock profile embedding
```

模型输出：

```text
Δred_strength_map
Δgreen_strength_map
Δradius_scale_map
Δbackground_visibility_map
residual_RGBA_halation_layer
```

硬约束：

```text
outside allowed_halation_mask, output = 0
```

也就是说，AI 只能在高光附近修正 halo，不能改脸、文字、建筑边缘、材质纹理。

---

# 10. 最终合成位置：不要在最终 sRGB 上 screen blend

正确位置：

```text
scene-linear exposure
→ film layer exposure
→ add red/green halation exposure
→ negative density curve
→ dye-density / scan transform
→ output
```

Kodak 的资料说明 sensitometric curves 描述红、绿、蓝光引起的 film density 变化，spectral dye-density curves 描述 processed film 染料的 spectral absorption，并用于优化扫描/印片设备。([Kodak][3]) 这意味着 halation 最好进入“胶片曝光/密度层”，而不是最后在 sRGB 上加红色 blur。

如果你必须做简化版，也至少应该：

```text
linear RGB
→ log exposure
→ add halation in log/linear domain
→ film curve / LUT
→ final display
```

不要：

```text
sRGB JPEG
→ red blur
→ screen
```

---

# 11. 推荐实现伪代码

```python
def film_halation(
    linear_rgb,
    stock_profile,
    strength=1.0,        # physical Amplify
    impact=1.0,          # final opacity/mix
    source_limiter=2.0,  # stops above middle gray
    local_diffusion=1.0,
    global_diffusion=0.2,
    hue_green=0.25,
    background_gain=1.0,
    no_remjet=1.0,
):
    # 1. scene exposure
    Y = luminance(linear_rgb)
    logE = log2(Y + 1e-6)

    # 2. source map: only strong highlights
    S = softplus((logE - source_limiter) / 0.4) ** 1.5
    S *= specular_confidence(linear_rgb)
    S *= highlight_edge_confidence(Y)

    # 3. background visibility
    bg = local_mean(Y, radius=32)
    V_dark = sigmoid((0.35 - bg) * background_gain)
    V_contrast = normalize(local_contrast(Y))
    V = V_dark * V_contrast

    # 4. kernels: physical radius mostly stock/format dependent
    K_R = multi_scale_red_kernel(
        sigma1=2.0 * local_diffusion,
        sigma2=8.0 * local_diffusion,
        tail_lambda=18.0 * local_diffusion
    )

    K_G = multi_scale_green_kernel(
        sigma1=0.8 * local_diffusion,
        sigma2=3.0 * local_diffusion
    )

    # 5. physical coupling
    beta_R = stock_profile.red_backscatter * no_remjet
    beta_G = stock_profile.green_backscatter * hue_green * no_remjet

    H_R = strength * beta_R * V * convolve(S, K_R)

    S_high = softplus((logE - (source_limiter + 1.5)) / 0.5) ** 1.8
    H_G = strength * beta_G * V * convolve(S_high, K_G)

    H_B = 0.0

    # 6. add in film exposure/density domain, not final sRGB
    halation_exposure = stack([H_R, H_G, H_B])
    halated = add_to_film_layer_exposure(linear_rgb, halation_exposure, stock_profile)

    # 7. final impact/mix
    return mix(render_film(linear_rgb, stock_profile), halated, impact)
```

---

# 12. 参数建议起点

## CineStill 800T-like / no-remjet

```yaml
stock: cinestill_800t_like
base_stock: vision3_500t_5219
remjet: removed
source_limiter_stops: 1.5-2.5
red_backscatter: high
green_backscatter: medium_low
blue_backscatter: zero
local_diffusion:
  sigma1_px_at_4k: 2-4
  sigma2_px_at_4k: 10-24
  tail_lambda_px_at_4k: 24-60
global_diffusion: 0.1-0.35
hue_green: 0.2-0.45
background_gain: high
strength_default: 0.8-1.2
impact_default: 0.6-0.9
```

## Kodak Vision3 500T ECN-2 / remjet present

```yaml
stock: vision3_500t_standard
remjet: present
source_limiter_stops: 2.5-4.0
red_backscatter: low_medium
green_backscatter: low
blue_backscatter: zero
local_diffusion:
  sigma1_px_at_4k: 1-2
  sigma2_px_at_4k: 5-12
  tail_lambda_px_at_4k: 10-30
global_diffusion: 0.05-0.15
hue_green: 0.1-0.25
background_gain: medium
strength_default: 0.4-0.8
impact_default: 0.5-0.8
```

Kodak 5219 技术资料明确说该 stock 有 acetate safety base with rem-jet backing；因此如果你模拟标准 Vision3 500T，halation 应该克制。([Kodak][3]) Dehancer 也区分 standard emulsion 和 no-remjet profile，并指出无 remjet 的 profile 下 halation 通常会非常明显。([Dehancer][4])

---

# 13. 评价指标：专门评估 halation，不要只看 LPIPS

你应该做一个 `HalationEvalSuite`：

| 指标                           | 目标                          |
| ---------------------------- | --------------------------- |
| `intensity_vs_exposure`      | 强度随 source exposure 增长      |
| `visible_radius_vs_exposure` | 半径随光强慢速增长，近似 log / sqrt-log |
| `radial_falloff`             | 多尺度长尾，不是单高斯                 |
| `hue_radius_curve`           | 中心橙红，外圈红                    |
| `dark_side_ratio`            | 暗背景侧明显强于亮背景侧                |
| `background_suppression`     | 亮背景上不乱出红边                   |
| `blue_leakage`               | 蓝通道 halo 接近 0               |
| `structure_preservation`     | 文字、脸、边缘不被改                  |
| `temporal_consistency`       | 视频不闪                        |

---

# 14. 最强方案一句话版

**在没有逐场景胶片/数码 paired data 的限制下，最接近真正胶片红色 halation 的方案是：**

> **用 Kodak 等官方 datasheet 建立 film exposure / density / dye / scan pipeline；用 no-remjet/CineStill-like profile 控制红层 backscatter；用“光强抬高固定散射核、可见半径按 log 或 sqrt-log 慢增长”的模型生成 red/green halation exposure；用公开真实胶片图像做 unpaired halo patch 统计校准；用公共 RAW/HDR 数据合成训练对；AI 只预测 halation 参数或 residual layer；并提供两个强度控制：physical Amplify 和 display Impact。**

这条路线的关键不是“做一个漂亮红 glow”，而是：

> **在正确的空间、正确的位置、用正确的强度—半径关系，生成一个可控的红/橙 halation layer。**

[1]: https://www.kodak.com/en/motion/page/glossary-of-motion-picture-terms/ "Glossary of Motion Picture Terms | Kodak"
[2]: https://blog.dehancer.com/articles/halation/ "Halation and its simulation in Dehancer – Dehancer Blog"
[3]: https://www.kodak.com/content/products-brochures/Film/VISION3_5219_7219_Technical-data.pdf "TI 7294 EKTACHROME 100D 190918.indd"
[4]: https://www.dehancer.com/learn/article/halation "Dehancer | Halation"
[5]: https://github.com/mikasenghaas/sillystill "GitHub - mikasenghaas/sillystill: Recreate the look of Cinestill-800T using Deep Learning · GitHub"
[6]: https://research.google/blog/introducing-the-hdr-burst-photography-dataset/?utm_source=chatgpt.com "Introducing the HDR+ Burst Photography Dataset"
[7]: https://arxiv.org/html/2411.15967v1 "CNNs for Style Transfer of Digital to Film Photography"
