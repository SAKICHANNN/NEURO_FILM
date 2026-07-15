# Roll2Film / FilmCase Ultimate 架构深度科研审计

**审计日期：2026-07-15（Asia/Singapore）**  
**结论状态：产品架构 conditional GO；Roll2Film 核心科研假设 NOT ESTABLISHED**

本报告把证据强度标为：**Fact**（附件、代码/文件计数或一手来源直接支持）、**Inference**（由事实推得但未被作者/数据发布者直接确认）、**Assumption**（实验设计所需但尚未验证）、**Recommendation**（建议动作或预注册门槛）、**Unknown/Blocked**（证据不足或当前不可获取）。外部时效性事实均按 2026-07-15 重新核查；附件证据引用 ZIP 内相对路径。

附件读取契约已满足：先读 `PACKAGE_README.md`、`CHECKSUMS.csv`、`visual_history/NUMBER_MAP.csv`，再检查 3900×9480 历史总表、54 张 normalized anchors、ID 11 反事实、E0 JSON/报告和项目上下文。`CHECKSUMS.csv` 的 **72/72** 个条目均通过字节数与 SHA-256 复核；没有缺失或损坏文件。

## 1. Executive verdict

### 一句话先给结论

**Recommendation：把“固定强专家库 → transform-aware hard selector → image-level support shrinkage → 显式算子渲染 → 全分辨率 severe veto”作为可独立交付的主线；Roll2Film 只作为必须击败 matched-count controls 的离线 challenger。当前 E0 不能证明 roll grouping 提供了超出更多样本/更宽颜色覆盖的特殊信息。**

### 决策表

| 对象 | 裁定 | 可声称 | 不可声称 |
|---|---|---|---|
| Ultimate 产品分工 | **Conditional GO** | 专家发现、硬选择、支持收缩、显式渲染和 severe veto 的职责边界合理 | Roll2Film 必然优于确定性专家库 |
| 当前 Roll2Film E0 | **方法单元测试通过；科学假设未成立** | restricted SPD-affine truth 下，更多 target pixels 降低估计误差；混合 operator 会造成 single-map misspecification | real-roll identifiability、非线性 LUT recovery、group labels 的额外信息、真实胶片风格或安全性 |
| Roll2Film real-roll 路线 | **Ambiguous** | 在冻结 source prior、受限 nuisance 和 canonical operator family 下研究 grouped-target unpaired transfer | 物理 stock/process calibration；未交叉 scanner/lab 时分离它们 |
| 固定确定性强专家库 | **GO，且允许最终获胜** | film-inspired / roll-look recipe，显式、可回放、12GB 轻松 | named-stock calibrated，除非以后有受控 pairs |
| FilmCase | **仅在 Oracle gap 存在时 GO** | safety/applicability routing；Top-1 避免 mode averaging | 无偏好标签时“学会吸引力”或审美真值 |
| support-aware layer | **GO，但先做 image-level validated strength path** | 在 OOD 输入上降低外推风险并记录 style loss | 数值可逆等于语义安全；用 identity fallback 美化研究结果 |
| L3/L4 非线性全局 operator | **逐级 gate** | 若 L2 有稳定、可重复的 hue/luminance residual，可升级容量 | 直接从 target marginal 唯一识别物理 film response |
| L5 bilateral/local residual | **Defer** | 只有全局方法稳定且局部失败类被 paired/stress evidence 证明后才开 | 为了追求更强风格而先上局部模型 |
| named-stock/process claim | **STOP** | 无 | 无受控数字/胶片 pairs、chart 与 scanner/process controls 时不可用 |

### 七个最重要发现

1. **Fact + Inference：53/55/56 是同一 strength path；五锚点尚未证明多个成熟 mode。** 01/09 与 Velvia family 的差异可能形成第二个候选实现族，但在 basic/strength-adjusted operator-grid clustering 前不能定论。02/03/33 只有 smoke cue，不能算专家。
2. **Fact：风格并非纯 saturation/contrast/WB。** ID 11 中红色环境偏橙、暗金属偏青而中性银仍较中性，说明存在 source-hue-dependent trajectory、cross-channel coupling 与 tone/chroma interaction；但当前附件只给 ID 11 原图，其他八张 replay 缺 source，无法对全部图做严格算子归因。
3. **Fact：ID 11 的 56 是 severe artifact，不是“合理的强风格”。** 可见高频 neon-red speckle/posterization；09 也出现同类 speckle，证明保守 strength/margin 并不自动保证 policy safety。
4. **Inference：56 chroma-margin4 challenger 在 ID 11 上没有明显“洗成 bland”。** 它去掉了突出的 speckle，仍保留清楚的橙/青 grade；但单图不能决定晋级，须 frozen multi-image、strength-matched、三次盲化全分辨率裁决。
5. **Fact：E0 把 frame count 与总 target sample count 完全绑在一起。** 每 frame 固定 128 pixels，所以 `n=1→32` 同时是 `128→4096` samples；无 nuisance 时 frame partition 本不应带来额外信息。
6. **Fact：无配对 marginal matching 一般不唯一。** 若 source prior 存在 measure-preserving automorphism，多个映射产生同一 target distribution；ICLR 2024 的 identifiable UDT 结果依赖多组**相互对应**的跨域 conditional distributions，只有 target-side roll groups 不满足该条件。[Identifiable UDT, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/8bb5a934785817f752e7f9322f9b4d54-Paper-Conference.pdf)
7. **Fact：FilmSet 运行时应使用 628 test，不是 638。** 论文写 5,285 originals、4,657 train、638 test，但算术不一致；当前官方分发树和本机四个 test 目录均为 628。`638` 应保留为 publication contradiction，不能静默改写历史。

## 2. 附件视觉分析

### 2.1 证据边界

- **Fact：** `visual_history/ALL_SCHEMES_NUMBERED.png` 为历史 contact sheet；`NUMBER_MAP.csv` 给出了编号到真实 scheme/manifest 的映射。
- **Fact：** `normalized_anchors/` 是 9 个冻结输入上的 anchor-inspired color-only replay，共 54 张；它用于 same-input 比较，不等于历史输出的逐像素复现。
- **Fact：** normalized replay 不含 grain；只有 `red_highlight_counterfactual/00_source.jpg` 提供了对应 source，所以除 ID 11 外不能做完整 source→output 算子归因。
- **Fact：** 历史 53 与 55/56 输入集不同；所有跨编号风格判断以 normalized same-input replay 为主，历史总表只作辅助。

### 2.2 共同视觉特征与可感知 mode

| 观察 | 证据类型 | 裁定 |
|---|---|---|
| 暖色向橙/黄推进，冷暗部与部分绿色向青/蓝绿推进 | Fact（same-input 视觉） | 共同的 warm–teal hue trajectory |
| 中间调对比与彩度增强，但中性区域相对保持 | Fact | 不只是统一 saturation；存在选择性 hue/cross-channel 作用 |
| 53/55/56 的画面关系高度稳定，差别主要是作用强度 | Fact | **一个 mode 的 strength path**，不是三个专家 |
| 01 与 09 的构图内颜色排序相近，09 更收敛、亮度作用更弱 | Fact | 是一个待检验的 safety/implementation family；是否与 Velvia 主轨迹同 mode 尚需 operator-grid clustering |
| 02/03/33 只有 1–2 张 smoke | Fact | 不足以证明稳定性、风格或可部署性 |
| 五个完整 anchor 是否构成多个 mode | Inference | 当前最多保留“一条确定 strength path + 至多一个待检验的第二 family”；**尚不支持 diverse bank** |

独立像素审计也支持这一视觉结论：以 bland control 为参照的线性 RGB 位移，53–56、55–56、53–55 的 cosine 分别约 `0.9967 / 0.9974 / 0.9901`；把 55 的位移只做标量缩放后，56 的残差约 7.2%，53 约 14.0%。这不是感知真值，但作为 operator-direction diagnostic，强烈说明 53/55/56 是近共线 strength path。相对 control 的 RMS 位移约为 09 `0.0346`、55 `0.0417`、56 `0.0494`、53 `0.0627`，顺序也与“同 mode、不同强度”一致。

### 2.3 哪些超越 saturation/contrast/WB

在 ID 11，同一红色环境表面被推向橙色，而低亮度金属部分转向 cyan/teal，中性银色仍相对中性。单一全局 saturation、contrast 或 diagonal WB 无法同时产生这三种方向；可解释来源按当前证据强弱排序为：

1. **Fact/strong inference：source-hue-dependent hue trajectory 与 cross-channel coupling。** 颜色变化方向随 source hue 改变，不是径向增饱和。
2. **Inference：luminance-conditioned chroma/hue。** 暗金属和亮红高光行为不同，但可能也来自 tone curve 与 gamut interaction；需要同 hue、不同 luminance 的 Hald slices 才能拆分。
3. **Inference：tone shape。** toe/midtone/shoulder 的变化可见，但未从完整 source set 量化。
4. **Fact：saturation 是成分之一，但不是全部。** 强度增加确实提高部分色域位移；它不能解释 hue rotation 和 neutral retention。

因此 L0/basic-adjustment adversary 必须先拟合曝光、WB、contrast 与平均 chroma，再比较 residual hue/cross-channel/style；否则“超越 saturation”不成立。

### 2.4 ID 11 反事实裁决

下表中的 `new hard clip` 是本审计统一定义：source 未在 0/255，而 output 任一通道新达到 0/255 的像素占比；它与项目文档中的其他 clipping 定义不可直接横比。`median ΔE76` 只度量变化量，不代表好看或安全。

| 输出 | new hard clip | median ΔE76 vs source | 全分辨率视觉裁决 |
|---|---:|---:|---|
| anchor01 | 25.284% | 18.17 | 颜色强、较平滑；clipping 是高风险诊断，需 full-res 视觉区分黑位、合法 shoulder 与破坏性 clipping |
| anchor09 | 0% | 16.13 | 数值 containment 好，但红色高光仍有明显 speckle；**severe fail** |
| anchor53 | 29.450% | 23.44 | 最强作用；红高光同样出现 speckle/clipping；**severe fail** |
| anchor55 | 23.981% | 16.44 | 较弱的同 mode 版本；仍出现红高光 speckle/hard-boundary；**severe fail** |
| anchor56 | 26.063% | 19.01 | 红色金属高光有突出 neon speckle/posterization；**severe fail** |
| 56 chroma-margin4 | 0% | 19.03 | speckle 明显消失，橙/青 grade 仍清楚；**candidate，不是自动 winner** |
| safe-rich control | 0% | 6.64 | 安全但明显更接近 bland control |

附件已有 diagnostic：56→challenger 的 candidate-pixel coverage 约 `0.314%→0.223%`，输出范围保持 `[4,251]` 且零新 hard clipping；所用诊断下 mean chroma 约 `30.93→21.46`。同时，candidate island count 并未单调改善，因此这些自动量只能定位风险区域，不能替代视觉裁决。

### 2.5 09 牺牲了什么；challenger 是否过淡

- **Fact：** 09 的 luma strength 与 overall strength 较低，视觉上牺牲了 midtone separation、暖/冷色差和局部色彩体积；它不是 identity，但比 53/56 收敛。
- **Fact：** 09 在 ID 11 仍有 red speckle，所以它牺牲风格并没有换来完整 semantic safety。
- **Inference：** challenger 在 ID 11 仍明显强于 safe-rich control，不能称为“过淡”；它更像把异常红高光压回 smooth orange，而非抹掉整个 mode。
- **Unknown：** challenger 在肤色、夜景 neon、天空 gradient、文字和高饱和织物上是否保持 anchor56 风格；单一反事实不能回答。
- **Fact：** 低色度纹理样本 18 上，尤其 53 会放大紫/黄/青色微噪声；这提示强度上升也可能把 neutral-texture noise 变成伪色，必须加入 stress set。

### 2.6 转化为工程 gate

1. **Style floor：** 不用绝对 saturation 阈值。对每个 candidate 求最优 basic operator（EV/WB/sat/contrast）后，要求 residual style distance 至少达到冻结 anchor 55/56 与 safe-rich control 的分离下界；阈值从全部 9 张 replay 与 Hald grid 冻结后得出。
2. **Saturation-only baseline：** 匹配 candidate 的 mean chroma、luma histogram、white point、contrast 和 median ΔE 后，再盲比 hue trajectory、cross-channel coupling 与 appeal。若 basic model 可解释 ≥90% 位移，或盲审不能稳定区分，不登记独立 expert。
3. **Expert-diversity gate：** 先沿合法 strength path 对齐，再计算 pairwise operator distance；53/55/56 应聚成一个 mode。若 survivor 只有一个 mode，关闭 FilmCase，交付一个 champion + bounded strength control。
4. **Operator capacity：** L1/L2 若已复现 hue trajectory 与 tone，停止升级；只有 residual 随 luminance 稳定变化才进 L4。不得因为一张图的 red speckle 就直接加 local residual。
5. **Severe gate：** ID 11 中 09 与 56 都进入 frozen worst-case gallery；任何候选只要在盲化 full-res 裁决被判定 speckle/posterization，即一票否决，不得由平均指标抵消。

## 3. Roll2Film 的支持与反对证据

### 3.1 当前 E0 真正证明了什么

| 条件 | n=1 correct RMSE | n=32 correct RMSE | n=32 mixed/shuffled RMSE |
|---|---:|---:|---:|
| 无 nuisance | 0.017172 | 0.004224 | 0.064019 |
| 轻 nuisance | 0.021609 | 0.005464 | 0.070120 |

**Fact：** 在 truth 恰好属于 orientation-preserving SPD-affine canonical family、source prior 正确且 target pixels 独立抽样时，sample mean/covariance 估计随总像素数增加而变准；实现的 affine、inverse、评估与 deterministic seed 基础可继续使用。

**Fact：** mixed control 表明一个 single-SPD map 无法拟合两个不同 operator 的 mixture。这是有用的 model-misspecification sanity check。

### 3.2 E0 的中心混淆

E0 每 frame 固定 128 pixels，因此总样本数为 `128×n`。无 nuisance 的 SPD-Gaussian OT 只依赖 target mean/covariance；把 4096 个像素分成 32 张 frame 或放入一张“大 frame”，在正确加权下应统计等价。

**Inference：** 当前下降曲线首先是 `more samples / broader Monte Carlo support`，而不是 repeated-measure group information。`mixed=bad` 也只说明 single-map 不适配 mixture，不证明正确 roll label 含特殊监督。

必须补的 fixed-budget controls：

- one frame `N` pixels vs `n` frames each `N/n`，无 nuisance；二者必须满足预注册 equivalence margin；
- 重复同一窄色域 frame vs 独立内容覆盖，固定 N；
- correct frame boundaries vs random partition，固定 N 与颜色覆盖；
- correct groups vs shuffled/mixed/same-film-wrong-roll，固定 group size 与 scene mix；
- nuisance capacity 逐级增强，观察 shared operator 是否被吸收；
- neutral prior swap、joint-training stop-gradient、metadata/ICC/JPEG/grayscale shortcut probes；
- frame/roll bootstrap，不准用 correlated pixels 伪造 CI。

### 3.3 支持中心假设的有限证据

- **Fact：** repeated observations 在一个已正确规定、共享参数的模型中会降低估计方差；E0 对这个工程直觉提供了 restricted synthetic evidence。
- **Inference：** 内容多样的独立 frames 可扩大颜色支持，减少 off-support operator 自由度；这种收益来自 support breadth，而非“roll”这个词本身。
- **Fact：** group shared latent / per-item latent 是成熟建模范式，不是新理论；[Neural Statistician](https://arxiv.org/abs/1606.02185) 与 [ML-VAE](https://cdn.aaai.org/ojs/11867/11867-13-15395-1-2-20201228.pdf) 已覆盖集合/分组共享因素。
- **Inference：** 若真实 roll 中 exposure/WB nuisance 可被小模型解释，而 group-shared residual 在 held-out frames 稳定，roll grouping 可能提供实用弱监督。

### 3.4 反对中心假设的强证据

- **Fact：** 无配对 marginal matching 有映射非唯一性；一个分布可被多种 measure-preserving transformation 保持。[Kernel of CycleGAN](https://openreview.net/pdf?id=B1eWOJHKvB)、[Density-changing regularization, NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/b7032a9d960ebb6bcf1ce9d73b5861f0-Paper-Conference.pdf)
- **Fact：** “多组条件分布可提升 identifiability”的现有定理要求 source/target 条件组彼此对应；当前 roll data 只有 target groups，不满足该前提。[Identifiable UDT, ICLR 2024](https://arxiv.org/abs/2401.09671)
- **Inference：** scene palette、scanner profile、lab process、uploader processing 与 film look 可完全混淆；一个 roll 若只经一个 scanner，二者不可分。
- **Inference：** flexible exposure/WB/curve nuisance 既可能吸收真实 style，也可能把错误 scene differences 留给 shared operator；没有 gauge 与 known-truth recovery 时无法判断。
- **Fact：** BlueNeg 是 restoration/archive data，不是未处理 digital scene 与 physical film pair；它能测试 roll ID 是否含稳定信号，不能给 operator truth。

### 3.5 结论性 claim ladder

| Claim | 当前状态 |
|---|---|
| “更多 target samples 改善 restricted SPD-affine distribution-map estimate” | **Supported** |
| “内容多样 target set 扩大颜色支持” | **Plausible；待 fixed-N support test** |
| “roll grouping 比 matched-count pooled/random grouping 多提供信息” | **Unknown；核心待证** |
| “可分离 shared look 与 exposure/WB nuisance” | **Unknown；待 synthetic + real held-out tests** |
| “可识别 source/scanner-specific roll-look composite” | **Conditional；须冻结 prior/family/metadata** |
| “可识别物理 stock/process response” | **Unsupported / impossible with current data** |

E0 group-information gate 通过前，推荐名称是 **`Canonical explicit colour-map estimation from grouped target observations`**。这些 frames 是不同 scenes 的 clustered/grouped observations，不是同一 latent scene 的严格 repeated measures。只有 fixed-N、nuisance-recovery 与 real-group controls 通过后，才可升级为 **`Grouped-target unpaired explicit colour-operator identification under per-frame nuisance`**；film 语境可用 `Roll-grouped target-only film-inspired explicit colour transfer`。绝不能写 stock-calibrated。

## 4. 推荐架构与不推荐路线

### 4.1 推荐的产品/研究解耦

```text
Frozen deterministic strong experts  ───────────────┐
                                                     ├─> diversity gate
Roll2Film offline candidate discovery (challenger) ─┘
     -> render all eligible experts on thumbnail
     -> hard safety/applicability veto
     -> transform-aware Top-1; low margin => champion
     -> image-level support/uncertainty strength shrink
     -> explicit global operator at full resolution
     -> hard numeric invariants + blinded full-res severe evaluation
     -> product fallback; research counts fallback as failure
```

职责必须严格分开：

- **Roll2Film：** 离线发现/拟合候选 operator，输出 uncertainty、support 与 provenance；不决定部署时审美。
- **Expert diversity gate：** 删除 strength duplicates 与 basic-adjustment duplicates，防止伪 bank。
- **FilmCase：** 在冻结 bank 上判断 applicability/safety；无标签时不能学习 appeal。只有 Evaluator Oracle 显著胜 global champion 才值得建 router。
- **Support layer：** 只管理输入颜色落在专家训练支持之外的外推风险；第一版用每个 operator family 已验证的 image-level strength path，fallback 到同 mode/global champion。
- **Renderer：** 只执行显式、可序列化、可检查算子；不得调用 image-to-image RGB generator。
- **Severe veto：** 只有实现/契约失败（NaN/Inf、serialization/round-trip mismatch、越过声明域、已证明的 folding/inverse failure）可自动阻断。普通 clipping%、gamut occupancy、Jacobian samples 只触发 risk review；脸/文字/halo/speckle 与输出 clipping 是否 severe 必须由预注册 full-res visual policy 决定，自动指标不能单独晋级或否决。

### 4.2 明确不推荐

- soft average 多个风格 expert：会混合 mode、平均化风格；Top-1 是正确默认，但需 margin 与未来视频 hysteresis。
- 直接 per-image histogram/OT 作为产品主线：强场景 palette shortcut 与高饱和 outlier 风险；只作 baseline/challenger。
- 首先训练 set encoder：在 MAP/known-truth solver 尚未成立前，amortization 只会更快地学 scanner/semantic shortcut。
- 先上 SA-LUT/bilateral local：高 halo、block、seam、flicker 风险，且会掩盖全球色彩管理 bug。
- 用 identity fallback 提升研究 severe rate：产品可以 fallback，论文必须把 fallback 计作 transfer failure 并报告 coverage。
- 重新引入 diffusion/GAN 直接生成最终 RGB：与硬约束冲突，且历史已证明身份/细节/显存风险。

## 5. 数学 formulation 与 identifiability

### 5.1 观测模型

对 roll `r`、frame `i`、pixel `p`，在冻结 working space 中写为：

\[
y_{rip}=C_{s(r)}\circ D_{d(r)}\circ T_{\ell(r)}\circ\Delta_r\circ A_{\eta_{ri}}(x_{rip})+\epsilon_{rip}.
\]

- \(T_{\ell(r)}\)：population stock/look；
- \(\Delta_r\)：roll-specific deviation；
- \(D_{d(r)}\)：development/lab；
- \(C_{s(r)}\)：scanner/camera/profile；
- \(A_{\eta_{ri}}\)：per-frame exposure/WB 等小 nuisance；
- \(x_{rip}\)：未观测的中性场景颜色。

没有跨 roll 交叉 stock/process/scanner 的设计时，只能估计复合量

\[
T_r^{look}=C_{s(r)}\circ D_{d(r)}\circ T_{\ell(r)}\circ\Delta_r,
\]

所以只能称 `roll-look` 或 `source/scanner-specific look`。

### 5.2 冻结 source prior 下的目标函数

Primary formulation 先取无条件冻结 prior `z_i=∅`。只有在扩展实验中，才允许使用独立训练、stop-gradient、经过 look/scanner invariance probe 的 descriptor \(z_i\)；source/target 必须有预先定义的 conditional matching，不能从 target colour histogram 临时构造条件并借用 ICLR 2024 的 identifiability 结论。若这样的 \(p_0(x\mid z_i)\) 已冻结，且 \(T_\theta,A_\eta\) 可逆，可用 frame-balanced likelihood：

\[
\mathcal L_r(\theta,\eta)=
\sum_i\frac{1}{N_i}\sum_p
\left[
\log p_0\!\left(T_\theta^{-1}A_{\eta_i}^{-1}(y_{ip})\mid z_i\right)
+\log\left|\det J_{T_\theta^{-1}A_{\eta_i}^{-1}}(y_{ip})\right|
\right]
+\log p(\theta)+\sum_i\log p(\eta_i).
\]

必须 frame-equal 或 capped weighting；不能让一张高分辨率天空图用数百万相关 pixels 产生伪精度。若用 density/normalizing flow，它只估计 likelihood/support，**不得输出最终 RGB**；最终仍由 \(T_\theta\) 或 bake 后 LUT 渲染。

### 5.3 不确定性与 support

Laplace 近似：

\[
q(\theta_r)\approx\mathcal N(\hat\theta_r,H_r^{-1}),\qquad
V_r(c)=J_\theta T_{\hat\theta}(c)H_r^{-1}J_\theta T_{\hat\theta}(c)^\top.
\]

Support 明确定义在 **inferred source space**。一个 frame 对一个颜色邻域最多贡献一次：

\[
m_r(c)=\sum_i \mathbf 1\!\left[\exists p:\left\|T_r^{-1}A_{\eta_i}^{-1}(y_{rip})-c\right\|<h\right].
\]

该量依赖当前 inverse model，不是观测真值，必须先用 synthetic known truth 校准。颜色 `c` 的可靠度组合：frame-capped coverage \(m_r(c)\)、source-prior density、support hull 距离、posterior predictive variance 与 local Jacobian/gamut risk。建议先取 query pixels 的低分位 reliability，再选一个全图 strength \(\gamma_x\)，而不是逐颜色 cell 生硬 blend。

直接在 RGB 输出上写

\[
T_{safe}(c)=T_0(c)+\rho(c)(T_r(c)-T_0(c))
\]

即使 \(\rho\) 是全局标量，也不保证一般 operator family 的 invertibility。第一版必须使用每个 family 已验证的 identity/champion→expert 路径 \(T_{r,\gamma}\)（例如 log-gain、Lie/velocity 或受约束 spline/LUT 参数路径），再重验 Jacobian、gamut 与 bake parity。后续若必须 color-cell shrink，应在 admissible parameter/velocity space 收缩并投影回 operator family；仅做 dense sampling 仍不是全局 injectivity 证明。

### 5.4 六类等价类/不可能边界

1. **Distribution automorphism：** 若 \(H_\#P_0=P_0\)，则 \(T_\#P_0=(T\circ H)_\#P_0\)。
2. **Scene/operator trade-off：** 对任意可逆 \(S\)，\(T_\#P=(T\circ S^{-1})_\#(S_\#P)\)。
3. **Nuisance/operator gauge：** 若 \(B\) 同时可由 shared operator 与 nuisance 表达，\(A_i\circ T=(A_i\circ B^{-1})\circ(B\circ T)\)。
4. **Factorization redundancy：** matrix gain、spline slope 与 LUT 低阶项可互相吸收。
5. **Colour-management equivalence：** 未知 scanner/camera/profile 可与 learned look 任意复合。
6. **Off-support equivalence：** 两个 operator 在已观测颜色上相同、未观测 neon/肤色/HDR 上任意不同，数据仍无法区分。

### 5.5 最低 identifiability assumptions

- working space、transfer、white point、ICC 和 decode 全已知且无隐式 clamp；
- neutral/content prior 独立训练并冻结，禁止与 operator 联合 collusion；
- operator family 有 canonical representative；Brenier/OT 结构只是选一个规范 map，不证明物理真值。[Monge Gap, ICML 2023](https://proceedings.mlr.press/v202/uscidda23a/uscidda23a.pdf)
- nuisance family 小、先验固定、与内容尽量独立，并设 `mean log exposure=0` 等 gauge；
- 同一 group 真正共享一个处理链，没有 frame-specific auto enhancement；
- 颜色支持令 Fisher information 近似满秩；
- group labels、duplicate clusters 与 split 无泄漏；
- 要分 stock/process/scanner，必须有 crossed metadata 或受控采集。

### 5.6 density/flow shortcut tests

- patch shuffle 保留颜色统计、破坏空间语义；若 set encoder 不变，它主要学 density shortcut；
- grayscale、metadata-only、ICC-only、JPEG quantization probes；
- 统一 strip metadata + re-encode 后重测；
- scanner/uploader/date/location adversary；
- source prior swap 与 held-out camera prior；
- matched scene-composition resampling；
- duplicate/near-duplicate exclusion；
- fixed total pixels 与 frame partition invariance；
- synthetic known truth over real-image content。

## 6. Operator capacity ladder

所有 level 共用不变量：float32 render、必要处 float64 fit/validation；无隐式 clamp/quantize；working space/white point/transfer/ICC 写入 canonical JSON；实现 deterministic forward 与适用的诊断；33³ 与 65³ bake parity；linear/HDR 使用 shaper LUT；gamut compression 是独立、可记录算子。L0–L4 的 density-compatible family 另要求可验证 inverse/Jacobian/logdet；L5 是独立 spatial renderer，不作该承诺。[OpenColorIO LUT baking](https://opencolorio.readthedocs.io/en/latest/tutorials/baking_luts.html)

| Level | 参数化与 inverse | Jacobian/gamut/regularization | 进入条件 | 停止条件 |
|---|---|---|---|---|
| L0 exposure/WB | \(T(x)=e^e\,diag(e^{w_R},e^{w_G},e^{w_B})x,\sum w=0\)；精确 inverse | gain/headroom bounds；neutral axis | 永远保留为 nuisance 和 baseline | candidate 可被 L0+basic 完整解释，则不称独特风格 |
| L1 orientation-preserving affine | \(Ax+b,\det A>0\)；建议 \(A=R\exp(S)\)；精确 inverse | condition number、det lower bound、neutral/headroom/bias | L0 无法解释稳定 cross-channel residual | hidden transfer 不改善或 matrix 只追 scene mean |
| L2 matrix + monotone splines | \(C_{out}\circ A\circ C_{in}\)；先只开一侧 3 条 rational-quadratic spline；解析 inverse/J | widths/heights/derivatives>0；固定 mid-grey slope 与 row-scale gauge | L1 有重复 toe/shoulder/channel response residual | spline 吸收 WB/EV，或 hidden/style 不改善 |
| L3 smooth invertible 3D residual | \(T_2\circ\exp(v)\)，或 \(T_2\circ(I+d)\) 且 \(\|\nabla d\|<1-\epsilon\)；flow inverse 或 fixed-point/Newton | 先 9³ 后 17³；二阶 smoothness、neutral/boundary/chroma gain；采用 diffeomorphic parameterization 或 cell-wise interval/Jacobian guarantee，有限 dense samples 只作诊断 | 同 luminance 仍有稳定 hue/cross-channel residual | 只降 NLL、不降 hidden error；folding/gamut fail |
| L4 luminance-conditioned chroma LUT | block-triangular：\(L'=f(L),c'=c+d_L(c)\)，K luminance knots；先反 L，再反该 L 的 chroma map | \(f\) 严格单调；每个 slice 及插值必须有全局 chroma injectivity（例如 diffeomorphic/interval bound）与非奇异 Jacobian；slice smoothness 本身不够 | residual 明确随 luminance 改变 | 只增加 saturation；高光 hue unstable；任一 slice folding |
| L5 bounded bilateral-grid residual | 全局 operator + 低分辨率 bilateral grid 的局部 affine residual；一般不可逆 | 限制 ΔL/ΔC/Δh、TV、edge-aware smoothness；整图一次预测 | L0–L4 稳定且重复 local failure 被证明 | 任一 halo/seam/block/flicker/severe；删除分支 |

L2 的 rational-quadratic spline 有解析 inverse/Jacobian，是合适基元。[Neural Spline Flows](https://papers.neurips.cc/paper/8969-neural-spline-flows.pdf) L5 的直接强 baseline 是 [Deep Bilateral Learning/HDRNet](https://groups.csail.mit.edu/graphics/hdrnet/data/hdrnet.pdf)，因此不是新结构。

### 序列化最低字段

`schema_version, operator_family, capacity_level, working_space, white_point, transfer, domain_min/max, parameters, gauge, regularizers, inverse_method, jacobian_bounds, fit_manifest_hash, source_group_hashes, license_provenance, analytic_hash, lut33_hash, lut65_hash, bake_error, software_commit`。

## 7. Roll2Film / FilmCase / hybrid / deterministic 路线对比

用户要求估计成功概率，但当前没有可校准的历史频率或 expert elicitation，因此不能诚实给出 5–10 个百分点精度。下表只给 **Assumption：ordinal engineering prior**：低 `<1/3`、中 `1/3–2/3`、高 `>2/3`；`明显风格` 指匹配 style floor 的粗先验，`severe` 指未经完整 frozen policy 筛选前出现至少一个严重视觉失败的粗风险。它只排序最小 pilot，不参与 go/no-go。

| 路线 | 明显风格先验 | saturation-only 风险 | severe 先验 | per-photo | 数据/解释性/12GB | 论文/产品价值；最可能失败 |
|---|---:|---|---:|---|---|---|
| 固定强专家库 | 高 | 中 | 低 | 无/强度 | 无训练；极高；轻松 | 产品最高；bank 其实只有一个 mode |
| histogram/quantile | 中 | 高 | 中 | 有 | 单参考；高；CPU | baseline；场景 palette 当 style、破 cross-channel |
| Gaussian/Bures/SPD OT | 中低 | 高 | 低中 | set/ref | 无配对；极高；CPU | E0 baseline；只像 WB/sat/contrast |
| sliced/IDT OT | 高 | 中 | 高 | 强 | 单参考；中；CPU | 强 classical baseline；outlier/speckle/off-support |
| canonical pivot | 中高 | 中 | 中 | 有 | 需预训；中；可行 | 近邻强；pivot bias/监督依赖 |
| pooled learned LUT | 中 | 中高 | 低中 | 无 | domain targets；高；轻松 | 易产品化；平均化 collapse |
| image-adaptive 3D LUT | 中高 | 中 | 中 | 有 | paired/adversarial；中高；可行 | 强 baseline；content shortcut/basis averaging |
| SepLUT | 高 | 中低 | 中 | 有 | 多为 paired；高；12GB 小 batch | 强工程 baseline；teacher/配对依赖、CUDA 扩展 |
| NILUT | 中高 | 中 | 中高 | style-cond. | LUT集；中；轻松 | 容量高；隐式 smooth 不等于单调/可逆，必须 bake |
| SA-LUT | 高 | 低 | 高 | 局部 | reference；中低；12GB 边缘 | 强但延期；halo/block/flicker |
| StatLUT | 未知（偏中） | 未知 | 未知 | ref/stat | 合成 LUT；中；待复现 | 2026-07 新预印本 challenger；成熟度未知 |
| Roll2Film per-roll MAP | 中 | 中 | 中 | roll | grouped target；极高；CPU/GPU | 科研核心；scene/scanner/nuisance confound |
| hierarchical Bayes | 中 | 中 | 低中 | roll+uncert. | 多 roll；高；可行 | uncertainty 价值高；数据少时 prior domination |
| permutation-invariant set encoder | 中高 | 中 | 中高 | amortized | 多 groups；中；可行 | 快；记忆 scanner/uploader/semantics |
| generic/photometric retrieval | bank 决定 | 高 | 中 | 选择式 | 无/少标签；中高；CPU | 必须 baseline；找相似场景而非适合 transform |
| transform-aware hard ranker | bank 决定 | 低中 | 低中 | 选择式 | 需 oracle/safety targets；高；轻量 | 有 oracle gap 才值；无 appeal labels |
| Roll2Film+FilmCase hybrid | 高（条件式） | 中 | 中 | roll+case | 最高复杂度；中高；可行 | 潜力高；错误 expert 与错误 shrink 级联 |
| support-aware shrinkage | 中（会降低） | 不解决 | 低中 | 有 | support/uncert.；高；CPU | tail safety；洗淡风格/边界 banding |
| bilateral/local residual | 高 | 低 | 高 | 强 | paired/local evidence；中；边缘 | 只解决已证局部失败；halo/seam/block |
| FilmSet paired supervised upper bound | 高 | 低中 | 中 | 有 | aligned pairs；模型决定；可行 | recipe upper bound；不是 physical film |
| 无神经 selector | bank 决定 | 中 | 低 | 规则式 | 无标签；极高；CPU | 产品/论文必备 baseline；可能过度保守 |

方法一手来源： [Image-Adaptive 3D LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT)、[SepLUT](https://github.com/ImCharlesY/SepLUT)、[NILUT](https://github.com/mv-lab/nilut)、[SA-LUT](https://github.com/Ry3nG/SA-LUT)、[StatLUT](https://arxiv.org/html/2607.08227v1)、[CanonCGT](https://github.com/Jinwon-Ko/CanonCGT)、[Reinhard mean/std](https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf)、[Pitié colour transfer](https://francois.pitie.net/colour/)。

## 8. 可获取数据、实时 access 与 license

### 8.1 四层状态必须分开

`本机存在 ≠ 远端可获取 ≠ 科学上适用 ≠ 可公开权重/示例`。以下表格分别裁决。FilmSet 仓库的 MIT license 明确覆盖代码；这不能自动外推到每张数据图片。Hugging Face/Kaggle 的 metadata license label 也不能替代 per-file lineage 与发布审查。

| 数据集 | 本机/远端实时状态 | 数量、格式、metadata/group/pair | 许可与发布边界 | 能支持 / 不能支持 | Pilot / 优先级 |
|---|---|---|---|---|---|
| **FilmSet** | **本机完整**；21,140 files，11,262,805,356 B；[论文](https://www.ijcai.org/proceedings/2023/0129.pdf)、[官方仓库](https://github.com/CXH-Research/FilmNet)、[Kaggle](https://www.kaggle.com/datasets/xuhangc/filmset) 当前可访问 | 5,285 digital originals；每张有 Cinema/ClassNeg/Velvia 三个 Capture One recipe target；精确同 basename pairs；无 roll/process/scanner metadata | Kaggle API 标 MIT、代码仓库 MIT；论文仅说来源为 “license-free samples”，无 per-file rights manifest。内部研究可继续；商业训练、公开 weights/examples **仍需 release clearance** | 支持 recipe imitation、paired hidden upper bound；**不支持 physical film、stock/process/scanner truth** | **P0，无新下载**：4,657 train identities 内建 content-disjoint dev；628 只作最终一次 lockbox |
| **BlueNeg** | 本机未下载；[HF card](https://huggingface.co/datasets/ttgroup/blueneg-release)、[file tree](https://huggingface.co/datasets/ttgroup/blueneg-release/tree/main)、[ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_BlueNeg_A_35mm_Negative_Film_Dataset_for_Restoring_Channel-Heterogeneous_Deterioration_ICCV_2025_paper.pdf)；公开 290 GB | 491 negative DNG 16-bit（约 10128×6840）、491 preview PNG、398 print TIFF、398 pseudo-GT；`meta.json` 有 date/roll/frame/location/film_type/paths/scene；53 rolls、13 film strings；negative↔print subset，不是 digital pair | [Custom LICENSE](https://huggingface.co/datasets/ttgroup/blueneg-release/blob/main/LICENSE) 明确允许 academic/commercial use；publication/reproduction/redistribution/derivative 必须写 `Copyrighted by Tien-Tsin Wong`。weights 未点名，商业 checkpoint 建议作者书面确认 | 支持 real-roll grouping 与 archive/scanner-specific pilot；**不支持 digital→film、正片 appeal、stock calibration** | **P1**：先 LICENSE+meta；批准后只下 preview+pseudoGT，955,748,961 B |
| **DigitalFilm_dataset** | 本机未确认；[HF card](https://huggingface.co/datasets/Richards-Sheehy-sudo/DigitalFilm_dataset)、[tree](https://huggingface.co/datasets/Richards-Sheehy-sudo/DigitalFilm_dataset/tree/main)、[code](https://github.com/SongZihui-sudo/digitalFilm)；5 ZIP，3,351,489,014 B | central-directory 审计：7,958 entries，3,979 digital+3,979 film；约 998 unique digital payload、3,896 unique film；只有 folders，无 roll/scanner/process；等数量不是 pairs | HF YAML MIT、card prose/代码 GPLv3、图片 license/lineage 缺失；film samples 明称来自 Internet | 最多 hostile internal exploration；不能训练可发布模型或支持 stock/roll claims | **Quarantine**；不下载、不训练 |
| **PhotoGAN C200** | [官方全文](https://www.techscience.com/cmc/v83n3/61027/html) 描述 1,964 Fuji-C200-labelled + 2,158 digital，512²；Availability 明确称 corpus 因 Unsplash 限制不公开；无 archive/IDs/code endpoint | unpaired；无 roll/process/scanner；query、uploader、scene 与 stock label 混淆 | 文章 CC BY 不覆盖图片；[Unsplash Terms](https://unsplash.com/terms) 的 ML dataset/training 需另走 Unsplash Data 授权 | 只能引用论文，不能合法复建关键训练集 | **Blocked** |
| **SillyStill CineStill pairs** | [repo](https://github.com/mikasenghaas/sillystill)、[paper](https://arxiv.org/html/2411.15967v1)；repo 仍写 Zenodo/HF “Not yet available” | 41 raw/38 processed same-scene Sony A7 vs Nikon F3/CineStill pairs；40mm/55mm、auto shutter、development/scan/hist-match confounds | 无 dataset release、无 LICENSE | 若以后发布，可作 tiny external sanity；不能支持泛化 stock calibration | **Unavailable，不可依赖** |
| **Emulating Emulsion** | [author PDF](https://musicofmusix.github.io/assets/misc/siggraph_abstract.pdf)、[ACM DOI](https://dl.acm.org/doi/10.1145/3721250.3743014)；无公开 data/code archive | 一卷、36 exposures、3 illuminants、11 exposures、96-patch chart，共 3,168 RAW patch pairs；30 参数 explicit model | 未发布数据无 dataset license | 最接近 controlled physical compact baseline；只能重实现公式/做 synthetic comparison | **Baseline-only**；索取数据或自采集需批准 |
| **FilmGrainStyle740k** | [官方页面/完整 license](https://www.interdigital.com/data_sets/filmgrainstyle740k-dataset)；邮件申请，无 direct/API/archive | 148,694 clean；每张 5 个 proprietary-Nuke synthetic grain 版本；240 grain styles；grain-only | 免费仅 noncommercial research/evaluation；commercial training/testing 禁止，商业需付费；禁止未批准重分发 | 只支持 grain，不支持 colour operator | **低优先级/非关键路径** |
| **MIT-Adobe FiveK** | 本机只有约 903-file/7.49 GB freeze pack，不是完整源；[官方页](https://groups.csail.mit.edu/graphics/fivek_dataset/) 提供 5,000 DNG 与 5 位专家 retouch，完整约 50 GB | 同数字原图五个 retouch pairs；非 film、无 roll | 图片按文件受两份 research license 管理；商业训练/weight release不可默认 | generic canonicalization、operator capacity、paired upper bound | **P2；不为当前路径补下完整包** |
| **HDR+ Burst** | [官方 dataset](https://hdrplusdata.org/dataset.html) anonymous download；3,640 bursts/28,461 DNG；curated 153 bursts 37 GiB，full 765 GiB | 2–10 同场景 burst frames；camera/WB/exposure metadata与 pipeline outputs | CC BY-SA；新大下载仍需项目批准 | 测 grouped sample-count、same-scene重复、camera/nuisance；不是 content-diverse roll 或 film | **仅 falsification 需要时 P2** |
| **Middlebury Color** | [官方 direct ZIP 页面](https://vision.middlebury.edu/color/data/)；chart RAW/JPEG 约 41–123 MB，85 natural RAW/JPEG pairs 约 136–139 MB | 35 camera models 的注册 ColorChecker，24 个有 RAW+JPEG；85 pairs/12 cameras；含 WB/exposure变化 | 官方明确允许 use and publish images，要求引用 | camera pipeline/scanner-style nuisance、matrix/curve unit tests；不是 film | **P1 小型 operator/nuisance benchmark** |
| **Rendered WB** | [官方页](https://yorkucvil.github.io/projects/public_html/sRGB_WB_correction/dataset.html)；65,416 rendered inputs+targets，7 DSLR | 多 WB preset、picture styles、chart coordinates/metadata；paired nuisance truth | 当前页可下载；发布/商用边界需随 archive license 快照复核 | L0 nuisance recovery 与 flexible-nuisance absorption；不是 film | **metadata/小 pilot 后再决定** |
| **BabelColor ColorChecker spectra** | [官方数据页](https://babelcolor.com/colorchecker-2.htm) direct CGATS/CxF/XLS，KB 级 | 30 charts 平均光谱与 D50/Lab；无图像 operator truth | 文件可下载；使用前保存条款快照 | synthetic support/grid、色彩管理单测；不能替代真实 capture | **P0 小型测试资产** |

### 8.2 BlueNeg 的泄漏与混淆

公开 `meta.json` 的直接审计给出：491 frames / 53 rolls；roll size `1–32`，中位数 6；13 个 film-type strings，`Kodak Gold 100-5` 单独占 226/491。每个 roll 只有一个 film type，因此 roll、stock、date、location 强耦合。

更关键的是：官方 30 个 test frames 来自 17 rolls，而这 17 rolls **全部也有 non-test frames**。所以 restoration 官方 split 不能用于 roll-level identification。E2 必须重新按完整 `roll_id` 切分，并报告 `film_type/date/location/partition` 预测 roll/operator 的 shortcut accuracy。

### 8.3 数据审计总判决

- **Fact：** 当前未找到同时满足“可直接下载、许可明确、真实同场景 digital/physical-film pair”的公开数据。SillyStill 最接近，但未发布。
- **Recommendation：** FilmSet 与 BlueNeg 是唯二关键路径；前者回答 recipe transfer，后者只回答 real-roll group 是否含稳定 archive/scanner-specific signal。两者的 claims 必须分开。
- **Recommendation：** `data/film_domain` 的 4,212 JPG 即使存在，lineage quarantine 仍有效；存在不等于科学可用或可发布权重。

## 9. FilmSet 628 / 638 冲突

这是一个可明确记录、但不能假装作者已确认的 publication contradiction：

- **Fact：** FilmSet 论文写每个 domain 有 5,285 张，同时写 4,657 train + 638 test。[IJCAI 2023 paper](https://www.ijcai.org/proceedings/2023/0129.pdf)
- **Fact：** `4,657 + 638 = 5,295`，与 5,285 相差 10。
- **Fact：** 当前官方 Kaggle archive 的本机完整树中，`input/Cinema/ClassNeg/Velvia` 的 train 都是 4,657，test 都是 628；`4×(4657+628)=21,140`，与文件总数完全一致。
- **Inference：** 论文中的 638 极可能是 typo；没有找到作者针对 split 数量的正式 erratum，所以不能写成作者已更正。
- **Recommendation：** 运行时代码、manifest、hidden evaluator 全部使用 **628**；文献综述仍原样记录 `paper_reported_test_n=638`。

建议冻结 schema：

```yaml
paper_reported_test_n: 638
archive_observed_test_n: 628
archive_observed_train_n: 4657
originals_n: 5285
archive_files_n: 21140
resolution: use_distributed_archive_and_retain_publication_contradiction
```

CI 必须校验 archive SHA-256、每目录计数、四 domain basename 一一对应、train/test basename 不相交。任何未来 archive 改版都产生新 split version，不静默覆盖。

## 10. 最小数据计划

### 10.1 现在即可执行：零新下载

1. 冻结附件 72-file checksum、normalized anchors、ID 11 worst-case gallery 与 E0 raw JSON。
2. 冻结本机 FilmSet 21,140-file manifest：SHA-256、bytes、dimensions、ICC、basename/pair ID、split、duplicate cluster。
3. 将 4,657 train identities 先按内容 ID/duplicate cluster 划为 source-only pool、target-only pool 与内部 paired-blind dev lockbox；同一 identity 不得同时出现在 source 与 target 训练池。除 stem/path/order/CRC 外，还以 pHash 与冻结 image embedding 排除近重复/内容检索泄漏。
4. 官方 628 aligned pairs 存入独立最终 evaluator；bank、capacity、router、shrink、阈值与 policy 全冻结前禁止读取。方法选择只用 4,657 内部 lockbox。
5. 实现 L0–L2、basic-adjustment adversary、fixed expert bank、classical OT、pooled operator、Exposure-style deterministic filter sequence。
6. 扩展 E0 到 fixed total pixels、coverage、nuisance capacity、L2/L3 truth 与 posterior calibration。

### 10.2 只取 metadata，不取图片

7. 保存 BlueNeg `LICENSE`、`README`、`meta.json` 的 URL、commit/hash、抓取时间；生成 roll-disjoint split 与 shortcut report。
8. 对 DigitalFilm/PhotoGAN/SillyStill/Emulating Emulsion 只保存 access/license verdict；不因网页存在而加入 loader。

### 10.3 通过 no-data gate 后才申请

9. 若 E0 group-information gate 通过，申请下载 BlueNeg 两条 8-bit lane，合计约 0.956 GB；不下载 290 GB 全量。
10. 只有需要 falsify same-scene group advantage 时，才申请 HDR+ 最小子集；不为“可能有用”下载 37–765 GiB。
11. named-stock calibration 若成为新目标，必须单独立项受控 digital RAW + physical film + chart + process + scanner 采集；不从 Internet photos 推断。

## 11. 实验 DAG

所有实验均输出：冻结 config、seed、input/group manifest hash、环境、raw per-image/frame metrics、bootstrap CI、worst-case gallery、decision JSON、软件 commit。`Pass` 不能靠一张 contact sheet；`Ambiguous` 一律不晋级。

### Gate freeze protocol

当前门槛中仍有需要 pilot 估计的量；在未冻结前，它们只能叫 **draft gate**，不能叫 preregistered result。流程固定为：仅用 synthetic pilot seeds / 4,657 内部 dev / non-gold stress 估计阈值 → 写入只读 `gate_contract.yaml` 与 hash → 再运行独立 confirmatory seeds / gold / 最终 628。

| Gate | Draft numeric rule（Recommendation） | 冻结时点 |
|---|---|---|
| E0 partition equivalence | TOST margin `δ=max(5%×pilot single-frame RMSE, 2×pilot Monte-Carlo SE)`；confirmatory CI 必须落在 ±δ | confirmatory synthetic seeds 前 |
| E0 group benefit | fixed-N 下 correct vs matched wrong 至少 10% relative hidden-error reduction，frame/roll bootstrap 95% CI lower >0 | confirmatory E0 前 |
| E1 paired-gap closure | `(best_unpaired−method)/(best_unpaired−paired_upper) ≥20%`，paired-image bootstrap lower CI >0 | internal dev confirmatory fold 前 |
| E3 routing value | Oracle loss 比 champion 至少低 10%；ranker 关闭 ≥25% generic→Oracle gap；CI lower >0 | gold/最终 policy 前 |
| Selector stability | 对 ±0.1EV、JPEG Q90/95、一次 down/up-resize 的非边界 cases，Top-1 switch ≤5%；低 margin cases 应 fallback | E3 confirmatory 前 |
| E4 style retention | in-support ≥90%，overall ≥80%；均对 best-basic/style-strength matched 定义 | E4 gold 前 |
| Severe | accepted transfers 的 full-res visual severe 为 0，且 cluster 95% upper CI ≤人工批准的 `B_product`；`B_product` 未批准时结果只能 ambiguous | E6 sample-size freeze 前 |

不使用可任意调权的 `style+appeal−λ·severe` 总分：先 severe lexicographic veto，再要求 style floor，最后比较 recipe fidelity/appeal。这样避免 hidden results 后移动 utility weights。

### E0 — known-truth falsification

| 项 | 预注册内容 |
|---|---|
| Hypothesis | 在固定总 sample budget 下，正确 group/frame structure 仅在 nuisance 分解或独立颜色覆盖时提供额外恢复信息 |
| Baselines | L0/L1、Gaussian/Bures OT、single-concatenated frame、pooled、repeated-copy、random partition |
| Variables | truth L1–L4；n=1/2/4/8/16/32；fixed N；support breadth；EV/WB/scene/scanner nuisance；mixed/shuffled；nuisance capacity |
| Controls | no-nuisance partition invariance；repeated vs independent-support；matched total pixels；wrong groups；source-prior swap；flexible-nuisance absorption |
| Metrics | operator-grid ΔE/RMSE、hidden RGB/ΔE、gauge-aligned parameter error、NLL、style ratio、Jacobian/gamut、90% interval empirical coverage |
| Pass | no-nuisance 1-frame 与 n-frame 在 equivalence margin 内；fixed-N correct grouping 在 nuisance/coverage 条件下胜 wrong grouping，roll/frame bootstrap 95% CI 排除 0；收益跟 independent support 而非 frame count；90% CI empirical coverage 约 85–95% |
| Fail | 只有总像素增加有效；L2/L3 truth 不恢复；稍强 nuisance 吸收 shared style；source-prior swap 导致 operator 大漂移 |
| Compute | L0–L2 CPU/M5 1–6 h；L3/L4 factorial 约 0.5–3 GPU-days |
| Next/stop | Pass→E1/E2；Fail→关闭 “roll grouping special information” claim，保留 canonical pooled operator baseline |

### E1 — 本机 FilmSet paired-blind unpaired transfer

| 项 | 预注册内容 |
|---|---|
| Hypothesis | 不看 pairs 的 grouped/pooled target operator 能在 4,657 identities 内的 content-disjoint paired-blind dev lockbox 上胜强 unpaired baselines，并在完整 policy 冻结后于官方 628 final lockbox 复现 |
| Pair blinding | source-only 与 target-only 训练 pools 的 content IDs/duplicate clusters 不相交；去 shared stem/path/order/CRC；加入 pHash/embedding nearest-neighbour canaries；内部 dev 与官方 628 都在独立 evaluator，后者最终只开放一次 |
| Baselines | identity、EV、WB、sat、contrast、WB+sat+contrast、Lab mean/std、histogram、single-ref/sliced OT、Gaussian/Bures、pooled operator、shuffled groups、fixed preferred anchor、functionally faithful [Exposure](https://github.com/yuanming-hu/exposure) operator-policy baseline、image-adaptive 3D LUT、SepLUT、NILUT、CanonCGT；StatLUT 若无可运行代码则明确 `not-run`；paired supervised upper bound |
| Group controls | same target sample count；one concatenated set；random pseudo-roll；pooled recipe；recipe-label shuffle；known synthetic multi-operator groups |
| Metrics | internal hidden ΔE00/PSNR/SSIM 只作 recipe fidelity；style floor、best-basic residual、operator stability、severe rate、fallback/coverage；paired-gap closure；最终 628 只做 frozen-policy confirmation |
| Pass | 内部 dev 胜 best unpaired baseline，paired-by-image bootstrap CI 排除 0；gap closure 达到冻结下界；通过 style floor 且 severe 不恶化；最终 628 仅验证、不再调参 |
| Fail | pooled/simple Exposure/anchor 同等或更好；pseudo-group gain 随 grouping 消失；只靠饱和度；hidden severe |
| Compute | classical/fixed CPU 4–12 h；pooled/adaptive LUT 6–24 h/模型；12GB 可行 |
| Claim | 只称 `film-recipe transfer`，不能称 real-film |

FilmSet 没有真实 roll hierarchy；一个 recipe domain 本身共享同一合成 recipe。E1 主要测试 target-set transfer 与 pair blindness，不单独证明 roll information。

若官方实现、数据许可或 12GB 兼容性使某 learned baseline 无法运行，报告必须逐项写 `not-run + blocker`，并把结论降为“胜过已运行 baselines”，不能写 strongest unpaired baseline。Exposure-style 手写 deterministic filters 只是额外消融，不能替代其 learned operator-policy baseline。

### E2 — BlueNeg metadata-first / minimal preview pilot

| 项 | 预注册内容 |
|---|---|
| Hypothesis | correct physical roll 的其他 frames 对 held-out frame 的 explicit operator，比 matched wrong-roll controls 提供稳定附加信息 |
| Data | 先仅 meta/license；批准后 8-bit preview+pseudoGT；按完整 roll-disjoint 切分，不用官方跨 roll 泄漏 split |
| Controls | same-film wrong-roll、date/location/scene matched wrong-roll、random、shuffled、same sample count；strip metadata/ICC、统一 re-encode、grayscale；film-type/date/location classifier |
| Metrics | held-out-frame likelihood/support、operator bootstrap stability、pseudo-GT/print subset error、shortcut predictability、severe/style |
| Pass | correct-roll advantage 在所有 matched controls 后仍存在，roll bootstrap CI 排除 0，且不由 deterioration/scanner/metadata probe 解释 |
| Fail | correct-roll gain 消失；scanner/date/location 解释更强；roll size/stock confounded；operator 对 content bootstrap 不稳定 |
| Claim | 最多 `archive/scanner-specific roll-look information`；绝不写 digital→film 或 stock truth |

### E3 — expert bank + FilmCase

| 项 | 预注册内容 |
|---|---|
| Hypothesis | bank 有多个 strength/basic-adjusted mode，且 post-hoc per-image Oracle 显著胜 global champion；不使用 appeal labels 的 transform-aware router 可关闭可观部分 safety/applicability Oracle gap |
| Baselines | champion、random、generic retrieval、photometric NN、no-neural rules、transform-aware ranker、Oracle |
| Data | frozen bank；训练与 gold 不交叉；thumbnail renders 与 operator/support/Jacobian features。主 ranker 只用 synthetic known-truth risk、self-consistency 与 paired-blind recipe applicability（若使用则另标 supervised）；human appeal 只作 post-hoc E6 evaluation，绝不训练 |
| Metrics | 先 lexicographic severe veto，再看 style floor/applicability；paired-recipe Oracle、synthetic-safety Oracle 与 human-appeal Oracle 分开报告；另报 top-1 margin、selection stability、coverage/fallback |
| Pass | diversity gate 留下 ≥2 实质 mode；至少一个合法监督契约的 Oracle 胜 champion 且 CI 排除 0；主 ranker 胜 generic/photometric/规则并关闭冻结比例的 safety/applicability Oracle gap；human appeal 结果只验证，不回灌训练；severe 不增加 |
| Stop | Oracle 无优势→关闭 FilmCase；只剩同一 strength path→单 champion+strength；ranker 不胜规则→用无神经 selector |
| Compute | 预渲 thumbnails 后，小 ranker 2–8 h；CPU inference |

### E4 — support shrinkage

| 项 | 预注册内容 |
|---|---|
| Hypothesis | 在匹配 style strength 后，support/uncertainty shrink 比 fixed gamut margin、global strength scaling 更能减少 red/neon/OOD severe |
| Arms | no shrink、沿已验证 champion→expert parameter/velocity path 的 image scalar \(T_{r,\gamma}\)、identity-path scalar、cellwise research-only、source-gamut margin、chroma-margin4；禁止直接 RGB-output linear blend 后假定可逆 |
| Stress | ID 11、neon、饱和织物、肤色、天空 gradient、暗部、HDR/headroom、wide gamut |
| Metrics | 以 inferred source-space frame-capped support 为条件的 supported/overall style retention、synthetic support calibration、severe rate+CI、candidate coverage、Jacobian/gamut、banding、fallback rate |
| Pass | supported style ≥90%、overall ≥80%（Recommendation：初始门槛）；相对 severe 显著下降；无新 Jacobian/banding；胜简单 margin/strength baseline |
| Fail | 安全只来自变 bland；不优于 chroma margin；cell boundary banding；研究结果依赖 identity fallback |

### E5 — bounded local residual

只在以下 DoR 全满足后开工：L0–L4 与 colour-management 已稳定；同类局部失败跨多图重复；failure 不是 gamut/support/ICC/8-bit bug；global operator 与 selector 都无法解决。

Baseline 为 global winner。Local residual 必须在预注册 failure masks 上改善至少约 15%（Recommendation：pilot threshold），全图 style 不下降；thin edge、checkerboard、tile、resize、video-jitter stress 无 halo/seam/block/flicker。任何 severe local artifact 即 fail-closed 并删除分支。

### E6 — hidden style / appeal / content / product policy

- 冻结 bank、ranker、shrink、fallback、renderer 与 policy hash；候选名/顺序随机化。
- 每幅 full-res 输出三次独立盲化裁决；`severe=yes` 一票触发 adjudication/fail-closed，`ambiguous` 不算 pass。
- 同时报告 style obviousness、appeal、content preservation、severe、fallback、coverage；不能只报 accepted subset。
- CI 按 input 与 rater cluster/bootstrap；同一图的三个 rating 不是三个独立 image samples。
- 若 0 severe / n images，95% 上界近似 `3/n`；32 张零失败仍只能说明上界约 9.4%。若产品预算为 1%，需约 300 个独立零失败 inputs 才有相应量级证据，且还要处理类别/聚类覆盖。

## 12. Style-not-saturation 评估

### 12.1 必备 baseline 集

`identity, exposure-only, WB-only, saturation-only, contrast-only, WB+saturation+contrast, Lab mean/std, per-channel histogram, single-reference OT, pooled operator, shuffled groups, same-film wrong-roll, fixed preferred anchor, generic retrieval, photometric retrieval, Exposure-style explicit filters, paired upper bound`。

### 12.2 Operator-space 指标

在冻结 Hald/natural-colour grid \(\mathcal G\) 上：

\[
d(T_a,T_b)=\mathbb E_{c\sim\mathcal G}\Delta E_{00}(T_a(c),T_b(c)).
\]

排除 basic adjustments：

\[
d_{nonbasic}(T)=\min_{B\in\mathcal B_{basic}}\mathbb E_c\Delta E_{00}(T(c),B(c)),
\]

\[
d_{mode}(T_a,T_b)=\min_{\alpha,B\in\mathcal B_{basic}}d(T_a,B\circ T_b^\alpha).
\]

其中 \(\mathcal B_{basic}\) 含 EV/WB/sat/contrast，\(T^\alpha\) 是合法 identity→operator strength path。必须同时报告：

- hue trajectory `Δh(L,h,C)`；
- Jacobian off-diagonal/cross-channel energy；
- neutral-axis drift；
- toe/midtone/shoulder 曲率；
- luminance×chroma interaction；
- colour-volume displacement 与 gamut occupancy；
- best-basic explained variance/residual；
- expert pairwise matrix、cluster 与 anchor relation；
- strength matching 后的 severe rate。

### 12.3 Style floor 与 collapse stop

- 先冻结 normalized 55/56、challenger 与 safe-rich 的 operator-grid距离；不要把 09 当自动安全 floor，因为它在 ID 11 有 severe speckle。
- Candidate 必须在多个输入类别上显著胜 best-basic baseline，且 `d_nonbasic` 不低于 anchor-derived floor。
- Candidate 与 identity/pooled/basic 的距离若低于 floor，判 identity/average collapse，即使 paired metric更好也停止。
- saturation adversarial test 必须匹配 mean chroma、white balance、tone histogram 后再盲比；只更鲜艳不算风格。
- 如果所有专家经 basic/strength 对齐后聚成一个 cluster，关闭 expert bank 和 FilmCase。

## 13. Severe-artifact evaluation

### 13.1 一票否决 taxonomy

`face/text/object/geometry corruption, posterization, banding, large clipping, gamut collapse, red/neon speckle, colour blocks, tile seams, halo, unstable local colour, NaN/Inf, future temporal flicker`。

### 13.2 自动诊断的职责边界

自动检查可以直接阻断实现契约失败：NaN/Inf、serialization/analytic-vs-bake mismatch、声明域外未定义行为、要求可逆的 family 出现已证明 folding/inverse failure、full/chunk 或 tile parity 不一致。Cross-channel hue rotation 不要求逐通道 monotonic；L3/L4 应检查 folding/Jacobian/injectivity guarantee，而不是泛化的 “LUT non-monotonic”。Clipping%、gamut occupancy、candidate coverage、sampled Jacobian 仅生成 risk flags、masks 与 worst-case crops，不能单独否决视觉候选。

自动检查不可以：仅凭 ΔE、LPIPS、clipping%、candidate coverage 或数值可逆决定脸/文字是否被破坏、speckle 是否严重、风格是否吸引。ID 11 已证明 candidate coverage 下降不等于所有 topology 指标改善。

### 13.3 Frozen gold/stress 结构

- gold：肤色、文字/标识、红高光、neon、天空/雾 gradient、暗部、饱和织物、植物、金属、中性灰、wide-gamut/HDR、24/100MP。
- synthetic stress：RGB/neutral ramps、Hald CLUT、checkerboard、thin edges、isoluminant hue sweep、near-boundary red/cyan、ICC/bit-depth permutations。
- 每次记录输入 hash、operator/policy hash、render precision、pre/post gamut、clipping、Jacobian、fallback reason。
- worst-case gallery 必须展示 champion、candidate、basic strength-matched、source、diagnostic mask；不能只展示赢家。

### 13.4 Policy-level统计

- 评估完整 selection→shrink→render→veto→fallback policy，而不是各 expert 的 oracle best cases。
- severe 分类别报告 rate 与 cluster-bootstrap CI；同时报告 fallback/abstention coverage。
- ambiguous fail-closed；不同裁决者三次盲化，冲突进入独立 adjudication。
- 预注册产品 severe budget 与样本量；没有足够 n 时写 inconclusive，不用“0 observed”冒充“0 risk”。

## 14. 工程 API、schema 与 tracker

### 14.1 建议目录

```text
src/roll2film/
  operators/       # L0-L5, gamut, LUT bake, canonical serialization
  simulator/       # known-truth operators, nuisance, support/scene/scanner factors
  identification/  # MAP/Bayes/set inference, gauge, uncertainty
  routing/         # expert bank, diversity, FilmCase scorers, hard policy
  support/         # coverage, posterior risk, scalar shrink, fallback
  evaluation/      # style, severe, blind adjudication, CI, galleries
  manifests/       # image/roll/operator/case/policy/provenance schemas
scripts/roll2film/
configs/roll2film/
tests/roll2film/
docs/roll2film/
outputs/roll2film/<experiment>/<config_hash>/
```

### 14.2 核心 protocols

```python
class GlobalColorOperator(Protocol):
    schema_version: str
    working_space: WorkingSpaceSpec
    domain: DomainSpec
    capacity_level: int
    invertible: bool
    supports_logdet: bool

    def forward(self, rgb_float) -> RGB: ...
    def jacobian(self, rgb_float) -> Matrix3x3: ...
    def serialize_canonical(self) -> bytes: ...
    def export_lut(self, cube_size: int, shaper: ShaperSpec) -> LUTBundle: ...
    def content_hash(self) -> str: ...

class InvertibleGlobalColorOperator(GlobalColorOperator, Protocol):
    invertible: Literal[True]
    supports_logdet: Literal[True]
    def inverse(self, rgb_float) -> RGB: ...
    def log_abs_det_jacobian(self, rgb_float) -> float: ...

class SpatialResidualRenderer(Protocol):
    # L5 only: depends on image coordinates/context and never enters density likelihood.
    def predict_field(self, lowres_working_image) -> FieldBundle: ...
    def render(self, fullres_working_image, field: FieldBundle) -> RGBImage: ...
    def serialize_canonical(self) -> bytes: ...
```

硬约束：无隐式 clamp/8-bit roundtrip；alpha 不参加拟合；CPU/CUDA/MPS/analytic/baked parity 可报告；render 与 fit 的 colour state 相同。L5 一般不可逆、依赖坐标/上下文，不实现 inverse/logdet，不得用于 §5.2 density likelihood。

### 14.3 Schema

| Schema | 必备字段 |
|---|---|
| `WorkingImageManifest` | source hash/bytes、decode/RAW params、ICC hash、working space、white point、transfer、bit depth、alpha |
| `RollManifest` | dataset/version、roll/frame、stock/process/scanner/lab/uploader/date/location、license snapshot、split、duplicate cluster、eligibility |
| `OperatorBundle` | family/level、params、gauge、fit config、source groups、claim level、support、uncertainty、provenance/license、analytic/LUT hashes |
| `InferenceReport` | solver trace、objective、posterior/Hessian、nuisance、support grid、seed、warnings、failure reason |
| `ExpertBankManifest` | bank version、operator hashes、diversity matrix/cluster、anchor relation、eligibility/veto、champion |
| `CaseDecision` | input hash、candidate scores/veto、margin、selected/fallback、support scalar、policy hash |
| `RenderReport` | pre/post gamut、range/clipping、Jacobian、bake error、precision/device、full-res diagnostics |
| `EvaluationManifest` | frozen IDs、blinding seed、rater sessions、metric definitions、policy freeze、CI method、decision |

### 14.4 最低测试集合

- unit/property：identity、round-trip、analytic inverse、det/orientation、monotone spline derivatives、dense Jacobian lower bound、NaN/Inf、out-of-domain、no hidden clip；
- LUT：canonical JSON deterministic roundtrip、33³/65³ bake ΔE、forward/inverse LUT、shaper、Hald/ramp/gradient；
- precision：float32/float64、CPU/CUDA/MPS parity，chunked vs full-image identical；
- colour management：ICC roundtrip、16-bit/wide-gamut/HDR、linear >1、legacy sRGB8 adapter explicitly detected；
- leakage：FilmSet pair-ID/path/order/CRC canaries、group-disjoint、duplicate cluster zero cross-split、BlueNeg roll-disjoint；
- routing：top-1 margin、perturbation stability、fallback reproducibility、bank freeze；
- local：整图一次 grid prediction，chunk apply parity，tile/edge/video-jitter stress。

### 14.5 Phase tracker

| Phase | 文件/DoR | DoD 与验证命令 | 估时/资源 | Gate / next |
|---|---|---|---|---|
| P0 Evidence freeze | `manifests/{anchors,filmset,e0}.jsonl`; DoR=附件/本机 files | 72/72 checksums；FilmSet 21,140计数；`pytest -q tests/roll2film/test_manifests.py` | 0.5–1 d CPU | mismatch→stop；pass→P1 |
| P1 Colour-state ingress | `operators/space.py`, `manifests/working_image.py`; DoR=WorkingImage API | 16-bit/profile fixtures；no implicit sRGB8；`pytest -q tests/roll2film/test_colour_state.py` | 2–4 d CPU/M5 | legacy adapter未隔离→stop |
| P2 L0–L2 core | `operators/{photometric,affine,splines,lut_io}.py` | roundtrip/Jacobian/bake/parity 全 pass；`pytest -q tests/roll2film/operators` | 4–7 d CPU | pass→P3/P4 |
| P3 E0 falsification | `simulator/`, `identification/`; fixed-budget configs | `python -m scripts.roll2film.e0 --config configs/roll2film/e0_fixed_budget.yaml`；raw+CI+decision | 3–7 d；L3另0.5–3 GPU-days | group info fail→关闭 Roll2Film core；保留 pooled |
| P4 E1 pair-blind dev | FilmSet adapters/evaluator；DoR=P0/P2 | 4,657 内 content-disjoint source/target/dev；pair/pHash/embedding leakage canaries=0；`pytest -q tests/roll2film/test_pair_blinding.py` | 1–2 wk；12GB可行 | internal best-unpaired/anchor不胜→学习路线 stop；628仍封存 |
| P5 Bank/diversity | `routing/bank.py`, `evaluation/style.py` | best-basic residual、cluster、Oracle report；无≥2 mode或无Oracle gap→stop | 3–5 d CPU/GPU thumbnails | pass→P6；fail→champion only |
| P6 FilmCase+support | `routing/ranker.py`, `support/scalar.py` | rules/generic/transform-aware；E4 stress；policy hash/replay | 1–2 wk；小GPU/CPU | ranker不胜规则→规则；shrink不胜margin→margin |
| P7 BlueNeg pilot | `manifests/blueneg.py`, E2 configs；DoR=批准+metadata gate | roll-disjoint、shortcut/matched controls、credit记录 | 3–7 d；下载0.956GB | confounded→inconclusive/stop real-roll claim |
| P8 L3/L4 | residual/L-conditioned operators；DoR=L2 systematic residual | hidden/style gain+Jacobian+severe pass | 1–3 wk；0.5–3 GPU-days | 无增益→回滚L2 |
| P9 L5 optional | bilateral grid；DoR=全局稳定+局部失败证据 | local gain、halo/seam/flicker=0；stress pass | 1–2 wk；12GB边缘 | 任一 severe→删除分支 |
| P10 E6 final lockbox | 完整 policy/config/gates/commit 已冻结；DoR=P4–P9 决策结束 | 官方 628 只开放一次；输出 final paired metrics、full-res severe/style、coverage；任何后续变更建立新研究代际，不回看本 lockbox | 1–2 d；deterministic render+eval | confirm→final claim；不复现→fail/ambiguous，不调阈值补救 |

每 phase 使用 scoped commit；manifest/config/report hash 写入输出。失败时回滚到前一个可重放 operator/policy，不修改已冻结 gold 与 hidden evaluator。

## 15. 测试复现与硬件成本

### 15.1 资源预算（工程估计，不是厂商 benchmark）

| 工作 | RTX 5070 Ti Laptop 12GB | Apple M5 32GB | CPU fallback |
|---|---|---|---|
| L0–L2 / Gaussian OT / manifests | 非必要 | 很适合 | 很适合 |
| E0 affine/curve，48 seeds | 数小时 | 数小时 | 1–6 h量级 |
| L3/L4 per-roll optimisation | 2–8 h/配置；factorial 0.5–3 GPU-days | 纯 PyTorch MPS 可试 | 数十小时 |
| FilmSet classical/fixed | 非必要 | 可跑 | 4–12 h量级 |
| pooled/adaptive LUT | 6–24 h/模型 | prototype | 慢 |
| SepLUT reproduction | 256/512 crop、batch1、AMP、grad accumulation；12–24 h量级 | custom CUDA extension 需 fallback | 不建议训练 |
| NILUT/bake | 轻量 | 轻量 | 可行 |
| set encoder | 12–48 h；8–12GB | MPS prototype | 慢 |
| FilmCase ranker | 2–8 h + thumbnail precompute | 可行 | 小模型可行 |
| L5 bilateral | 12–48 h/候选 | 纯 Torch 可试 | 很慢 |
| full-res global LUT render | 秒/图量级 | 很快 | 可接受 |

训练只用 256–512 px crop/thumbnail；full resolution 只做 deterministic render/eval。CUDA 可用 AMP/gradient accumulation，但 Jacobian、inverse、bake 与 final LUT validation 保持 float32/必要处 float64。[PyTorch AMP](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html) M5 只假设纯 PyTorch MPS 路径；custom CUDA ops 不会自动兼容。[Apple MPS](https://developer.apple.com/metal/pytorch/)

### 15.2 复现命令建议

```bash
pytest -q tests/roll2film
python -m scripts.roll2film.audit_filmset --config configs/roll2film/data/filmset.yaml
python -m scripts.roll2film.e0 --config configs/roll2film/e0_fixed_budget.yaml
python -m scripts.roll2film.e1_train --config configs/roll2film/e1_pair_blind.yaml
python -m scripts.roll2film.e1_hidden_eval --frozen-run outputs/roll2film/e1/<hash>
python -m scripts.roll2film.build_bank --config configs/roll2film/e3_bank.yaml
python -m scripts.roll2film.evaluate_policy --config configs/roll2film/e6_gold.yaml
```

### 15.3 关键性能/精度边界

- 当前 early sRGB8 adapter 是科学与产品 blocker：它会隐藏 wide-gamut/HDR 行为、制造/掩盖 clipping 与 banding；在任何 L3/L4 结论前必须建立 high-precision path。
- global LUT 可以 pixel chunking；bilateral grid 必须整图预测低分辨率 field 后再 chunk apply，不能独立 tile。
- 目标设备 12GB 并不限制 global matrix/curve/LUT；真正风险是 spatial cross-attention、高分辨率 field 与无界 batch。
- CPU fallback 至少支持 L0–L2、fixed bank、FilmCase rules、3D LUT interpolation、support scalar 与所有 manifest/diagnostic。

## 16. Nearest work 与 novelty

### 16.1 最近工作矩阵

| 工作 | 年份/监督/group | 显式算子与实际迁移 | 与本项目重叠 / 剩余 gap | 威胁 |
|---|---|---|---|---|
| [Exposure](https://arxiv.org/abs/1709.09602) / [code](https://github.com/yuanming-hu/exposure) | TOG/SIGGRAPH 2018；unpaired source + target preference collection；无 roll subgroups | RL+GAN 学 exposure/WB/gamma/sat/tone/colour curves/contrast 的 white-box 序列，输出分辨率无关滤镜 | 已覆盖“target collection→explicit photo operator policy”；本项目只剩 shared roll operator、nuisance、support/safety 差异 | **Critical** |
| [Emulating Emulsion](https://musicofmusix.github.io/assets/misc/siggraph_abstract.pdf) | SIGGRAPH Poster 2025；一卷、3,168 controlled paired chart patches | 两个 3×3 matrix + 三个 4-param sigmoid，显式 scanner stage | 已覆盖“一卷真实胶片→紧凑物理 operator”；Roll2Film 仅在 target-only natural images 与无配对弱监督上不同 | **Critical** |
| [Identifiable UDT](https://arxiv.org/abs/2401.09671) | ICLR 2024；多组对应 cross-domain conditionals | 通用理论 | 单 marginal 有 automorphism；只有 target roll groups 不能直接借用其可辨识保证 | **Critical theory** |
| [CanonCGT](https://openaccess.thecvf.com/content/CVPR2026/papers/Ko_CanonCGT_Reference-Based_Color_Grading_via_Canonical_Pivot_Representation_CVPR_2026_paper.pdf) / [code](https://github.com/Jinwon-Ko/CanonCGT) | CVPR 2026；监督预训；对 unpaired photos 的不同 crops 加 synthetic grading perturbation 做 self-supervision；单 reference | canonicalization LUT + grading LUT | 占据 canonical pivot、explicit LUT、reference transfer；但不是从真实 target marginal 做 roll operator identification | **Very high** |
| [StatLUT](https://arxiv.org/html/2607.08227v1) | arXiv 2026-07；10k 已知 LUT 合成监督 | Lab statistics→Transformer residual 3D LUT | 直接威胁 statistics→LUT；patch-shuffle 必须作为 shortcut baseline | **Very high；未评审** |
| [cmKAN](https://openaccess.thecvf.com/content/ICCV2025/papers/Nikonorov_Color_Matching_Using_Hypernetwork-Based_Kolmogorov-Arnold_Networks_ICCV_2025_paper.pdf) / [code](https://github.com/gosha20777/cmKAN) | ICCV 2025；paired/unpaired distribution matching/instance fitting | KAN neural spline colour map，可加 spatial maps；**概念近邻，非自动符合 production operator allowlist** | 已占 unpaired nonlinear camera/colour mapping；无 roll/nuisance identifiability | **High** |
| [ModFlows](https://arxiv.org/abs/2503.19062) / [code](https://github.com/maria-larchenko/modflows)；[Nonlinear Color Transfer](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Nonlinear_Color_Transfer_via_Learnable_Bezier_Flows_CVPR_2026_paper.html) | AAAI 2025 / CVPR 2026；source/style distributions | neural RGB flow / Bézier trajectory+MoE；**概念近邻，须 bake/约束才可部署** | 占据 OT/flow、可逆非线性 hue trajectory；可逆不等于 semantic identifiability | **High** |
| [Image-Adaptive 3D LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) | TPAMI 2022；paired/adversarial unpaired | 小 CNN 混合 basis 3D LUT | adaptive explicit LUT 本身不新；无 roll shared inference | **High baseline** |
| [SepLUT](https://github.com/ImCharlesY/SepLUT) | ECCV 2022；paired enhancement | 1D + 3D LUT，分开 channel-independent/correlated | L2/L3 的直接强 baseline | **High baseline** |
| [NILUT](https://ojs.aaai.org/index.php/AAAI/article/view/27901) / [code](https://github.com/mv-lab/nilut) | AAAI 2024；多 LUT fitting/mixing | continuous implicit LUT，可 bake | implicit LUT 容量不新；缺部署级 monotone/inverse/safety guarantee | **Medium-high** |
| [SA-LUT](https://openaccess.thecvf.com/content/ICCV2025/papers/Gong_SA-LUT_Spatial_Adaptive_4D_Look-Up_Table_for_Photorealistic_Style_Transfer_ICCV_2025_paper.pdf) / [code](https://github.com/Ry3nG/SA-LUT) | ICCV 2025；reference-conditioned | style-guided 4D LUT + content/style context map | 占据 luminance/context-conditioned local LUT；无 grouped roll/identifiability | **High** |
| [Hist2Style](https://openaccess.thecvf.com/content/CVPR2026/papers/Galor_Hist2Style_Histogram-Guided_Stylization_with_Bilateral_Grids_CVPR_2026_paper.pdf) | CVPR 2026；teacher 用同一 prompt 产生跨内容 synthetic same-style variants；推理仍是单 histogram reference | monotone LUT + bilateral grid + uncertainty | 威胁 histogram style embedding、local operator 与 uncertainty；不是 grouped-observation inference | **High** |
| [INRetouch](https://openaccess.thecvf.com/content/WACV2026/papers/Elezabi_INRetouch_Context_Aware_Implicit_Neural_Representation_for_Photography_Retouching_WACV_2026_paper.pdf) | WACV 2026；one/multiple before-after pairs | 小 context-aware INR colour function；**概念近邻，非 allowlist 中的可检查 matrix/curve/LUT** | 覆盖 multiple-reference→operator；本项目差异是 target-only、无 source counterpart | **High** |
| [ChameleonTuner](https://openaccess.thecvf.com/content/WACV2026/papers/Tan_ChameleonTuner_Automatic_ISP_Color_Tuning_in_Subjective_Scenarios_WACV_2026_paper.pdf) / [code](https://github.com/ZjTan4/ChameleonTuner) | WACV 2026；多场景 source-target pairs | NSGA-II 拟合 17³ LUT | 多图 explicit LUT identification 的强 paired upper bound | **High** |
| [Parameterized Color Enhancement](https://openaccess.thecvf.com/content_WACV_2020/papers/Chai__Supervised_and_Unsupervised_Learning_of_Parameterized_Color_Enhancement_WACV_2020_paper.pdf) | WACV 2020；supervised/unpaired | global parameterized transform | “unpaired 但最终显式 transform”早有先例 | **High conceptual** |
| [Neural Statistician](https://arxiv.org/abs/1606.02185) / [ML-VAE](https://cdn.aaai.org/ojs/11867/11867-13-15395-1-2-20201228.pdf) | ICLR 2017 / AAAI 2018；sets/groups | shared-set/group latent | set encoder、shared/per-item factor inference 不新 | **High conceptual** |
| [Weakly-Supervised Disentanglement Without Compromises](https://proceedings.mlr.press/v119/locatello20a/locatello20a.pdf) | ICML 2020；non-i.i.d. pairs/groups 共享部分 latent factors | representation identifiability theory | 直接说明 grouped observations 可能有信息，但依赖 smooth invertible generator、factor-sharing 等强假设，不能自动迁移到 roll photos | **High theory** |
| [FUNIT](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Few-Shot_Unsupervised_Image-to-Image_Translation_ICCV_2019_paper.pdf) | ICCV 2019；multiple unseen target exemplars | 生成式 RGB translation | few-target set-conditioned unpaired transfer 已存在；本项目只在显式安全 operator 上不同 | **Medium-high** |
| [Learning Personalized Photographic Style](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_Learning_Personalized_Photographic_Style_from_Pairwise_User_Preferences_CVPR_2026_paper.pdf) | CVPR 2026；Transformer 聚合多 preference pairs/triplets | photographic-style embedding + implicit enhancement | 直接削弱宽泛 set-conditioned photographic style claim；其监督正是本项目禁止的新偏好数据 | **High conceptual** |
| [D-LUT](https://openaccess.thecvf.com/content/WACV2025/papers/Li_D-LUT_Photorealistic_Style_Transfer_via_Diffusion_Process_WACV_2025_paper.pdf) / [code](https://github.com/leemojiang/D-LUT) | WACV 2025；reference/unpaired score fitting | 训练时 diffusion/score，最后抽取 3D LUT | 不符合本项目推荐训练路线，但证明“生成/score过程预测最终 LUT”已有先例 | **Medium-high** |
| [Illuminant-aware gamut transfer](https://diglib.eg.org/items/ace39860-a6ac-4926-9351-7a6ca451eae6) | CGF 2014；single reference | WB + gamut-constrained colour transfer | safety/gamut 直接近邻；未覆盖 posterior source-support→champion shrinkage | **Medium safety** |
| [FilmNet / FilmSet](https://www.ijcai.org/proceedings/2023/0129.pdf) | IJCAI 2023；paired digital recipe targets | CNN RGB + LUT refinement | film-recipe dataset/upper bound；不是 real film identity | **Medium** |

### 16.2 Surviving、weakened 与失败 novelty

**Surviving（仍需实验支持，不得写 first）：**

> **Conditional statement（只有 E0/E2 gates 通过后才可使用）：** target-only、roll-grouped、content-diverse natural photos 作为 grouped-observation weak supervision；显式分解 per-frame photometric nuisance；每组估计一个可序列化 global colour operator；并用 matched-count pooled/shuffled controls 主动证伪。Posterior uncertainty、input-colour support 与 hard routing 是可能有价值的**系统组合**，当前不单独声称方法 novelty。

**当前安全标题：** `Canonical explicit colour-map estimation from grouped target observations`。  
**Gate 通过后才可用：** `Grouped-target unpaired explicit colour-operator identification under per-frame nuisance`。

**已被削弱：** `set-conditioned unpaired explicit colour transfer` 仍过宽；Exposure、set latent、FUNIT、StatLUT 与多参考 retouching 都覆盖其中部分。

**不可用：**

- “first unpaired explicit photo enhancement/operator learning”；
- “first set-conditioned colour/style transfer”；
- “first roll-level film operator model”；
- “film stock calibration from unpaired roll images”；
- “first statistical feature-to-LUT”。

### 16.3 Novelty failure conditions

任一满足，核心方法论文应转为工程系统/确定性产品路线：

- correct groups 不胜 matched-count random/pool；
- content-balanced bootstrap 下 operator 不稳定；
- nuisance 稍增即吸收 shared style；
- scanner/lab/uploader/date 对 operator 的解释力高于 roll signal；
- Exposure、StatLUT、fixed champion 在 matched style strength 下同等质量且 severe 更低；
- support shrink 不优于 simple gamut margin/global strength；
- FilmCase 不胜 no-neural selector/global champion；
- 没有 controlled pairs 却需要 named-stock claim。

## 17. 风险登记

| 风险 | 概率/影响 | 早期触发器 | 缓解与停止 |
|---|---|---|---|
| group 只有 sample-count 效应 | 高/致命科研 | fixed-N partition 无差异 | 关闭 roll-information claim；保留 pooled canonical map |
| scene palette 被记为 style | 高/高 | patch shuffle/scene resample 不变；bootstrap漂移 | conditional prior、matched content、report instability；不发布 |
| scanner/lab 与 stock/roll 混淆 | 高/致命 claim | scanner/date/location probe 很强 | 只称 source/scanner look；需要 crossed capture 才升级 |
| nuisance 吸收 style或反之 | 高/高 | nuisance capacity 改变 shared operator | gauge、known-truth recovery、固定 priors；失败即停 |
| expert bank 是 strength duplicates | 高/高 FilmCase | basic/strength-adjusted clustering=1 mode | 关闭 bank/router；champion+strength |
| selector 无 appeal supervision | 确定/高 | safe experts 间无法定真值 | 只学 safety/applicability；global aesthetic prior；不宣称 personalization |
| support shrink 洗淡风格 | 中高/高 | severe下降伴 style低于 floor | matched-strength comparison；champion fallback；报告 transfer failure |
| hard selector case-boundary跳变 | 中/中 | 小 EV/JPEG perturbation 切换 | margin、champion fallback、未来 hysteresis |
| red/neon speckle、posterization | 已发生/致命产品 | ID 11 09/56 | frozen worst-case、一票否决、gamut/support/chroma ablations |
| 8-bit adapter掩盖/制造 artifact | 高/高 | 16-bit parity失败、gradient banding | high-precision ingress/export 先于 L3/L4 |
| local residual引入 halo/seam/flicker | 高/高 | edge/tile/video stress | defer L5；任一 severe 删除分支 |
| FilmSet rights不等于代码MIT | 中高/高发布 | 无 per-file rights manifest | 内部研究与公开发布分离；release review |
| BlueNeg 官方 split roll泄漏 | 确定/高 | test roll出现在train | 完整 roll-disjoint resplit |
| hidden evaluator被调参污染 | 中/高 | 多次打开628 targets | 独立进程/权限；一次 freeze 后 eval；access log |
| 小 gold 得到虚假“0 severe” | 高/高 | n<300却声称<1% | rule-of-three、分层stress、写 inconclusive |
| 12GB OOM | 低（global）/中（local） | spatial model/crop过大 | global-first、AMP、batch1/accumulation；local延期 |

## 18. Go / stop / ambiguous

### GO

- attachment/manifest evidence freeze；
- WorkingImage high-precision colour-state contract与16-bit/profile-aware I/O；
- L0–L2 canonical operators、analytic inverse/Jacobian、LUT bake；
- fixed strong deterministic expert bank，允许最终独立获胜；
- E0 fixed-sample/coverage/nuisance falsification；
- FilmSet 4,657 内 content-disjoint paired-blind development、paired upper bound，以及完整 policy 冻结后的最终 628 lockbox；
- classical OT、Exposure、best-basic、no-neural selector 基线；
- diversity gate、Oracle gate、image-level support scalar；
- full-resolution policy-level severe evaluation。

### AMBIGUOUS / conditional GO

- Roll2Film per-roll MAP：当前只能称受限 canonical colour-map estimation；fixed-N/real-group gates 通过后才称 operator identification；
- hierarchical Bayes：需要多个可用 roll 与足够 metadata；
- set encoder：仅在 MAP/known-truth 通过后 amortize；
- L3/L4：逐级 gate；
- BlueNeg 8-bit pilot：no-data gates 通过且批准后；
- StatLUT：新 preprint challenger，先复现再判断。

### STOP / defer

- 用当前 E0 宣称 real-roll identifiability；
- target-only group 分离 stock/process/scanner；
- named-stock calibrated claim；
- soft expert averaging；
- identity fallback 美化论文成功率；
- 无标签的 appeal/personalization learning；
- SA-LUT/bilateral local 作为第一架构；
- 未 bake/Jacobian/gamut 验证就让 NILUT 神经函数直接进 production renderer；
- DigitalFilm/PhotoGAN/未发布 SillyStill 进入关键路径；
- 重新使用 diffusion/GAN 直接生成最终 RGB。

## 19. 下一步十项任务

1. **冻结证据包：** 记录 72/72 checksum、9×6 normalized manifest、ID 11 worst-case、E0 raw JSON 与本报告 hash。
2. **修正 tracker：** 全部运行态 `638→628`；保留 `paper_reported=638` 矛盾字段；更新 FilmSet 已在本机、`data/processed/manifest.jsonl` 已存在、film_domain=4,212 的 Windows 实况。
3. **实现 colour-state gate：** WorkingImage linear-sRGB/high-precision 到 operator/render/export 全程无隐式 sRGB8；冻结 ICC/16-bit/HDR fixtures。
4. **实现 L0–L2 protocol：** canonical JSON、inverse/Jacobian、33³/65³ bake、CPU/CUDA/MPS parity 与 property tests。
5. **重做 E0：** fixed total pixels、partition equivalence、independent support、strong nuisance/scanner、mixed/shuffled、flexible nuisance、prior swap、uncertainty calibration。
6. **冻结 FilmSet pair-blind manifest：** 21,140 files、四域 basename parity、duplicate clusters；4,657 内 source-only/target-only/dev 内容不相交；pHash/embedding leakage canaries；628 final evaluator 继续封存。
7. **建立 deterministic/style baselines：** anchors、best-basic、hist/OT/Bures/sliced、pooled LUT、Exposure filter sequence；所有比较先匹配 style strength。
8. **做 diversity+Oracle gate：** 删除 53/55/56 strength duplicates；若没有 ≥2 modes 或 Oracle 不胜 champion，立即关闭 FilmCase。
9. **做 E4 support pilot：** ID 11 与 frozen stress；比较 validated champion→expert parameter/velocity path、global strength、source/chroma margin；不先做 cell/local residual。
10. **BlueNeg metadata preflight：** roll-disjoint split、license/credit、stock/date/location shortcut report；只有 E0 通过后才申请 0.956GB preview lanes。

## 20. 需要人工批准的动作

| 动作 | 当前是否需要 | 批准前必须提供 |
|---|---|---|
| 使用本机 FilmSet 做内部 E1 | **不需新下载批准**；但 hidden access 要流程控制 | archive hash、rights caveat、pair-blind manifest、628 evaluator |
| 下载 BlueNeg preview+pseudoGT 约0.956GB | **需要明确批准** | no-data E0 gate、license snapshot、roll-disjoint plan、存储/删除计划 |
| 下载 BlueNeg 290GB 全量 | **不推荐；另行批准** | 为什么8-bit lanes不足、存储与计算预算 |
| 下载 HDR+ 37GB+ | **需要批准** | 不能用 synthetic/local数据完成的具体 falsification |
| 申请 FilmGrainStyle740k/商业许可 | **不推荐；邮件/付费需批准** | 独立 grain work package、license/预算 |
| 联系 SillyStill/Emulating Emulsion 作者索取数据 | **外部联络需批准** | 联系文本、用途、rights/retention、失败替代 |
| 自采 controlled digital+film chart/scene pairs | **新 scope/费用/参与者需批准** | capture design、consent/rights、stock/process/scanner crossing、预算 |
| 云 GPU / 大规模训练 | **需要批准** | 本地12GB失败证据、预计GPU-hours/费用、stop rule |
| 外部 blind human study | **需要批准** | IRB/consent适用性、样本/问题/排除/分析、图片发布权 |
| 冻结最终产品 severe budget `B_product` | **需要产品负责人批准** | failure taxonomy、目标 rate、CI 上界、样本量、fallback/coverage 成本 |
| 发布 weights、示例图或 dataset-derived artifacts | **逐数据源 release review** | FilmSet per-file rights、BlueNeg exact credit/作者确认、model card/lineage |

## 21. 一句话建议

**先把 fixed strong explicit experts、L0–L2、FilmSet content-disjoint paired-blind development、最终 628 lockbox、fixed-budget E0、diversity/Oracle gate 和 image-level support-path shrink 做成可独立获胜的确定性系统；只有正确 roll groups 在 matched-count、nuisance、scanner 与 support controls 下仍有稳定增益时，才继续把 Roll2Film 写成科研核心，否则关闭学习路线并由确定性 champion 获胜。**
