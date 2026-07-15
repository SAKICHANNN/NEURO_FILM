# Roll2Film / FilmCase Ultimate 架构深度科研提示词

> 使用方式：将本文件与 `ROLL2FILM_RESEARCH_HANDOFF_20260715.zip` 一起上传给网页端推理专家。要求专家先读取 ZIP 内 `PACKAGE_README.md`、`CHECKSUMS.csv` 和视觉附件，再开始回答。

你是一位同时熟悉计算摄影、色彩科学、胶片成像、无配对图像迁移、显式 3D LUT、最优传输、统计可辨识性、机器学习与科研工程的高级研究员。

请对下面的项目进行一次深入、敌对式、以可执行开发为目标的研究。你必须使用深度搜索，优先查找论文原文、官方项目页、官方代码仓库、官方数据集页面和数据 API。不要依赖搜索摘要、二手博客或未经验证的数据集名称。

最终请用中文回答，专业术语可以保留英文。所有外部关键事实必须附可点击的一手来源。不要为了支持既定方向而迎合我们；如果 Roll2Film 的核心假设不成立，请明确否定并给出更好的非生成式替代方案。

## 0. 附件读取契约

本提示词随附一个研究交接 ZIP。你必须先检查：

1. `visual_history/ALL_SCHEMES_NUMBERED.png`：3900×9480 的 56 方案历史总表；
2. `visual_history/NUMBER_MAP.csv`：编号到真实 scheme/manifest 的映射；
3. `normalized_anchors/`：9 张冻结输入上的 `01/09/53/55/56 + bland control` 共 54 张同图重渲染；
4. `red_highlight_counterfactual/`：ID 11 原图、09、56、56 chroma-margin challenger；
5. `roll2film_e0/`：CT1/E0 配置、原始 JSON 和证据报告；
6. `project_context/`：当前项目约束、tracker 和 Roll2Film 研究计划；
7. `CHECKSUMS.csv`：附件 SHA-256 与字节数。

如果 ZIP、视觉文件或映射无法读取，停止对历史偏好的视觉推断，明确列出缺失文件。不得仅根据编号或文字描述虚构风格结论。

## 1. 项目目标

我们正在开发非生成式胶片色彩迁移系统。真实产品标准是：

> 在没有严重 glitch/artifact 的前提下，尽量产生明显、强烈、具有辨识度且有吸引力的胶片风格。

不接受：只增加 saturation/contrast/WB；数据集平均化后的寡淡滤镜；指标提高但肉眼无明确风格；为安全退化到 identity；重画脸、文字、物体、纹理或几何；高光爆色、红色 speckle、紫色色块、posterization、banding、tile seam、halo、异常 clipping 或不稳定局部颜色。

## 2. 硬约束

1. 禁止 diffusion、GAN 或其他生成式模型直接生成最终 RGB。
2. 最终全分辨率图像必须由显式、可检查、可回放的算子产生：exposure/WB、单调曲线、3×3 matrix、1D/3D LUT、luminance-conditioned LUT，或严格受约束的低分辨率参数场。
3. 学习模型可以预测或选择 operator 参数，但不能直接输出最终 Style-safe RGB。
4. 不能依赖用户继续提供新照片、数字/胶片配对、per-image 标签、偏好投票或人工标注。
5. 公开在线数据只有在具体文件/API/archive 当前可获取时才能进入计划。
6. 付费数据、云 GPU、大型新下载和外部参与者实验需要单独批准；先执行 metadata-only/no-data falsification。
7. 目标硬件：RTX 5070 Ti Laptop 12GB、Apple M5 32GB、CPU fallback。
8. 无配对数据只能支持 `film-inspired`、`roll-look` 或 `film-recipe transfer`；命名 stock/process calibrated claim 必须有受控配对证据。
9. 允许 Roll2Film、FilmCase、固定确定性专家库或无神经网络方案获胜，也允许关闭整个学习路线。

## 3. 当前工程事实

```text
WorkingImage linear-sRGB input
  -> temporary sRGB8 compatibility adapter
  -> PIL 8-bit legacy renderer
  -> deterministic CIELAB mean/std safe_lab / safe-rich
  -> optional grain / halation / dust
  -> bounded 8-bit PNG
```

优点：不生成式重写内容；已有 deterministic effects、WorkingImage、ICC-aware raster helpers、基础 RAW decode 和 CPU-safe CI。

缺点：仍过早退回 sRGB8；没有完整 16-bit/wide-gamut/HDR/profile-preserving export；safe_lab 容易让不同 stock 相似；旧 Neural LUT/SepLUT/NILUT 多由 pseudo-teacher 蒸馏，常退化为 saturation/contrast/WB。

旧 SD/IP2P/SDXL/SDEdit 已因身份、细节和几何重写或 12GB OOM 被淘汰，不要重新建议同类最终 RGB 生成方案。

## 4. 当前候选 Ultimate 架构

```text
roll/look grouped target images
  -> Roll2Film shared explicit operator identification
  -> diverse strong operator bank
  -> FilmCase / transform-aware hard applicability ranker
  -> Top-1 operator, never default soft averaging
  -> input colour-support / uncertainty analysis
  -> support-aware shrinkage toward identity/global champion
  -> deterministic curves + matrix + 3D LUT render
  -> optional bounded local residual only after a proven global failure
  -> full-resolution severe-artifact veto
  -> low-confidence/OOD fallback
```

请判断这是不是合理分工：Roll2Film 负责产生不同专家，FilmCase 负责针对具体输入选择专家，support-aware layer 负责强风格下的颜色外推安全。

## 5. 历史视觉偏好：必须结合附件分析

用户从历史总表中选择 `53,55,56,33,09,03,02,01`。这些不是八个独立专家：

| 编号 | Scheme | 样本数 | 已知事实 |
|---|---|---:|---|
| 01 | `baseline_current` | 20 | strength 0.55、luma 0.35；历史示例 01 有 5.81% 新 clipping |
| 02 | `baseline_current_smoke` | 2 | 01 的 smoke subset，不是独立风格 |
| 03 | `baseline_current_smoke2` | 1 | 01 的单图 smoke，不是独立风格 |
| 09 | `baseline_noclip_s0p50` | 20 | strength 0.50、luma 0.25、source-gamut、margin 4；较保守 |
| 33 | `evaluate_color_pipeline_smoke` | 1 | 单图 smoke，不能证明稳定风格 |
| 53 | `velvia50_digital20_s0p72_gamutsafe` | 20 | strength 0.72；历史输入与 55/56 不同 |
| 55 | `velvia50_rawpixls20_s0p50_gamutsafe` | 20 | strength 0.50；与 56 同一 RAWPixls 家族 |
| 56 | `velvia50_rawpixls20_s0p58_gamutsafe` | 20 | strength 0.58；较强，ID 11 红色高光有 speckle |

冻结规则：`01/09/53/55/56` 是完整 preference anchors；`02/03/33` 仅 smoke cues。历史 53 与 55/56 输入不同，必须以 same-input normalized replay 为主要比较。

Normalized recipes：01=`s0.55,l0.35`；09=`s0.50,l0.25,source-gamut,margin4`；53=`s0.72,l0.35,source-gamut`；55=`s0.50,l0.35,source-gamut`；56=`s0.58,l0.35,source-gamut`；control=`safe-rich`；56 challenger=`s0.58,l0.35,chroma-gamut,margin4`。

已知 ID 11 trade-off：原 56 红色金属高光有明显高频 speckle；chroma+margin4 将 diagnostic candidate-pixel coverage 从约 0.314% 降至 0.223%，并保持 `[4,251]`/零新 hard clipping，但 mean chroma 从约 30.93 降至 21.46，可能洗掉风格。diagnostic 不能替代视觉裁决。

请实际检查附件并回答：共同视觉特征是什么；哪些超越 saturation/contrast/WB；五个完整锚点是否形成多个可感知 mode；哪些只是强度变化；差异是否由输入集造成；53/56 的强风格来自 hue trajectory、tone shape、cross-channel coupling、luminance-conditioned chroma 还是 saturation；09 牺牲了什么；ID 11 的 56 是合理强风格还是 severe artifact；challenger 是否过淡；如何转化为 style floor、saturation-only baseline、expert-diversity gate 和 operator capacity。

## 6. Roll2Film 假设与已完成 E0

中心假设：同一物理 roll/稳定处理链中的内容多样图像可作为 repeated-measure weak supervision，使共享 roll-look operator 比单张参考更可辨识。

已完成 CPU synthetic E0：orientation-preserving SPD affine truth；估计器只看独立 neutral prior 与无序 target frames，不看 pairs；group sizes 1/2/4/8/16/32；48 replicates；混合两个 operator 为 hostile control。

结果：无 nuisance correct RMSE `0.017172 -> 0.004224`，32-frame mixed `0.064019`；轻 nuisance `0.021609 -> 0.005464`，mixed `0.070120`。这只是 restricted SPD-affine synthetic pass，不证明真实 roll、非线性 LUT、per-photo adaptation、风格或 artifact safety。

必须审查：group 是否有特殊信息还是只有更多样本；operator 与 exposure/WB/content/scanner/lab/uploader 是否可分；等价类；neutral prior 来源；density/flow shortcut；何时只能称 source/scanner-specific look。

## 7. 当前 Windows 数据实况与冲突

2026-07-15 当前工作区实查：

- FilmSet 已完整存在：`21,140` files，`11,262,805,356` bytes；
- train 每个 `input/Cinema/ClassNeg/Velvia` 均 4,657；test 每个目录均 628；总 originals 5,285；
- 旧计划多处写 `638 test`，与当前文件及 `5285-4657=628` 冲突。请查明论文/官方数据真实 split，不得静默采用 638；
- `data/processed/manifest.jsonl` 当前存在；
- `data/film_domain` 当前有 4,212 JPG，仍受 lineage quarantine；文件存在不等于可训练；
- BlueNeg 当前未下载；
- FiveK 完整源不在本机，但 903-file、约 7.49GB freeze pack 存在；FiveK 不是 film identity；
- 没有受控真实同场景 digital/physical-film pair。

请将本机存在、远端可获取、科学可用、允许发布权重分开。

## 8. 数据集实时深搜要求

重新核查 FilmSet、BlueNeg、DigitalFilm_dataset、PhotoGAN C200、SillyStill CineStill pairs、Emulating Emulsion、FilmGrainStyle740k，并搜索其他当前确实可获取的数据：同 roll 多帧、roll/process/scanner metadata、真实 digital/film pairs、同数字原图多 recipe、受控 chart/exposure/scanner 数据。

每个数据集必须报告：官方 URL、direct/API/archive 状态、登录/申请/费用、大小、图像数、格式、metadata、grouping unit、pairs、license、训练/权重/示例发布许可、attribution、可支持与不可支持 claim、leakage/confound、最小 pilot、优先级。

只有 direct-download/API/file-tree/public archive verified 才能进入关键路径；request-only 不能成为依赖。网页存在不等于数据可下载，metadata license 标签不等于 release clearance。

## 9. 必须比较的非生成式路线

比较固定强专家库；histogram/quantile/Gaussian/Bures/sliced OT/canonical-pivot；pooled learned LUT、image-adaptive LUT、SepLUT、NILUT、SA-LUT、StatLUT；FilmCase generic/photometric retrieval 与 transform-aware hard ranker；Roll2Film per-roll optimisation、hierarchical Bayesian inference、permutation-invariant set encoder；Roll2Film+FilmCase hybrid；support-aware shrinkage；bounded bilateral grid/local residual；FilmSet paired supervised upper bound；无神经网络 selector。

对每条路线估计明显风格概率、saturation-only 风险、严重 artifact 风险、per-photo 适配、数据要求、解释性、12GB 可行性、论文/产品价值和最可能失败模式。

## 10. 数学 formulation 与 operator ladder

给出 shared roll operator、per-frame nuisance、neutral/content prior、uncertainty、support shrinkage 的数学 formulation；明确 identifiability assumptions、impossibility boundaries、equivalence classes、如何防止 nuisance 吸收 style 和 shared operator 记忆 scene/scanner。若用 density/flow，说明它不生成最终 RGB并设计 shortcut tests。

容量阶梯至少包括：0 exposure/WB；1 orientation-preserving matrix+bias；2 matrix+strictly monotone splines；3 smooth/invertible 3D LUT residual；4 luminance-conditioned LUT；5 bounded low-resolution bilateral-grid residual。每级给参数化、forward/inverse、Jacobian/gamut/serialization/precision/regularisation/拟合、进入和停止条件。

## 11. Style-not-saturation 与 severe artifact

必须有 identity、exposure-only、WB-only、saturation-only、contrast-only、WB+saturation+contrast、Lab mean/std、histogram、single-reference OT、pooled operator、shuffled groups、same-film wrong-roll、fixed preferred anchor、generic retrieval 和 paired upper bound。

评估 hue trajectory、cross-channel coupling、neutral axis、toe/midtone/shoulder、luminance-conditioned chroma、colour-volume displacement、operator diversity、distance from identity/saturation、与附件锚点关系。匹配 style strength 后比较 artifact；设置 style floor、identity-collapse stop 和 saturation adversarial test。

严重 artifact 一票否决：face/text/object/geometry corruption、posterization、banding、large clipping、gamut collapse、red/neon speckle、colour blocks、tile seams、halo、unstable local colour、未来 flicker。设计 frozen gold/stress、diagnostics、三次盲化全分辨率裁决、ambiguous fail-closed、worst-case gallery、rate/CI。自动指标不能单独晋级或否决，评估最终完整 selection policy。

## 12. 实验 DAG

每个实验给 hypothesis、baseline、variable、controls、data、split、metrics、预注册 gate、pass/fail/ambiguous、next/stop、compute、artifacts：

- E0：扩展 matrix/curves/LUT/luminance-conditioning、support breadth、强 exposure/WB/scene/scanner nuisance、mixed/shuffled/flexible-nuisance absorption、uncertainty；
- E1：使用本机 FilmSet 做 paired-blind unpaired transfer；训练 loader 禁止 pair ID，628 aligned test 仅 hidden evaluation；区分 group information 与 sample-count；
- E2：BlueNeg metadata-only preflight 和最小 preview/pseudo-GT correct-roll matched-control；
- E3：operator bank+FilmCase，generic/photometric/transform-aware/Oracle；
- E4：support shrinkage，匹配 style strength 测红高光/饱和物；
- E5：只有全局稳定局部失败才允许 local residual；
- E6：hidden style/appeal/severe/content/product evaluation。

FilmSet pair-blinding tests 必须保证训练对象不同时暴露 input 与对应 target 的 shared ID/path/order；group leakage 为零；paired supervised 仅 upper bound。

## 13. 工程开发方案

转化为可执行结构：`src/roll2film/{operators,simulator,identification,routing,support,evaluation,manifests}`、scripts/configs/tests/docs/outputs。给出模块职责、operator/roll/case/manifest/provenance/report schema、unit/property/integration tests、seed/config/hash、checkpoint/resume、no-pair/group leakage、visual workflow、commit/rollback。每 phase 给文件、DoR、DoD、验证命令、耗时、资源、批准 gate 和下一分支。

## 14. Novelty 深搜

检索 unpaired colour transfer from grouped observations、repeated-measure weak supervision、set-conditioned enhancement、set-to-parameter inference、grouped domain translation、roll-level film modelling、multi-image camera pipeline identification、scanner/process identification、unpaired LUT estimation、OT map identifiability、support-aware colour transfer、hard expert selection。

制作 nearest-work matrix：论文、年份/会议、监督、group、显式算子、实际迁移、identifiability/nuisance/real-film、重叠、剩余 gap、威胁等级。不得轻率声称 first。给 surviving/weakened novelty statement、failure condition，并判断是否应改称 `set-conditioned unpaired explicit colour transfer`。

## 15. 最终输出结构

1. Executive verdict；2. 附件视觉分析；3. Roll2Film 支持/反对证据；4. 推荐架构和不推荐路线；5. 数学与 identifiability；6. operator ladder；7. Roll2Film/FilmCase/hybrid/deterministic 对比；8. 可获取数据和实时 license/access；9. FilmSet 628/638 冲突；10. 最小数据计划；11. 实验 DAG；12. style-not-saturation；13. severe evaluation；14. 工程 API/tracker；15. 测试复现/硬件成本；16. nearest-work/novelty；17. 风险；18. go/stop/ambiguous；19. 下一步十项任务；20. 人工批准动作；21. 一句话建议。

## 16. 回答纪律

区分 Fact、Inference、Assumption、Recommendation。外部当前事实实时验证。不得把 FilmSet recipe 当真实胶片、BlueNeg restoration 当 digital-to-film pair、E0 pass 当完整 Roll2Film 成立、更高 saturation 当风格成功、数值可逆当 semantic safety、candidate safety 当 policy safety。证据不足写 unknown/inconclusive/blocked。输出必须足够具体，让工程团队可直接据此写代码、测试和实验配置。
