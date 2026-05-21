# Yao 论文配图逻辑与本文配图规定

本文不直接复刻 Yao 等人的齿轮算例配图，而是复用其力学论文的图件组织方式：先用少量高信息密度示意图建立方法可信度，再在数值实验部分用工况图、场图、时程曲线和误差曲线构成验证链条。

## 1. Yao 论文配图结构审计

Yao 论文共 22 个主图。图件按功能可分为四类。

### 1.1 问题与理论图

| Yao 图号 | 位置 | 图件功能 | 对本文的启发 |
| --- | --- | --- | --- |
| Fig. 1 | 引言/第 2 节开头 | 移动接触界面、潜在载荷和活动载荷随时间变化的概念图 | 本文第一张图必须先交代“接触区沿表面移动、活动补丁随时间变化”的问题，而不是先放算法流程 |
| Fig. 2 | FFR 建模 | 坐标系、刚体运动、弹性变形的几何定义 | 本文需要一张降阶坐标与全阶位移场关系图，但不要重复 FFR 基础，应突出主动态状态与 active patch ILC 的分离 |
| Fig. 3 | free-interface CMS | 模态集合、低频正常模态、高频准静态贡献和 residual modes 的构造 | 本文应有一张“低阶模态 + patch residual correction”的机制图，明确 residual mode 的来源和创新边界 |

### 1.2 算法流程图

| Yao 图号 | 位置 | 图件功能 | 对本文的启发 |
| --- | --- | --- | --- |
| Fig. 4 | 计算流程 | offline preparation、online prediction、recovery 三阶段流程 | 本文需要一张 offline/online 图，但必须避免普通流程图，应把 SDF query、patch activation、alpha projection 和 DAE dimension invariant 放到同一叙事中 |
| Fig. 5 | 接触算法 | 接触对检测、接触边界检测、接触力计算 | 本文需要一张 SDF 到 patch load basis 的局部机制图，替代齿面接触对检测图 |

### 1.3 基准验证图

| Yao 图号 | 位置 | 图件功能 | 对本文的启发 |
| --- | --- | --- | --- |
| Fig. 6 | 数值算例开头 | 工程 FE 模型总览和局部接触放大图 | 本文数值实验开头应放统一 benchmark geometry 图，显示梁/块体、复杂三角网格表面、接触路径和观测量 |
| Fig. 7 | 静态完备性 | 工况示意 + FE 与 proposed 的 von Mises 应力和位移场对比 | 本文必须把静态完备性从纯表格升级为“力-位移曲线 + 真实连续体场图/误差云图” |
| Fig. 8 | 截断准则 | 激励示意 + 多节点频谱对比 | 本文不应设置同等主图；模态截断不是本文创新点，放补充材料或一句说明即可 |
| Fig. 9 | 接触算法验证 | 接触工况示意 + FE/proposed 应力场对比 | 本文需要对应的 CalculiX/SFC 接触场图，重点是非线性接触 pressure-overclosure 与应力场一致性 |

### 1.4 动力学、性能和应用图

| Yao 图号 | 位置 | 图件功能 | 对本文的启发 |
| --- | --- | --- | --- |
| Fig. 10 | 动力学验证 1 | 接触力、角速度、传动误差三类全局时间历程 | 本文的动力学三方验证应画位移/接触力/误差随时间，而不是只画一个误差指标 |
| Fig. 11-12 | 动力学验证 1 | 根部/接触点 von Mises 应力与位移时程，含局部节点 inset | 本文应补真实连续体应力时程或关键点应力时程，避免只给位移三方验证 |
| Fig. 13 | 计算效率 | residual vector 与求解方程耗时对比 | 本文可以保留性能图，但按用户要求不用柱状图，改为 scaling curve、Pareto curve 或 runtime-vs-accuracy 曲线 |
| Fig. 14-16 | 高速动力学验证 | 全局量 + 局部应力，比较不同模态集合 | 本文不需要照搬“高速”结构；对应的是复杂移动接触和三方接触验证中的多工况对比 |
| Fig. 17 | 模态形状 | 典型自由界面模态云图 | 本文可放补充材料；正文只在方法图中示意 residual basis，不把模态形状作为主证据 |
| Fig. 18-22 | 工程应用/TWV | 时域、频谱、波形传播和模态贡献 | 本文若要有应用级压轴图，应替换为复杂三角网格表面移动接触或真实 SFC 接入算例，而不是再做抽象 toy case |

## 2. 本文主图总体规定

前一版 8 张主图过于保守，不符合 Yao 论文的图件密度。Yao 论文正文有 22 个主图，且数值算例部分占据绝大多数图件。本文应在数量和功能上参照这一结构：方法部分用 5 张图建立视觉语言，算例部分用 17 张图形成密集证据链。

正文主图数量规定为 **22 张**：

- Figure 1-5：问题、理论和算法图，对应 Yao Fig. 1-5。
- Figure 6-9：模型、静态完备性、模态/残差基和 SDF 映射验证，对应 Yao Fig. 6-9。
- Figure 10-14：准静态接触、移动接触、复杂表面和真实连续体场图，对应 Yao 的静态/接触验证图。
- Figure 15-17：线性动力学、非线性接触和关键点应力/位移时程，对应 Yao Fig. 10-12。
- Figure 18-19：性能和精度-成本曲线，对应 Yao Fig. 13，但不用柱状图。
- Figure 20-22：扩展工况、模态/残差形状和工程级应用演示，对应 Yao Fig. 14-22 的应用压轴。

正文不设置参数敏感性分析图。性能图不使用柱状图，统一采用 scaling curve、runtime-accuracy Pareto curve、stacked line 或 cumulative curve。曲线图参数点不少于 10 个。真实连续体应力/应变图必须由 SFC 或 CalculiX 后处理得到，当前 surrogate 云图只能作为占位或设计稿。

## 3. 逐章节配图规定

### 摘要图/Graphical Abstract

**图件：Graphical abstract 或 Figure 1a 的简化版。**

内容必须包含四个视觉元素：moving contact samples、SDF nearest-triangle query、active patch set、low-mode dynamics plus residual ILC recovery。该图只讲核心创新，不放数值曲线。

### 1 引言

**Figure 1: moving contact problem and active patch concept.**

建议面板：

- a. 全阶柔性体移动接触问题：接触区沿外表面移动。
- b. 潜在接触 patch 与当前 active patch 的差异。
- c. 低阶 ROM 在局部接触柔度上的缺失。
- d. 本文思想：SDF samples -> active patch ILC -> residual recovery。
- e. 本文验证链：static completeness -> SDF projection -> quasistatic contact -> moving contact -> true continuum fields -> three-way validation -> scaling。

作用：对应 Yao Fig. 1。第一张图必须先讲清楚问题，而不是先讲算法。

**Figure 2: reduced coordinates and DAE state separation.**

建议面板：

- a. 全阶位移场、低阶模态坐标和局部接触修正之间的关系。
- b. 主动态状态向量不含 active alpha。
- c. 活动接触区变化时，alpha 集合变化但主 DAE 维度不变。
- d. 与 node-level contact DOFs 和 ordinary patch static modes 的维度对比曲线。

作用：对应 Yao Fig. 2。本文不重复 FFR 基础，而是把“主动态状态与 active ILC 分离”画清楚。

### 2 方法：SDF 驱动的补丁剩余 ILC 公式

**Figure 3: patch residual ILC basis construction.**

建议面板：

- a. retained low modes 表示整体动力学。
- b. patch load basis 表示局部接触载荷。
- c. full static attachment response。
- d. retained-mode static contribution。
- e. residual mode = full static attachment response - retained-mode static contribution。
- f. reconstructed response 与 full static response 闭合。

作用：对应 Yao Fig. 2-3，但要明确创新边界：residual mode 数学定义来自 free-interface CMS，本文新增的是 SDF/area-weighted patch load basis 和 active patch coupling。

### 3 运动方程与主 DAE 维度不变性

**Figure 4: offline-online-recovery workflow.**

建议面板：

- a. Offline: mesh/surface extraction、patch hierarchy、patch load bases、residual basis。
- b. Online: SDF query、active patch detection、alpha projection、ROM step。
- c. Recovery: low-mode displacement + active residual correction。
- d. Contact loop: contact force update 与 DAE integration 的交互。
- e. 不同阶段的主要计算量归属。

必须报告的指标：

- dynamic dimension contribution = 0。
- active patch alpha 不进入主动态状态。
- residual dynamic DOF ratio vs ordinary patch static modes = 0.0769231。

作用：对应 Yao Fig. 4。该图是方法实现的主流程图，但要避免画成软件流程图，必须保留力学量和矩阵量。

### 4 SDF 投影与多尺度 active patch 管理

**Figure 5: SDF projection and multiscale active patch contact algorithm.**

建议面板：

- a. SDF 查询几何：query point、closest triangle、gap、normal。
- b. 三角形到 patch 的 primary assignment 与 overlap weights。
- c. coarse/medium/fine patch hierarchy。
- d. active set hysteresis、delayed deactivation、warm-start alpha 的时间线。
- e. area-weighted pressure/penetration 到 patch normal load 的单位一致性。
- f. single patch、two-patch boundary、outside contact 三个最小映射算例。

作用：对应 Yao Fig. 5，但将齿面对检测换成本文自己的 SDF/patch 映射机制。该图必须说明单位和面积一致性，避免审稿人质疑 penalty 与 pressure-overclosure 不对应。

### 5 接触处理与 CalculiX penalty 标定

**不单独放方法主图，但必须在算例 Figure 16 中出现。**

本节不把 penalty 标定画成独立方法图；但在非线性接触三方验证图中必须出现：

- CalculiX pressure-overclosure 曲线。
- Python area-weighted penalty 对应线。
- 穿透量/接触压力定义示意。

原因：力学期刊正文可以接受接触参数标定作为三方验证证据的一部分，但不应把实现细节膨胀成独立方法主图。

### 6 数值实验

数值实验必须形成 Yao 式链条：模型图 -> 工况图 -> 场图 -> 时程曲线 -> 误差/效率证据。第 6 节应是全文配图最密集的部分，至少 17 张主图。

#### 6.0 算例模型总览

**Figure 6: benchmark models and observation definitions.**

建议面板：

- a. 基础悬臂块体/梁模型：约束、载荷、观测节点。
- b. 规则表面移动接触路径和 patch 划分。
- c. 复杂三角网格表面模型、warped surface、contact band。
- d. CalculiX 非线性接触模型：rigid plane/contact pair、接触面、载荷和边界。
- e. SFC 接入目标模型或真实连续体模型。

作用：对应 Yao Fig. 6。算例部分必须先给模型总览，否则后面的曲线和云图缺乏力学背景。

#### 6.1 静态完备性

**Figure 7: static completeness under patch loads.**

建议面板：

- a. 静态 patch load benchmark 工况。
- b. full static response 云图。
- c. low-mode-only error 云图。
- d. patch residual reconstructed error 云图。
- e. reconstruction error over all patch loads 曲线。
- f. completeness error distribution 曲线或累计分布。

作用：对应 Yao Fig. 7。不能只给表，必须用场图和误差曲线证明静态完备性。

**Figure 8: retained modes and residual response shapes.**

建议面板：

- a. 前几阶 retained low modes 形状。
- b. 典型 patch residual mode 形状。
- c. residual response 的局部化程度曲线。
- d. retained mode count 与 static reconstruction error 曲线。

作用：对应 Yao Fig. 8 和 Fig. 17。虽然模态截断不是创新点，但读者需要看到 retained modes 与 residual modes 的物理形态。

#### 6.2 SDF 映射验证

**Figure 9: SDF-to-patch load conservation tests.**

建议面板：

- a. single-patch contact sample 映射。
- b. two-patch boundary sample 映射。
- c. outside/no-contact sample。
- d. total force error、normal force error、moment error 曲线。
- e. alpha dimension 和 dynamic dimension contribution。

作用：对应 Yao Fig. 5 的接触算法验证，但本文要把 SDF 映射是否守恒画成独立证据图。

#### 6.3 准静态接触

**Figure 10: quasistatic indentation and force-displacement validation.**

建议面板：

- a. 压入模型图：indenter/contact patch、约束、观测位移。
- b. force-displacement 曲线：full FEM、low modes、ordinary patch static modes、patch residual ILC。
- c. contact pressure/penetration 分布。
- d. compliance error 随压入量变化曲线。
- e. active alpha 或 active patch count 随压入量变化。

作用：对应 Yao Fig. 7 和 Fig. 9 的静态/接触验证。该图证明 patch residual ILC 恢复接触刚度。

#### 6.4 移动接触与复杂三角网格表面

**Figure 11: moving contact on regular patch boundaries.**

建议面板：

- a. 规则表面接触路径和 patch boundary crossing 示意。
- b. projected normal force、gap 或 alpha jump 随步数变化。
- c. active patch count 随步数变化。
- d. contact force jump、gap jump、energy drift 曲线。
- e. alpha warm-start 或 hysteresis 状态随时间变化。

作用：对应 Yao 动力学图中的全局量时程。该图专门证明跨 patch 边界不会引入不可接受跳变。

**Figure 12: moving contact on a complex triangular surface.**

建议面板：

- a. warped triangular surface 模型图。
- b. contact band、triangle normals 和 normal variation 可视化。
- c. contact path snapshots，不少于 5 个时刻。
- d. coarse-only vs multiscale active ILC projection error 曲线。
- e. active patch fraction、unique requested patches、contact-band triangle count 曲线。

作用：对应 Yao 工程模型图和动态应用图。该图是本文区别于 Yao 齿轮接触背景的重要证据。

#### 6.5 真实连续体应力/应变场

**Figure 13: true continuum strain and stress fields.**

这是后续必须补强的主图，不能继续使用弹簧格架 surrogate 作为最终投稿主证据。

建议面板：

- a. SFC/CalculiX full FEM equivalent strain cloud。
- b. SFC/CalculiX full FEM von Mises stress cloud。
- c. adaptive ROM recovered equivalent strain cloud。
- d. adaptive ROM recovered von Mises stress cloud。
- e. contact pressure 或 penetration cloud。
- f. full FEM 与 ROM 场图差异的局部放大。

作用：对应 Yao Fig. 7 和 Fig. 9 中 FE/proposed 应力场对比，是本文从“算法验证”进入“力学可信度”的关键图。

**Figure 14: continuum error fields and stress-path comparison.**

建议面板：

- a. low-mode ROM displacement error cloud。
- b. patch residual ILC displacement error cloud。
- c. low-mode stress error cloud。
- d. patch residual ILC stress error cloud。
- e. through-contact-line stress comparison 曲线。
- f. path-wise relative error 曲线。

硬性要求：

- 云图来自真实连续体单元应力/应变后处理，优先 SFC 接入后的 full FEM；CalculiX 作为外部对照。
- error cloud 使用共同色标。
- 三维云图连续渲染，不用离散块状截图作为最终主图。
- 若只能从 CalculiX 获得积分点/节点应力，应明确说明投影/平均方式。

#### 6.6 CalculiX 三方验证

**Figure 15: linear dynamic three-way validation.**

建议面板：

- a. 线性动力学工况示意：悬臂 C3D8 块体、固定端、分布载荷、观测节点。
- b. 线性动力学位移时程：Python/SFC full FEM、adaptive ROM、CalculiX full FEM。
- c. 线性动力学累计相对误差曲线。
- d. velocity 或 energy balance 曲线。
- e. observed nodes 的误差分布曲线。

作用：对应 Yao Fig. 10。当前 Nature-style Fig. 3 可作为该图初版，但还要补模型图和更多误差曲线。

**Figure 16: nonlinear contact three-way validation with CalculiX contact active.**

建议面板：

- a. 非线性接触工况示意：rigid plane/contact pair、接触面、约束、载荷。
- b. Python/SFC full FEM、adaptive ROM、CalculiX full FEM 位移时程。
- c. contact force time history 三方对比。
- d. penetration/gap/contact pressure time history。
- e. pressure-overclosure slope 标定曲线。
- f. contact activity map 或 contact status snapshots。

作用：对应 Yao Fig. 9、Fig. 11 和 Fig. 12。该图必须明确 CalculiX 非线性接触参与三方验证，而不是只用 CalculiX 矩阵。

**Figure 17: local stress and displacement histories at selected points.**

建议面板：

- a. root-like observation point 和 contact point 的位置示意。
- b. contact point von Mises stress time history。
- c. non-contact/root point von Mises stress time history。
- d. contact point displacement time history。
- e. root point displacement time history。
- f. 每个观测点的 ROM/CalculiX relative error 曲线。

作用：直接参照 Yao Fig. 11-12。不能只画整体位移，必须给局部应力/位移时程以证明接触局部响应可信。

#### 6.7 性能与 scaling estimate

**Figure 18: SDF and active-patch computational scaling.**

建议面板：

- a. SDF scalar exact query vs batched exact query wall time 曲线。
- b. runtime vs query/contact points 曲线，至少 10 个点。
- c. runtime vs active patch fraction 曲线。
- d. SDF query、projection、residual recovery 的 per-step cost curves。

作用：对应 Yao Fig. 13 的耗时比较，但不用柱状图。

**Figure 19: accuracy-preserving runtime-DOF Pareto comparison.**

建议面板：

- a. accuracy vs runtime Pareto curve：low modes、ordinary patch static modes、patch residual ILC、node-level contact。
- b. dynamic DOFs vs relative error 曲线。
- c. active alpha DOFs vs projection/contact error 曲线。
- d. memory proxy 或 solver dimension proxy vs model scale 曲线。

硬性要求：

- 不使用柱状图，替代 Yao Fig. 13 的柱状耗时图。
- 性能结论必须绑定误差约束，不能只说快。
- 报告中位加速、最大加速、误差上限和测试硬件。

#### 6.8 扩展工况和应用级演示

**Figure 20: extended moving-contact case under longer trajectory or higher speed.**

建议面板：

- a. 扩展工况模型图。
- b. contact force time history。
- c. displacement time history。
- d. stress/strain peak time history。
- e. cumulative error and energy drift。

作用：对应 Yao Fig. 14-16 的第二动力学算例。本文需要一个不同于基本 moving contact 的扩展工况，证明方法不是单一 toy case。

**Figure 21: mode and residual contribution interpretation.**

建议面板：

- a. retained low-mode contribution。
- b. patch residual contribution。
- c. combined recovered response。
- d. mode/residual contribution coefficients over time。
- e. selected frames showing residual localization moving with the contact zone。

作用：对应 Yao Fig. 17 和 Fig. 21-22。它不是创新主证据，但能解释为什么方法能恢复局部响应。

**Figure 22: final application demonstrator with SFC integration.**

建议面板：

- a. SFC 接入后的工程级模型图。
- b. contact path snapshots，不少于 5 个时刻。
- c. full FEM vs adaptive ROM displacement/stress history。
- d. contact pressure/penetration snapshots。
- e. runtime/accuracy summary curve。
- f. final validation map：哪些证据来自 SFC，哪些来自 CalculiX，哪些来自内部 ROM。

作用：对应 Yao Fig. 18-22 的压轴应用图。该图用于证明本文方法可以进入后续 SFC 项目，而不只是当前仓库里的简化算例。

### 7 讨论与结论

正文结论不新增图。若需要图，只引用 Figure 1 的 validation ladder，并在结论中逐项回扣每张主图支持的 claim。

## 4. 补充材料配图规定

以下内容建议进入 Supplementary Figures，而不是正文主图：

- S1: 更多 retained low modes 和 patch residual modes。
- S2: 更细 retained-mode 数量检查。
- S3: pressure-overclosure penalty 标定细节。
- S4: 更多观测节点的三方动力学时程。
- S5: 更多接触路径和复杂表面 case。
- S6: 生成图件的 mesh、source data 和后处理流程。
- S7: 更多硬件/网格规模下的性能曲线。

## 5. 当前已有图件归位

| 已有文件 | 当前状态 | 正文归位 |
| --- | --- | --- |
| `paper/nature_figures/fig1_validation_curves.*` | 初版组合图，内容混合静态、移动接触和性能 | 拆分到 Figure 10、Figure 11、Figure 12、Figure 18 |
| `paper/nature_figures/fig2_continuous_field_clouds.*` | 预期图/占位图，非最终真实连续体证据 | 被 Figure 13 和 Figure 14 替换，必须改为 SFC/CalculiX 后处理 |
| `paper/nature_figures/fig3_external_three_way_curves.*` | 线性动力学三方验证图 | 归入 Figure 15，但需补模型图、更多观测节点和误差曲线 |
| `results/paper_patch_ilc/figures/three_way_nonlinear_contact_displacement.png` | 已有非线性接触位移证据 | 归入 Figure 16 |
| `results/paper_patch_ilc/figures/three_way_nonlinear_contact_activity.png` | 已有接触活动证据 | 归入 Figure 16，证明 CalculiX 非线性接触参与验证 |
| `results/paper_patch_ilc/figures/complex_surface_moving_contact.png` | 已有复杂表面证据 | 归入 Figure 12 |
| `results/paper_patch_ilc/figures/active_patch_history.png` | 规则移动接触证据 | 归入 Figure 11 |
| `results/paper_patch_ilc/figures/sdf_patch_projection_map.png` | SDF 映射证据 | 归入 Figure 9 |
| `results/paper_patch_ilc/figures/patch_hierarchy.png` | patch hierarchy 证据 | 归入 Figure 5 |
| `results/paper_patch_ilc/figures/performance_optimization.png` 和 `runtime_scaling.png` | 性能证据初版 | 改造成 Figure 18 和 Figure 19 的曲线图 |

## 6. 最终 22 张主图清单

| 主图 | 对应 Yao 图件功能 | 章节 | 核心 claim | 必须包含的证据 |
| --- | --- | --- | --- | --- |
| Figure 1 | Fig. 1 问题图 | 引言 | 移动接触导致活动界面载荷位置变化 | moving contact、potential patches、active patches、validation ladder |
| Figure 2 | Fig. 2 坐标图 | 方法 | 主动态状态与 active ILC 分离 | reduced coordinates、alpha outside DAE、DOF curves |
| Figure 3 | Fig. 3 residual modes | 方法 | patch residual ILC 恢复局部静态响应 | low modes、patch load basis、static attachment、residual response |
| Figure 4 | Fig. 4 workflow | 方法 | 方法可分为 offline-online-recovery 三阶段 | offline basis、online SDF、ROM step、recovery loop |
| Figure 5 | Fig. 5 contact algorithm | 方法/接触 | SDF 采样可映射到面积加权 active patch ILC | SDF query、patch hierarchy、overlap weights、minimal tests |
| Figure 6 | Fig. 6 FE model | 数值实验 | 所有算例有清晰模型、边界和观测量 | model library、contact paths、observed nodes、CalculiX contact model |
| Figure 7 | Fig. 7 static field validation | 6.1 | 静态完备性成立 | full static field、low error cloud、residual error cloud、error curves |
| Figure 8 | Fig. 8/17 modes | 6.1 | retained modes 与 residual modes 具有明确物理形态 | mode shapes、residual shapes、mode-count convergence |
| Figure 9 | Fig. 5/9 contact algorithm validation | 6.2 | SDF-to-patch 映射守恒 | single/two/outside cases、force/moment errors、dynamic contribution |
| Figure 10 | Fig. 7/9 quasistatic contact | 6.3 | patch residual ILC 恢复准静态接触刚度 | indentation model、force-displacement、pressure/penetration、compliance error |
| Figure 11 | Fig. 10 global dynamic histories | 6.4 | 规则移动接触跨 patch 边界保持连续 | force/gap/active patch/alpha/energy curves |
| Figure 12 | Fig. 14-16 extended contact | 6.4 | 复杂三角网格表面上多尺度 active patch 仍降低误差 | warped mesh、contact snapshots、coarse vs multiscale error、active fraction |
| Figure 13 | Fig. 7/9 stress fields | 6.5 | 真实连续体应力/应变场支持方法可信度 | SFC/CalculiX strain、stress、ROM recovered fields、pressure cloud |
| Figure 14 | Fig. 11-12 local error fields | 6.5 | patch residual ILC 降低空间误差 | displacement/stress error clouds、line stress comparison、path error |
| Figure 15 | Fig. 10 linear dynamics | 6.6 | 线性动力学三方结果一致 | model diagram、Ux/Uy/Uz histories、cumulative errors、energy |
| Figure 16 | Fig. 9/11/12 nonlinear contact | 6.6 | CalculiX 非线性接触参与三方验证 | nonlinear contact model、displacement/force/penetration、pressure-overclosure、contact activity |
| Figure 17 | Fig. 11-12 local histories | 6.6 | 局部应力和位移时程也与 full FEM/CalculiX 对齐 | contact/root point stress、displacement histories、local errors |
| Figure 18 | Fig. 13 cost | 6.7 | SDF 和 active patch 计算具有可量化 scaling 优势 | SDF batch curves、runtime vs points、cost breakdown curves |
| Figure 19 | Fig. 13 efficiency summary | 6.7 | 精度约束下本文方法位于更优 runtime-accuracy 区域 | Pareto curves、DOF-error curves、memory/runtime proxy |
| Figure 20 | Fig. 14-16 second dynamic case | 6.8 | 扩展移动接触工况下方法仍稳定 | extended model、contact force、stress peak、energy/cumulative error |
| Figure 21 | Fig. 17/21/22 modal interpretation | 6.8 | residual contribution 随接触区移动而局部化 | retained/residual/combined fields、coefficient histories、moving residual snapshots |
| Figure 22 | Fig. 18-22 engineering application | 6.8 | 方法可进入 SFC 工程级算例 | SFC model、contact snapshots、full/ROM histories、pressure snapshots、runtime-accuracy summary |

