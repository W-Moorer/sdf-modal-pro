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

本文正文建议控制在 8 张主图以内，补充材料承接模态形状、参数检查和冗余时程。主图顺序如下：

1. Figure 1: problem and method narrative.
2. Figure 2: patch residual ILC formulation and DAE dimension invariance.
3. Figure 3: SDF-to-patch projection and multiscale active patch algorithm.
4. Figure 4: static and quasi-static completeness validation.
5. Figure 5: moving contact on regular and complex triangular surfaces.
6. Figure 6: true continuum stress/strain and error fields from SFC/CalculiX post-processing.
7. Figure 7: three-way validation with Python/SFC full FEM, adaptive ROM and CalculiX, including nonlinear contact.
8. Figure 8: performance scaling under accuracy constraints.

正文不单独设置“参数敏感性分析图”；若审稿需要，可放补充材料。正文也不放 Yao 式柱状耗时图，所有性能结论用曲线或相图表达。

## 3. 逐章节配图规定

### 摘要图/Graphical Abstract

**图件：Graphical abstract 或 Figure 1a 的简化版。**

内容必须包含四个视觉元素：moving contact samples、SDF nearest-triangle query、active patch set、low-mode dynamics plus residual ILC recovery。该图只讲核心创新，不放数值曲线。

### 1 引言

**Figure 1: SDF-driven active patch residual ILC: problem, mapping and validation ladder.**

建议面板：

- a. 移动接触柔性体问题：接触区沿三角网格表面移动，潜在 patch 多、活动 patch 少。
- b. 低阶 ROM 的局限：局部接触柔度缺失，用一个简化变形示意表达。
- c. SDF query：sample point -> closest triangle -> normal/gap/contact area。
- d. Patch load basis：三角面片接触压力面积加权到 patch resultant。
- e. Residual ILC：active alpha 在主 DAE 之外恢复局部响应。
- f. Validation ladder：static completeness -> SDF projection -> quasistatic contact -> moving contact -> CalculiX three-way -> scaling。

作用：承担 Yao Fig. 1、Fig. 3、Fig. 4 的合并功能。不能做成普通流程框图，必须让读者第一眼知道本文解决的是“移动接触区导致活动界面坐标变化”的问题。

### 2 方法：SDF 驱动的补丁剩余 ILC 公式

**Figure 2: formulation of patch residual interface load coordinates.**

建议面板：

- a. 全阶位移、低阶模态坐标和 patch load coordinate 的分解关系。
- b. Patch load basis 的矩阵列结构：每个 patch 一列或少数几列面积加权法向载荷。
- c. Residual mode 构造：full static attachment response 减去 retained-mode static contribution。
- d. 静态完备性闭环：low-mode contribution + residual correction -> full static response。

作用：对应 Yao Fig. 2-3，但要明确创新边界：residual mode 数学定义来自 free-interface CMS，本文新增的是 SDF/area-weighted patch load basis 和 active patch coupling。

### 3 运动方程与主 DAE 维度不变性

**不建议新增独立主图。**

该节以公式和一张小表支撑即可。若必须配图，使用 Figure 2 的 d 面板或 Figure 3 的 online loop 面板引用，不再新建整页图。

必须报告的指标：

- dynamic dimension contribution = 0。
- active patch alpha 不进入主动态状态。
- residual dynamic DOF ratio vs ordinary patch static modes = 0.0769231。

### 4 SDF 投影与多尺度 active patch 管理

**Figure 3: SDF projection and multiscale active patch activation.**

建议面板：

- a. SDF 查询几何：query point、closest triangle、gap、normal。
- b. 三角形到 patch 的 primary assignment 与 overlap weights。
- c. coarse/medium/fine patch hierarchy。
- d. active set hysteresis、delayed deactivation、warm-start alpha 的时间线。
- e. SDF projection verification：single patch、boundary two-patch、outside contact 三个最小算例。
- f. active patch count 与 projected force 随移动接触步变化的曲线。

作用：对应 Yao Fig. 5，但将齿面对检测换成本文自己的 SDF/patch 映射机制。该图必须说明单位和面积一致性，避免审稿人质疑 penalty 与 pressure-overclosure 不对应。

### 5 接触处理与 CalculiX penalty 标定

**不单独放主图，合并到 Figure 7 或补充图。**

本节若需要图，只放一张小型补充图：

- CalculiX pressure-overclosure 曲线。
- Python area-weighted penalty 对应线。
- 穿透量/接触压力定义示意。

原因：力学期刊正文不应把实现细节膨胀成独立主图；真正重要的是它在三方验证中是否对齐。

### 6 数值实验

数值实验必须形成 Yao 式链条：工况图 -> 场图/曲线 -> 误差/效率证据。建议按以下主图放置。

#### 6.1 静态完备性

**Figure 4: static completeness and quasistatic contact response.**

建议面板：

- a. 静态 patch load benchmark 工况。
- b. full static response、low-mode response、patch residual reconstructed response 的误差曲线。
- c. force-displacement curve：full FEM vs low modes vs patch residual ILC。
- d. compliance error 或 reconstruction error 随 patch/load case 的曲线。

作用：对应 Yao Fig. 7 的“静态完备性图”，但不要只给表。该图证明 residual ILC 能恢复低阶模态遗漏的局部接触柔度。

#### 6.2 SDF 映射验证

**并入 Figure 3，不单独设主图。**

保留 `sdf_patch_projection.csv` 作为源数据表。正文只汇报 single patch、two-patch boundary、outside contact 全部通过，总力/法向力/力矩误差 < 1e-12。

#### 6.3 准静态接触

**并入 Figure 4。**

准静态接触图必须是曲线，不使用柱状图。至少包含 full FEM、low modes、patch residual ILC 三条线。

#### 6.4 移动接触与复杂三角网格表面

**Figure 5: moving contact continuity and complex-surface projection.**

建议面板：

- a. 规则表面接触路径和 patch boundary crossing 示意。
- b. projected normal force、gap 或 alpha jump 随步数变化。
- c. complex triangular surface 工况：warped mesh、contact band、normal variation。
- d. coarse-only vs multiscale active ILC projection error 曲线。
- e. active patch fraction 或 active patch count 随步数变化。

作用：对应 Yao 的动态验证前置图，但本文的重点是移动接触跨 patch 边界是否平滑，以及复杂三角网格表面是否仍能降低投影误差。

#### 6.5 真实连续体应力/应变场

**Figure 6: continuum stress, strain and error fields from SFC/CalculiX.**

这是后续必须补强的主图，不能继续使用弹簧格架 surrogate 作为最终投稿主证据。

建议面板：

- a. SFC/CalculiX full FEM equivalent strain cloud。
- b. SFC/CalculiX full FEM von Mises stress cloud。
- c. low-mode ROM displacement/stress error cloud。
- d. patch residual ILC error cloud。
- e. contact pressure 或 penetration cloud。
- f. selected line/path stress comparison through the contact zone。

硬性要求：

- 云图来自真实连续体单元应力/应变后处理，优先 SFC 接入后的 full FEM；CalculiX 作为外部对照。
- error cloud 使用共同色标。
- 三维云图连续渲染，不用离散块状截图作为最终主图。
- 若只能从 CalculiX 获得积分点/节点应力，应明确说明投影/平均方式。

作用：对应 Yao Fig. 7 和 Fig. 9 中 FE/proposed 应力场对比，是本文从“算法验证”进入“力学可信度”的关键图。

#### 6.6 CalculiX 三方验证

**Figure 7: three-way full-order FEM / adaptive ROM / CalculiX validation.**

建议面板：

- a. 线性动力学工况示意：悬臂 C3D8 块体、固定端、分布载荷、观测节点。
- b. 线性动力学位移时程：Python/SFC full FEM、adaptive ROM、CalculiX full FEM。
- c. 线性动力学累计相对误差曲线。
- d. 非线性接触工况示意：rigid plane 或 contact pair，显示接触面、载荷和约束。
- e. 非线性接触位移/接触力时程三方对比。
- f. penetration/contact pressure/activity 曲线，证明 CalculiX 非线性接触确实参与验证。

作用：对应 Yao Fig. 10-12，但本文必须同时覆盖线性动力学和 CalculiX 非线性接触。当前 Nature-style Fig. 3 只覆盖线性动力学，不能作为最终 Figure 7 的全部内容。

#### 6.7 性能与 scaling estimate

**Figure 8: accuracy-preserving performance scaling.**

建议面板：

- a. SDF scalar exact query vs batched exact query wall time 曲线。
- b. runtime vs query/contact points 曲线，至少 10 个点。
- c. runtime vs active patch fraction 曲线。
- d. accuracy vs runtime Pareto curve：low modes、ordinary patch static modes、patch residual ILC、node-level contact。

硬性要求：

- 不使用柱状图，替代 Yao Fig. 13 的柱状耗时图。
- 性能结论必须绑定误差约束，不能只说快。
- 报告中位加速、最大加速、误差上限和测试硬件。

### 7 讨论与结论

正文结论不新增图。若需要图，只引用 Figure 1 的 validation ladder，并在结论中逐项回扣每张主图支持的 claim。

## 4. 补充材料配图规定

以下内容建议进入 Supplementary Figures，而不是正文主图：

- S1: retained low modes 和 patch residual modes 的典型形状，对应 Yao Fig. 17。
- S2: 模态截断或 retained-mode 数量检查，对应 Yao Fig. 8。
- S3: pressure-overclosure penalty 标定细节。
- S4: 更多观测节点的三方动力学时程。
- S5: 更多接触路径和复杂表面 case。
- S6: 生成图件的 mesh、source data 和后处理流程。

## 5. 当前已有图件归位

| 已有文件 | 当前状态 | 正文归位 |
| --- | --- | --- |
| `paper/nature_figures/fig1_validation_curves.*` | 可作为 Figure 4/5/8 的初版组合图 | 后续应拆分或重排，避免一张图同时承担过多验证任务 |
| `paper/nature_figures/fig2_continuous_field_clouds.*` | 只能作为预期图或占位图 | 最终投稿需替换为 SFC/CalculiX 真实连续体应力/应变云图 |
| `paper/nature_figures/fig3_external_three_way_curves.*` | 线性动力学三方验证图 | 可作为 Figure 7 的一部分，但必须补入非线性接触三方验证 |
| `results/paper_patch_ilc/figures/three_way_nonlinear_contact_displacement.png` | 已有非线性接触位移证据 | 并入最终 Figure 7 |
| `results/paper_patch_ilc/figures/three_way_nonlinear_contact_activity.png` | 已有接触活动证据 | 并入最终 Figure 7，证明 CalculiX 非线性接触参与验证 |
| `results/paper_patch_ilc/figures/complex_surface_moving_contact.png` | 已有复杂表面证据 | 并入最终 Figure 5 |
| `results/paper_patch_ilc/figures/performance_optimization.png` 和 `runtime_scaling.png` | 性能证据初版 | 改造成无柱状、点数不少于 10 的 scaling/Pareto 曲线 |

## 6. 最终主图清单

| 主图 | 章节 | 核心 claim | 必须包含的证据 |
| --- | --- | --- | --- |
| Figure 1 | 引言 | SDF-driven active patch residual ILC 解决移动接触局部柔度和活动界面变化问题 | 问题示意、SDF 映射、active patch、residual recovery、验证链 |
| Figure 2 | 方法 | Patch residual ILC 可以恢复低阶模态遗漏的局部静态响应，且 alpha 不进入主 DAE | 位移分解、patch load basis、residual construction、DAE 外耦合 |
| Figure 3 | 算法 | SDF 采样可稳定映射到多尺度 active patch load coordinates | SDF query、patch hierarchy、hysteresis/warm-start、最小映射验证 |
| Figure 4 | 6.1-6.3 | 静态完备性和准静态接触精度成立 | full/low/residual 曲线、compliance/reconstruction error |
| Figure 5 | 6.4 | 移动接触跨 patch 边界保持连续，并能推广到复杂三角网格表面 | force/gap/alpha history、complex surface、coarse vs multiscale error |
| Figure 6 | 6.5 | 局部应力/应变场和误差场支持方法的力学可信度 | SFC/CalculiX stress、strain、pressure、low-mode error、patch error |
| Figure 7 | 6.6 | 结果不是同源代码自证，线性动力学和非线性接触均通过三方验证 | Python/SFC full FEM、adaptive ROM、CalculiX；位移、接触力、penetration/activity |
| Figure 8 | 6.7 | 在精度约束下，本文方法具有可量化性能优势 | SDF batch scaling、active patch scaling、runtime-accuracy Pareto |

