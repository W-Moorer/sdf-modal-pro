# 仿照 Yao 等论文格式的中文论文大纲

## 1. Yao 等论文写作结构分析

参考文献 9 更接近一篇力学方法论文，而不是单纯的软件或算法论文。它的写作主线是：

1. 从工程中的移动接触问题出发：全阶有限元精度高，但在动态接触和大规模模型中计算代价过高。
2. 回顾已有的降阶、CMS、接触界面处理方法，并指出核心瓶颈：局部接触精度需要大量界面自由度。
3. 提出一种既能保持局部接触精度、又不显著增加 DAE 维度的降阶表达。
4. 按层次推导方法：运动学、模态收敛判据、模态集合构造、剩余模态计算、运动方程、界面载荷坐标高效求解、接触算法。
5. 逐级验证：静态完备性、截断准则、接触算法、与全阶 FEM 或已有方法的动态对比，以及一个更大规模的工程应用。
6. 结论部分紧扣精度、效率和对其他移动接触系统的推广性。

原文的章节流是：

- 摘要：问题背景、全阶 FEM 局限、自由界面 CMS、剩余模态、界面载荷坐标、验证结果。
- 第 1 节 引言：移动接触动机、FEM 成本、仅用低阶模态的局限、静态修正模态、SMS/ALE/PMOR 对比、本文贡献和章节安排。
- 第 2 节 移动接触多体系统的自由界面 CMS：
  - 2.1 移动接触柔性体的 FFR 运动学。
  - 2.2 将低频动态模态和高频准静态模态分开的收敛判据。
  - 2.3 模态集合：保留的正规模态加由界面载荷驱动的剩余模态。
  - 2.4 剩余模态计算：全静态附着响应减去保留模态的静态贡献。
- 第 3 节 运动方程简化：坐标、能量项、质量和刚度矩阵简化、最终 DAE 形式。
- 第 4 节 ILC 高效求解和计算流程：界面载荷坐标的直接更新与实现流程图。
- 第 5 节 接触算法：候选接触对检测、边界检测、接触力和等效节点载荷计算。
- 第 6 节 数值算例：静态完备性、频率截断准则、接触算法验证、与 FEM/ALE 的动态验证、高速齿轮行波应用。
- 第 7 节 结论：方法回顾、数值证据、效率提升、对其他移动接触系统的应用前景。

因此，本文可以复用 Yao 的论文组织方式：先理论推导，再算法实现，再接触处理，最后通过逐级数值算例支撑方法结论。但本文不能把 free-interface CMS、residual modes 和 ILC 本身写成原创贡献，它们应被定位为理论基础。

## 2. 拟定论文题目

面向表面移动接触的 SDF 驱动补丁剩余界面载荷坐标方法

英文备选题目：

SDF-Driven Patch Residual Interface Load Coordinates for Surface Moving Contact Dynamics

## 3. 摘要大纲

中心论证句：

> 对移动接触柔性体，本文证明 SDF 驱动的补丁剩余坐标可以在主动态状态维度不变的前提下恢复局部接触柔度，并通过静态完备性、SDF 到补丁载荷守恒、移动接触平滑性和 CalculiX 三方验证支撑该结论；当前范围限于无摩擦法向接触。

第一段：问题与缺口。

- 柔性多体系统中的移动接触需要同时描述整体动力学和局部表面柔度。
- 全阶 FEM 可以捕捉局部接触响应，但在接触状态反复变化时计算成本过高。
- 低阶 ROM 难以表示局部接触柔度；普通静态修正模态虽然能恢复局部响应，却会扩大动态基或引入模态切换问题。
- 既有 free-interface CMS 和 ILC 方法为移动接触降阶提供了理论基础，但仍缺少面向三角网格 SDF 表面的活动补丁映射、多尺度补丁激活和外部求解器验证框架。

第二段：本文方法。

- 提出 SDF 驱动的补丁级界面载荷坐标方法，将表面移动接触采样映射为面积加权补丁载荷基。
- 构造由补丁法向载荷基驱动的剩余模态，用补丁静态附着响应恢复局部接触柔度。
- 将活动补丁 ILC 放在主动态状态之外求解，使接触区域移动和活动补丁集合变化时主 DAE 维度保持不变。
- 设计多尺度重叠补丁激活策略，通过 coarse/medium/fine 层级、partition-of-unity 权重、滞回、延迟失活和 alpha 暖启动降低跨补丁边界的力和间隙跳变。

第三段：验证证据。

- 通过静态完备性、SDF 映射、准静态接触、规则与复杂三角网格移动接触、CalculiX 三方验证、实测 SDF 查询复用加速和 scaling estimate 进行验证。
- 报告最能支撑摘要结论的硬数字：静态完备性误差 `2.180325e-17`；低阶模态准静态力误差 `0.831666`，补丁剩余 ILC 恢复到 `0`；移动接触 force jump `0.019911`，gap jump `0.00897824`；非线性接触中 Python full FEM vs CalculiX 位移相对误差 `0.005014`；patch alpha DOFs `40`，低于 node-level contact DOFs `69`。
- 建立全阶 FEM、自适应 ROM 和外部开源 FEM 求解器的三方验证路线，用于降低同源代码自证的风险。
- 消融对比作为内部一致性检查和附录材料，不在正文中单独设置验收式章节。

## 4. 与 Yao 论文的创新边界

本文的创新边界必须写得非常清楚：Yao 的 free-interface CMS、residual modes 和 interface load coordinates 是理论来源，不是本文原创。本文的原创性来自将这些思想改造成面向 SDF 表面移动接触的补丁级、自适应、多尺度接触降阶框架。

| 问题维度 | Yao 论文已有内容 | 本文新增内容 |
| --- | --- | --- |
| 理论基础 | free-interface CMS；低频模态表示整体动力学，高频截断模态通过剩余模态补偿局部准静态响应。 | 继承该分解思想，但把局部接触载荷组织为 SDF 驱动的补丁级载荷基。 |
| ILC 思想 | 用 interface load coordinates 表示剩余模态贡献，避免大量界面自由度直接扩大主系统。 | 不能把 ILC 本身作为原创；应强调 active patch ILC 在接触区域移动和活动集合变化时仍保持主 DAE 维度不变。 |
| 接触几何 | 面向移动接触多体和齿轮接触，接触界面具有明确机械结构和候选接触对。 | 面向三角网格外表面，使用 SDF 查询将接触采样点映射到表面三角形和活动补丁。 |
| 接触载荷表示 | 更接近界面/节点层级的载荷坐标。 | 构造面积加权 patch load basis，将节点接触压力/穿透等效到补丁载荷坐标。 |
| 剩余模态构造 | 剩余模态等于全阶静态附着响应减去保留模态静态贡献。 | 保留该数学定义，但静态附着响应由补丁载荷基驱动，用于压缩局部接触柔度表达。 |
| 活动区域变化 | 重点不在 SDF 表面活动补丁管理。 | 提出 SDF-driven active patch set，并处理活动补丁随接触位置移动而变化的问题。 |
| 平滑策略 | 不以跨补丁边界连续性为核心。 | 提出多尺度重叠补丁激活、partition-of-unity、滞回、延迟失活和 alpha 暖启动。 |
| 验证体系 | 静态完备性、接触算法、FEM/ALE 对比和齿轮应用。 | 加入全阶 FEM、自适应 ROM、外部开源 FEM 求解器三方验证；消融结果作为内部证据或附录，不作为正文独立章节。 |

一句话创新定位：

> 本文不是提出新的 CMS 或新的 ILC 理论，而是提出一种面向 SDF 表面移动接触的补丁级剩余 ILC 框架，使局部接触柔度、活动接触区域变化和主 DAE 维度不变性可以同时成立。

## 5. 分章节大纲

### 1 引言

中心论证句：

> 本文研究的问题不是一般 CMS 是否有效，而是当接触区域沿三角网格表面移动时，如何在不改变主动态系统维度的前提下恢复局部接触柔度并保持接触载荷转移平滑。

1. 移动接触柔性体需要同时保留整体动力学响应和局部表面响应。
2. 全阶 FEM 精度高，但当接触区域在细分表面上移动时，时间积分和接触搜索成本很高。
3. 低阶模态 ROM 计算高效，但不能准确表达局部接触柔度。
4. 普通静态修正模态可以恢复局部响应，但会增加动态坐标数量，或者需要在接触区域变化时切换模态。
5. Free-interface CMS、剩余模态和 ILC 为移动接触降阶提供了可借鉴基础，但这些方法通常没有解决 SDF 表面上接触采样到活动补丁的映射、活动集合变化时的平滑性，以及外部求解器三方验证问题。
6. 本文提出 SDF 驱动的补丁剩余 ILC 框架，在活动接触区域移动时保持主 DAE 维度不变，并用多尺度活动补丁恢复局部接触柔度。
7. 本文贡献限定为以下四项：
   - 提出 SDF-driven patch ILC，将表面移动接触采样映射为面积加权补丁载荷坐标，避免直接使用节点级接触自由度。
   - 构造 patch-load-driven residual modes，用补丁静态附着响应恢复局部接触柔度，同时保留低阶自由界面模态描述整体动力学。
   - 设计 active patch ILC coupling outside the main DAE，使活动补丁数量和位置变化时主动态系统维度保持不变。
   - 提出 multi-scale overlapping patch activation，通过粗/中/细层级、partition-of-unity、滞回、延迟失活和 warm start 改善跨补丁边界的接触连续性，并用全阶 FEM/ROM/CalculiX 三方验证支撑结果可信度。
8. 给出全文结构安排。

### 2 移动接触柔性体的补丁剩余 ILC 公式

#### 2.1 柔性体运动学与降阶坐标

- 定义全阶位移场和保留的自由界面模态坐标。
- 将主动态状态与接触/界面载荷坐标分开。
- 明确继承关系：低频模态与剩余模态分解来自 free-interface CMS 思路。
- 明确本文差异：alpha 是活动补丁载荷坐标，不是节点级界面自由度，也不是主 DAE 状态变量。

预期图：

- `method_pipeline.png` 应设计成主叙事图，而不是简单流程图。
- 图型：schematic-led composite。
- Panel a：full FEM moving-contact problem，显示局部接触区域沿柔性体表面移动。
- Panel b：SDF query，将接触采样点映射到最近表面三角形、gap 和 normal。
- Panel c：surface triangles 聚合到 patch load basis，突出面积加权和总法向力守恒。
- Panel d：patch residual modes 恢复低阶模态遗漏的局部静态柔度。
- Panel e：active patch ILC 位于主 DAE 状态之外，活动补丁变化不改变主动态维度。
- Panel f：validation ladder，依次显示 static completeness -> SDF projection -> quasistatic contact -> moving contact -> warped triangular surface -> CalculiX three-way -> performance/scaling estimate。

#### 2.2 局部接触响应的收敛判据

- 沿用 Yao 论文的基本逻辑：低频模态承担整体动力学，被截断的高频模态在局部接触载荷下主要体现为准静态贡献。
- 说明仅用低阶模态无法收敛到准确的局部接触柔度。
- 将目标静态附着响应定义在补丁载荷基上，而不是节点级界面载荷上。
- 说明这一步是本文相对 Yao 的关键压缩：用少量补丁载荷基代表局部接触压力分布。

预期表：

- `static_completeness.csv`：各补丁的静态重构误差。

#### 2.3 SDF 驱动的补丁载荷基与界面载荷坐标

- 通过 SDF 查询获得接触采样点、最近表面三角形、法向和穿透量。
- 将表面三角形映射到补丁，并构造活动补丁集合。
- 构造面积加权的法向 patch load basis，使 pressure/penetration 与补丁等效载荷在单位和作用面积上匹配。
- 定义 alpha 向量，并说明补丁级 ILC 如何降低节点级界面坐标规模。

对应验证：

- SDF 接触采样可以稳定映射到活动补丁。
- 多尺度补丁 alpha 自由度为 40，节点级接触自由度为 69。

#### 2.4 补丁载荷驱动的剩余模态

- 定义全阶静态响应 `G_B`。
- 定义保留模态的静态贡献。
- 将补丁剩余模态定义为补丁载荷静态附着响应与保留模态静态贡献之差。
- 说明静态完备性：模态静态贡献加补丁剩余响应可以重构补丁载荷下的全阶静态附着响应。
- 明确创新边界：剩余模态定义不是本文原创；本文原创在于它由 SDF/面积加权补丁载荷基驱动。

预期表：

- `compliance_error.csv`：低阶模态柔度误差、剩余 ILC 误差、压缩剩余模态误差。

### 3 运动方程与活动 ILC 的 DAE 维度不变性

#### 3.1 不含 alpha 的动态状态

- 用保留模态坐标定义降阶位移、速度和加速度。
- 说明 alpha 只通过恢复的剩余变形或接触修正发挥作用，不进入主动态状态。
- 避免直接写成“ILC 不进入 DAE 是本文首创”；应写成“活动补丁集合变化时，主动态系统维度仍保持不变”。

#### 3.2 在线剩余耦合

- 给出接触残差到补丁 alpha 的投影。
- 给出由活动剩余块恢复局部变形的方式。
- 说明当前实现中采用的冻结 alpha 或切线式迭代处理。
- 讨论活动补丁增删时如何保持时间积分变量维度不变。

#### 3.3 DAE 维度论证

- 对比三类模型：
  - 仅低阶模态。
  - 将普通补丁静态模态作为动态模态加入。
  - 活动补丁剩余 ILC。
- 强调普通静态模态会增加动态自由度，而活动补丁剩余 ILC 保持主 DAE 维度不变。
- 将论证重点放在移动接触导致活动补丁集合变化的情况下，而不是重复一般 ILC 论点。

对应指标：

- DAE independence flag = 1；相对普通静态模态的 residual dynamic DOF ratio = 0.0769231。

### 4 活动 ILC 高效求解与多尺度补丁激活

#### 4.1 SDF 接触采样到补丁 ILC 的投影

- 通过 SDF 查询检测接触采样点。
- 将最近表面三角形映射到主补丁。
- 将法向接触力聚合到补丁 alpha。
- 验证总力守恒、法向力一致性和面积权重一致性。

#### 4.2 自适应活动补丁管理

- 当前接触补丁必须激活。
- 邻域补丁用于一环平滑。
- 滞回和延迟失活用于避免活动集合不连续跳变。
- alpha 暖启动用于平滑补丁切换。

#### 4.3 多尺度重叠补丁激活

- 粗补丁提供全局兜底。
- 中尺度补丁覆盖当前接触邻域。
- 当投影误差或接触力梯度较大时激活细尺度重叠补丁。
- 用 partition-of-unity 权重在重叠补丁之间分配接触力。
- 将多尺度策略写成本文最强差异化贡献之一，因为 Yao 论文不处理 SDF 表面上的活动补丁平滑问题。

预期图表：

- `patch_hierarchy.png`
- `active_patch_history.png`
- `active_patch_history.csv`

### 5 接触处理

#### 5.1 基于 SDF 的接触检测

- 定义查询点、最近表面三角形、间隙、法向和穿透量。
- 说明 SDF/BVH 查询适合三角网格外表面的移动接触检测。
- 强调这与 Yao 的齿轮接触背景不同：本文目标是更一般的外表面移动接触。

#### 5.2 力投影与 penalty 标定

- 将 pressure/penetration 转换为法向节点载荷或补丁载荷。
- 使用面积加权使 penalty 单位与 pressure-overclosure 接触关系一致。
- 说明与 CalculiX 对齐验证之间的关系。

#### 5.3 接触边界平滑性

- 定义跨补丁边界时的接触力跳变、间隙跳变和 alpha 跳变指标。
- 将接触点跨越补丁边界作为主要平滑性测试。
- 说明多尺度补丁和 warm start 如何降低跳变。

预期证据：

- moving force jump = 0.019911
- moving gap jump = 0.00897824
- energy drift excess ratio = 0.954554

### 6 数值算例与验证

#### 6.1 静态完备性验证

Claim-first 开头句：

> 为了检验补丁载荷驱动的剩余模态是否可以恢复全阶静态附着响应，我们比较全阶静态响应、保留模态静态贡献和补丁剩余修正后的重构误差。

指标：

- 最大静态完备性误差。
- 剩余柔度误差。
- 压缩剩余模态柔度误差。

预期结果：

- max completeness error = 2.180325e-17
- compressed residual compliance error = 2.456128e-02

#### 6.2 SDF 到补丁载荷投影验证

Claim-first 开头句：

> 为了检验 SDF 接触采样是否能够稳定映射为活动补丁 ILC，我们构造单补丁、跨补丁边界和无接触三类采样，并检查活动补丁数、等效总力和主动态维度贡献。

算例：

- 单补丁接触采样。
- 双补丁边界接触采样。
- 无接触采样。

指标：

- 活动补丁数量与预期补丁数量。
- 总力误差、总法向力误差和力矩误差。
- alpha 维度和主动态系统维度贡献。

预期结果：

- 三类 SDF 映射算例全部通过。
- 总力、法向力和力矩误差 < 1e-12。
- dynamic dimension contribution = 0。

预期图表：

- `sdf_patch_projection_map.png`
- `sdf_patch_projection.csv`

#### 6.3 准静态接触验证

Claim-first 开头句：

> 为了检验补丁剩余 ILC 是否恢复低阶模态遗漏的局部接触刚度，我们在准静态压入算例中比较低阶模态、普通补丁静态模态和补丁剩余 ILC 的力-位移响应。

对比模型：

- 仅低阶模态。
- 低阶模态加普通补丁静态模态。
- 低阶模态加补丁剩余 ILC。

指标：

- 力-位移曲线。
- 局部压入量。
- 补丁柔度。
- 间隙。
- 接触区域大小。
- 活动自由度数量。

预期图表：

- `force_displacement_curve.png`
- `contact_force_error.csv`

#### 6.4 移动接触验证

Claim-first 开头句：

> 为了检验活动补丁集合变化是否会引入不连续接触响应，并检验该策略能否从规则表面推广到复杂三角网格表面，我们让接触区域分别在规则侧面和 warped triangular surface 上跨越补丁边界移动，并测量接触力、间隙、活动补丁数量、投影误差和能量漂移。

指标：

- 接触力连续性。
- 间隙连续性。
- 活动补丁数量。
- 运行时间比例。
- 能量漂移。
- 复杂三角网格表面的法向变化角、接触带三角形数量和活动补丁比例。

预期结果：

- force jump < 5%。
- gap jump < 5%。
- 自适应活动补丁的能量漂移不高于全活动补丁基准。
- 复杂三角网格表面算例：surface triangles = 328，contact-band triangles = 58，max normal variation = 47.55 deg，unique requested patches = 54，active patch fraction = 0.1328。
- 复杂三角网格表面算例中 multi-scale projection error = 0.358，coarse-only projection error = 0.860，说明复杂表面上细化活动补丁仍能降低投影误差。

预期图表：

- `active_patch_history.png`
- `active_patch_history.csv`
- `complex_surface_moving_contact.png`
- `complex_surface_moving_contact.csv`

#### 6.5 CalculiX 三方验证

Claim-first 开头句：

> 为了检验本文结果是否不仅是项目内全阶 FEM 和 ROM 的同源一致性，我们将自适应 ROM、项目内全阶 FEM 和 CalculiX 外部全阶 FEM 在动力学和非线性接触算例中进行三方对比。

对比对象：

- 项目内全阶 FEM 动力学解。
- 自适应补丁剩余 ROM 解。
- CalculiX 全阶 FEM 动力学/非线性接触解。

验证算例：

- 线性动力学三方位移时程对比。
- CalculiX surface-to-surface 非线性接触三方对比。

指标：

- 位移时程相对误差。
- 接触力时程相对误差。
- SFC 全阶 FEM 与 CalculiX matrix storage 的刚度/质量矩阵一致性。
- 非线性接触中的穿透量、压力峰值和 contact pressure-overclosure 斜率。

预期结果：

- 线性动力学中 Python full FEM vs CalculiX displacement relative L2 = 0.011581。
- 线性动力学中 adaptive ROM vs Python full FEM relative L2 = 0.003701。
- 非线性接触中 SFC stiffness/mass matrix vs CalculiX relative L2 < 1.2e-14。
- 非线性接触中 Python full FEM vs CalculiX displacement relative L2 = 0.005014。
- 非线性接触中 adaptive ROM vs Python full FEM force relative L2 = 1.765600e-13。
- CalculiX pressure-overclosure slope = 499.999996，对应目标 stiffness = 500.0。

预期图表：

- `three_way_external_fem.csv`
- `three_way_dynamic_displacement.png`
- `three_way_nonlinear_contact_displacement.png`
- `three_way_nonlinear_contact_activity.png`

#### 6.6 Scaling estimate 与性能优化证据

Claim-first 开头句：

> 为了检验活动补丁策略是否降低接触坐标规模和接触映射代价，我们比较全补丁、节点级 ILC、粗补丁和多尺度活动补丁的自由度、runtime proxy、memory proxy，并对优化前后的多尺度 SDF 映射执行实测 wall-time 对比。

对比对象：

- baseline modal-SDF。
- patch residual ILC。
- node-level ILC 估计。
- coarse-only patch。
- multi-scale patch。
- uncached multi-scale SDF mapping。
- cached multi-scale SDF mapping。
- optimized active patch sequence。

指标：

- dynamic DOFs。
- active patch/ILC DOFs。
- node-level contact DOFs。
- active patch fraction。
- solver-dimension speedup。
- runtime proxy 和 memory proxy。
- baseline/optimized SDF query count。
- cached-vs-uncached SDF mapping measured speedup。
- optimized active patch sequence wall time。

写作定位：

- 这是 scalability estimate、active-patch cost analysis 和接触映射内核的实测优化证据，不把小型 Python 原型的全流程 wall time 当作最终工业级全阶 FEM 基准。
- 性能结论应落在两个硬证据上：一是 cached SDF mapping 将 coarse/medium/fine 三层最近三角形查询从 105 次降到 35 次，实测加速约 3.02x；二是复杂三角网格算例中活动补丁比例为 0.1328，对应 active-vs-full patch runtime proxy speedup = 7.53x。
- 因此最终结论可以突出本文优势：方法不是只降低主 DAE 维度，也降低了移动接触过程中需要评估和更新的局部补丁坐标规模。

预期图表：

- `runtime_scaling.csv`
- `runtime_scaling.png`
- `performance_optimization.csv`
- `performance_optimization.png`

### 7 结论

第一段：方法回顾。

- 本文没有提出新的 CMS 或 ILC 理论，而是提出了面向 SDF 表面移动接触的补丁级剩余 ILC 框架。
- SDF 活动补丁投影支持三角网格外表面上的移动接触。
- 补丁载荷驱动的剩余模态恢复局部接触柔度。
- 多尺度重叠补丁激活提高跨补丁边界的连续性和精度/效率平衡。

第二段：证据回顾。

- 静态完备性和柔度恢复已经验证。
- 移动接触中的力跳变和间隙跳变受到约束。
- SDF 映射验证、准静态接触、规则与复杂三角网格移动接触、CalculiX 三方验证、实测 SDF 映射优化和 scaling estimate 支撑该方法。
- 消融对比可作为附录或补充材料，用于说明各模块对精度和效率的贡献。

第三段：局限与未来工作。

- 当前方法主要聚焦无摩擦法向接触。
- 后续工作应扩展到摩擦切向基、非线性材料/接触律、更强的外部 FEM 基准，以及更大规模的工业几何模型。

## 6. 图表计划

### 图

1. `method_pipeline.png`：主叙事图，采用 schematic-led composite，串联 full FEM moving contact、SDF query、patch load basis、residual modes、active ILC outside DAE 和 validation ladder。
2. `patch_hierarchy.png`：粗/中/细补丁层级和重叠区域。
3. `sdf_patch_projection_map.png`：SDF 接触采样映射到活动补丁 ILC 的算例证据。
4. `force_displacement_curve.png`：准静态接触力-位移对比。
5. `active_patch_history.png`：移动接触中的活动补丁数量和接触力。
6. `three_way_dynamic_displacement.png`：线性动力学三方位移时程对比。
7. `three_way_nonlinear_contact_displacement.png`：非线性接触三方位移时程对比。
8. `three_way_nonlinear_contact_activity.png`：非线性接触穿透量和接触力活动证据。
9. `complex_surface_moving_contact.png`：复杂三角网格表面上的移动接触轨迹和接触带。
10. `performance_optimization.png`：cached SDF mapping 实测加速和活动补丁比例。
11. `runtime_scaling.png`：scaling estimate 和活动补丁代价对比。

### 表

1. `yao_boundary_table.csv`：Yao 已有内容与本文新增内容边界。
2. `static_completeness.csv`：补丁静态重构。
3. `sdf_patch_projection.csv`：单补丁、双补丁边界和无接触三类 SDF->patch ILC 映射算例。
4. `compliance_error.csv`：柔度恢复。
5. `contact_force_error.csv`：压入和动态接触力误差。
6. `complex_surface_moving_contact.csv`：warped triangular surface 移动接触算例。
7. `three_way_external_fem.csv`：项目内全阶 FEM、自适应 ROM、CalculiX 外部 FEM 三方验证算例。
8. `runtime_scaling.csv`：scaling estimate 和活动补丁代价对比。
9. `performance_optimization.csv`：SDF 查询复用和活动补丁评估的性能优化证据。
10. `ablation_summary.csv`：A-E 模块对比，建议作为附录或补充材料。

## 7. 内部证据映射（不作为正文章节）

| 论文结论点 | 证据 | 状态 |
| --- | --- | --- |
| SDF 接触采样可以稳定映射到活动补丁。 | `sdf_patch_projection.csv` 覆盖单补丁、双补丁边界和无接触三类算例；活动补丁数匹配预期，总力/法向力/力矩误差 < 1e-12，动态维度贡献为 0。 | 已补充 |
| 仅低阶模态无法表示局部接触柔度。 | quasi-static low-mode force error = 0.831666。 | 已支撑 |
| 补丁载荷驱动的剩余模态可以恢复局部准静态响应。 | static completeness error = 2.180325e-17；residual force error = 0。 | 已支撑 |
| 补丁级 ILC 降低节点级规模。 | patch alpha DOFs = 40，node-level contact DOFs = 69。 | 已支撑 |
| 活动补丁 ILC 不进入主 DAE，活动集合变化时主动态系统维度保持不变。 | DAE independence flag = 1；dynamic DOF ratio vs ordinary modes = 0.0769231。 | 已支撑 |
| 多尺度重叠补丁激活改善跨补丁边界连续性和精度/效率平衡。 | force jump = 0.019911；gap jump = 0.00897824；multi-scale projection error = 0.269226，single-scale = 0.299569。 | 已支撑 |
| 方法可以处理复杂三角网格外表面上的移动接触，而不只是在规则侧面上有效。 | `complex_surface_moving_contact.csv`：surface triangles = 328，contact-band triangles = 58，normal variation = 47.55 deg，unique requested patches = 54，active patch fraction = 0.1328，multi-scale projection error < coarse-only。 | 已补充 |
| 三方验证说明结果不是同源 FEM/ROM 自证。 | `three_way_external_fem.csv` 覆盖线性动力学和 CalculiX 非线性 surface-to-surface contact；线性三方位移相对误差 < 5%，非线性接触矩阵一致性 < 1e-10，pressure-overclosure 标定误差 < 1e-6。 | 已补充 |
| 性能优势不是只靠 proxy，而有接触映射内核实测优化证据。 | `performance_optimization.csv`：cached SDF mapping query count 105 -> 35，measured speedup = 3.023x，sample count delta = 0；active patch fraction = 0.1328，proxy speedup = 7.53x。 | 已补充 |

## 8. 自查清单

- 引言是否明确说明：Yao 的 free-interface CMS、residual modes 和 ILC 是理论基础，不是本文原创？
- 贡献是否压缩为 SDF-driven patch ILC、patch-load-driven residual modes、active patch ILC outside main DAE、multi-scale overlapping patch activation 四项？
- 是否避免把“ILC 不进入 DAE”孤立写成原创，而是绑定到 SDF 移动接触下活动补丁集合变化？
- 剩余模态是否被定义为补丁载荷静态附着响应减去保留模态静态贡献？
- alpha 是否始终被描述为活动补丁界面载荷坐标，而不是动态状态？
- 摘要和引言中的每个结论点是否都能映射到数值算例、附录对比或三方外部 FEM 验证输出？
- 第 6 节每个结果小节是否以 “为了检验……我们……” 的 claim-first 句式开头？
- 是否没有把参数敏感性分析作为力学主线证据，而是把篇幅留给完备性、接触映射、三方验证和性能优化？
- 图 1 是否是主叙事图，而不是只列模块名称的简单流程图？
- 所有图表是否都服务于核心结论，而不是泛泛展示程序结果？
- 局限性是否足够明确：无摩擦法向接触、原型规模动态 FEM 对比、未来工业规模验证？
