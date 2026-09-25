# 下一阶段 Virtual GT 研究意见（GPT-5.6 Sol）

Updated: 2026-09-25  
Author: **GPT-5.6 Sol（研究意见，不代表已验证实验结论）**

> 本文基于当前 Nikon 三轮实验、LSMI / One-Net 协议，以及对 multi-illuminant color constancy、synthetic relighting、dense illumination estimation 等相关工作的广泛调研形成。建议项在真实实验完成前不应写成性能结论。

## 1. 核心判断

当前路线不应继续被理解为“再设计一种 alpha mask”。真正值得推进的问题是：

**能否只依赖真实单光源图像和廉价 global illuminant labels，构造可控的 virtual pixel-wise illuminant GT，并利用 coverage、controllability、combinatorics 与 targeted sampling，最终超过固定 One-Net 原有的真实单/多光源监督训练范式？**

One-Net 应继续固定，作为快速 supervision probe。只有 virtual GT 在固定 One-Net 上过关后，再迁移到真正 dense-output 网络，才能把数据监督收益与 architecture 收益分开。

已有文献已经覆盖 single/unique-illuminant → synthetic dual/multi illumination、pixel-wise illuminant estimation、relighting augmentation 等方向。因此 novelty 不能建立在“第一次从单光源合成多光源”上。更可防御的贡献是把 virtual dense supervision 看成一个**可设计的训练分布**，系统分解 endpoint relation、global dominance、local contrast、spatial scale、intensity field 和 GT-aware sampling，并最终验证 virtual-GT-only supervision 能否超过有限真实 multi-light supervision。

## 2. 当前结果给出的信号

第一轮 G/R/M/P 的 test patch mean 为 2.98131 / 2.87320 / 2.72580 / 2.76464°。空间变化从 G 到 M 明显帮助真实 multi 子集，但 P 没有继续改善，因此 spatial supervision 是有效信息，继续堆叠 anchor-patch trick 的优先级不高。

第二、三轮说明恢复真实/单端点样本覆盖很重要。AC 达到 2.43142°，O 达到 2.42386°，相较 M 的 2.72580° 已明显缩小 gap，但仍没有超过原 baseline 1.96924°。

我的判断是：主要瓶颈越来越不像“有没有 spatial GT”，而更像联合分布仍不匹配：

p(L1, L2, alpha, S, scale, content, sampling).

LSMI 的真实 dense illumination GT 本身也可理解为独立 illuminant chromaticity 与 pixel contribution coefficient 的组合。因此当前

E(x) = alpha(x)L1 + [1-alpha(x)]L2

这个核心表示不应首先被推翻。更应该研究 endpoints 如何配对、alpha 如何分布、空间尺度如何控制、总照度如何变化，以及 virtual GT 最后如何进入 patch sampling。

## 3. 下一版：factorized virtual-GT generator

建议正式抽象为：

G_VGT(theta), theta = (L1, L2, q, tau, s, beta)

其中：

- L1, L2：illuminant endpoints；
- q：整幅图 global dominance；
- tau：固定 global dominance 后的 local contrast；
- s：illumination spatial correlation scale；
- beta：总入射强度场变化；
- 同时保留 pixel-wise alpha(x), S(x), E(x)。

### 3.1 第一优先级：exact-q global dominance

沿用 M 的 base field Z_M，改为：

alpha(x) = sigmoid(tau * Z_M(x) + b_q)

先抽：

q ∈ {0.1, 0.3, 0.5, 0.7, 0.9}

再数值求解 b_q，使 image-level mean alpha 精确等于 q。

第一组 Q 实验保持 tau=1、M 的 6×6/18×18 base field、One-Net、patch 数和 69,600 updates 全部不变，只测试 exact-q 本身。

### 3.2 第二优先级：解耦 local contrast

在 exact-q 不变的前提下：

tau ∈ {0.5, 1, 2}.

每次改变 tau 后重新求 b_q，保证 mean alpha 仍为目标 q。

这样可以区分两个以前纠缠的问题：谁是主光，以及在相同主次关系下局部 illumination deviation 有多强。这比继续增加随机 alpha texture 更有解释力。

### 3.3 第三优先级：GT-aware patch sampling

Virtual GT 的独特优势是生成器知道每个候选 patch 的 illumination regime。

建议每个 view 的 64 patches 改为：

- 32 uniform；
- 32 GT-stratified。

按 patch mean alpha 初始分三类：

- A-dominant: mean alpha < 0.2
- mixed: 0.2 <= mean alpha <= 0.8
- B-dominant: mean alpha > 0.8

stratified quota 可先用 11/10/11；某类不存在时动态分配，不为了完成 quota 人工造 patch。

同时记录 patch 内 std(alpha)。因为两个 patch 即使 mean alpha 都是 0.5，一个可能处处接近 0.5，另一个可能一半接近 0、一半接近 1，后者对 One-Net 的训练含义完全不同。

### 3.4 高优先级新增：same-scene endpoint co-occurrence

这是本轮文献调研后我最值得新增的方向之一。

在不读取 mixture image / mixture map / dense GT 的前提下，可以利用 train metadata 的 Light1/2/3 global chromaticity 建立同场景 endpoint pair pool：

P_cooccur = {(Li, Lj) | Li, Lj 来自同一个 train scene}.

它保留“什么样的两盏灯在真实场景中共同出现过”的 joint prior，同时不破坏 single-source / no-real-dense-GT 边界。

最干净的实验是：

Q-tau + global random pair
vs
Q-tau + same-scene pair.

不要一开始混合两种 pair；先判断 co-occurrence 本身有没有价值。

### 3.5 Factorized intensity field

第二轮 S 的方向值得保留，但下一版应把 brightness、dominance、local contrast 解耦。

独立生成 S(x)，再定义：

u1(x) = S(x) alpha(x)
u2(x) = S(x) [1-alpha(x)]

因此：

S = u1 + u2
alpha = u1 / (u1 + u2)

最终仍有：

E(x) = alpha L1 + (1-alpha)L2
I_syn = WB * S * E.

第一轮 brightness 只比较 beta=0 与一个非零值（如 0.5），不要大范围扫参。

## 4. 两项零训练成本审计

### 4.1 Black-level / linear-domain audit

需要确认 Nikon 单光源 TIFF 在执行 WB=RAW/Light1 前后的 black-level convention 与 One-Net baseline 输入完全一致。

如果输入仍带 additive black offset，纯乘法 diagonal relighting 会把 additive offset 转成 illuminant-dependent color term。

这不是断言当前 runner 有错，而是公开协议不足以排除它。继续 generator search 前应直接审计生产代码和真实输入。

### 4.2 Normalization audit

当前协议写明 whole image 和 patch 分别 z-score，但应明确统计维度。

若 RGB 三通道各自独立 z-score，则正 diagonal illuminant gain 会被数学上消除；如果 RGB tensor 共用 scalar mean/std，则没有这个精确不变性。

应写自动测试，比较同一 WB source 在不同 illuminant relighting 后，经过完整 preprocess 的输入张量差异。

## 5. Patch-scale diagnosis

One-Net 假设小 patch 可近似由一个 illuminant 描述；而 M 的 fine field 是 18×18 grid 上采样到 256×256，典型 spacing 与 16×16 patch 已处于同一量级。

先不训练，在 frozen synthetic val 对每个 patch 记录：

- mean alpha
- std(alpha)
- mean |grad alpha|

在固定 mean-alpha bin 内画 angular error vs intra-patch variation。

如果误差随 patch 内变化显著上升，再做 matched diagnostic：保持 patch mean GT 不变，只缩放 patch 内 illumination variation。这个实验能直接回答当前 virtual pixel GT 的空间尺度是否与 One-Net 的监督接口匹配。

## 6. 推荐实验顺序

已有 C/A/AC/L/S 和第三轮结果保留。下一阶段建议：

| 优先级 | 实验 | 主要问题 |
|---|---|---|
| P0 | black-level / normalization audit | relighting 与 illuminant signal 是否正确 |
| P0 | O/E diagnostic controls | global relighting 本身损失多少 |
| P1 | Q | global dominance coverage 是否是主要缺口 |
| P1 | Q-tau | dominance 与 local contrast 是否需要独立控制 |
| P1 | Q-tau-Samp | minority-light / hard-region 是否被充分训练 |
| P1 | Q-tau-Pair | endpoint co-occurrence realism 是否重要 |
| P2 | Q-tau-S | total intensity field 是否进一步缩小 sim-to-real gap |
| P2 | scale-controlled | illumination spatial scale 是否匹配 patch assumption |
| P3 | content-aware fields | 最后再引入 geometry/object-aware support |

所有正式 One-Net 比较继续固定 **69,600 optimizer updates**。

## 7. 最小但信息量最大的下一轮

如果实验预算有限，我建议只跑四个新的主实验：

1. **Q**：M + exact-q；
2. **Q-tau**：Q + controlled local contrast；
3. **Q-tau-Samp**：Q-tau + GT-aware patch sampling；
4. **Q-tau-Pair**：Q-tau + same-scene endpoint pair。

同时先完成无需训练的 black/z-score audit 和 M 的真实 generator-distribution histogram。

不要一次把 q、tau、pair、sampling、brightness 全揉在一起。我们现在需要的不只是更低数字，而是知道**为什么 virtual GT 变强**。

## 8. 论文最终应该争取的三个命题

### 命题 A：可控 virtual supervision 可以超过有限真实 multi-light supervision

最硬的目标仍然是固定 One-Net 下 virtual-GT training < 1.96924°。最终应以多 seed、paired scene statistics 和额外 camera/holdout 确认，而不是单 seed test 最小值。

### 命题 B：优势来自 controllability，而不只是 synthetic sample 数量

通过 q、tau、pair、S、sampling 的逐层消融证明。

### 命题 C：virtual GT 是 supervision source，而不是 One-Net trick

winning generator 之后迁移到官方 LSMI U-Net / PWCC 等 dense network；如果仍有收益，才真正说明方法与网络解耦。

## 9. 与已有工作的边界

相关工作已经包括 pixel-wise illuminant recovery、LSMI 的真实 dense multi-illuminant dataset 与 pixel-level relighting augmentation、physics-driven multi-illumination generation、unique/single-illumination → dual-illumination synthetic generation、white-balanced agent + artificial illuminant relighting、dense illumination-map estimation 与 structured illumination decomposition。

因此不要声称“首次从单光源生成多光源监督”。

更稳妥的研究定位是：

> 将 virtual dense illuminant supervision 建模为一个可控的 latent illumination distribution，并系统研究 endpoint co-occurrence、global dominance、local contrast、spatial scale、intensity coupling 与 GT-aware sampling 如何决定 synthetic-to-real transfer；最终验证这种只依赖 single-light/global-label 的 supervision 是否能够超过有限真实 multi-light supervision。

## 10. 我的最终意见

如果只能押一条主线，我押：

**Controlled Contribution-Field Virtual GT**

即：

realistic endpoint relation
× global dominance q
× local contrast tau
× spatial scale
× total intensity S
× GT-aware sampling.

而不是继续发明更多 random alpha shapes。

真实 dense GT 最大的优势是**真实**；virtual GT 如果想超过它，必须利用真实数据不容易拥有的优势：

**coverage + controllability + combinatorics + targeted sampling.**

目前 M 已证明 spatial virtual GT 有信息量；第二、三轮又证明保留真实/单光源分布很重要。下一步应该把这些经验统一成一个**因子化、可测量、可消融的 virtual dense supervision system**。

这是我认为最有机会把项目从 synthetic augmentation engineering 收敛成明确方法论贡献的路线。

## 11. 重点参考工作

- Kim et al., Large Scale Multi-Illuminant (LSMI) Dataset for Developing White Balance Algorithm, ICCV 2021.
- Das et al., Generative Models for Multi-Illumination Color Constancy, ICCV Workshops 2021.
- Xing et al., Dual-Illumination Weighting and Estimation, ICPR 2022.
- Cun et al., Learning Enriched Illuminants for Color Constancy, 2022.
- Domislović et al., multi-illuminant extension of One-Net, 2023.
- Li et al., MIMT / multi-illuminant multitask work, 2022–2023.
- Entok et al., PWCC: Pixel-Wise Color Constancy, ICIP 2024.
- Kim et al., Attentive Illumination Decomposition Model for Multi-Illuminant White Balancing, CVPR 2024.
- Recent multi-scale pixel-wise illuminant estimation work, 2025–2026.

引用与 novelty 判断在正式论文写作前仍应逐篇回到原论文核验；本文用于指导下一阶段实验，不是 related-work 定稿。
