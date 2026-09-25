# 下一阶段 Virtual GT 策略（GPT-5.6 Sol）

Updated: 2026-09-25  
Author: **GPT-5.6 Sol**  
Status: **研究意见 / 下一阶段实验策略，不代表已验证结论**

> 本版吸收 Nikon round 1–3、AN/PS 最新结果，以及一个关键修正：**content/source 与 illumination state 不应互斥分配。** 同一个单光源 source 可以在训练过程中反复生成 original、global-relit 和 spatial mixed 等不同光照状态。Virtual GT 的优势应来自 on-the-fly illumination-state coverage，而不是把有限 source 切成若干互斥子集。

## 1. 当前证据如何重新理解

当前固定 One-Net 的主要结果：

| 模型 | test all patch | test single | test multi |
|---|---:|---:|---:|
| 原 baseline | 1.96924 | 1.34651 | 2.53378 |
| M | 2.72580 | 2.00299 | 3.38104 |
| AC | 2.43142 | 1.68796 | 3.10540 |
| O | **2.42386** | **1.52070** | 3.24261 |
| B | 2.42972 | 1.67393 | **3.11488** |
| AN | 2.56261 | — | — |
| PS | 2.70852 | — | — |

O 的操作不是新的 spatial GT。它相对 AC 把 25% 单端点 global-relit 分支替换成原始单光源，因此：

- AC = 25% original + 25% global-relit + 50% M mixed；
- O = 50% original + 50% M mixed。

O 相对 AC 的主要变化是 single 从 1.688° 改善到 1.521°，但 multi 从 3.105° 退化到 3.243°。因此 **O 不证明“50/50 是最优比例”**。更合理的解释是：

> **identity/original illumination state 是一个必须稳定保留的真实 anchor；增加 synthetic illumination 不应该以减少 source 在 identity state 下的覆盖为代价。**

当前与 baseline 的 gap 也已经很不对称：

- single gap：约 **+0.174°**；
- multi gap：约 **+0.709°**。

因此下一阶段真正要攻的是 synthetic mixed-light branch，而不是继续优化 single-light distribution。

B 改变每图主导光源比例后，multi=3.115°，优于 O 的 3.243°，但 single 较差。这提示 global dominance 可能值得继续研究；它不是因果证明，但比继续增加随机 alpha 纹理更有针对性。

AN/PS 负结果说明简单的 source-image / whole-endpoint-pair association 没有解决当前问题。它不能排除更一般的 endpoint prior，但 **same-scene endpoint pairing 不再作为 P1 主线**。

## 2. 核心策略：Content Bank × Illumination-State Sampler

不要把 668 个 source 分成 original 与 mixed 两部分。

把每个 source 视为一个可重复使用的 content / reflectance-like base：

[
B_i = I_i / L_i.
]

然后在训练时 on-the-fly 采样 illumination state：

[
(B_i,; z) ightarrow (I_{i,z}, E_{i,z}).
]

同一个 (B_i) 可以经历无限多个状态：

[
B_i ightarrow
{	ext{identity},	ext{global relit},	ext{weak mixed},	ext{strong mixed},ldots}.
]

因此方法框架应从 categorical dataset mixture 改成：

[
oxed{	ext{Content Bank} 	imes 	ext{On-the-fly Illumination-State Distribution}}
]

668 个场景的 content diversity 固然有限，但 illumination conditional diversity 可以非常大。

## 3. 第一优先级实验：每个 source 强制 paired identity + mixed

当前训练本来就是每 source / cycle 生成两个 view。最干净的下一组实验不增加预算：

[
V_i^{(0)}=	ext{identity}(B_i,L_i)
]

[
V_i^{(1)}=	ext{synthetic-mixed}(B_i,E_i(x)).
]

两个 view：

- 来自同一个 source；
- 使用同一 crop / resize / flip / geometry；
- 一个保留原始 Light1；
- 一个 on-the-fly 生成 spatial mixed illumination；
- 各自使用准确 GT；
- 不新增 consistency loss；
- 仍为两个 view、每 view 64 patches；
- optimizer updates 仍固定 69,600。

这与 O 的区别非常重要。

O 是训练分布意义上的 50% original / 50% mixed；paired 方案则保证：

[
oxed{orall; source,quad identity;state;+;mixed;state}
]

都被覆盖。

因此它检验的不是“比例”，而是：

> **每一个有限 content 是否都应该同时承担真实 identity anchor 与 synthetic illumination expansion。**

如果 paired coverage 优于 O，即可把后续研究正式建立在“source reuse × illumination-state randomization”上。

## 4. 第二优先级：只改 paired 方案中的 mixed generator

Identity view 保持完全真实，不再反复调整。

研究资源全部投入第二个 synthetic view。

### 4.1 Controlled global dominance

从 M 的 base field (Z(x)) 出发：

[
alpha(x)=sigma(	au Z(x)+b_q).
]

先采样目标 image-level dominance：

[
q in {0.1,0.3,0.5,0.7,0.9},
]

再求 (b_q)，使：

[
operatorname{mean}_xalpha(x)=q.
]

这样主光 / 辅光比例成为显式受控变量，而不是随机场的副产品。

B 的结果使这一方向比之前更值得优先验证。

### 4.2 Local contrast 与 global dominance 解耦

固定 q 后再改变：

[
	auin{0.5,1,2},
]

每次重新求 (b_q)。

这样分别控制：

- q：谁在整幅图中占主导；
- tau：相同主次关系下，局部 illumination variation 有多强。

不要再通过增加更多 random alpha shape 间接改变这两个因素。

### 4.3 Spatial scale 必须显式记录

对 synthetic view 记录：

- patch mean alpha；
- patch std(alpha)；
- mean |grad alpha|；
- illumination correlation scale。

M 的 18×18 fine field 与 One-Net 16×16 patch 已处于相近尺度。若 patch 内 illumination variation 太大，patch-average GT 可能成为较困难的监督接口。

先做诊断，再决定是否把 scale 作为下一训练因素。

## 5. 第三优先级：GT-aware patch sampling

Virtual GT 的一个真正独有优势是：生成器知道哪里是主光区、辅光区和 transition 区。

在 synthetic mixed view 中，可以把 64 patches 拆成：

- 32 uniform；
- 32 GT-stratified。

例如按 patch mean alpha 分为：

- A-dominant：<0.2；
- mixed：0.2–0.8；
- B-dominant：>0.8。

某类不存在时动态重分配，不为了 quota 人工制造区域。

目标不是改变 GT，而是避免小面积 secondary-light 区域因为面积小而几乎不贡献梯度。

这应该在 paired identity+mixed 和 controlled-q 之后测试，而不是一开始与 generator 混在一起。

## 6. 关于空间强度：不要把 chromaticity 与 intensity 混成一个变量

仓库准备的候选是在 mixed branch 上：

[
I'(x)=I(x),[1-alpha(x)/2]
]

而 illuminant chromaticity label 不变。

如果这个乘子是对 RGB 三通道相同的 scalar，它本身不会改变 chromaticity GT；可以解释为一个与 alpha 相关的总照度 / shading field。因此它不必然构成 label inconsistency。

但它把 intensity 与 alpha 强绑定：

[
S(x)=1-alpha(x)/2.
]

这样如果实验有效，很难知道收益来自“真实场景确实存在强度变化”，还是来自这一特定负相关。

更合理的长期形式是 factorize：

[
E_c(x)=alpha(x)L_1+[1-alpha(x)]L_2
]

[
I_{m syn}(x)=B(x),S(x),E_c(x),
]

其中：

- (E_c(x))：chromaticity field，作为 virtual GT；
- (S(x)>0)：独立 scalar intensity / shading field，不进入 chromaticity GT。

先比较 (S=1) 与一个独立低频 (S(x))，再研究 (S) 与 alpha 的相关性。

## 7. 暂时降级的方向

### Same-scene endpoint pairing

AN/PS 已经说明简单的 source / whole-pair association 没有带来改善。它不能彻底否定 endpoint joint prior，但当前不值得占用主实验预算。

### 更多随机 alpha shape

M 已证明 spatial variation 有用，但 P、S 等结果没有支持“越复杂越好”。下一阶段优先控制 distribution，而不是继续增加 texture family。

### 三光源

LSMI 有三光源场景，但当前 two-light synthetic branch 还比真实 baseline multi 高约 0.7°。在 two-light 的 contribution field、intensity 与 sampling 尚未厘清前，直接加入三光源会同时增加多个自由度，暂列后续。

### 新 consistency loss

paired same-content views 天然允许 consistency/equivariance，但第一轮不要加。两个 view 已有准确 GT，应先证明 **paired virtual supervision 本身**有效，避免把贡献变成 loss engineering。

## 8. 最小实验矩阵

固定 One-Net、初始化协议、69,600 optimizer updates、两个 views/source/cycle、每 view 64 patches。

| 实验 | View 1 | View 2 | 唯一主要变化 |
|---|---|---|---|
| O | stochastic original/mixed | stochastic original/mixed | 当前 parent |
| Pair-M | identity | M mixed | 每个 source 强制双状态覆盖，共享 geometry |
| Pair-Q | identity | controlled-q mixed | 显式控制 global dominance |
| Pair-QT | identity | controlled q + tau | 再解耦 local contrast |
| Pair-QT-Samp | identity | QT + GT-aware sampling | 检验 minority/transition patch coverage |

如果预算更紧，只跑前三个：**Pair-M → Pair-Q → Pair-QT**。

每一步都同时报告 single / multi。主目标是：

- single 不丢掉 O 已恢复的能力；
- multi 从 O 的 3.2426° 明显向 baseline 2.5338° 靠近。

## 9. 决策规则

### 如果 Pair-M > O

说明 O 的收益不只是“每个 source 同时看到两个状态”；随机 illumination-state sampling 可能本身更适合当前优化。停止把 paired coverage 当主线，回到 mixed generator distribution。

### 如果 Pair-M < O，且 single 不退化

说明“每个 content 都覆盖 identity + mixed”是有效结构。后续所有 generator 实验都以 paired protocol 为 parent。

### 如果 Pair-Q 主要改善 multi

继续研究 q / tau / scale / sampling，这是最理想的信号。

### 如果 Pair-Q / QT 都无法明显改善 multi

不要继续堆 alpha 参数。优先转向：

1. intensity / contribution-field factorization；
2. linear-domain / black-level / normalization audit；
3. synthetic-vs-real mixed-light statistics；
4. 再决定是否需要 geometry/content-aware illumination support。

## 10. 最终方法论定位

我现在最推荐的主线不是：

> 50% original + 50% synthetic。

而是：

[
oxed{
	ext{有限 Content Bank}
	imes
	ext{无限 On-the-fly Illumination States}
}
]

其中每个 source 的真实 identity state 被稳定保留，synthetic branch 负责扩展真实数据难以覆盖的 illumination space。

Virtual GT 想超过真实 mixed-light supervision，应该利用的不是“合成图片更多”这个粗粒度优势，而是：

[
oxed{
	ext{content reuse}
+
	ext{exact supervision}
+
	ext{controllable illumination coverage}
+
	ext{targeted sampling}
}
]

这也是目前我认为最值得用固定 One-Net 验证的下一阶段策略。

## 11. 相关工作边界

LSMI 已展示 pixel-level relighting augmentation 能提升 multi-illuminant white balance，并提供 illuminant chromaticity、pixel-wise mixture ratio 与 dense GT。后续 pixel-wise color constancy 工作也强调 illumination map 的空间连续性。

因此不要把“同一 scene 做 relighting augmentation”本身作为 novelty。

更值得争取的贡献是：

> **在不使用真实 mixed image / mixture map / dense GT 构造训练监督的前提下，把单光源 content bank 与 illumination-state distribution 解耦，并系统证明受控 virtual dense supervision 如何在固定模型和固定训练预算下缩小乃至超过真实 multi-light supervision。**

重点参考仍包括 LSMI (ICCV 2021)、One-Net multi-illuminant extension、Dual-Illumination Weighting and Estimation、PWCC (ICIP 2024) 及 illumination decomposition / multi-scale pixel-wise estimation 工作。正式论文写作前应逐篇核验 novelty 与实验协议。
