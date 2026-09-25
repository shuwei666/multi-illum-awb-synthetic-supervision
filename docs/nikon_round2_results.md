# Nikon 第二轮：冻结选择与实测结果

Updated: 2026-09-25 · Agent: codex · Status: completed; performance target not met

第二轮五个新增训练 C/A/AC/L/S 各完成 69,600 次更新、417 cycles。M 复用第一轮 checkpoint，baseline 复用历史真实混光训练 checkpoint，二者本轮未重训。真实混光最终评分前按冻结开发分数选出的候选为 **AC**；AC 与其他全部候选均为 **0/18** 格严格低于 baseline，多 seed 目标尚未完成。独立复核确认该失败结论：性能契约 FAIL、契约 ADEQUATE，支持 S1 可追溯失败实验，不是性能通过；复核与执行共享磁盘、同模型族，隔离有限。

## 选择先于最终评分

开发数据是冻结的 190 个 val 单光源场景及其四类合成开发图，合成端点只取 train 全局白点。选择分数为 `0.5 × real_single/M_real_single + 0.5 × mean(四类 synthetic/M 对应类)`，越低越好；不读取真实混光 GT 作开发选型。

| 候选 | 开发分数 ↓ | 构造 |
|---|---:|---|
| M | 1.000000 | 既有多尺度场 |
| C | 1.000693 | M + crop |
| A | 0.893802 | M + 原图/纯端点分支 |
| AC | **0.886318** | crop 与分支组合，最终预选 |
| L | 0.890364 | AC + 端点小扰动 |
| S | 0.979857 | AC + 空间贡献与明暗 |

AC 先成为 L/S 的共同构造参照，最终开发比较仍选择 AC。最终选择登记于 **2026-09-25 13:21:49 +08:00**，随后执行统一真实 val/test 评分，完成状态于 **13:24:29 +08:00**登记。时间来自执行记录，选择与评测清单哈希见[机读摘要](../results/nikon_round2/summary.json)。不得按 test 结果改选 winner。

## 全部 test 结果

单位为平均角误差（度），越低越好。test 共 204 张 composition 图：single 97 张、multi 107 张；共有 52,224 个有效 patch。patch 为有效 patch pooled mean；pixel 为每图像素 AE 均值的等权平均。预测 patch 块广播到 dense GT 尺寸进行像素评分，未以 GT 平滑预测。

| 模型 | patch all | patch single | patch multi | pixel all | pixel single | pixel multi |
|---|---:|---:|---:|---:|---:|---:|
| 原 baseline | 1.96924 | 1.34651 | 2.53378 | 2.16713 | 1.34658 | 2.91099 |
| M | 2.72580 | 2.00299 | 3.38104 | 2.82495 | 2.00306 | 3.57002 |
| C | 2.74188 | 2.03288 | 3.38462 | 2.83762 | 2.03293 | 3.56711 |
| A | 2.45066 | 1.63999 | 3.18558 | 2.56590 | 1.64006 | 3.40521 |
| AC（预选） | 2.43142 | 1.68796 | 3.10540 | 2.55264 | 1.68802 | 3.33644 |
| L | 2.45039 | 1.68073 | 3.14811 | 2.56975 | 1.68079 | 3.37563 |
| S | 2.74676 | 1.91797 | 3.49808 | 2.86273 | 1.91802 | 3.71915 |

数字来自保存的 float32 patch 误差经 float64 重聚合，以及逐图 pixel 均值。原 summary 使用 float32 汇总时会有末位差异；判定使用未四舍五入值。本次每图均有 256 个有效 patch，故重聚合的 patch pooled 和 patch image-balanced 相同；两种口径仍单独保留，不将 18 格当成 18 个独立统计检验。

val 共 394 张（single 190、multi 204），与 test 场景互斥。每个 split 中所有模型使用相同图像清单。AC val all patch/pixel 为 2.24221° / 2.38935°，baseline 为 1.79442° / 2.04053°；全部 val/test 数值、支持样本数和 artifact 哈希已纳入[机读摘要](../results/nikon_round2/summary.json)。

## 当前验收与结论边界

当前目标是 `val/test × all/single/multi × patch pooled / patch image-balanced / pixel image-balanced` 共 18 格严格低于同机原 seed 0 baseline。开发预选 seed 0 模型必须先通过全部 18 格，再固定配置追加 seed 1/2；三 seed 算术均值亦须逐格达标，并披露每个 seed 全部结果。

AC test all patch/pixel 比 baseline 分别高 0.46218° / 0.38551°；所有候选均有 18/18 格高于 baseline。AC 相比 M 的 test patch mean 下降 0.29437°，仅支持本次固定 seed/预算下的观察；A 同时改变原图与纯端点样本组成，不能单独归因于其中一个分支。不能通过更换 baseline、放宽阈值、按 test 换 winner 或挑 seed 最小值改写失败。

旧执行流程在 L 训练后因输出 BrokenPipe 中断，失败记录保留。冻结训练代码未变，恢复流程完成余下开发评分、S 训练、选择和统一评分；“执行完成”和“性能达标”是不同结论。启动阶段的 PASS / ADEQUATE 仅覆盖启动条件，结果不应标为性能通过。

允许所有 train 全局 Light1/2/3 白点，包括多光源场景中各光源颜色；候选训练图像只用 train `_1.tiff` 与 Light1。真实混光图、dense GT、混合权重及派生信息禁止进入构造、训练和开发选型，人工生成 map 可以监督。AE 评分依赖评测 GT，真实无 GT 推理时无法据此挑结果。

历史 test 已被查看，本轮是探索迭代；baseline 使用不同训练来源，比较不是唯一变量因果消融。单 seed 结果不证明统计优越性，也不证明单光源路线普遍不可行。本次仅公开文档与数值摘要，私有数据、权重、逐图文件和训练实现没有随本次发布，哈希不等于可独立复现。
