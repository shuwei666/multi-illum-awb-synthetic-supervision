# Diffuse Virtual GT v1：实际最终指标独立验收

updated: 2026-09-26; agent: codex independent reviewer

**独立验证通过，范围仅限已保存评分产物的数值、身份、支持集和冻结绑定。** 产物 PASS；契约 ADEQUATE。最终性能为 **FINAL_TARGET_NOT_MET**，不能将数值验收通过写成方法成功。O、A、B、C 相对 baseline 均为 **0/18** 格严格更低。

## 范围与方法

依据 [原任务](../USER_TASK.txt)、[用户修订](../REVISION_1.md) 和 [预先审计计划](final_metrics_audit_prepare.md)，在全新上下文用 Python 标准库 csv/math/hashlib 与 NumPy 直接读取 NPZ/CSV，未导入或调用生产 aggregate，未读取新 RAW/GT、运行 GPU 或训练。此前训练完成验收是 [completed_cohort_review](completed_cohort_review.md)，本次重新核对其哈希绑定，不以评分退出码替代训练证据。

执行：`python3 reviews/audit_final_metrics.py`，从本任务根目录运行，退出 0，24,276 项断言通过。独立脚本为 [audit_final_metrics.py](audit_final_metrics.py)，完整 90 格、72 个候选对 baseline 差值、36 个机制差值、逐场景汇总和输入 SHA256 见 [final_metrics_review.json](final_metrics_review.json)。

隔离有限：同模型、共享磁盘，全新上下文；只读边界靠操作约束，未声称操作系统强制隔离。仅写本 reviews 下的独立脚本与报告。

## 证据核对

- 完成清单恰含 30 个 model/split summary、CSV、NPZ，SHA256 全部匹配。评分实现、父项代码、5 个 checkpoint、各臂完成证据、评分许可和独立许可复核的绑定一致；逐图与逐场景 CSV 哈希一致。
- O/A/B/C 开发分数按独立公式 `0.5×real_single/reference + 0.125×四个 synthetic/reference 之和` 重算为 0.857398657031、0.925380521205、0.965579201470、0.961475202035。冻结最小者 O，与 O/A/B/C 平手顺序一致。记录的选择时间早于真实评分开始 259.250626 秒；这验证记录顺序，不证明从未看过历史 test。原任务已经披露历史 test 查看。
- val 为 394 图、190 场景，single/multi 为 190/204 图；test 为 204 图、97 场景，single/multi 为 97/107 图。图像顺序和唯一身份与官方参照一致；train、val、test 场景不交叉。
- 五模型的 valid patch 位置、每图有效 patch/pixel 数、灯数完全一致；每图均 256 个有效 patch。val 每图 262,144 个有效像素，test 每图 65,536 个有效像素，保留父项原有分辨率差异。官方 baseline patch 支持集另与 round2 NPZ 核对一致。
- 90 个最终均值最大重算差为 **1.3322676295501878e-15°**，低于预定 1e-12°。NPZ patch 与序列化 CSV/单模型 summary 最大差 **1.8700957298278809e-06°**，低于已有 2e-5° 容差，未修改阈值。
- baseline/O 与 round3 历史 NPZ/CSV 重算的全部 18 格各自完全相同，差为 **0°**；单模型历史 summary 所检查的 patch/pixel 均值差也为 0°。它们的 checkpoint 哈希与冻结记录一致。
- 1,196 条逐图 B-A/B-C 差值和 1,722 条逐场景、子集差值全部重算通过。场景内先对 composition 等权，跨场景汇总另行等权；不把同场景多 composition 当独立场景样本。

## 实际指标与判断

角度单位为度，越小越好。下表每格为 patch pooled / pixel image-balanced。所有图有效 patch 数相同，因此本批 patch image-balanced 与 patch pooled 数值相同；完整三种口径均保存在 JSON 中。

| 模型 | test all | test single | test multi |
|---|---:|---:|---:|
| baseline | 1.969242 / 2.167129 | 1.346506 / 1.346577 | 2.533778 / 2.910994 |
| O | 2.423858 / 2.537886 | 1.520702 / 1.520779 | 3.242606 / 3.459937 |
| A | 2.460820 / 2.587331 | 1.618154 / 1.618220 | 3.224732 / 3.465871 |
| B | 2.419867 / 2.545138 | 1.576391 / 1.576470 | 3.184514 / 3.423276 |
| C | 2.345681 / 2.472964 | 1.486679 / 1.486781 | 3.124402 / 3.366980 |

独立严格 `< 0` 判断确认：O/A/B/C 均未在任一冻结指标格超过 baseline，预选 O 也不能作为新臂成功。未达到 seed0 gate，更未完成多 seed 性能目标。

B-A 在 val 的全部 9 格为正，在 test 的全部 9 格为负：去旧明暗本次呈现开发/val 与 test 方向不一致的描述性差异。test all patch 为 -0.040953°、pixel 为 -0.042193°；val all patch 为 +0.067615°、pixel 为 +0.063600°。不能根据 test 的局部好转追认选型或宣称稳健机制成立。

B-C 在 val/test 全部 18 格均为正，场景等权后的 all/single/multi 两指标也全部为正。本批未提供“保留场景空间对应优于旋转错位”这一预测的有利信号。test all B-C patch 为 +0.074187°、pixel 为 +0.072174°；这不证明场景对应普遍无用，也不产生跨 seed 显著性结论。

## 覆盖边界与成熟度

正确性：保存数组到最终统计、选择公式、配对差值与 gate 通过。证据：哈希链、身份与支持集可追溯。复现性：独立 CPU 脚本可重跑上述审计。任务适配性：完整保留 baseline/O/A/B/C、18 格与负结果，未用 test 更换 winner。

**pixel 检查仅为已保存 CSV pixel 均值的再聚合**，另核对按有效像素加权的全 split summary；没有从原预测与真实 pixel GT 独立重算 native pixel 角误差。本次也未独立重建 patch GT，patch 验证起点是保存的逐 patch angular-error NPZ。

达到可追溯数值产物的 S1，不意味着达到性能效用 S2。没有验证真实物理 shading/normal 恢复，没有补做真实换光验证；TIFF 转换/黑电平仍为用户接受的工程假设。单 seed、历史 test 暴露、外部先验和 C 干预强度边界保持有效。本批应保留负结果并结束，不能依据这里的 test 子项追加调参或训练。
