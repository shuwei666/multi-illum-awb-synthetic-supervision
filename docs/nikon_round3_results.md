# Nikon 第三轮：完成评测，严格目标未通过

Updated: 2026-09-25 15:10 +08:00 · Agent: codex · Status: performance FAIL; contract ADEQUATE; operational correctness PASS

开发预选 **O** 的 18 项均值全部高于原 baseline，严格门槛 **0/18**；AC/B/ER/EP 也均为 0/18。O/B/ER/EP 四个新增训练各完成 69,600 次更新、417 cycles，AC 和原 baseline 复用既有 checkpoint。本轮没有新增 COMBO，**没有触发追加 seed 1/2 的授权条件**；整体目标尚未完成。

## 先开发选择，再真实评分

开发只用冻结的 190 个 val 单光源 RAW/Light1 及四类合成数据，合成端点只取 train 全局白点。分数为 `0.5 × real_single/M_real_single + 0.5 × mean(四类 synthetic/M 对应类)`，越低越好。

| 候选 | 冻结开发分数 ↓ |
|---|---:|
| AC | 0.886318 |
| O | 0.857399 |
| B | 0.897479 |
| ER | 0.907763 |
| EP | 0.903814 |

只有 O 严格优于 AC，故事前组合规则得到 O 原配置并复用 O checkpoint。最终选择于 **14:47:21 +08:00** 冻结；获独立完整产物验收后，统一真实 val/test 评测于 **14:59:03** 开始、**15:01:11** 完成。上述时间来自保存记录，不是防篡改时间认证。最终候选保持 O，没有按 test 改选。

O 将 AC 的 25% 纯端点 global 分支替换为原图，得到 50% 原始、0% 纯端点、50% 混光。其开发分数下降不代表所有开发子项改善，更不建立 test 达标结论。ER/EP 全程及逐轮实际消费的端点边缘贡献计数匹配，包含最后 128 步；这不保证 alpha 加权贡献或每 batch loss 权重相同。

## 全部模型 test 结果

平均角误差，单位度，越低越好。test 为 204 张图（97 single、107 multi），52,224 个有效 patch。patch pooled 是所有有效 patch 的均值；pixel image-balanced 是先求每图有效像素均值，再对图像等权平均。patch 预测广播到原生 GT 网格评分，不借 GT 平滑预测。

| 模型 | test patch pooled | test pixel image-balanced | 严格 18 格通过数 |
|---|---:|---:|---:|
| 原 baseline | 1.969242 | 2.167129 | — |
| AC | 2.431420 | 2.552636 | 0/18 |
| O（开发预选） | 2.423858 | 2.537886 | 0/18 |
| B | 2.429720 | 2.553529 | 0/18 |
| ER | 2.530040 | 2.651859 | 0/18 |
| EP | 2.490060 | 2.607614 | 0/18 |

## 冻结 O 与原 baseline 的全部子集对照

| split/subset | baseline patch | O patch | baseline pixel | O pixel |
|---|---:|---:|---:|---:|
| val/all | 1.794425 | 2.164356 | 2.040534 | 2.306758 |
| val/single | 1.240101 | 1.260342 | 1.240146 | 1.260383 |
| val/multi | 2.310706 | 3.006331 | 2.785995 | 3.281324 |
| test/all | 1.969242 | 2.423858 | 2.167129 | 2.537886 |
| test/single | 1.346506 | 1.520702 | 1.346577 | 1.520779 |
| test/multi | 2.533778 | 3.242606 | 2.910994 | 3.459937 |

当前契约完整保留 `val/test × all/single/multi × patch pooled / patch image-balanced / pixel image-balanced`，共 18 格。patch image-balanced 先求每图有效 patch 均值再等权平均；本次每图恰有 256 个有效 patch，所以两种 patch 口径数值一致，但仍分别检查，不能称为 18 个独立统计检验。val 有 394 张图（190 single、204 multi），100,864 个有效 patch；各模型身份、顺序与有效位置一致，val/test 场景互斥。

表格为保存的 float32 patch 误差经 float64 重聚合后的显示舍入值；判定使用未舍入数值。O test all patch/pixel 分别比 baseline 高 **0.454616° / 0.370757°**。开发预选 seed 0 必须先在 18 格全部严格低于同一个原 seed 0 baseline，才能固定配置追加 seed 1/2，并要求三 seed 算术均值仍逐格达标、披露每个 seed。本轮首道门槛失败，不能靠挑 seed、换 winner 或放宽阈值改写。

## 来源边界、独立复核与限制

训练图像只来自 668 个 official Nikon train `_1.tiff` 及 Light1 白平衡；允许 train 全局 Light1/2/3，包括真实多光源场景各光源的白点颜色。禁止真实混光图、pixel-wise GT、混合权重及其派生信息进入构造、训练、开发选型或调参；禁止混光训练 checkpoint/teacher 初始化。人工生成的 dense 光照 map 允许监督。白平衡后的底图仍含原 shading，不等于恢复反射率。

独立复核重聚合全部 6 模型 × 18 格，最大差异 8.89e-16°；核对 40 份本地/远端产物及 70 份完整训练审计产物哈希，确认 baseline/AC 跨轮预测与误差不变；另外在 val/test 首个单光/混光样例对六模型做 24 次原生像素均值检查，最大差异 3.25e-14°。结论为 **性能 FAIL、契约 ADEQUATE、运行正确性 PASS**，仅支持 S1 可追溯失败实验，不是 S2 性能成功。

复核未重训、未重新运行网络预测、未完整重算所有原始 patch GT，也未进行历史文件访问系统调用审计；训练来源结论依据冻结代码、来源清单和完整训练审计。新只读上下文与执行者仍共享文件系统/模型族，隔离有限。真实 GT 在冻结后的最终评分与其复核中使用，不能把这些分数转作开发数据。历史 test 早已查看，整个研究过程不能宣称 GT-blind；单 seed 失败不证明所有单光源构造方法无效。原 baseline 使用真实混光训练且只有一个固定 seed，本比较不建立统计显著性或唯一变量因果结论。

## 后续已预声明的 AN/PS 对照

AN/PS 协议于 **14:05:54 +08:00**、第三轮真实评分之前预声明。它测试源图与整对白点的关联：AN 使用本源 Light1 与合法同场景 Light2/3；PS 在角色及实际消费贡献匹配层中置换整对白点，保持两端点及联合对白点计数相同。10 个无合法伙伴的源保持 parent 构造不变。该对照不是仅针对第一端点的因果识别，也不是完整物理重光照或新颖性结论。

截至本快照，端点 helper 和 toy renderer 仅获 CPU 范围独立验证；完整 runner 接入、真实 Nikon 校准与正式训练仍待完成。构造 parent 按第三轮开发选择固定为 O，不用 O 权重初始化；是否启动由“严格目标未达成”这一二元结果触发，明确属于评测反馈。参数与两组设计不是依据第三轮 test 数值新拟定的方案。任何实际启动仍需实现、真实校准和独立验收，不因准备组件通过而自动放行。

## 证据身份

| 内部产物 | SHA256 |
|---|---|
| final selection | `bed54f28daf3f437cb605998700d5a15d82aace6ffdfbdd434f039f8db45c073` |
| evaluation manifest | `c6375f1565e89cc0e5b0a18c367c3abde7dd69905c71a050c38bc2cf6472053e` |
| strict comparison | `f1dd91bc8a6e5bc586485ab565d9f118c19d44a3497f8e1cd6a5b704bc6cfd17` |
| evaluation approval | `10ca1d0d62a925f701cb5b8acdcf8ea6b0285fb0a974f7b5288cae678a5046e4` |
| baseline checkpoint | `94d1cbf18b550156d46b6a201f87cc5d05619ab7e73c19983d7785bd382d39bb` |
| O checkpoint | `7670d875b96331df7e3c937748aee40d006e8147710ae77ad4962eabb7291351` |

完整 18 格、所有 checkpoint/code 哈希和支持样本数见[机读摘要](../results/nikon_round3/summary.json)；机制与冻结选择规则见[第三轮方案](nikon_round3_plan.md)。本次只发布文档与小型数值摘要，不发布私有数据、逐图输出、权重或训练实现；哈希只标识内部产物，不等于公开可独立复现。证据范围见 [EVIDENCE](../EVIDENCE.md)。
