# Diffuse Virtual GT v1：seed0结果与证据边界

agent: codex

## 01 判定与边界

冻结开发预选：O；最终判定：FINAL_TARGET_NOT_MET。单 seed0 闸门通过：False；多 seed 总目标未完成。

通过条件是预选的新候选在全部18格严格低于原baseline；任何一格未通过均不能称总体成功。即使seed0通过，也不代表统计显著、完成多seed目标或证明新颖性。本批评分后结束，不按test细节改参数。历史test曾被查看，见USER_TASK。


## 02 对照与资源边界

| 方案 | 定义 | 解释 |
| --- | --- | --- |
| baseline | 本项目冻结的modified One-Net baseline | 固定性能参照；非重新复现作者完整协议 |
| O | 既有数据组成父项 | 不新增本轮外部先验的既有参照 |
| A | W × H | 保留旧明暗，叠加新照明 |
| B | (W / s0) × H | 与A使用相同H及色度GT |
| C | (W / s0) × rot180(H) | 重新计算图像、GT和中性目标 |

B-A检验有限度去旧明暗；B-C检验新照明与场景的空间对应。B-O包含外部先验及有效性变化，不能全部归因于去旧明暗。三臂保留相同技术有效性和父项未加权patch标签。


## 03 冻结开发五项与预选

五项误差单位为度，越低越好；开发score无量纲，沿冻结公式。只按开发score选型，同分依次O、A、B、C，最终test不参与改选。

| 子项 | O | A | B | C | B-A | B-C |
| --- | --- | --- | --- | --- | --- | --- |
| real_single | 1.2604235410690308 | 1.2808669805526733 | 1.3360438346862793 | 1.307713508605957 | 0.05517685413360596 | 0.028330326080322266 |
| global | 1.9031167030334473 | 1.8491690158843994 | 2.0485661029815674 | 2.079348564147949 | 0.19939708709716797 | -0.030782461166381836 |
| sigmoid | 2.921599864959717 | 3.1792025566101074 | 3.3419950008392334 | 3.336125135421753 | 0.16279244422912598 | 0.005869865417480469 |
| blob | 3.0955817699432373 | 3.5657711029052734 | 3.6325743198394775 | 3.622471809387207 | 0.0668032169342041 | 0.010102510452270508 |
| lowfreq | 2.945733070373535 | 3.6504838466644287 | 3.6966915130615234 | 3.726715564727783 | 0.04620766639709473 | -0.030024051666259766 |

| 候选 | score | 预选 |
| --- | --- | --- |
| O | 0.8573986570311704 | True |
| A | 0.9253805212050032 | False |
| B | 0.9655792014703064 | False |
| C | 0.9614752020351813 | False |


## 04 完整18格最终误差

单位：度，越低越好。patch pooled汇总全部有效patch；patch image-balanced先算每图patch均值再等权平均；pixel image-balanced先算每图pixel均值再等权平均。all为全部图，single为单光源，multi为多光源。所有方案沿原评价身份和测试mask；这些误差依赖评价GT，真实推理不能获得。数值保留JSON原精度。

| split / group / metric | baseline | O | A | B | C |
| --- | --- | --- | --- | --- | --- |
| val/all/patch_pooled | 1.7944247106351383 | 2.164356416630464 | 2.170340199324514 | 2.2379554520908815 | 2.159853979902173 |
| val/all/patch_image_balanced | 1.7944247106351383 | 2.164356416630464 | 2.170340199324514 | 2.2379554520908815 | 2.159853979902173 |
| val/all/pixel_image_balanced | 2.0405344519578694 | 2.30675834694191 | 2.313835930100291 | 2.3774357776598043 | 2.3091061521999925 |
| val/single/patch_pooled | 1.240101148980083 | 1.2603419357118821 | 1.2808423601885903 | 1.3359993077647232 | 1.307617628679643 |
| val/single/patch_image_balanced | 1.240101148980083 | 1.2603419357118821 | 1.2808423601885903 | 1.3359993077647232 | 1.307617628679643 |
| val/single/pixel_image_balanced | 1.2401455894147886 | 1.2603825585365738 | 1.280893117524677 | 1.336072000998325 | 1.3076484372069073 |
| val/multi/patch_pooled | 2.310706459235435 | 3.0063306880742413 | 2.9987940691079724 | 3.0780126453358334 | 2.9536035227074713 |
| val/multi/patch_image_balanced | 2.310706459235435 | 3.0063306880742413 | 2.9987940691079724 | 3.0780126453358334 | 2.9536035227074713 |
| val/multi/pixel_image_balanced | 2.785994667071524 | 3.281324032221391 | 3.2758905104403233 | 3.3473334127856926 | 3.241836376948454 |
| test/all/patch_pooled | 1.9692417243970304 | 2.4238576383274504 | 2.460819726738348 | 2.419867154830689 | 2.345680527507605 |
| test/all/patch_image_balanced | 1.9692417243970304 | 2.4238576383274504 | 2.460819726738348 | 2.419867154830689 | 2.345680527507605 |
| test/all/pixel_image_balanced | 2.1671289093194748 | 2.5378864060742408 | 2.587331360275501 | 2.5451380509784767 | 2.472963674597706 |
| test/single/patch_pooled | 1.3465057009695753 | 1.5207020019352995 | 1.6181535584197713 | 1.5763910071266192 | 1.4866785563896987 |
| test/single/patch_image_balanced | 1.3465057009695753 | 1.5207020019352995 | 1.6181535584197713 | 1.5763910071266192 | 1.4866785563896987 |
| test/single/pixel_image_balanced | 1.3465772262103686 | 1.5207788399969573 | 1.6182204559929607 | 1.5764704697639278 | 1.486781046075546 |
| test/multi/patch_pooled | 2.5337781194667794 | 3.2426062058979044 | 3.224732047550516 | 3.1845137560203596 | 3.1244019405771084 |
| test/multi/patch_image_balanced | 2.5337781194667794 | 3.2426062058979044 | 3.224732047550516 | 3.1845137560203596 | 3.1244019405771084 |
| test/multi/pixel_image_balanced | 2.9109935192408134 | 3.4599371902751415 | 3.4658711520082712 | 3.423275951705684 | 3.366979702323403 |


## 05 机制差值与局限

差值单位：度。B-A或B-C小于0表示B误差更低，大于0表示B更高；仅为单seed描述性观察。相同场景的多个composition不是独立样本，场景bootstrap亦不能代表训练seed不确定性。没有干预强度证据时，不宣称空间对应机制得到充分验证。

B-A：B在18格中的9格误差更低；test/all/patch_pooled差值为-0.040952571907658886度。这是描述性差异，不是统计显著性结论。

B-C：B在18格中的0格误差更低；test/all/patch_pooled差值为0.07418662732308423度。这是描述性差异，不是统计显著性结论。

| split / group / metric | B-A | B-C |
| --- | --- | --- |
| val/all/patch_pooled | 0.06761525276636737 | 0.07810147218870833 |
| val/all/patch_image_balanced | 0.06761525276636737 | 0.07810147218870833 |
| val/all/pixel_image_balanced | 0.0635998475595132 | 0.0683296254598118 |
| val/single/patch_pooled | 0.055156947576132875 | 0.028381679085080158 |
| val/single/patch_image_balanced | 0.055156947576132875 | 0.028381679085080158 |
| val/single/pixel_image_balanced | 0.055178883473647966 | 0.02842356379141764 |
| val/multi/patch_pooled | 0.07921857622786099 | 0.12440912262836212 |
| val/multi/patch_image_balanced | 0.07921857622786099 | 0.12440912262836212 |
| val/multi/pixel_image_balanced | 0.07144290234536932 | 0.10549703583723868 |
| test/all/patch_pooled | -0.040952571907658886 | 0.07418662732308423 |
| test/all/patch_image_balanced | -0.040952571907658886 | 0.07418662732308423 |
| test/all/pixel_image_balanced | -0.04219330929702414 | 0.07217437638077051 |
| test/single/patch_pooled | -0.041762551293152095 | 0.08971245073692047 |
| test/single/patch_image_balanced | -0.041762551293152095 | 0.08971245073692047 |
| test/single/pixel_image_balanced | -0.04174998622903292 | 0.08968942368838184 |
| test/multi/patch_pooled | -0.040218291530156325 | 0.06011181544325117 |
| test/multi/patch_image_balanced | -0.040218291530156325 | 0.06011181544325117 |
| test/multi/pixel_image_balanced | -0.042595200302587344 | 0.05629624938228073 |

[per_image_deltas.csv](results/per_image_deltas.csv)

[per_scene_deltas.csv](results/per_scene_deltas.csv)


## 06 离线成本、训练成本与校准

| 阶段 | 耗时秒 | 峰值allocated bytes | 其他 |
| --- | --- | --- | --- |
| 全来源缓存 | 824.2384779453278 | 2126633472 | 668来源；3011816286 bytes；不可用比例0.0 |
| 渲染preflight | 106.31759510189295 | 19186227200 | render_preflight_only |
| A正式训练 | 1405.7233691513538 | 19186913280 | 69600成功更新；417cycles |
| B正式训练 | 1415.929714076221 | 19186913280 | 69600成功更新；417cycles |
| C正式训练 | 1478.7248073071241 | 19186913280 | 69600成功更新；417cycles |

| 校准臂 | 状态 | 更新数 | 耗时秒 | 峰值allocated bytes |
| --- | --- | --- | --- | --- |
| A | calibration_complete | 2 | 48.96136770397425 | 19186398720 |
| B | calibration_complete | 2 | 48.66125974059105 | 19186398720 |
| C | calibration_complete | 2 | 50.47614850103855 | 19186398720 |

校准独立于正式初始化与预算；preflight不是optimizer或性能证据。峰值allocated不等于整卡占用。训练耗时含本轮在线构造，不能将历史耗时作为本次全流程耗时。

| 训练臂 | 在线加权均值秒/update | 在线最小秒/update | 在线最大秒/update |
| --- | --- | --- | --- |
| A | 0.01928009741661278 | 0.018885339403937676 | 0.021792019644897142 |
| B | 0.019426129315819204 | 0.01911699664806891 | 0.0217816864926658 |
| C | 0.020218280044480643 | 0.018947694651380985 | 0.022442091547418386 |

在线均值按每cycle实际更新数加权；范围为各cycle的秒/update。包含该cycle构造与更新时间，不包含全流程离线缓存与所有保存开销。

| 训练臂 | 来源图数 | 生成视图slots | 实际消费视图slots | 实际有效patch slots | 实际混光视图slots | 实际混光有效patch slots |
| --- | --- | --- | --- | --- | --- | --- |
| A | 668 | 557112 | 556800 | 35423630 | 278403 | 17712121 |
| B | 668 | 557112 | 556800 | 35423630 | 278403 | 17712121 |
| C | 668 | 557112 | 556800 | 35423630 | 278403 | 17712121 |

来源图数为cache_manifest中的668。每cycle每来源生成两视图，因此生成数=cycles×1336；实际消费四项直接累加history字段。尾轮截断使生成数与实际消费数不同。视图slot和patch slot是重复生成/消费记录，不是独立照片或独立观测；来源图数也不能直接等同独立场景数。


## 07 外部先验与未验证事项

| 记录 | 证据内容 |
| --- | --- |
| licenses | {"Intrinsic": "academic-only custom", "DSINE": "Imperial academic/noncommercial custom", "MiDaS": "MIT", "chrislib": "no LICENSE in frozen snapshot; no redistribution claim"} |
| external_training_sources | {"Intrinsic": ["CGIntrinsics", "OpenRooms", "Hypersim", "GTA", "MID-Intrinsics (Murmann)"], "DSINE": ["Cleargrasp", "3D Ken Burns", "Hypersim", "SAIL-VOS 3D", "TartanAir", "MVS-Synth", "BlendedMVG", "Taskonomy", "Replica", "Replica+GSO", "EfficientNet pretraining"]} |
| nikon_overlap | "UNKNOWN; not independently audited" |

Intrinsic灰度分解及DSINE法线为外部学习先验，不是实测albedo或法线；camera-linear proxy为域外近似，近似内参不等于准确几何。未恢复完整BRDF、镜面/透明、互反射或真实投影阴影。TIFF黑电平与转换来源仍UNVERIFIED，按REVISION_1接受工程假设。真实换光物理验证：NOT_PERFORMED。风险区域可进入整图上下文，技术mask不保证纯漫反射。


## 08 独立审查与可追溯证据

下列内容为独立审查文件原文；本生成器不自行授予PASS，不将报告生成视为验收通过。

# Diffuse Virtual GT v1：实际最终指标独立验收

updated: 2026-09-26; agent: codex independent reviewer

**独立验证通过，范围仅限已保存评分产物的数值、身份、支持集和冻结绑定。** 产物 PASS；契约 ADEQUATE。最终性能为 **FINAL_TARGET_NOT_MET**，不能将数值验收通过写成方法成功。O、A、B、C 相对 baseline 均为 **0/18** 格严格更低。

## 范围与方法

依据 [原任务](USER_TASK.txt)、[用户修订](REVISION_1.md) 和 [预先审计计划](reviews/final_metrics_audit_prepare.md)，在全新上下文用 Python 标准库 csv/math/hashlib 与 NumPy 直接读取 NPZ/CSV，未导入或调用生产 aggregate，未读取新 RAW/GT、运行 GPU 或训练。此前训练完成验收是 [completed_cohort_review](reviews/completed_cohort_review.md)，本次重新核对其哈希绑定，不以评分退出码替代训练证据。

执行：`python3 reviews/audit_final_metrics.py`，从本任务根目录运行，退出 0，24,276 项断言通过。独立脚本为 [audit_final_metrics.py](reviews/audit_final_metrics.py)，完整 90 格、72 个候选对 baseline 差值、36 个机制差值、逐场景汇总和输入 SHA256 见 [final_metrics_review.json](reviews/final_metrics_review.json)。

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

[独立最终指标审查](reviews/final_metrics_review.md)

| 证据文件 | SHA256 |
| --- | --- |
| USER_TASK.txt | 7488b5be071894e25144554ffee225188872e5656270b42eba501f699942424f |
| REVISION_1.md | ad6b4f6da46b2a8ed72a548320a09d5a4ff85a71c8b334097e564e2f6325a176 |
| run_manifest.json | af5371a9efa15564074aa8cc1264fd95469a77fe900b4383fa6fdb8c75b7442f |
| frozen/manifest.json | 4886059347f1ddf4f9d9f285cc21cb8a2be9a47d948ec940d98a71e1365b7803 |
| development_selection.json | 674ede698fb96310b3ec985c3453e3513657e5737c3a2afaa34dae05e1d73081 |
| results/summary.json | d2f72d42dbe50941062ea38ddf3f3866d4d650836a7cd0edda05206efcc7fc36 |
| results/evaluation_manifest.json | 5c7c094ac6265c1b49f01a7929f804e052dac7e8925d9bc8ed98514a5af16c1d |
| cache_manifest.json | 807bf9573ecd5cef4d01e5e96502461797f242dcd20023a8b54af4106cc00ae0 |
| cache_progress.json | 02131a1b7572296eb24960830f11013a8d483d0778ca62e3a60da79e72ec1049 |
| preflight_001.json | 568e595204939a8e54414d56151bf529aa97fa0e6575a82476350485d9e5e54a |
| priors_manifest.json | 565ddb655741922a3ef4db5d5eb4288c5b47b8c517fdee832d7f1da02ee401e2 |
| build_report.py | 96a3a3eb755239621b5769d5f17590304a979387bb1681ea41b29310365773c9 |
| reviews/final_metrics_review.md | 07e035c5f6af5f1834d8502f729cdef376b8666a84dd23af842c67074d2b8439 |

[USER_TASK.txt](USER_TASK.txt)

[REVISION_1.md](REVISION_1.md)

[run_manifest.json](run_manifest.json)

[frozen/manifest.json](frozen/manifest.json)

[development_selection.json](development_selection.json)

[results/summary.json](results/summary.json)

[results/evaluation_manifest.json](results/evaluation_manifest.json)

[cache_manifest.json](cache_manifest.json)

[cache_progress.json](cache_progress.json)

[preflight_001.json](preflight_001.json)

[priors_manifest.json](priors_manifest.json)

[build_report.py](build_report.py)

[reviews/final_metrics_review.md](reviews/final_metrics_review.md)
