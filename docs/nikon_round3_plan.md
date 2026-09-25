# Nikon 第三轮：单光源占比、光源主导比例与端点配对

Updated: 2026-09-25 13:46 +08:00 · Agent: codex · Status: protocol, implementation and calibration gates passed; formal O training running; no complete results

本页是 2026-09-25 13:46（Asia/Shanghai）的阶段快照，不是结果报告。协议已批准，实施与独立校准闸门均为 PASS / ADEQUATE，O 组已进入正式运行，尚无第三轮完整结果。这些通过仅覆盖相应启动条件，不代表性能验收通过。第二轮开发预选的 **AC** 是共同构造参照；不得因已查看的 test 表现更换参照或最终选择规则。以下机制都是待检验假设，未证明能改善真实混光泛化。

## 数据边界与成像假设

训练底图仅为 Nikon official train 的 668 个 `_1.tiff`，用 Light1 做 `B=RAW/Light1`。允许所有 train 全局白点，包括多光源场景 Light1/2/3 各光源颜色；不读取真实混光图像、dense GT、混合权重及派生信息来构造、训练、开发选型或调参。人工生成光照 map 可作监督。旧真实混光训练 baseline 只作比较，不作初始化或 teacher。[LSMI 官方说明](https://github.com/DY112/LSMI-dataset) 区分全局光源色度与逐像素 mixture map。

在线性相机 RGB 对角近似下，`I(x) ≈ rho(x) × sum_j[s_j(x) ell_j]`。B 仍含原始 shading，不是恢复后的反射率；`J=B×E` 是非负混光增强，不是完整光传输重建。One-Net 使用全局与局部特征，因而值得检验全图光源组成是否影响局部估计，但该推断不等于增强已经有效。[作者方法介绍](https://ilijad.github.io/zane.html)

固定 42,179 参数 modified One-Net、Eq.5、AdamW、69,600 次成功更新、原按步学习率日程、seed 0、每源两视图、每图 64 个 patch 及原几何增强。校准不计正式预算；正式训练重新加载冻结随机初始权重并新建优化器。

## 四个独立对照

| 组 | 相对 AC 的改动 | 可检验的问题与边界 |
|---|---|---|
| O | 用原始 Light1 恢复分支替换 25% 纯端点 global 分支，得到 50% 原始、0% 纯端点、50% M 混光；crop 与混光位置不变。 | 原始图与随机 global 样本的组成取舍，不增加场景或预算。 |
| B | 仅混光分支变为 `alpha'=alpha*exp(b)/(1-alpha+alpha*exp(b))`，每图 `b~U[-2,2]`；M 尺度不变且额外明暗 S=1。 | 改变每图主导光源比例，不把变化与新增 shading 混在一起。 |
| ER | 按合法白点对角色分层抽样，再在同角色、同实际消费有效 patch 贡献的组内打乱第二端点；其他分支不变。 | 相对 AC 是角色约束与边缘重加权的联合干预，不能只归因于排除同角色配对。 |
| EP | 采用与 ER 相同的抽样清单，不打乱第二端点；白点对随机分配给源图，不绑定回原场景。 | 对照 ER，控制端点边缘频数后检验同场景白点配对依赖，避免图像绑定混入因素。 |

角色指 metadata 的 Light1/2/3 编号，不是已知物理灯具类别。合法清单有 712 对：L1–L2 658、L1–L3 27、L2–L3 27；沿用既有端点有效性规则，暂不生成三端点空间场。白点对不都对应数据中独立的双光 composition 图。

ER/EP 每轮使用相同 pair index。ER 在分层组内对第二端点作随机非零循环位移，不足两条时保留并记录；有放回抽样会产生重复，打乱后仍可能同场景，必须记录残留率。每轮及全程的两端点实际有效 patch 贡献须与 EP 严格相同，包括最后不足整轮的截断部分。有效 mask 仅来自允许的源图和固定 patch 采样，并核查正端点不改变 mask。此控制不保证每个 batch 的 loss 平均权重逐项相同。

原图/global 分支在 B/ER/EP 中应与 AC 逐张一致，新随机流不得扰动 patch/crop/旋转翻转。关闭新增开关时须精确复现 AC 的五个训练张量；否则不能继承既有 AC 参照。

## 开发选择与最多一个组合

沿用第二轮冻结的 190 个 val 单光源场景、四类合成开发数据、seed 10001；端点仅取 train。分数仍是 `0.5 × real_single/M_real_single + 0.5 × mean(四类 synthetic/M 对应类)`，越低越好，M 为固定归一化参照。真实混光 GT 不参与这一步。

四组完成后，O 或 B 仅在开发分数严格低于 AC 时进入组合；ER/EP 中选分数较低者且须严格低于 AC，同分优先 ER。无因素支持则不新增组合，组合与已有单因素配置相同则复用 checkpoint，否则从相同 seed 0 初始化训练一个 COMBO，预算仍为 69,600 步。组合收益不能由单因素收益相加推定。

最终在 AC/O/B/ER/EP 及若存在的 COMBO 中按同一开发分数选唯一候选，精确同分依上述顺序。先登记全部开发结果、checkpoint 哈希和选择时间，再统一真实 val/test 评分所有组；不按 test 改选 winner。

## 启动闸门与验收

正式训练前须核查 AC 张量兼容、O 分支组成、B 的范围/恒等/有限性、ER/EP 角色与实际有效 patch 端点贡献严格相等、map 与 patch 标签对应、有限梯度和真实参数更新。代码、参数、来源访问清单和开发数据先冻结；异常留证停止，不跳过失败组。新干预的真实单步或一轮校准不得记为正式训练完成。

当前验收是 `val/test × all/single/multi × patch pooled / patch image-balanced / pixel image-balanced` 共 18 格。开发预选 seed 0 模型须全部严格低于原 seed 0 baseline 的同机复算值，再冻结配置补 seed 1/2；三 seed 算术均值也须逐格达标，并披露每个 seed 的全部结果。最多五个新增 seed 0 训练，达标后再追加两次重复；不宣称每个 seed 或统计上都优越。

历史 test 已被查看，本轮属于探索性迭代。官方 split、各模型图像清单和评测聚合保持一致，val/test 场景互斥；完成须有 checkpoint、完整评测和独立复核，不能用进程退出替代。几何/深度引导与额外明暗建模不属于本批实施范围。
