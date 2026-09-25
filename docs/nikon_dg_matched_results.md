# Nikon One-Net：双高斯匹配实验 R1

updated: 2026-09-25; agent: codex

两组匹配训练和统一评分均完成，但未达到原 baseline。加入受限明暗的 DG-CS 在本次单 seed 比 DG-C 略好；两组整体仍不如 O。冻结开发集仍选择 O，没有根据 test 更换候选。

## 1. Test 结果

| 配置 | Test all patch | Test all pixel | Test single patch | Test multi patch |
|---|---|---|---|---|
| 原 One-Net baseline | 1.969242 | 2.167129 | 1.346506 | 2.533778 |
| O | 2.423858 | 2.537886 | 1.520702 | 3.242606 |
| DG-C：高斯比例 | 2.495608 | 2.624675 | 1.695535 | 3.220908 |
| DG-CS：比例＋明暗 | 2.460909 | 2.587955 | 1.687082 | 3.162417 |

角误差单位为度，越低越好。Patch 为有效 patch 汇总均值；pixel 为先对每张图的有效像素求均值、再对图像等权平均。上述真实误差需要评测 GT，不是推理时可获得的置信度。完整 val/test × all/single/multi × 三种均值，共18格，见 strict_comparison_001.json。DG-C、DG-CS 各0/18低于原baseline。

DG-CS 的多光源 test patch 均值低于 O（3.162417 对3.242606），但单光源更差（1.687082 对1.520702），整体仍更差。这里只能说本次受控实验中明暗项比不加明暗略好，不能据单seed宣称稳定收益，也不能将二维高斯称为真实三维光照。

## 2. 构造与批准修订

来源限668张Nikon train单光源图与Light1；允许1353个train白点，包括多光源场景的全局白点，不使用真实混光图、混光权重或pixel-wise GT构造训练数据。保留O的网络、原图/合成比例、crop和patch采样；双高斯独立宽度/方向生成alpha，S为median归一化总强度并限制到[0.25,4]。两臂E、GT和采样相同，仅DG-CS输入增加S。

用户同意后，两臂对同一视图采用共同整图二次幂曝光系数，按C/CS反事实峰值保留60000以内的数值余量；不逐像素clip，不改变GT。原图分支不变。每臂417周期中的115个对应生成视图触发相同缩放，最小系数1/4。原pipeline_001失败保留；CPU原溢出视图与GPU full/tail复检通过。现有TIFF黑电平/线性来源仍未闭环，属于明确接受的O继承假设，不是物理来源验证。

## 3. 执行与证据边界

两组均从同一冻结seed0随机初始化和fresh AdamW开始，各69600次更新、417周期，不从O或校准权重续训。DG-C用时937.981秒，DG-CS952.339秒。完整endpoint采样记录逐字节一致，共同构造统计逐周期一致。开发分数O=0.8573986570、DG-C=0.9150515595、DG-CS=0.9002775599，越低越好；评分前已冻结选择。没有追加seed、改参数或重选test最优。

独立复核分开覆盖契约、入口、真实单步、完成队列和最终数值；同模型与共享磁盘，隔离有限。最终pixel复核是CSV均值重聚合，不冒充重新读取逐像素GT评分。此前已查看历史test，不能声称整个研究过程完全test-blind。此次已完成授权两组实验，原性能目标未达成。

## 4. 单光源与多光源 test

Test 单光源 97 张（24,832 patches），多光源 107 张（27,392 patches）。

| 方案 | 单光源 Patch | 单光源 Pixel | 多光源 Patch | 多光源 Pixel |
|---|---:|---:|---:|---:|
| 原 baseline | 1.346506 | 1.346577 | 2.533778 | 2.910994 |
| O | 1.520702 | 1.520779 | 3.242606 | 3.459937 |
| DG-C | 1.695535 | 1.695603 | 3.220908 | 3.466917 |
| DG-CS | 1.687082 | 1.687143 | 3.162417 | 3.404579 |

## 5. 早期 Sony 实验补充（不同协议，不横向比较）

历史记录中的扩源是约 794 张单光源来源图，加入约 794 张双光源来源图，合计约 1,588 张；不是增加 794 个独立场景，也不是最终合成图数量。仅单光源 physical 分支 3.225°，扩源并继续 physical 微调后 2.922°，再训练 50 epoch 后 2.8338°；全链条约下降 0.391°（12.1%），不能全部归因于扩源。random-alpha 分支扩源前后 3.289° → 3.176°，但训练预算并非严格匹配。双光源底图预处理 provenance 未闭环，不能排除真实 dense GT 的上游使用。详见 [Sony 历史报告](sony_historical_report.md)。

## 6. 记录入口

- [完整18格及哈希](../results/nikon_dg_matched/summary.json)
- [冻结开发选择](../results/nikon_dg_matched/final_selection.json)
- [最终数值独立复核](../results/nikon_dg_matched/final_metrics_review.md)
- [独立复核机读结果](../results/nikon_dg_matched/final_metrics_independent.json)
- 完整 checkpoint、训练日志、预览和完成队列复核保存在 DATA_4T 的 `Research/onenet_nikon_single_source_20260925/dg_matched/`，本次仅同步轻量报告与结果，不上传数据集及模型权重。
