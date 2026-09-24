---
title: "从单光源参考构造多光源监督：LSMI Sony 实验链复盘"
date: 2026-09-24
draft: false
slug: "multi-illuminant-synthetic-supervision"
tags: ["AWB", "多光源", "合成监督", "颜色恒常"]
categories: ["Imaging"]
toc:
  enable: true
  auto: true
---

# 从单光源参考构造多光源监督：LSMI Sony 实验链复盘

## 结论先说

这组实验的核心不是换了一个更复杂的网络，而是把真实多光源逐像素标注的采集问题，转化成一个可控的成像模型与数据构造问题：

1. 从 LSMI Sony 的单光源白平衡参考图获得近似反射率；
2. 从光源颜色池抽取两盏端点光源；
3. 人工构造空间混合权重 `alpha(x)`；
4. 重新生成多光源输入及其精确已知的逐像素光源图；
5. 用这些合成样本训练同一个 `UNetDirect`，再在真实 LSMI Sony 多光源场景上验证和测试。

最干净的 `_1`-only 合成监督结果是 **3.225°**；加入 `_12` 分支扩展反射率来源后，最佳 pilot 达到 **2.834°**。但后者存在 `_12` 反射率 provenance 不完整、真实 dense validation 选模和 test 多轮开发复用等边界，因此不能表述为严格的 “raw-only、label-free、独立测试” 结果。

## 九项冻结事实

1. 全部核心结果属于 **LSMI Sony 单传感器**，不是 Galaxy AIDM 实验。
2. **NUS 只提供早期 relighting 的端点光源颜色池，不提供反射率图像。**
3. 反射率主体从一开始就来自 LSMI Sony；最初主要是 `_1` 单光源白平衡参考。
4. 后续的“扩反射率”是把来源从 `_1` 扩到 `_1 + _12`，不是第一次引入 LSMI。
5. random、depth、physical、confidence-guided 和 shadow-aware 改变的是空间混光 `alpha(x)`；基本 relighting 方程不变。
6. 训练的梯度监督来自人工生成的逐像素光源图；真实多光源 dense train map 不进入 synth-only 主线的训练 loss。
7. **真实 Sony val 227 场景的 dense GT 每轮参与最佳 checkpoint 选择。**它不是 synthetic validation。
8. 最终 `t4_clean_ft` 是从 `t3_expand_phys/best.pt` 出发，在同一扩反射率 physical-alpha 合成集上让全部网络参数继续训练 50 epoch，而不是新建一种数据或冻结部分网络。
9. 真实 Sony test 114 场景用于报告结果，但也在多轮方法开发中被反复查看，因此 2.834° 是最佳 practical pilot，不是严格封存后的独立最终测试。

## 1. 三个容易混淆、但彼此独立的轴

整个实验同时改变过三件不同的东西。

| 轴 | 早期设置 | 后续设置 | 它实际改变什么 |
|---|---|---|---|
| 反射率来源 | LSMI Sony `_1` | LSMI Sony `_1 + _12` | 场景/材质外观的多样性 |
| 端点光源池 | NUS 光源颜色 | LSMI Sony train 的 `ell_1/ell_2` | 合成光源色度与目标域的匹配程度 |
| 空间混合图 | random alpha | depth / physical alpha 等 | 两盏光在图像空间中的分布方式 |

因此，“换 NUS 为 LSMI”和“从 `_1` 扩到 `_1 + _12`”不是同一件事：前者换的是光源颜色分布，后者扩的是反射率来源。

## 2. 单个合成样本怎样生成

### 2.1 反射率近似

在简化的线性成像模型下：

\[
I(x)=r(x)\odot \ell(x)
\]

早期 `_1` 分支以 LSMI Sony 的单光源白平衡参考作为反射率近似。代码归档把它描述为：`LSMI _1_gt white-balanced`。因此它不是“任意未标注 RAW”，而是依赖单光源参考资产。

在保存过的合成样本中，也可以通过

\[
r(x)\approx I(x)/(\ell(x)+\epsilon)
\]

恢复出 `r`。

### 2.2 两盏端点光源

选择两个全局光源颜色 `L1`、`L2`。早期 baseline 从 NUS 光源池抽取；后来改为从 794 个 LSMI Sony train 场景的 `ell_1/ell_2` 建立目标域光源池，共记录约 1581 个有效光源。

这一步只换光源颜色，不换反射率图像。

### 2.3 空间混合图

构造逐像素权重 `alpha(x)`：

\[
\ell(x)=\alpha(x)L_1+[1-\alpha(x)]L_2
\]

实验过的方案包括：

- random-alpha：水平、垂直、Perlin、blob 等随机空间模式；
- depth-alpha：把深度压成单调空间权重；
- physical-alpha：`depth -> 3D -> surface normal -> Lambertian shading`；
- confidence-guided depth-alpha；
- shadow-aware physical-alpha；
- 额外 RAW noise。

### 2.4 同时生成输入和标签

合成输入为：

\[
I_{syn}(x)=r(x)\odot \ell(x)
\]

因为 `L1`、`L2` 和 `alpha(x)` 都由生成器控制，`ell(x)` 本身就是精确已知的逐像素训练 GT。relighting 操作在各阶段基本相同，变化的是 `r`、光源池或 `alpha(x)` 的来源。

## 3. `_1` 与 `_12` 到底是什么关系

### 3.1 第一波：`_1`-only

第一波合成数据主要使用 LSMI Sony 单光源 `_1` 白平衡参考作为反射率源，794 个 train 场景形成约 794 份基础反射率。

这一阶段已经完整执行 relighting：抽取两盏光、构造 `alpha(x)`、生成合成多光源图和逐像素 GT。并不是到了 `_12` 阶段才开始 relighting。

### 3.2 第二波：`_1 + _12`

后来发现单光源反射率来源的多样性有限，于是又读取 processed `_12` 分支中的 `r`，把反射率源从约 794 扩展到约 1588：

```text
_1  white-balanced reference  ─┐
                               ├─ reflectance pool ─ relighting ─ synthetic input + dense GT
_12 processed r               ─┘
```

两波数据使用的是同一类 relighting 逻辑；主要区别是反射率池扩大了。

但当前归档只确认 `build_t3.py` 读取 processed `_12` 的 `d['r']`，缺少生成这个 `r` 的完整上游 provenance。不能排除它在预处理阶段依赖真实 `_12_gt`。因此：

- **3.225°**：较干净的 `_1`-only、dense-training-map-free feasibility result；
- **2.834°**：`_1 + _12` 最佳 pilot，不能在 provenance 补齐前作为最严格的无真实 dense GT 证据。

## 4. 按时间顺序重建训练过程

### 阶段 A：NUS 光源池的早期基线

- 反射率：LSMI Sony `_1` 来源；
- 光源池：NUS 端点光源颜色；
- alpha：random-alpha；
- 网络：`UNetDirect`，约 31M 参数；
- 测试：真实 LSMI Sony test 114 场景；
- test mean：**3.800°**。

早期完整训练脚本已不在当前归档中，因此其所有超参数与 checkpoint 选择细节标记为未验证；现有实验表确认的是数据角色和测得结果。

### 阶段 B：把光源池换成 LSMI Sony

`build_synth_lsmi.py` 保持同一批 `_1` 反射率，重新从 LSMI Sony train 的 `ell_1/ell_2` 抽取端点光源，并分别生成 random-alpha 与 physical-alpha 数据。

| run | 初始化与训练 | test mean |
|---|---|---:|
| `B0_lsmi` | random-alpha，from scratch 120 epoch | 3.351° |
| `B0 -> randomFT` | 从 B0 出发，同类 random 数据继续 50 epoch | 3.289° |
| `B0 -> phys` | 从 B0 出发，physical-alpha 数据继续 50 epoch | **3.225°** |
| `phys-scratch` | physical-alpha，from scratch 120 epoch | 3.227° |

NUS 3.800° 到 LSMI 3.351° 的约 0.45° 改善，是整个链条中最大的单项提升，说明目标域光源分布对齐比复杂 alpha 设计更重要。

### 阶段 C：扩展反射率来源

在 LSMI 光源池基础上，把反射率源从 `_1` 扩展为 `_1 + _12`，约 794 变为 1588。记录中的 `t1_expand_ft` 达到 **3.176°**，相对对应的 3.289° 改善约 0.113°。

历史记录写明该阶段属于从 LSMI-light 实验 checkpoint 继续训练的 100-epoch/early-stop-40 任务，但当前归档缺少当时的完整 `build_task12.py/run_task12.sh`，所以更细的初始化 checkpoint 与最佳 epoch 不作推测。

### 阶段 D：扩反射率 + physical-alpha

`t3_expand_phys` 从 `t1_expand_ft/best.pt` 初始化，使用 `_1 + _12` 扩展反射率与 physical-alpha 合成数据，训练 50 epoch：

- batch size 48；
- lr `1e-4`；
- Adam `beta1=0.5`；
- linear decay，epoch 25 开始；
- 全部网络参数继续优化；
- real Sony val 选择 best checkpoint。

真实 test mean 达到 **2.922°**。与训练量匹配的 random-FT control 比较表明，physical-alpha 的受控独立贡献约为 0.109°，而不是把全部改善都归因于几何先验。

### 阶段 E：`t4_clean_ft` 再继续 50 epoch

“再训练 50 轮”指的是：

```text
t3_expand_phys/best.pt
        ↓ load all model weights
同一 synth_t3_phys_expand 数据，全部参数可训练
        ↓ 50 epochs, bs=48, lr=1e-4
每轮用真实 Sony val dense GT 计算 AE 并选择 best
        ↓
t4_clean_ft/best.pt
```

它没有引入第三批新数据，也没有冻结 encoder 或只训练某个 head；训练脚本加载模型权重后新建 Adam，并对全部参数反向传播。

归档 checkpoint 的内嵌信息为：

- 计划训练 50 epoch；
- 最佳 checkpoint 位于 epoch index 43，即第 44 轮；
- 保存时 val AE 为 2.8851°；
- 最终真实 test mean 为 **2.8338°**。

## 5. Validation 和 test 的真实作用

用户记忆中的“validation 也使用真实 test 数据”需要拆成两个概念：

- **Validation**：真实 LSMI Sony val，227 场景。训练期间每个 epoch 都计算真实 dense GT 误差，并据此保存 `best.pt`。
- **Test**：真实 LSMI Sony test，114 场景。它不参与梯度，也不直接选择单次训练中的 best checkpoint；但历史开发中多次用于比较 random、physical、expanded、shadow、noise 等方案，实际上已承担部分开发集作用。

因此，真实 dense GT 的使用应分三层报告：

| 环节 | 是否使用真实多光源 dense GT | 作用 |
|---|---:|---|
| synth-only 梯度训练 | 否 | loss 使用人工构造的 `ell(x)` |
| checkpoint 选择 | 是 | 真实 val dense GT 选择每个 run 的 best |
| 方法开发与结果报告 | 是 | 真实 test 被多轮查看并报告 |

准确术语是：**no real multi-illuminant dense maps in gradient-based parameter training**，而不是 fully label-free、raw-only 或 completely annotation-free。

## 6. 结果总表

| 阶段 | 反射率 | 光源池 | alpha / 训练动作 | 真实 test mean |
|---|---|---|---|---:|
| A | LSMI `_1` | NUS | random-alpha baseline | 3.800° |
| B1 | LSMI `_1` | LSMI train | random-alpha scratch | 3.351° |
| B2 | LSMI `_1` | LSMI train | B0 后 random-FT 50ep | 3.289° |
| B3 | LSMI `_1` | LSMI train | B0 后 physical-FT 50ep | **3.225°** |
| C | LSMI `_1 + _12` | LSMI train | 扩反射率 fine-tuning | 3.176° |
| D | LSMI `_1 + _12` | LSMI train | expanded + physical，50ep | 2.922° |
| E | 同 D | LSMI train | 从 D best 再继续 50ep | **2.834°** |

最终 114 场景 test 分布：

- mean：2.8338°；
- median：2.3920°；
- W25：4.9774°；
- W5：7.9473°；
- worst：13.5654°。

所有结果目前均为单 seed。real-only 4.177°、real-mixed 3.099° 与 synth-only 2.834° 的训练预算和历史不同，不能直接组成公平的论文主表。

## 7. 哪些增强有效，哪些没有

### 已观察到有效

- 合成增强本身：real-only 4.177° 到约 3.2°；
- 目标域光源池：NUS 3.800° 到 LSMI 3.351°；
- physical-alpha：在训练量匹配的对照中约改善 0.065–0.135°；
- 扩反射率：约改善 0.113°；
- 在已收敛 checkpoint 上继续适量训练：2.922° 到 2.834°。

### 当前没有稳定收益

- confidence-guided depth-alpha；
- shadow-aware cast shadow；
- 额外 RAW noise。

这说明主要能力来自数据构造、数据多样性和目标域分布对齐；physical-alpha 是有统计支持但幅度较小的精修项，网络始终是同一个 `UNetDirect` 验证载体。

## 8. 可防御的研究结论

当前最稳妥的结论是：

> 给定同传感器的单光源线性参考场景、端点光源先验和可控的空间混光先验，可以不把真实多光源 dense map 放入梯度训练，构造出有效的逐像素多光源监督，并在真实同传感器场景上取得接近或优于历史 real-mixed run 的 pilot 结果。

还不能声称：

- 任意 RAW、零标签即可训练；
- 2.834° 已经证明严格 dense-GT-free；
- 114 张 test 仍是完全独立测试；
- 单 seed 结果可以外推到其他相机；
- 2.834° 与 3.099° 构成公平训练预算下的正式胜负。

真正的下一版严格协议应使用 `_1`-only source、固定训练步数或 synthetic validation 选模、三 seeds、冻结 test，并补第二传感器验证。

## 9. 公开证据索引

本报告依据的归档文件包括：

- `README.md`：项目状态与结果链；
- `experiments.md`：实验表、配对检验与历史 run；
- `code/build_synth_lsmi.py`：LSMI 光源池、random/physical alpha 和 relighting；
- `code/build_t3.py`：`_1 + _12` 扩反射率与 physical-alpha；
- `code/train_direct.py`：训练、真实 validation 与 best checkpoint 选择；
- `code/run_t3.sh`：t3 初始化和50轮训练配置；
- `server_artifacts/eval_lsmi_exp.json`：3.351°、3.289°、3.225°、3.227°；
- `server_artifacts/eval_phys_results.json`：NUS 光源池基线与 physical-alpha 对照；
- `viz/test_stats.json`：2.834° 的 test 分布；
- 最佳 checkpoint SHA256：`f59d53302550a8e59da3bbc58c6dca7d2971ba18b485f3e684932cd6040c829f`。

当前没有公开原始数据、模型权重或服务器凭据。本报告是对已有实验资产的可追溯复盘，不代表重新运行或独立复现了全部结果。

