# Stage-zero independent acceptance

updated: 2026-09-26; agent: codex (fresh review context)

产物 verdict：**PASS，仅限阶段0的受阻审计报告与清单**。契约 verdict：**PROVISIONAL**。整体结论：**验证受阻（缺输入来源和训练兼容决策）**；不是训练闸门通过，也不是方法有效性验证。当前报告准确区分已做的来源盘点/代码检查/合成数学测试，与未做的外部先验推断、实际网络输入敏感性、24图QC、缓存、校准、训练和最终评分。

## 独立检查

- 重新枚举 train `*_1.tiff` 文件名并对照 split：668个，scene唯一、无漏项或额外项。逐条核对清单 Light1 与全局 metadata 的 float32 表示；重新计算允许端点及排除规则：1,353个端点，10个排除，全部匹配。meta/split SHA256、24图固定选择及访问清单均匹配。
- 使用固定随机种子20260926随机抽核3个来源文件SHA256：Place187 = `40309818573d98e268a86d0b0b4e9398af59d4457f073acd20a44ecd19f2e35f`；Place153 = `3d539a6856dbe418f64880d03189d3eab70b21c7ac88360bf2a2686c67117b97`；Place116 = `167100826d4a8163c2cfff9cc1de179070cddbacb34bc69d329483eb8454c0e0`。全部一致。未重新解码/哈希全部668张；全量尺寸、dtype、范围仍来自被审查脚本及其结果。
- 直接检查 audit_inputs.py：只构造 train单光源文件、meta和split路径；不是操作系统访问隔离或系统级历史读文件证明。此次复核也未读真实mixed图、mixture map或dense GT。
- 独立进程设置 PYTHONDONTWRITEBYTECODE=1，以 importlib 加载 test_math.py、unittest 运行，4项通过，0错误/失败；没有进入脚本main，没有覆盖 test_report.json。这里只覆盖合成代数，不覆盖官方uninvert、DSINE轴向、实际材质、随机流配对或真实输入通道。
- 阅读父代码 round2.py:49–90、round3_synthesis.py:77–155、frozen/domislovic_v2.py:303–306。实际O路径、RGB转换、/Light1/4、half存储、crop及C/H/W共同z-score与最新版00_audit一致。父GT是全16×16均值后向量归一化，确实不是有效像素加权/掩码均值；报告指出的兼容缺口成立。

## 未覆盖及下一闸门

未独立重跑服务器连接、官方预处理仓库检查、全盘RAW来源搜索、历史运行/发布核查；相关内容属于执行者已有记录，本验收未补作强证明。清单不能证明TIFF线性与黑电平来源，4项代数测试也不能替代生产管线验收。原始任务内“共同灯功率只改I尺度”的表述与中位数归一公式不一致，报告已公开纠正其解释。继续执行前仍须闭环来源/光度证据，明确父标签兼容、C有效性传播、曝光上限与随机流；禁止将本记录用于授权正式训练。

复核版本：source_manifest.json SHA256 `107377500af89fc67a95a9f8095a92826957057880e2f0371baff81980c96924`；audit_inputs.py `74e0859334d5d5cf96433df495ef5e7033f0be020663931e7152359eef9d3413`；test_math.py `0257cbb2db3360fb33d7562980159f263fa2de716d95279e20b8aa2c9117bd97`；test_report.json `92eb5c94d364b29a00665042a92604b03be302cd00a1e0c4e59d6b33de8b2d46`。00_audit采用本次修正为round2实际loader的版本。

隔离说明：这是新的只读验收上下文，但使用同模型、共享磁盘，**隔离有限**，没有操作系统强制只读隔离。仅写入本验收记录，未修改实现、阈值、数据或训练状态；未进行训练。
