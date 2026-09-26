# 全668离线缓存入口独立审查

updated: 2026-09-26; agent: codex (read-only reviewer)

结论：PASS，许可范围仅 `all_source_cache_only`，可启动 `--all` 生成668来源离线先验缓存；不批准校准或正式训练。本审查继承 qc_artifact_review.md 的物理/材质/暗部可视性限制，不把扩大缓存视为解决这些限制。

本次逐项检查 `diff -u qc_code_002.py cache_priors.py`，并用AST确认 load_priors 与 normal_inference 完全相同，当前脚本语法解析通过。读源、全局Light1白平衡、proxy、gray模型调用、uninvert后resize、median及[0.25,4]保护、法线与镜像诊断的推断主体无变更。preview仅增加是否保存PNG开关；其数值计算和固定scene种子未改，因此缓存批次顺序不会改变QC灯场种子。神经网络GPU算子的位级确定性未另验证，不声称重跑逐位一致。

冻结审查代码hash：

- qc_code_002.py：`8f6a6c6d2ab4aaa3ff8f63feb029838b19fb18e881889c1e1a87f7675597304f`。
- cache_priors.py：`2887970af598093412c3bcbc7d2be77859cd5dd8fd1fece6fbea048d8c3505f6`。
- render_diffuse.py沿QC审查：`b1c5514311ceb7ef4278a922ad33512272682801a04c43f1b5fb68f2c2e478a4`。

`--all` 仅按source_manifest已有668来源顺序运行，没有新增混光/GT路径。输出隔离到 cache_all、cache_metadata、cache_progress.json、prior_runtime_all.json、cache_manifest.json；不覆盖旧24 PNG、JSON、NPZ或prior_runtime。非空NPZ输出目录直接拒绝，不能静默覆盖失败批次。每源继续核输入hash，权重和来源manifest继续核hash，strict加载保持不变。

入口要求 cache_approval.json 的PASS及scope=`all_source_cache_only`，并核evidence中所有文件hash。审查时该批准文件尚未创建，故目前运行会fail-closed；主控应在启动前将本审查、qc_artifact_review、冻结runtime及实际cache_priors/render_diffuse代码hash纳入evidence，并核远端副本一致。该文件由主控据此审查生成，无需再次询问用户；它不是训练批准。

进度每图写入，最终cache_manifest列每来源cache/input hash、来源和先验manifest hash、代码hash、总字节和耗时。不可用定义为256固定QC格点中0个共同可用patch，这是工程缓存检查口径，不能代替训练随机crop/patch消费账。超过20%不可用最终抛错、不产生本次complete manifest；此阈值在全量推断后执行，非提前中止。cache_progress 的INFERENCE_COMPLETE仅指模型推断结束，不能单独当成功条件：还须进程exit 0、最终manifest存在且含668唯一身份、unusable_source_fraction≤0.2、文件hash与实际内容一致。失败状态检测由主控运行封装负责，本脚本异常会保留已有文件但不会自行写FAILED。

潜在较保守支持集、未知传感器饱和、camera RGB域外代理、近似内参、骨干前向等价未证明及外部训练重叠UNKNOWN均保持原披露。此增量不通过更改mask或挑样消除这些风险。训练runner正另行修订，与本缓存许可分离。
