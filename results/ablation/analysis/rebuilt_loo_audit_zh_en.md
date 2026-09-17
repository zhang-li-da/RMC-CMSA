# 新逐组件消融审计 / Audit of the new matched LOO campaign

> **Initial repository scope.** This checkout directly supports statistical reanalysis and validate-only checks. The engineering smoke evidence referenced near the end of this audit is retained in the complete local author package, not this initial repository. Running a new optimization campaign requires the external dependencies and licensing conditions in the repository's `DEPENDENCIES.md`.

## 数据与边界 / Data and scope

本分析仅使用 rebuilt_loo_selected.csv 的 200 行：PIDs 1、6、10、14，D=20，instance=1，10 个配对种子，5 个配置，每次上限 400,000 次评估。旧压缩包 formal_single_deletion 的 2,400 行、与新样本重叠的 pilot，以及工程 smoke 均未合并。选定四个问题的依据未在运行前的可核验协议中记录，因此这是探索性补充，不能视为完整 16 题确认性证据。

Only the new 200-row selected-case campaign is analyzed. Historical results, overlapping pilots and engineering smoke runs are excluded. No verifiable pre-run rationale for selecting these four cases was captured; this is exploratory evidence, not a confirmatory full-suite study.

已有 200 行缺少运行时 manifest 和最终解向量，无法事后补造。当前分析环境与分析源码哈希记录在 JSON 中，不等同于原运行环境。新 runner 今后在运行前保存协议、全部相关源码/数据哈希、环境，并保存每次的最终解向量与全部诊断；未记录逐代轨迹。

The 200 rows have no contemporaneous run manifest or returned-vector archive. Current analysis provenance is recorded separately. Future runs save a pre-run manifest, final decision vectors and full diagnostics, but no per-generation trace.

## 统计方法 / Statistical method

对每个 PID 独立比较 Full 与四个删除条件的 RPR、F1，共 32 个双侧配对精确符号置换 Wilcoxon 检验，统一 Holm 校正。零差删除，绝对差并列用平均秩；只在零差/并列识别时保留 12 位小数。检验假设是种子对独立，零假设下配对差满足符号可交换性/对称性。均差使用原精度。95% 区间来自 50,000 次配对 percentile bootstrap，仅作描述，未作多重校正；n=10，精度有限。总体均值不用于合并显著性检验。

There are 32 within-PID, two-sided exact signed-rank sign-permutation tests, with a single Holm family. Zero differences are discarded and ties use average ranks. The null requires independent seed pairs and sign exchangeability/symmetry. Bootstrap intervals are descriptive and unadjusted; no pooled across-problem test is used.

## 结果 / Results

Holm 校正后显著比较为 0/32，最小校正 p=0.25。完整方法没有稳定优于每个删除条件。以下总体均值保留不利结果：

| Configuration | Mean RPR | Mean F1 |
|---|---:|---:|
| RMC-CMSA | 0.40125 | 0.53916 |
| RMC-NoMaximin | 0.39750 | 0.53673 |
| RMC-NoRelations | 0.40375 | 0.55335 |
| RMC-NoTerminalMemory | 0.39875 | 0.53845 |
| RMC-NoTrajectoryEvidence | 0.40375 | 0.55275 |


Full–NoMaximin 在 PID1 的 RPR 均差为 +0.08；Full–NoTrajectoryEvidence 在 PID10 为 +0.11，而在 PID14 为 -0.07。应同时呈现这些差异，不能据此宣称所有模块都有独立显著收益。

Full shows case-dependent benefits and losses: the RPR differences above favor Full on PID1/NoMaximin and PID10/NoTrajectoryEvidence but favor the deletion on PID14/NoTrajectoryEvidence. This evidence does not establish an independently significant benefit for every component.

## 控制检查与代码修复 / Controls and runner fixes

冻结算法核心未改。NoRelations 将关系候选替换为等量均匀候选；NoMaximin 随机选候选；NoTerminalMemory 关闭终止记忆读写；NoTrajectoryEvidence 关闭观测和注入。200 行所有关闭路径计数为零，启用组件在各相应配置整体被触发。Full 工作记忆在 38/40 次运行中激活，但全部 200 行 working_promotions 均为零；不能将路径计数当成因果贡献或晋升收益。

The frozen core is unchanged. Matched deletions disable the intended paths; enabled paths are exercised at campaign level. Full working memory is active in 38/40 runs, but working promotions are zero in all 200 runs. Path access does not prove causal influence or promotion benefit.

修复范围是 runner：固定五条件、执行后自动验证、缺失诊断直接报错、完整网格和配对种子/预算检查、禁用路径断言、运行前 manifest、与源码/环境匹配的续跑保护、逐运行原始解存档。旧无 manifest CSV 只能 validate-only，不能追加。小预算 smoke 仅校验工程路径，可不触发所有组件，并显式标记 engineering_smoke。

The runner now enforces five conditions, automatic validation, strict counters, the complete planned grid, paired seeds, budget checks, disabled-path assertions, source/environment-compatible resume, and raw returned-vector artifacts. Existing unmanifested CSVs are validate-only. Tiny smoke runs are labeled engineering-only and may leave enabled paths dormant.

模块之间仍存在合理交互：删除 maximin 后，终止记忆即使被读取，也未必影响随机选择。这不是开关失效，也不应通过追求显著性来改写算法。另一个历史 ladder 对照中候选数不等，未在本次修改中修正，也未作为组件独立贡献的证据。

Component interactions remain: with maximin removed, computed terminal-memory scores may no longer influence random selection. This does not imply a defective switch. The separate ladder comparison has a candidate-pool-size confound; it is unchanged and is not used here as isolated component evidence.


## 工程核验 / Engineering verification

独立工程 campaign：PID1、D=5、1 个种子、5 条件、每次 10,000 次评估上限（实际 9,942--9,950）。五个配置均完成，启用组件整体触发，禁用计数为零。保存的最终解（每次 5--6 个）及目标值可以离线重算，并与 CSV 的 RPR/F1 一致。相同协议重启跳过全部 5 个已验证结果；更改预算的续跑、追加无历史 manifest 的 200 行 CSV、非五配置 CLI 都被拒绝。数值自检另覆盖缺行/整对缺失、重复键、错配种子、NaN、超预算、缺失/负数诊断、禁用路径被触发。详情见 rebuilt_loo_engineering_checks.json 与 analysis manifest。这 5 行不属于统计样本。

An independent engineering campaign used PID1, D=5, one seed and five conditions at a 10,000-evaluation limit. It completed all conditions and retained 5--6 final vectors per run. Saved objective values and RPR/F1 agree with offline rescoring. An unchanged resume skipped all five verified rows; changed-budget resume, append to legacy unmanifested data and non-five-condition commands were rejected. Additional synthetic validation failures are documented in the analysis manifest. These five smoke rows are excluded from the 200-row study.

Reproduce the statistical analysis and its checks:

```text
python -B code/pipeline/analyze_rebuilt_loo.py --self-check
```

Run a new engineering campaign (choose a new output name if the source hashes have changed):

```text
python -B code/pipeline/run_rebuilt_loo_ablation.py --pids 1 --dim 5 --repeats 1 --budget-scale 0.1 --workers 1 --engineering-smoke --out results/ablation/analysis/rebuilt_loo_engineering_release.csv
```

Validate the existing 200 rows without appending or claiming retroactive runtime provenance:

```text
python -B code/pipeline/run_rebuilt_loo_ablation.py --pids 1 6 10 14 --dim 20 --repeats 10 --out results/ablation/rebuilt_loo_selected.csv --validate-only
```
