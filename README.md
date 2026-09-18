# RMC-CMSA

Research materials for **Relational-Memory State-Driven Restart Allocation for Multimodal Optimization**.

This repository contains the revision's experiment drivers, the author-controlled RMC-CMSA controller source, selected result records, and reproducible statistical analysis. It is a partial research release: external adapters, the CMSA/RSC kernel, and third-party benchmark implementations are not included. Statistical reanalysis works with this checkout; running new optimization experiments requires the dependencies described below.

The original controller source is in `code/controller_src/`. It is provided for inspection and extension but is not a standalone executable checkout because its external imports are intentionally documented separately in [ORIGINAL_SOURCE_STATUS.md](ORIGINAL_SOURCE_STATUS.md).

## Reproduce the statistical analysis

From the repository root, using Python 3.13 and a virtual environment:

```text
python -m pip install -r requirements-revision.txt
python -B code/pipeline/analyze_rebuilt_loo.py --self-check --outdir reproduced/ablation --tex-dir reproduced/tables
```

This command validates all 200 rows and recomputes the descriptive summaries, component diagnostics, 32 paired tests, Holm correction, bootstrap intervals, and LaTeX tables. It does not import or run the excluded optimizer or benchmark. Generated analysis metadata describe the current reanalysis environment.

To validate the recorded results without running optimization:

```text
python -B code/pipeline/run_rebuilt_loo_ablation.py --validate-only --pids 1 6 10 14 --dim 20 --repeats 10 --seed-base 20260808 --out results/ablation/rebuilt_loo_selected.csv
```

## What is included

- `code/pipeline/`: the rebuilt LOO driver, its shared experiment driver, and the statistical analysis script. Their contents match the revised local research package.
- `code/controller_src/`: the four author-controlled controller modules used by the method; see `ORIGINAL_SOURCE_STATUS.md` for the import boundary.
- `results/ablation/rebuilt_loo_selected.csv`: 200 new runs, comprising four problem IDs (1, 6, 10, 14), ten paired seeds, and five conditions, at D=20 and instance=1. The nominal budget is 400,000 evaluations per run.
- `results/ablation/analysis/`: recorded descriptive statistics, diagnostics, paired comparisons and the retrospective analysis manifest.
- `results/ablation/*pilot*.csv`: earlier exploratory pilot records, retained for transparency. These overlap with the selected campaign and must not be merged into its statistical sample.
- `results/revision_allocator_scaling/`: all 120 exploratory geometric trials across D=2, 5, 10, 20, 50, 100, their summaries, protocol and figures. These are allocator geometry diagnostics, not objective-optimization results; the generating optimizer is not included in this initial release.
- `RELEASE_FILES.sha256`: checksums of this repository's initial research files.

## Evidence and limitations

The LOO study compares Full against removal of maximin selection, relational proposals, terminal memory, and trajectory evidence. None of the 32 within-problem component/metric comparisons is significant after a single Holm correction; the minimum adjusted p-value is 0.25. The pooled Full means are RPR=0.40125 and F1=0.53916. These results do not establish an independent benefit for every component.

The four selected problems and ten seeds constitute exploratory evidence. The 200 runs lack contemporaneous source/environment manifests and archived final solution vectors; the retrospective analysis manifest cannot reconstruct them. The rebuilt driver saves those artifacts for future campaigns and refuses incompatible resumes. Existing results are retained in full, including unfavorable comparisons.

The geometric trials show increasing boundary clipping at larger dimension and do not demonstrate an advantage of combined relational proposals over equal-budget uniform proposals under the tested states. They do not establish high-dimensional optimization performance.

Historical benchmark and baseline results from the full local package are not part of this initial repository. In particular, this release does not resolve the historical CEC2013 external-evaluator discrepancy or establish equality of target-count information across baselines.

## Optimizer dependencies and licensing

See [DEPENDENCIES.md](DEPENDENCIES.md) for the files needed to execute new optimization campaigns and [LICENSE_STATUS.md](LICENSE_STATUS.md) for the publication boundary. Some external components have noncommercial conditions; the CMMOP redistribution terms have not been confirmed. No blanket license is applied to those components. An author-selected license for the original scripts is still pending, so public visibility should not be interpreted as an unrestricted reuse grant.

The paper and internal revision report are distributed separately from this initial source-and-results repository. Citation metadata are in [CITATION.cff](CITATION.cff).
