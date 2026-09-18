# RMC-CMSA source release

This repository contains the author-controlled source code and experiment drivers for **RMC-CMSA (Relational-Memory Coverage for Multimodal Optimization)**.

The public release is intentionally source-only. It includes the four controller modules and the scripts used to launch and analyze experiments. Benchmark data, run outputs, manuscript files, review reports, and third-party optimizer/source trees are not published in this repository.

## Source layout

- `code/controller_src/`: author-controlled controller modules.
- `code/pipeline/`: experiment drivers and statistical analysis scripts.
- `DEPENDENCIES.md`: required external packages and redistribution boundaries.
- `ORIGINAL_SOURCE_STATUS.md`: source ownership and completeness boundary.
- `LICENSE_STATUS.md`: current licensing status.

The controller modules are not a standalone executable checkout. Running a new campaign additionally requires the excluded CMSA/RSC kernel, CMMOP benchmark package, and external adapters described in `DEPENDENCIES.md`. The repository therefore makes no claim of one-command independent reproduction.

## Full-PID leave-one-component-out driver

`code/pipeline/run_rebuilt_loo_ablation.py` supports the protocol used in the manuscript: CEC2026 PID1--PID16, instance 1, dimension 20, five configurations, and matched repeated seeds. `code/pipeline/analyze_full_pid_loo.py` validates and analyzes a complete campaign when the required CSV, per-run artifacts, benchmark metadata, and source manifests are supplied locally. No campaign outputs are included here.

## Citation

Please cite the associated manuscript using [`CITATION.cff`](CITATION.cff).

## License and dependencies

No blanket license is applied to the complete dependency stack. The author-controlled scripts remain subject to the authors' rights until a reuse license is selected. Third-party components retain their own terms; consult [`DEPENDENCIES.md`](DEPENDENCIES.md) before use.
