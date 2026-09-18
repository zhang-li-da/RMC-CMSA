# Dependencies for new optimization campaigns

This source-only repository includes experiment drivers and analysis scripts, but no experiment CSV files or run artifacts. Statistical reanalysis requires separately supplied local inputs; a new optimization campaign also requires the excluded optimizer and benchmark dependencies. The published files do not constitute a complete optimization implementation on their own.

Running a new LOO campaign additionally requires the following layout from the full author research package:

```text
code/frozen_four_component/code/
    rmc_cmsa_final_four_component.py
    modular_rmc_cmsa_v12.py
    modular_rmc_cmsa_v3.py
    base_algorithm.py
    external_baselines.py
    CMMOP/...
code/baseline_src/RSCMSAESII_v1/...
```

The four author-controlled modules are published for inspection under `code/controller_src/`; copy them into the author-package layout above only after obtaining the dependencies. The full-PID analyzer also requires the matching CSV, per-run artifacts, source manifests, and benchmark PID metadata, which are not part of this public snapshot.

The original `external_baselines.py` contains implementations or adaptations of other baselines in addition to the small RSC adapter used by RMC-CMSA. Its redistribution terms must be checked before it is published, or the adapter must be separated and that packaging change validated. `base_algorithm.py` imports CMMOP at module level, so CMMOP is also required to import the optimizer.

RSCMSA-ESII supplies the CMSA search kernel, including `OptimOption.py`, `OptimProcess.py` and `Subpopulation.py`. The supplied `license.pdf` states CC BY-NC-SA without a version, permits noncommercial/academic use and requires citation of DOI [10.1109/TEVC.2021.3117116](https://doi.org/10.1109/TEVC.2021.3117116). Commercial use requires contacting its rightsholder. Obtain the source and applicable terms from that rightsholder; this repository does not grant rights to it.

The supplied CEC2026/CMMOP sources credit Ali Ahrari (2026), but no redistribution license was found in the audited files. Obtain that benchmark and its terms from its author or official distribution. No unverified download URL is supplied here.

Once these dependencies are lawfully available in the expected layout, a fresh campaign can be launched with:

```text
python -B code/pipeline/run_rebuilt_loo_ablation.py --pids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --pin 1 --dim 20 --repeats 10 --seed-base 20260808 --safeguard --workers 4 --out results/ablation/new_campaign.csv
```

Use a new output filename and configure BLAS to use one thread per worker for the manuscript protocol. This command requires the full local dependency layout and has not been executed against the partial public checkout. Earlier statistical reanalysis was validated using research-package inputs that are not included in this source-only release.
