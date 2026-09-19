# Four-variant component-deleted profile

This folder contains the supplementary-only profile requested for the revised
paper. It is intentionally independent of the main-text D=20 Full-versus-
deletion audit. The four runnable configurations are exactly:

```text
RMC-NoMaximin RMC-NoRelations RMC-NoTerminalMemory RMC-NoTrajectoryEvidence
```

Run CEC2013 (official dimensions are read from each FID):

```powershell
python -B code/supplementary_component_profile/run_cec2013_deleted_profile.py `
  --configs RMC-NoMaximin RMC-NoRelations RMC-NoTerminalMemory RMC-NoTrajectoryEvidence `
  --fids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 `
  --runs 30 --seed-base 20260808 --workers 4 `
  --out results/supplementary_component_profile/cec2013_deleted_profile.csv
```

Run one CEC2026 dimension at a time:

```powershell
python -B code/supplementary_component_profile/run_cec2026_deleted_profile.py `
  --configs RMC-NoMaximin RMC-NoRelations RMC-NoTerminalMemory RMC-NoTrajectoryEvidence `
  --pids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 `
  --pin 1 --dim 2 --repeats 30 --seed-base 20260808 `
  --budget-scale 1.0 --safeguard --workers 24 `
  --out results/supplementary_component_profile/cec2026_d02.csv
```

Repeat the CEC2026 command for `--dim 5`, `10`, and `20`, changing the output
file to `cec2026_d05.csv`, `cec2026_d10.csv`, and `cec2026_d20.csv`.
The present `cec2026_d20.csv` is a source-compatible archived 30-run profile
filtered from the historical formal summary; its provenance note is stored in
`cec2026_d20.provenance.txt` and it may be replaced by a current-run file.

After all files are complete, build the ranking tables with the authoritative
macro-score analyzer:

```powershell
python -B code/pipeline/make_component_profile_material.py `
  --cec2013 results/supplementary_component_profile/cec2013_deleted_profile.csv `
  --cec2026 results/supplementary_component_profile/cec2026_d02.csv `
             results/supplementary_component_profile/cec2026_d05.csv `
             results/supplementary_component_profile/cec2026_d10.csv `
             results/supplementary_component_profile/cec2026_d20.csv `
  --names CEC2026-D2 CEC2026-D5 CEC2026-D10 CEC2026-D20 `
  --repeats 30 --outdir results/supplementary_component_profile/material
```

The analyzer ranks only the four deleted variants within each problem. Its
tables are exploratory compatibility profiles: with Full omitted, a high rank
describes the performance of the remaining mechanism set and is not a causal
estimate of the deleted component's marginal contribution.
