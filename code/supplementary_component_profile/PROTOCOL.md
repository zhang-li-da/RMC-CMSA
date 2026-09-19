# Supplementary component-deleted performance profile

This campaign is separate from the main-text leave-one-component-out audit.
The main text keeps its existing Full-versus-deletion experiment unchanged.
This supplementary campaign runs only the four deleted configurations:

- `RMC-NoMaximin`
- `RMC-NoRelations`
- `RMC-NoTerminalMemory`
- `RMC-NoTrajectoryEvidence`

Each benchmark case and configuration uses 30 independent runs. Seeds are
derived from `seed_base=20260808` using the frozen runner rules. CEC2013 uses
F1--F20 at each function's official dimension. CEC2026 uses PID1--PID16,
instance 1, at `D=2,5,10,20`, with the nominal `20,000 D` evaluation budget.
CEC2026 PID1--PID8 have 20 target optima and PID9--PID16 have 10 targets;
this target-count metadata is recorded for interpretation and is not used to
change the controller budget or ranking metric.

The ranking tables compare the four deleted configurations within each case:

- CEC2013: descending mean PR and SR at `epsilon_f=1e-3`;
- CEC2026: descending mean RPR, F1, and their reported `score`.

The tables are exploratory compatibility profiles. A high rank for a deleted
configuration means that the *remaining* mechanisms performed well on that
case; it is not a causal estimate of the deleted component's marginal effect.
No Full configuration is included in these rankings, and no Full-minus-
deletion claim is made from this campaign.

The CEC2013 driver is `run_cec2013_deleted_profile.py`; the CEC2026 driver is
`run_cec2026_deleted_profile.py`. Both write resumable CSV output. After all
four CEC2026 dimension files and the CEC2013 file complete, run
`analyze_deleted_profile.py` to create per-case and dimension-level ranking
tables.
