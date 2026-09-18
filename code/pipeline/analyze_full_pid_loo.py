#!/usr/bin/env python3
"""Analyze the complete matched CEC2026 PID1--PID16 LOO campaign.

The release protocol is 16 PIDs x 5 configurations x 10 paired seeds at
D=20.  Full is compared with each deletion for RPR and F1, giving one Holm
family of 128 paired tests.  This script refuses pilot or incomplete inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import run_rebuilt_loo_ablation as runner

PROTOCOL = {"pids": list(range(1, 17)), "pin": 1, "dim": 20,
            "repeats": 10, "budget_scale": 1.0, "seed_base": 20260808}
METRICS = ("rpr", "f1")
N_TESTS = 16 * 4 * 2
LABELS = {"RMC-CMSA": "Full", "RMC-NoMaximin": "No maximin",
          "RMC-NoRelations": "No relations",
          "RMC-NoTerminalMemory": "No terminal memory",
          "RMC-NoTrajectoryEvidence": "No trajectory evidence"}


def signed_rank(differences: np.ndarray) -> float:
    values = np.round(np.asarray(differences, dtype=float), 12)
    values = values[values != 0]
    if len(values) == 0:
        return 1.0
    ranks = rankdata(np.abs(values), method="average")
    observed = abs(float(np.dot(np.sign(values), ranks)))
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(values))))
    return float(np.mean(np.abs(signs @ ranks) >= observed - 1e-12))


def holm(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.maximum.accumulate((len(order) - np.arange(len(order))) * pvalues[order])
    result = np.empty(len(order), dtype=float)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def paired_bootstrap(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def verify_artifacts(data: pd.DataFrame, input_path: Path) -> dict:
    folder = input_path.parent / "analysis"
    manifest_path = folder / f"{input_path.stem}.run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in PROTOCOL.items():
        if manifest["protocol"].get(key) != value:
            raise ValueError(f"run manifest protocol differs at {key}")
    if not manifest["protocol"].get("safeguard"):
        raise ValueError("campaign requires the recorded rejection safeguard")
    manifest_hash = runner.sha256(manifest_path)
    protocol_hash = hashlib.sha256(json.dumps(manifest["protocol"], sort_keys=True).encode()).hexdigest()
    artifact_dir = folder / f"{input_path.stem}_runs"
    if len(list(artifact_dir.glob("*.json"))) != 800:
        raise ValueError("expected exactly 800 per-run artifacts")
    hashes = {}
    benchmark_relative = "code/frozen_four_component/code/CMMOP/data/pid-data.xlsx"
    benchmark_path = runner.ROOT / benchmark_relative
    if runner.sha256(benchmark_path) != manifest["sources"].get(benchmark_relative):
        raise ValueError("benchmark PID metadata differ from pre-run manifest")
    metadata = pd.read_excel(benchmark_path).set_index("pid")
    full_protocol = manifest["protocol"]
    for row in data.to_dict("records"):
        path = runner.artifact_path(artifact_dir, row)
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("manifest_sha256") != manifest_hash or artifact.get("protocol_sha256") != protocol_hash:
            raise ValueError(f"artifact provenance mismatch: {path.name}")
        for field in runner.FIELDS:
            a, b = artifact["row"][field], row[field]
            equal = bool(np.isclose(a, b, rtol=1e-12, atol=1e-12)) if isinstance(a, (int, float)) else a == b
            if not equal:
                raise ValueError(f"artifact/CSV mismatch: {path.name}:{field}")
        shape = (int(row["n_sol"]), int(row["dim"]))
        values = np.asarray(artifact["solutions"], dtype=float).reshape(shape)
        objectives = np.asarray(artifact["objective_values"], dtype=float)
        if artifact["solutions_shape"] != list(shape) or not np.isfinite(values).all() or objectives.shape != (shape[0],) or not np.isfinite(objectives).all():
            raise ValueError(f"invalid returned vectors: {path.name}")
        if artifact.get("n_minima") != int(metadata.loc[int(row["pid"]), "n_minima"]):
            raise ValueError(f"incorrect target count: {path.name}")
        expected_budget = int(20000 * full_protocol["dim"] * full_protocol["budget_scale"])
        if artifact.get("max_eval") != expected_budget or artifact.get("metric_ftol") != [1e-5, 1.0]:
            raise ValueError(f"budget or evaluation tolerance mismatch: {path.name}")
        for field in runner.COUNTERS:
            if artifact.get("diagnostics", {}).get(field) != row[field]:
                raise ValueError(f"diagnostic/row mismatch: {path.name}:{field}")
        hashes[path.name] = runner.sha256(path)
    return {"manifest_path": str(manifest_path), "manifest_sha256": manifest_hash,
            "protocol_sha256": protocol_hash, "artifact_count": len(hashes), "artifact_sha256": hashes,
            "verification": "CSV/artifact equality, finite final vectors, exact run protocol and manifest hashes; not an independent objective re-evaluation"}


def write_tables(texdir: Path, data: pd.DataFrame, comparisons: pd.DataFrame, bootstrap_draws: int = 50000) -> None:
    texdir.mkdir(parents=True, exist_ok=True)
    end = r" \\" + "\n"
    table = "% Generated by analyze_full_pid_loo.py; descriptive sample SD.\n"
    table += r"\begin{tabular}{llrrrrr}" + "\n\\toprule\n"
    table += "PID & Metric & " + " & ".join(LABELS.values()) + end + r"\midrule" + "\n"
    for pid in PROTOCOL["pids"]:
        for metric in METRICS:
            cells = []
            for config in runner.LOO_CONFIGS:
                values = data.loc[(data.pid == pid) & (data.config == config), metric]
                cells.append(f"${values.mean():.3f} \\pm {values.std(ddof=1):.3f}$")
            table += f"{pid} & {metric.upper()} & " + " & ".join(cells) + end
    table += r"\midrule" + "\nPooled & RPR & " + " & ".join(
        f"{data.loc[data.config == c, 'rpr'].mean():.3f}" for c in runner.LOO_CONFIGS) + end
    table += "Pooled & F1 & " + " & ".join(
        f"{data.loc[data.config == c, 'f1'].mean():.3f}" for c in runner.LOO_CONFIGS) + end
    table += r"\bottomrule" + "\n\\end{tabular}\n"
    (texdir / "rebuilt_loo_ablation_full_pid.tex").write_text(table, encoding="utf-8")

    table = "% Generated by analyze_full_pid_loo.py; Delta = Full minus deletion. Holm family = 128.\n"
    table += r"\begin{longtable}{lllrrr}" + "\n"
    table += r"\caption{All 128 paired full-minus-deletion contrasts at $D=20$. Exact two-sided signed-rank sign-permutation tests use one Holm family.}\label{tab:rebuilt-loo-paired}\\" + "\n\\toprule\n"
    table += r"PID & Deletion & Metric & $\Delta$ & $p_{raw}$ & $p_{Holm}$" + end + r"\midrule" + "\n"
    table += r"\endfirsthead\toprule" + "\n"
    table += r"PID & Deletion & Metric & $\Delta$ & $p_{raw}$ & $p_{Holm}$" + end + r"\midrule\endhead" + "\n"
    table += r"\midrule\multicolumn{6}{r}{Continued on the next page}\\\endfoot\bottomrule\endlastfoot" + "\n"
    for row in comparisons.itertuples():
        table += f"{row.pid} & {LABELS[row.deletion]} & {row.metric.upper()} & {row.delta:+.4f} & {row.p_raw:.4f} & {row.p_holm:.4f}" + end
    table += "\\end{longtable}\n"
    (texdir / "rebuilt_loo_paired_full_pid.tex").write_text(table, encoding="utf-8")

    counters = ("mrmc_relation_uses", "mrmc_maximin_selections", "mrmc_terminal_memory_peak", "mrmc_evidence_observed")
    table = r"\begin{tabular}{lrrrrr}\toprule" + "\n"
    table += "Configuration & Relation uses & Maximin selections & Sum of terminal peaks & Evidence observations & Runs" + end + r"\midrule" + "\n"
    for config in runner.LOO_CONFIGS:
        group = data[data.config == config]
        table += LABELS[config] + " & " + " & ".join(f"{int(group[c].sum()):,}" for c in counters) + f" & {len(group)}" + end
    table += "\\bottomrule\n\\end{tabular}\n"
    (texdir / "rebuilt_loo_diagnostics_full_pid.tex").write_text(table, encoding="utf-8")

    overall = data.groupby("config")[list(METRICS)].mean().reindex(runner.LOO_CONFIGS)
    results = "The rebuilt campaign completed all 800 runs. Pooled RPR/F1 (descriptive means over 160 runs per configuration) is "
    results += "; ".join(f"${v.rpr:.4f}/{v.f1:.4f}$ for {LABELS[c].lower()}" for c, v in overall.iterrows()) + ". "
    results += f"Across the 128 within-PID contrasts, the minimum Holm-adjusted $p$ is ${comparisons.p_holm.min():.4f}$. "
    results += r"With at most ten nonzero paired differences, the smallest attainable two-sided exact $p$ is $2/2^{10}=0.001953125$. The first Holm threshold is $0.05/128=0.000390625$, so this protocol cannot reject any member of the family. We therefore interpret effect sizes and their directions as descriptive conditional evidence; the absence of adjusted rejections is not evidence of component equivalence or absence of benefit."
    (texdir / "full_pid_results_text.tex").write_text(results + "\n", encoding="utf-8")

    directions = []
    for deletion in runner.LOO_CONFIGS[1:]:
        for metric in METRICS:
            subset = comparisons[(comparisons.deletion == deletion) & (comparisons.metric == metric)]
            directions.append(f"For {LABELS[deletion].lower()} ({metric.upper()}), full-minus-deletion means are positive/zero/negative on {int((subset.delta > 1e-12).sum())}/{int((subset.delta.abs() <= 1e-12).sum())}/{int((subset.delta < -1e-12).sum())} PIDs.")
    note = " ".join(directions) + f" The paired percentile-bootstrap intervals in the CSV are descriptive, unadjusted 95\\% intervals based on {bootstrap_draws:,} resamples. The null test requires independent seed pairs and sign exchangeability (symmetry) of paired differences; it is not a pooled across-problem test.\n"
    (texdir / "full_pid_directions_text.tex").write_text(note, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=runner.ROOT / "results/ablation/all_pid_loo_d20_singlethread.csv")
    parser.add_argument("--outdir", type=Path, default=runner.ROOT / "results/ablation/full_pid_analysis")
    parser.add_argument("--bootstrap-draws", type=int, default=50000)
    parser.add_argument("--tex-dir", type=Path, default=runner.ROOT / "paper/tables_edc")
    args = parser.parse_args()
    if args.bootstrap_draws < 1000:
        parser.error("at least 1000 bootstrap draws required")
    data = pd.read_csv(args.input)
    validation = runner.validate_frame(data, PROTOCOL)
    artifacts = verify_artifacts(data, args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)
    summaries = []
    comparisons = []
    for pid in PROTOCOL["pids"]:
        case = data[data.pid == pid]
        full = case[case.config == "RMC-CMSA"].set_index("repeat").sort_index()
        for config in runner.LOO_CONFIGS:
            group = case[case.config == config]
            summaries.append({"pid": pid, "config": config, "n": len(group),
                              **{f"{m}_mean": group[m].mean() for m in METRICS},
                              **{f"{m}_sd": group[m].std(ddof=1) for m in METRICS}})
        for deletion in runner.LOO_CONFIGS[1:]:
            other = case[case.config == deletion].set_index("repeat").sort_index()
            for metric in METRICS:
                differences = (full[metric] - other[metric]).to_numpy(dtype=float)
                seedtext = f"rmc-full-pid-loo|{pid}|{deletion}|{metric}"
                seed = int.from_bytes(hashlib.blake2b(seedtext.encode(), digest_size=8).digest(), "little")
                lo, hi = paired_bootstrap(differences, args.bootstrap_draws, seed)
                comparisons.append({"pid": pid, "deletion": deletion, "metric": metric,
                                    "n_pairs": len(differences), "full_mean": full[metric].mean(),
                                    "deletion_mean": other[metric].mean(), "delta": differences.mean(),
                                    "bootstrap_low": lo, "bootstrap_high": hi,
                                    "full_wins": int((differences > 1e-12).sum()),
                                    "ties": int((np.abs(differences) <= 1e-12).sum()),
                                    "full_losses": int((differences < -1e-12).sum()),
                                    "p_raw": signed_rank(differences)})
    summaries = pd.DataFrame(summaries)
    comparisons = pd.DataFrame(comparisons)
    if len(comparisons) != N_TESTS:
        raise RuntimeError(f"expected {N_TESTS} comparisons, got {len(comparisons)}")
    comparisons["p_holm"] = holm(comparisons.p_raw.to_numpy())
    comparisons["reject_holm_005"] = comparisons.p_holm <= 0.05
    summaries.to_csv(args.outdir / "full_pid_per_pid_summary.csv", index=False)
    data.groupby("config")[list(METRICS)].mean().reindex(runner.LOO_CONFIGS).to_csv(args.outdir / "full_pid_descriptive_overall.csv")
    comparisons.to_csv(args.outdir / "full_pid_paired_comparisons.csv", index=False)
    data.groupby(["pid", "config"])[list(runner.COUNTERS)].sum().to_csv(args.outdir / "full_pid_diagnostic_sums.csv")
    write_tables(args.tex_dir, data, comparisons, args.bootstrap_draws)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "input": str(args.input),
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(), "rows": len(data),
                "protocol": PROTOCOL, "validation": validation, "tests": N_TESTS,
                "artifacts": artifacts, "analysis_script_sha256": runner.sha256(Path(__file__)),
                "analysis_environment": runner.environment(),
                "multiplicity": "one Holm family: PID1--PID16 x four deletions x RPR/F1",
                "minimum_attainable_raw_p": 2 / 2**10,
                "holm_first_threshold": 0.05 / 128,
                "test_resolution_limit": "No Holm rejection is attainable with this exact test, 10 pairs and 128 comparisons. Interpret deltas and uncertainty, not failure to reject, as descriptive evidence.",
                "bootstrap": f"paired percentile, {args.bootstrap_draws} draws, descriptive only"}
    (args.outdir / "full_pid_analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    overall = data.groupby("config")[list(METRICS)].mean().reindex(runner.LOO_CONFIGS)
    result = {"rows": len(data), "comparisons": N_TESTS,
              "holm_significant": int(comparisons.reject_holm_005.sum()),
              "minimum_holm_p": float(comparisons.p_holm.min()),
              "overall": overall.to_dict(orient="index"), "outdir": str(args.outdir)}
    (args.outdir / "full_pid_result_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
