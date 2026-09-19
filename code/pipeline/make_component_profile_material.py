#!/usr/bin/env python3
"""Build exploratory rankings for the four component-deleted variants.

This material is deliberately separate from the manuscript's matched Full
versus deletion ablation.  It ranks only the four deletion configurations and
therefore reports a *compatibility profile* rather than a causal component
effect.  Inputs are CSV files produced by the revision runners.  CEC2013
uses the mean of its six PR/SR columns; CEC2026 uses ``(rpr + f1) / 2``.

The expected protocol is 30 runs per problem/configuration.  The checker is
strict by default so an accidentally mixed campaign cannot be silently
ranked.  ``--allow-partial`` is intended only for checking a pilot.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


CONFIGS = (
    "RMC-NoMaximin",
    "RMC-NoRelations",
    "RMC-NoTerminalMemory",
    "RMC-NoTrajectoryEvidence",
)
LABELS = {
    "RMC-NoMaximin": "No maximin",
    "RMC-NoRelations": "No relations",
    "RMC-NoTerminalMemory": "No terminal memory",
    "RMC-NoTrajectoryEvidence": "No trajectory evidence",
}
SHORT = {
    "RMC-NoMaximin": "No-Mx",
    "RMC-NoRelations": "No-Rel",
    "RMC-NoTerminalMemory": "No-Term",
    "RMC-NoTrajectoryEvidence": "No-Traj",
}
CEC2013_METRICS = (
    "pr_1e-03", "sr_1e-03", "pr_1e-04", "sr_1e-04", "pr_1e-05", "sr_1e-05"
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def rank_desc(values: dict[str, float]) -> dict[str, float]:
    """Descending midranks, with deterministic configuration tie order."""
    ordered = sorted(values.items(), key=lambda item: (-item[1], CONFIGS.index(item[0])))
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and math.isclose(
            ordered[i][1], ordered[j][1], rel_tol=0.0, abs_tol=1e-12
        ):
            j += 1
        rank = (i + 1 + j) / 2.0
        for config, _ in ordered[i:j]:
            ranks[config] = rank
        i = j
    return ranks


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def load_campaign(
    path: Path,
    benchmark: str,
    key_field: str,
    metrics: tuple[str, ...],
    expected_problems: list[int] | None,
    repeats: int,
    allow_partial: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict]:
    rows = read_rows(path)
    required = {"config", key_field, *metrics}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    seen: set[tuple[str, int, int]] = set()
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    by_run: dict[tuple[int, str, int], dict[str, float]] = defaultdict(dict)
    problems = set()
    dimensions: dict[int, int] = {}
    for row in rows:
        config = row["config"]
        if config not in CONFIGS:
            raise ValueError(f"{path}: Full or unknown configuration {config!r}; four deletions only")
        run_key = "run" if "run" in row else "repeat"
        if run_key not in row:
            raise ValueError(f"{path}: missing run/repeat column")
        problem = int(row[key_field]); run = int(row[run_key])
        if "dim" in row and row["dim"] not in ("", None):
            dimensions[problem] = int(float(row["dim"]))
        key = (config, problem, run)
        if key in seen:
            raise ValueError(f"{path}: duplicate key {key}")
        if run < 1 or run > repeats:
            raise ValueError(f"{path}: run outside 1..{repeats}: {key}")
        seen.add(key); problems.add(problem)
        for metric in metrics:
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError(f"{path}: non-finite {metric}: {key}")
            grouped[(problem, config, metric)].append(value)
            by_run[(problem, config, run)][metric] = value
    if expected_problems is not None and problems != set(expected_problems):
        raise ValueError(f"{path}: problems differ from protocol: expected {expected_problems}, observed {sorted(problems)}")
    expected = {(c, p, run) for c in CONFIGS for p in problems for run in range(1, repeats + 1)}
    if not allow_partial and seen != expected:
        raise ValueError(f"{path}: incomplete four-configuration grid; missing {len(expected - seen)}, extra {len(seen - expected)}")
    detail: list[dict[str, object]] = []
    rankings: list[dict[str, object]] = []
    for problem in sorted(problems):
        scores: dict[str, float] = {}
        stats: dict[str, dict[str, float]] = {}
        for config in CONFIGS:
            values = {m: grouped[(problem, config, m)] for m in metrics}
            # CEC2013's six values are all on [0,1].  CEC2026's RPR and F1
            # are averaged into the same [0,1] score.
            score = mean([mean(values[m]) for m in metrics])
            scores[config] = score
            stats[config] = {m: mean(values[m]) for m in metrics}
            stats[config]["score"] = score
            # Score SD is calculated run-wise, not from metric SDs.
            run_scores = [mean([by_run[(problem, config, run)][m] for m in metrics])
                          for run in range(1, repeats + 1)
                          if (problem, config, run) in by_run]
            stats[config]["score_sd"] = sample_sd(run_scores)
            detail.append({
                "benchmark": benchmark, "problem": problem, "config": config,
                "n": len(values[metrics[0]]), "score": score,
                "score_sd": stats[config]["score_sd"],
                **{f"{m}_mean": stats[config][m] for m in metrics},
            })
        ranks = rank_desc(scores)
        for config in CONFIGS:
            rankings.append({
                "benchmark": benchmark, "problem": problem,
                "dimension": dimensions.get(problem, ""), "config": config,
                "score": scores[config], "rank": ranks[config],
            })
    manifest = {
        "benchmark": benchmark, "input": str(path.resolve()), "rows": len(rows),
        "problems": sorted(problems), "configs": list(CONFIGS), "repeats": repeats,
        "ranking": "descending midranks within problem; lower mean rank is better",
        "interpretation": "deletion-variant compatibility profile; no Full baseline and no causal component effect",
    }
    return detail, rankings, manifest


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0])
        for row in rows[1:]:
            fields.extend(field for field in row if field not in fields)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def aggregate(rankings: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rankings:
        grouped[(str(row["benchmark"]), str(row["config"]))].append(row)
    rows = []
    for benchmark in sorted({str(r["benchmark"]) for r in rankings}):
        for config in CONFIGS:
            subset = grouped[(benchmark, config)]
            rows.append({
                "benchmark": benchmark, "config": config,
                "problems": len(subset),
                "mean_score": mean([float(r["score"]) for r in subset]),
                "mean_rank": mean([float(r["rank"]) for r in subset]),
            })
    # Overall rank among the four configurations for each benchmark.
    for benchmark in sorted({str(r["benchmark"]) for r in rows}):
        subset = [r for r in rows if r["benchmark"] == benchmark]
        ranks = rank_desc({str(r["config"]): -float(r["mean_rank"]) for r in subset})
        # rank_desc is descending, so negate mean rank first; report lower rank
        # as the better final ordering using a deterministic explicit sort.
        order = sorted(subset, key=lambda r: (float(r["mean_rank"]), CONFIGS.index(str(r["config"]))))
        for i, row in enumerate(order, 1):
            row["overall_rank"] = i
    return rows


def aggregate_by_dimension(rankings: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize compatibility ranks by benchmark and measured dimension."""
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rankings:
        dimension = int(row["dimension"]) if row.get("dimension", "") != "" else -1
        grouped[(str(row["benchmark"]), dimension, str(row["config"]))].append(row)
    rows = []
    for benchmark, dimension, config in sorted(grouped):
        subset = grouped[(benchmark, dimension, config)]
        rows.append({
            "benchmark": benchmark, "dimension": dimension, "config": config,
            "problems": len(subset),
            "mean_score": mean([float(r["score"]) for r in subset]),
            "mean_rank": mean([float(r["rank"]) for r in subset]),
        })
    return rows


def tex_table(rows: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{longtable}{llrrrr}",
        r"\caption{Four component-deleted variants ranked by problem-wise score. Lower mean rank is better.}",
        r"\toprule Benchmark & Variant & Problems & Mean score & Mean rank & Overall rank\\",
        r"\midrule\endfirsthead",
        r"\toprule Benchmark & Variant & Problems & Mean score & Mean rank & Overall rank\\",
        r"\midrule\endhead",
    ]
    for row in rows:
        lines.append(
            f"{row['benchmark']} & {LABELS[str(row['config'])]} & {int(row['problems'])} & "
            f"{float(row['mean_score']):.4f} & {float(row['mean_rank']):.3f} & {int(row['overall_rank'])}\\\\"
        )
    lines += [r"\bottomrule", r"\end{longtable}", ""]
    return "\n".join(lines)


def tex_problem_table(
    benchmark: str,
    detail: list[dict[str, object]],
    rankings: list[dict[str, object]],
) -> str:
    """One compact longtable with all problem-wise means, SDs and ranks."""
    rows = [r for r in detail if str(r["benchmark"]) == benchmark]
    ranks = {(str(r["problem"]), str(r["config"])): r for r in rankings
             if str(r["benchmark"]) == benchmark}
    lines = [
        r"\begin{longtable}{r*{4}{c}}",
        f"\\caption{{{benchmark} component-deleted scores (mean $\\pm$ SD; rank in parentheses).}}\\",
        r"\toprule Problem & " + " & ".join(SHORT[c] for c in CONFIGS) + r"\\",
        r"\midrule\endfirsthead",
        r"\toprule Problem & " + " & ".join(SHORT[c] for c in CONFIGS) + r"\\",
        r"\midrule\endhead\bottomrule\endlastfoot",
    ]
    for problem in sorted({r["problem"] for r in rows}, key=lambda x: int(x)):
        cells = []
        for config in CONFIGS:
            row = next(r for r in rows if r["problem"] == problem and r["config"] == config)
            rank = ranks[(str(problem), config)]["rank"]
            cells.append(f"{float(row['score']):.4f} $\\pm$ {float(row['score_sd']):.4f} ({float(rank):.1f})")
        lines.append(str(problem) + " & " + " & ".join(cells) + r"\\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def tex_dimension_table(rows: list[dict[str, object]]) -> str:
    """Summarize compatibility ranks by benchmark and measured dimension."""
    lines = [
        r"\begin{longtable}{llrrrr}",
        r"\caption{Dimension-level summary of the four component-deleted compatibility profiles. Lower mean rank is better.}\\",
        r"\toprule Benchmark & Dimension & Variant & Problems & Mean score & Mean rank\\",
        r"\midrule\endfirsthead",
        r"\toprule Benchmark & Dimension & Variant & Problems & Mean score & Mean rank\\",
        r"\midrule\endhead\bottomrule\endlastfoot",
    ]
    for row in sorted(rows, key=lambda r: (str(r["benchmark"]), int(r["dimension"]), CONFIGS.index(str(r["config"])))):
        lines.append(
            f"{row['benchmark']} & {int(row['dimension'])} & {LABELS[str(row['config'])]} & "
            f"{int(row['problems'])} & {float(row['mean_score']):.4f} & {float(row['mean_rank']):.3f}\\\\"
        )
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cec2013", type=Path, required=True)
    parser.add_argument("--cec2026", type=Path, nargs="+", required=True,
                        help="one or more files; use --names to label dimensions")
    parser.add_argument("--names", nargs="+", required=True,
                        help="benchmark names corresponding to --cec2026 files, e.g. CEC2026-D2")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if len(args.cec2026) != len(args.names):
        parser.error("--cec2026 and --names must have equal lengths")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    all_detail: list[dict[str, object]] = []
    all_rankings: list[dict[str, object]] = []
    manifests = []
    detail, rankings, manifest = load_campaign(
        args.cec2013, "CEC2013", "fid", CEC2013_METRICS,
        list(range(1, 21)), args.repeats, args.allow_partial,
    )
    all_detail += detail; all_rankings += rankings; manifests.append(manifest)
    for name, path in zip(args.names, args.cec2026):
        detail, rankings, manifest = load_campaign(
            path, name, "pid", ("rpr", "f1"), list(range(1, 17)),
            args.repeats, args.allow_partial,
        )
        all_detail += detail; all_rankings += rankings; manifests.append(manifest)
    summary = aggregate(all_rankings)
    write_csv(args.outdir / "component_deleted_profile_values.csv", all_detail)
    write_csv(args.outdir / "component_deleted_profile_by_problem.csv", all_rankings)
    write_csv(args.outdir / "component_deleted_profile_summary.csv", summary)
    dimension_summary = aggregate_by_dimension(all_rankings)
    write_csv(args.outdir / "component_deleted_profile_dimension_summary.csv",
              dimension_summary)
    (args.outdir / "component_deleted_profile.tex").write_text(
        "\\section*{Exploratory component-deleted compatibility profiles}\n"
        "This section ranks only the four deletion variants; the manuscript's Full-versus-deletion ablation is unchanged. "
        "Higher score is better and ranks are descriptive. Without a Full baseline, these rankings do not estimate a component's causal marginal effect.\n\n"
        + tex_table(summary) + "\n"
        + tex_dimension_table(dimension_summary) + "\n"
        + "\n".join(tex_problem_table(name, all_detail, all_rankings)
                    for name in sorted({str(r["benchmark"]) for r in all_detail})),
        encoding="utf-8",
    )
    (args.outdir / "component_deleted_profile_manifest.json").write_text(
        json.dumps({"repeats": args.repeats, "configs": list(CONFIGS), "campaigns": manifests,
                    "cec2026_target_counts": {"PID1-PID8": 20, "PID9-PID16": 10},
                    "interpretation": "Four-variant compatibility ranking only; Full omitted; no causal marginal effect."}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
