#!/usr/bin/env python3
"""Matched five-condition LOO runner with strict diagnostics and safe resume.

The frozen algorithm is unchanged. Historical CSVs can be validated, but may
not be appended to without a compatible manifest created before the run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_rmc_revision_cec2026_experiments as common

ROOT = Path(__file__).resolve().parents[2]
CONTROLS = {
    "RMC-CMSA": "full",
    "RMC-NoMaximin": "no_maximin",
    "RMC-NoRelations": "no_relations",
    "RMC-NoTerminalMemory": "no_terminal_memory",
    "RMC-NoTrajectoryEvidence": "no_trajectory_evidence",
}
LOO_CONFIGS = tuple(CONTROLS)
FIELDS = tuple(common.FIELDS)
COUNTERS = tuple(f for f in FIELDS if f.startswith("mrmc_"))
KEY = ("config", "pid", "pin", "dim", "repeat")
DISABLED = {
    "RMC-NoMaximin": ("mrmc_maximin_selections",),
    "RMC-NoRelations": ("mrmc_relation_uses",),
    "RMC-NoTerminalMemory": ("mrmc_terminal_memory_peak", "mrmc_persistent_guidance_uses", "mrmc_sparse_guidance_uses"),
    "RMC-NoTrajectoryEvidence": ("mrmc_evidence_observed", "mrmc_working_memory_peak", "mrmc_working_additions", "mrmc_working_promotions"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict:
    return {"python": sys.version, "platform": platform.platform(),
            "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")},
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas")}}


def source_hashes() -> dict:
    paths = [Path(__file__).resolve(), Path(common.__file__).resolve()]
    for folder in (ROOT / "code/frozen_four_component/code", ROOT / "code/baseline_src/RSCMSAESII_v1"):
        paths.extend(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(paths))}


def json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def expected_keys(protocol: dict) -> set:
    return set(itertools.product(LOO_CONFIGS, protocol["pids"], [protocol["pin"]],
                                 [protocol["dim"]], range(1, protocol["repeats"] + 1)))


def validate_frame(data: pd.DataFrame, protocol: dict | None = None, *,
                   complete: bool = True, require_activation: bool = True) -> dict:
    missing = set(FIELDS) - set(data.columns)
    if missing or data.empty:
        raise ValueError(f"empty campaign or missing columns: {sorted(missing)}")
    numeric = [f for f in FIELDS if f not in ("config", "control")]
    values = data[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("non-finite numeric value")
    integers = ["pid", "pin", "dim", "repeat", "seed", "n_sol", "used_eval", *COUNTERS]
    if any((values[c] != np.floor(values[c])).any() for c in integers):
        raise ValueError("non-integer identifiers or diagnostics")
    if (values[["pid", "pin", "dim", "repeat", "used_eval"]] <= 0).any().any():
        raise ValueError("nonpositive identifier or evaluation count")
    if (values[["seed", "n_sol", "elapsed_sec", *COUNTERS]] < 0).any().any():
        raise ValueError("negative count, seed or elapsed time")
    if (values.seed >= 2**32 - 1).any():
        raise ValueError("seed exceeds generator range")
    if ((values[["rpr", "f1", "score", "precision"]] < 0) | (values[["rpr", "f1", "score", "precision"]] > 1)).any().any():
        raise ValueError("metric outside [0, 1]")
    if not np.allclose(values.score, (values.rpr + values.f1) / 2, atol=1e-12, rtol=0):
        raise ValueError("score is inconsistent with RPR and F1")
    if data.duplicated(list(KEY)).any():
        raise ValueError("duplicate full run key")
    if set(data.config) - set(LOO_CONFIGS):
        raise ValueError("unknown configuration")
    if any(str(row.control) != CONTROLS[str(row.config)] for row in data.itertuples()):
        raise ValueError("configuration/control mismatch")
    if data.budget_scale.nunique() != 1 or (values.budget_scale <= 0).any():
        raise ValueError("mixed or nonpositive budget scale")
    budget = np.floor(20000 * values.dim * values.budget_scale + 0.5).astype(int)
    if (values.used_eval > budget).any():
        raise ValueError("evaluation budget exceeded")
    if protocol is None:
        if data.pin.nunique() != 1 or data.dim.nunique() != 1:
            raise ValueError("mixed instance or dimension requires an explicit separate campaign")
        protocol = {"pids": sorted(map(int, data.pid.unique())), "pin": int(data.pin.iloc[0]),
                    "dim": int(data.dim.iloc[0]), "repeats": int(data.repeat.max()),
                    "budget_scale": float(data.budget_scale.iloc[0]), "seed_base": None}
    observed = set(data[list(KEY)].itertuples(index=False, name=None))
    expected = expected_keys(protocol)
    if observed - expected or (complete and observed != expected):
        raise ValueError(f"grid mismatch: missing {len(expected - observed)}, unexpected {len(observed - expected)}")
    if not np.allclose(values.budget_scale, protocol["budget_scale"], atol=0, rtol=0):
        raise ValueError("budget scale differs from protocol")
    if data.groupby(["pid", "pin", "dim", "repeat"]).seed.nunique().max() != 1:
        raise ValueError("paired seed mismatch")
    if protocol.get("seed_base") is not None:
        for r in data.itertuples():
            if int(r.seed) != common.derive_seed(protocol["seed_base"], int(r.pid), int(r.pin), int(r.dim), int(r.repeat)):
                raise ValueError("seed differs from protocol schedule")
    for config, counters in DISABLED.items():
        if (data.loc[data.config == config, list(counters)] != 0).any().any():
            raise ValueError(f"disabled component path activated: {config}")
    dormant = []
    for config, group in data.groupby("config"):
        for counter in ("mrmc_relation_uses", "mrmc_maximin_selections", "mrmc_terminal_memory_peak", "mrmc_evidence_observed"):
            if counter not in DISABLED.get(config, ()) and group[counter].sum() == 0:
                dormant.append(f"{config}:{counter}")
    if complete and require_activation and dormant:
        raise ValueError(f"enabled components unexercised at campaign level: {dormant}")
    return {"rows": len(data), "expected_rows": len(expected), "complete": observed == expected,
            "budget_unused_min": int((budget - values.used_eval).min()),
            "budget_unused_max": int((budget - values.used_eval).max()),
            "dormant_enabled_paths": dormant,
            "working_promotions_total": int(values.mrmc_working_promotions.sum())}


def validate(path: Path, protocol: dict | None = None, **kwargs) -> dict:
    return validate_frame(pd.read_csv(path), protocol, **kwargs)


def strict_stats(stats: dict) -> dict:
    missing = set(COUNTERS) - set(stats)
    if missing:
        raise RuntimeError(f"algorithm omitted required diagnostics: {sorted(missing)}")
    output = {}
    for name in COUNTERS:
        value = stats[name]
        if not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(float(value)) or value < 0 or int(value) != value:
            raise RuntimeError(f"invalid algorithm diagnostic {name}: {value!r}")
        output[name] = int(value)
    return output


def artifact_path(folder: Path, job: dict) -> Path:
    return folder / f"{job['config']}_p{job['pid']}_i{job['pin']}_d{job['dim']}_r{job['repeat']}.json"


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def run_one(job: dict) -> dict:
    # LOO needs none of the ladder/no-state patches. Only the shared budget
    # patch and the explicitly recorded optional rejection safeguard apply.
    if job["safeguard"]:
        common.install_rejection_safeguard()
    common.apply_patches(1.0, job["budget_scale"])
    from rmc_cmsa_final_four_component import ModularRMCCMSAFinal
    seed = common.derive_seed(job["seed_base"], job["pid"], job["pin"], job["dim"], job["repeat"])
    algo = ModularRMCCMSAFinal(seed=seed, causal_control=CONTROLS[job["config"]])
    started = time.perf_counter()
    result = algo.run(job["pid"], job["pin"], job["dim"])
    elapsed = time.perf_counter() - started
    stats = dict(getattr(algo, "mrmc_stats", {}) or {})
    diagnostics = strict_stats(stats)
    returned = np.asarray(result.solutions, dtype=float)
    # BaseAlgorithm appends evaluator-side objective values as the last column.
    if returned.ndim != 2 or returned.shape[1] != job["dim"] + 1 or not np.isfinite(returned).all():
        raise RuntimeError("invalid raw solution matrix")
    solutions, fitness = returned[:, :-1], returned[:, -1]
    row = {key: job[key] for key in KEY}
    row.update(seed=seed, budget_scale=job["budget_scale"], rpr=float(result.rpr),
               f1=float(result.f1), score=float(result.score), precision=float(result.precision),
               n_sol=int(solutions.shape[0]), used_eval=int(result.used_eval),
               elapsed_sec=round(elapsed, 3), control=CONTROLS[job["config"]], **diagnostics)
    validate_frame(pd.DataFrame([row]), job["protocol"], complete=False, require_activation=False)
    artifact = {"schema_version": 1, "row": row, "solutions": solutions.tolist(),
                "solutions_shape": list(solutions.shape),
                "objective_values": fitness.tolist(), "n_minima": int(result.n_minima),
                "max_eval": int(result.max_eval), "metric_ftol": np.asarray(algo.FTOL).tolist(),
                "diagnostics": json.loads(json.dumps(stats, default=json_default)),
                "protocol_sha256": job["protocol_sha256"], "manifest_sha256": job["manifest_sha256"],
                "note": "Final returned decision vectors and all reported diagnostics; restart traces are not retained."}
    json_write(artifact_path(Path(job["artifact_dir"]), job), artifact)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=list(LOO_CONFIGS))
    parser.add_argument("--pids", type=int, nargs="+", required=True)
    parser.add_argument("--pin", type=int, default=1)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=20260808)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--safeguard", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--engineering-smoke", action="store_true", help="Record engineering-only status; allow enabled paths to remain dormant at tiny budgets")
    args = parser.parse_args()
    if len(args.configs) != 5 or set(args.configs) != set(LOO_CONFIGS):
        parser.error("exactly the five matched LOO configurations are required")
    if len(set(args.pids)) != len(args.pids) or any(p not in range(1, 17) for p in args.pids):
        parser.error("PIDs must be unique integers in 1..16")
    if min(args.pin, args.dim, args.repeats, args.workers) <= 0 or not math.isfinite(args.budget_scale) or args.budget_scale <= 0:
        parser.error("instance, dimension, repeats, workers and budget scale must be positive")
    protocol = {"configs": list(LOO_CONFIGS), "controls": CONTROLS, "pids": sorted(args.pids),
                "pin": args.pin, "dim": args.dim, "repeats": args.repeats, "seed_base": args.seed_base,
                "budget_scale": args.budget_scale, "safeguard": args.safeguard,
                "engineering_smoke": args.engineering_smoke,
                "seed_rule": "blake2b8(base|pid|pin|dim|runN), little-endian, mod(2^32-1)"}
    if args.validate_only:
        print(json.dumps(validate(args.out, protocol, require_activation=not args.engineering_smoke), indent=2))
        return
    out = args.out.resolve()
    folder = out.parent if out.parent.name == "analysis" else out.parent / "analysis"
    manifest_path = folder / f"{out.stem}.run_manifest.json"
    artifacts = folder / f"{out.stem}_runs"
    compatible = {"schema_version": 1, "protocol": protocol, "sources": source_hashes(),
                  "environment": environment(), "output_filename": out.name}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(k) != v for k, v in compatible.items()):
            raise RuntimeError("resume refused: protocol, source, environment or output differs from manifest")
    else:
        if out.exists() or artifacts.exists():
            raise RuntimeError("resume refused: existing output has no prior run manifest; use --validate-only or a new output name")
        manifest = compatible | {"created_utc": datetime.now(timezone.utc).isoformat(),
                                 "command": sys.argv, "expected_rows": len(expected_keys(protocol)),
                                 "provenance": "Captured before this campaign; does not attest to earlier CSVs."}
        json_write(manifest_path, manifest)
    protocol_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest_sha = sha256(manifest_path)
    done = set()
    if out.exists():
        previous = pd.read_csv(out)
        if not previous.empty:
            validate_frame(previous, protocol, complete=False, require_activation=False)
            for row in previous.to_dict("records"):
                artifact = json.loads(artifact_path(artifacts, row).read_text(encoding="utf-8"))
                if artifact.get("protocol_sha256") != protocol_sha or artifact.get("manifest_sha256") != manifest_sha:
                    raise RuntimeError("resume refused: run artifact provenance mismatch")
                for key in FIELDS:
                    a, b = artifact["row"][key], row[key]
                    equal = math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) if isinstance(a, (int, float)) else a == b
                    if not equal:
                        raise RuntimeError(f"resume refused: artifact/CSV mismatch at {key}")
                shape = (int(row["n_sol"]), int(row["dim"]))
                solutions = np.asarray(artifact["solutions"], dtype=float).reshape(shape)
                objectives = np.asarray(artifact["objective_values"], dtype=float)
                if artifact.get("solutions_shape") != list(shape) or not np.isfinite(solutions).all() or objectives.shape != (shape[0],) or not np.isfinite(objectives).all():
                    raise RuntimeError("resume refused: invalid saved solutions")
            done = set(previous[list(KEY)].itertuples(index=False, name=None))
    jobs = []
    for key in sorted(expected_keys(protocol) - done):
        job = dict(zip(KEY, key))
        job.update(budget_scale=args.budget_scale, seed_base=args.seed_base, safeguard=args.safeguard,
                   protocol=protocol, artifact_dir=str(artifacts), protocol_sha256=protocol_sha,
                   manifest_sha256=manifest_sha)
        jobs.append(job)
    print(f"jobs to run: {len(jobs)} (skipped {len(done)} verified runs)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    if jobs:
        with out.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
                handle.flush()
            with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
                futures = [pool.submit(run_one, job) for job in jobs]
                for index, future in enumerate(as_completed(futures), 1):
                    writer.writerow(future.result())
                    handle.flush()
                    print(f"progress {index}/{len(jobs)}", flush=True)
    report = validate(out, protocol, require_activation=not args.engineering_smoke)
    json_write(folder / f"{out.stem}.validation.json", report | {"csv_sha256": sha256(out), "manifest_sha256": manifest_sha})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
