#!/usr/bin/env python3
"""Matched CEC2013 leave-one-component-out campaign.

This runner is intentionally separate from the historical 30-run CEC2013
benchmark driver.  It uses the frozen four-component implementation and
records one JSON artifact for every run, including the complete ``mrmc_stats``
diagnostic map.  The supplementary protocol is F1--F20, four component-
deleted configurations, and thirty repetitions per configuration/function
(2400 runs total).  A run is resumed only when its artifact and CSV row agree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import importlib.metadata
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code" / "frozen_four_component" / "code"
RSC = ROOT / "code" / "baseline_src" / "RSCMSAESII_v1"
CEC = ROOT / "code" / "benchmarks" / "CEC2013-master" / "python3"
PIPELINE = ROOT / "code" / "pipeline"
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(RSC)); sys.path.insert(0, str(CEC)); sys.path.insert(0, str(PIPELINE))

import run_rmc_revision_cec2026_experiments as common  # noqa: E402
from run_rmc_revision_cec2013_experiments import CEC2013Problem  # noqa: E402

CONFIGS = {
    "RMC-NoMaximin": "no_maximin",
    "RMC-NoRelations": "no_relations",
    "RMC-NoTerminalMemory": "no_terminal_memory",
    "RMC-NoTrajectoryEvidence": "no_trajectory_evidence",
}
CONFIG_ORDER = (
    "RMC-NoMaximin",
    "RMC-NoRelations",
    "RMC-NoTerminalMemory",
    "RMC-NoTrajectoryEvidence",
)
DIAGNOSTIC_FIELDS = (
    "mrmc_calls", "mrmc_relation_uses", "mrmc_maximin_selections",
    "mrmc_random_selections", "mrmc_relation_builds", "mrmc_relation_working_uses",
    "mrmc_terminal_memory_peak", "mrmc_persistent_guidance_uses",
    "mrmc_sparse_guidance_uses", "mrmc_evidence_observed", "mrmc_evidence_eligible",
    "mrmc_working_memory_peak", "mrmc_working_additions", "mrmc_working_promotions",
    "mrmc_working_noncompetitive", "mrmc_working_same_domain_blocks",
    "mrmc_new_peaks", "mrmc_repeated_peaks", "mrmc_completed_restarts",
    "mrmc_early_stops", "mrmc_used_eval",
)
FIELDS = (
    "config", "fid", "run", "seed", "dim", "n_goptima", "max_eval", "used_eval",
    "n_sol", "best_f", "pr_1e-03", "sr_1e-03", "pr_1e-04", "sr_1e-04",
    "pr_1e-05", "sr_1e-05", "elapsed_sec", *DIAGNOSTIC_FIELDS,
)
DISABLED = {
    "RMC-NoMaximin": ("mrmc_maximin_selections",),
    "RMC-NoRelations": ("mrmc_relation_uses", "mrmc_relation_builds", "mrmc_relation_working_uses"),
    "RMC-NoTerminalMemory": ("mrmc_terminal_memory_peak", "mrmc_persistent_guidance_uses", "mrmc_sparse_guidance_uses"),
    "RMC-NoTrajectoryEvidence": ("mrmc_evidence_observed", "mrmc_working_memory_peak", "mrmc_working_additions", "mrmc_working_promotions"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def derive_seed(base: int, fid: int, run: int) -> int:
    digest = hashlib.blake2b(f"{base}|{fid}|{run}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def artifact_path(folder: Path, config: str, fid: int, run: int) -> Path:
    return folder / f"{config}_f{fid:02d}_r{run:02d}.json"


def json_default(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    raise TypeError(type(v).__name__)


def atomic_json(path: Path, value: object) -> None:
    """Write a complete artifact before its CSV row becomes visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_one(job: dict) -> dict:
    # Every worker starts with a fresh interpreter in normal campaign use.
    # These idempotent patches register controls and cap pathological taboo
    # rejection loops without changing the frozen algorithm source.
    common.install_no_state(); common.install_ladder_controls(); common.install_rejection_safeguard()
    from rmc_cmsa_final_four_component import ModularRMCCMSAFinal
    from cec2013.cec2013 import how_many_goptima

    problem = CEC2013Problem(job["fid"])
    seed = derive_seed(job["seed_base"], job["fid"], job["run"])
    algo = ModularRMCCMSAFinal(seed=seed, causal_control=CONFIGS[job["config"]])
    started = time.perf_counter(); solutions = algo._run(problem); elapsed = time.perf_counter() - started
    X = np.atleast_2d(np.asarray(solutions, dtype=float))
    if X.size == 0: X = np.empty((0, problem.dim))
    if X.ndim != 2 or X.shape[1] != problem.dim or not np.isfinite(X).all():
        raise RuntimeError("invalid returned solution matrix")
    X = np.clip(X, problem.low_bound, problem.up_bound)
    raw = np.asarray([problem.raw_eval(x) for x in X], dtype=float)
    stats = dict(getattr(algo, "mrmc_stats", {}) or {})
    row = {"config": job["config"], "fid": job["fid"], "run": job["run"],
           "seed": seed, "dim": problem.dim, "n_goptima": problem.n_minima,
           "max_eval": problem.max_eval, "used_eval": problem.used_eval,
           "n_sol": int(len(X)), "best_f": float(np.max(raw)) if len(raw) else float("nan"),
           "elapsed_sec": round(elapsed, 3)}
    for eps in (1e-3, 1e-4, 1e-5):
        count, _ = how_many_goptima(X, problem.benchmark, eps)
        row[f"pr_{eps:.0e}".replace("+", "")] = float(count / problem.n_minima) if problem.n_minima else 0.0
        row[f"sr_{eps:.0e}".replace("+", "")] = int(count == problem.n_minima)
    missing = [field for field in DIAGNOSTIC_FIELDS if field not in stats]
    if missing:
        raise RuntimeError(f"algorithm omitted required CEC2013 diagnostics: {missing}")
    for field in DIAGNOSTIC_FIELDS:
        value = stats[field]
        if not isinstance(value, (np.integer, int, np.floating, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise RuntimeError(f"invalid diagnostic {field}: {value!r}")
        row[field] = int(value) if float(value).is_integer() else float(value)
    for config, counters in DISABLED.items():
        if row["config"] == config and any(row[counter] != 0 for counter in counters):
            raise RuntimeError(f"disabled CEC2013 path activated: {config}: {counters}")
    return {"row": row, "solutions": X.tolist(), "objective_values": raw.tolist(),
            "n_minima": problem.n_minima, "max_eval": problem.max_eval,
            "diagnostics": json.loads(json.dumps(stats, default=json_default))}


def protocol(args) -> dict:
    return {"configs": list(CONFIG_ORDER), "fids": list(args.fids), "runs": args.runs,
            "seed_base": args.seed_base, "safeguard": True,
            "seed_rule": "blake2b8(base|fid|run), little-endian, mod(2^32-1)",
            "metrics": ["pr_1e-03", "sr_1e-03", "pr_1e-04", "sr_1e-04", "pr_1e-05", "sr_1e-05"]}


def source_manifest() -> dict[str, str]:
    """Hash the runner, frozen implementation, CMSA source, and CEC data."""
    roots = (CODE, RSC, CEC)
    paths = [Path(__file__).resolve(), Path(common.__file__).resolve()]
    for root in roots:
        paths.extend(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in sorted(set(paths))}


def environment_manifest() -> dict:
    package_names = ("numpy", "scipy", "pandas", "openpyxl")
    packages = {}
    for name in package_names:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def validate_rows(rows: list[dict[str, str]], args, *, require_complete: bool) -> set[tuple[str, int, int]]:
    expected = {(c, f, r) for c in CONFIG_ORDER for f in args.fids for r in range(1, args.runs + 1)}
    observed = set()
    for row in rows:
        key = (row.get("config", ""), int(row["fid"]), int(row["run"]))
        if key in observed: raise ValueError(f"duplicate CSV key: {key}")
        if key not in expected: raise ValueError(f"out-of-protocol CSV key: {key}")
        observed.add(key)
        for field in FIELDS:
            if field not in row: raise ValueError(f"CSV missing field {field}")
            if field != "config" and not math.isfinite(float(row[field])): raise ValueError(f"non-finite CSV field {field}: {key}")
        for config, counters in DISABLED.items():
            if row["config"] == config and any(float(row[counter]) != 0 for counter in counters): raise ValueError(f"disabled path activated in CSV: {key}")
    if require_complete and observed != expected: raise ValueError(f"exact grid mismatch: missing={len(expected-observed)}, extra={len(observed-expected)}")
    return observed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=list(CONFIG_ORDER))
    p.add_argument("--fids", nargs="+", type=int, default=list(range(1, 21)))
    p.add_argument("--runs", type=int, default=30); p.add_argument("--seed-base", type=int, default=20260808)
    p.add_argument("--workers", type=int, default=1); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()
    if tuple(args.configs) != CONFIG_ORDER: p.error("use exactly the four deleted configurations in CONFIG_ORDER")
    if sorted(set(args.fids)) != sorted(args.fids) or any(fid < 1 or fid > 20 for fid in args.fids): p.error("FIDs must be unique integers in 1..20")
    if args.runs != 30: p.error("the supplementary profile protocol requires exactly 30 runs per function/configuration")
    if args.workers < 1: p.error("workers must be positive")
    prot = protocol(args); out = args.out.resolve(); folder = out.parent / f"{out.stem}_runs"; manifest_path = out.parent / f"{out.stem}.manifest.json"
    source_list = [Path(__file__), Path(common.__file__), *[x for x in (CODE / "rmc_cmsa_final_four_component.py", CODE / "modular_rmc_cmsa_v12.py", CODE / "modular_rmc_cmsa_v3.py", CODE / "base_algorithm.py")]]
    manifest = {"schema_version": 2, "protocol": prot, "sources": source_manifest(),
                "environment": environment_manifest(), "created_utc": datetime.now(timezone.utc).isoformat()}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"));
        if old.get("protocol") != prot or old.get("sources") != manifest["sources"]: raise RuntimeError("manifest mismatch; refusing resume")
    elif not args.validate_only: manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    def read_rows_and_verify(*, require_complete: bool = False) -> list[dict[str, str]]:
        if not out.exists():
            raise SystemExit(f"missing CSV: {out}")
        with out.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        keys = validate_rows(rows, args, require_complete=require_complete)
        manifest_sha = sha256(manifest_path) if manifest_path.exists() else ""
        for row in rows:
            key = (row["config"], int(row["fid"]), int(row["run"]))
            path = artifact_path(folder, *key)
            if not path.exists():
                raise ValueError(f"missing run artifact: {path}")
            artifact = json.loads(path.read_text(encoding="utf-8"))
            if artifact.get("manifest_sha256") != manifest_sha:
                raise ValueError(f"artifact manifest mismatch: {path}")
            saved = artifact.get("row", {})
            for field in FIELDS:
                if field not in saved:
                    raise ValueError(f"artifact row missing {field}: {path}")
                # CSV parsers return strings; compare numerics at tight tolerance.
                if field in ("config",):
                    equal = saved[field] == row[field]
                else:
                    equal = math.isclose(float(saved[field]), float(row[field]), rel_tol=1e-12, abs_tol=1e-12)
                if not equal:
                    raise ValueError(f"CSV/artifact mismatch at {field}: {path}")
        return rows

    if args.validate_only:
        rows = read_rows_and_verify(require_complete=True); expected = len(args.configs)*len(args.fids)*args.runs
        keys = {(r["config"], int(r["fid"]), int(r["run"])) for r in rows}
        print(json.dumps({"rows":len(rows), "expected":expected, "complete":len(rows)==expected and len(keys)==expected,
                          "artifacts":len(list(folder.glob("*.json"))), "manifest":str(manifest_path)}, indent=2)); return
    done = set()
    if out.exists():
        for r in read_rows_and_verify(): done.add((r["config"], int(r["fid"]), int(r["run"])))
    # Interleave configurations inside each matched (function, repetition)
    # block, so a long-running configuration cannot starve diagnostics for the
    # other controls and the paired schedule remains auditable.
    jobs = [{"config": c, "fid": f, "run": r, "seed_base": args.seed_base}
            for f in args.fids for r in range(1, args.runs + 1) for c in args.configs
            if (c, f, r) not in done]
    out.parent.mkdir(parents=True, exist_ok=True); folder.mkdir(parents=True, exist_ok=True); write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header: writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
            futures = [pool.submit(run_one, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                bundle = future.result(); row = bundle["row"]
                artifact = {"schema_version": 1, "row": row, "solutions": bundle["solutions"], "objective_values": bundle["objective_values"], "n_minima": bundle["n_minima"], "max_eval": bundle["max_eval"], "diagnostics": bundle["diagnostics"], "manifest_sha256": sha256(manifest_path)}
                # The artifact is committed first.  If the process is killed
                # between these two operations, resume sees no orphan CSV row.
                atomic_json(artifact_path(folder, row["config"], int(row["fid"]), int(row["run"])), artifact)
                # Publish the CSV row only after the artifact commit.
                writer.writerow(row); handle.flush()
                print(f"progress {i}/{len(jobs)}", flush=True)
    # A normal completion must satisfy the exact protocol; partial outputs are
    # only tolerated while a process is interrupted and later resumed.
    complete_rows = read_rows_and_verify(require_complete=True)
    print(f"done: {out} ({len(complete_rows)} verified rows; {len(jobs)} new runs, {len(done)} resumed)")


if __name__ == "__main__": main()
