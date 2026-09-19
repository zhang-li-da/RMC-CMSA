#!/usr/bin/env python3
"""Four-deletion CEC2026 component-profile campaign.

Runs the frozen RMC-CMSA implementation directly under the four deleted
controls used by the supplementary compatibility profile. Full, ladder, and
no-state controls are rejected by the CLI. The remaining process-local patch
machinery is retained from the revision driver:
- RMC scale parameters: candidate_pool_scale, sparse_gate_horizon_scale,
  trajectory_target_scale, trajectory_keep_scale, trajectory_capacity_scale,
  trajectory_separation_scale, relation_shell_scale
- local-search depth coefficient s_r (``local_depth_scale``): scales the
  inherited CMSA stagnation window (stopCr.stagPar) and the convergence
  history window (_RestartState.tolHistSize) that together bound how many
  generations one local search may run
- budget_scale: multiplies the CEC2026 per-case evaluation budget

Seeds are derived exactly like the frozen runner
(blake2b "<base>|<pid>|<pin>|<dim>|run<repeat>"), so runs are matched across
configurations and comparable with the recorded formal results.

Each configuration is executed in a fresh worker process; parameter patches
are process-local and never modify the archived sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CODE = str(Path(__file__).resolve().parents[2] / "code" / "frozen_four_component" / "code")
RSC = str(Path(__file__).resolve().parents[2] / "code" / "baseline_src" / "RSCMSAESII_v1")
sys.path.insert(0, CODE)
sys.path.insert(0, RSC)


def derive_seed(base: int, pid: int, pin: int, dim: int, repeat: int) -> int:
    text = f"{base}|{pid}|{pin}|{dim}|run{repeat}"
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


# ----------------------------------------------------------------------------
# Process-local patches (depth coefficient and budget scale)
# ----------------------------------------------------------------------------

_DEPTH_SCALE = 1.0
_DEPTH_MODE = "both"
_BUDGET_SCALE = 1.0
_CAP_MULT = None
_PATCHED = False


def apply_patches(
    depth_scale: float,
    budget_scale: float,
    depth_mode: str = "both",
    cap_mult: float | None = None,
) -> None:
    global _DEPTH_SCALE, _DEPTH_MODE, _BUDGET_SCALE, _CAP_MULT, _PATCHED
    _DEPTH_SCALE, _BUDGET_SCALE = float(depth_scale), float(budget_scale)
    _DEPTH_MODE = depth_mode
    _CAP_MULT = cap_mult
    if _PATCHED:
        return
    _PATCHED = True

    import OptimOption as _OO

    orig_opt_init = _OO.OptimOption.__init__

    def opt_init(self, problem):
        orig_opt_init(self, problem)
        if _DEPTH_SCALE != 1.0:
            self.stopCr.stagPar = self.stopCr.stagPar * _DEPTH_SCALE

    _OO.OptimOption.__init__ = opt_init

    import modular_rmc_cmsa_v3 as _v3

    orig_restart_init = _v3._RestartState.__init__

    def restart_init(self, process, opt, problem):
        orig_restart_init(self, process, opt, problem)
        if _DEPTH_SCALE != 1.0 and _DEPTH_MODE == "both":
            self.tolHistSize = max(5, int(round(self.tolHistSize * _DEPTH_SCALE)))

    _v3._RestartState.__init__ = restart_init

    import Subpopulation as _SP

    orig_term_flag = _SP.SubpopulationCMSA.update_term_flag

    def term_flag(self, restart, archive, process, opt, problem):
        orig_term_flag(self, restart, archive, process, opt, problem)
        if _CAP_MULT is not None and self.terminationFlag == 0:
            stag0 = (
                float(opt.stopCr.stagPar[0])
                + float(opt.stopCr.stagPar[2]) * problem.dim / process.subpopSize
            )
            cap = max(5, int(_CAP_MULT * stag0 + 0.5))
            if int(self.iterNo) >= cap:
                self.terminationFlag = -1

    _SP.SubpopulationCMSA.update_term_flag = term_flag

    from CMMOP.ProblemMM import ProblemMM

    orig_form = ProblemMM.form

    def form(self):
        orig_form(self)
        if _BUDGET_SCALE != 1.0:
            self.max_eval = int(0.5 + self.max_eval * _BUDGET_SCALE)

    ProblemMM.form = form




_NO_STATE_READY = False
_SAFEGUARD = False
_SAFEGUARD_LIMIT = 10_000


def install_rejection_safeguard() -> None:
    """Bound pathological taboo-rejection spinning (part of the substrate
    control definition): after 10^4 consecutive rejected candidate samples
    within one generation, the next sample is accepted unconditionally.
    Inactive on runs whose searches never hit the consecutive limit."""
    global _SAFEGUARD
    if _SAFEGUARD:
        return
    _SAFEGUARD = True
    import Subpopulation as _SP

    orig_se = _SP.SubpopulationCMSA.sample_and_eval
    orig_accept = _SP.SubpopulationCMSA.is_taboo_acceptable
    streaks: dict[int, int] = {}

    def sample_and_eval(self, restart, archive, process, opt, problem):
        streaks[id(self)] = 0
        try:
            orig_se(self, restart, archive, process, opt, problem)
        finally:
            streaks.pop(id(self), None)

    def is_taboo_acceptable(self, sample, tempRedRatio, opt):
        if orig_accept(self, sample, tempRedRatio, opt):
            streaks[id(self)] = 0
            return True
        streak = streaks.get(id(self), 0) + 1
        streaks[id(self)] = streak
        if streak >= _SAFEGUARD_LIMIT:
            streaks[id(self)] = 0
            return True
        return False

    _SP.SubpopulationCMSA.sample_and_eval = sample_and_eval
    _SP.SubpopulationCMSA.is_taboo_acceptable = is_taboo_acceptable


def install_no_state() -> None:
    """Add a composed 'no_state' control: uniform-only pool, uniform
    selection, no terminal memory, no trajectory evidence (the inherited
    peak archive and its taboo/merge interfaces remain active)."""
    global _NO_STATE_READY
    if _NO_STATE_READY:
        return
    _NO_STATE_READY = True
    import numpy as _np
    import rmc_cmsa_final_four_component as _rmc

    if "no_state" not in _rmc._CONTROLS:
        _rmc._CONTROLS = frozenset(set(_rmc._CONTROLS) | {"no_state"})

    class NoStateAllocator(_rmc._TrustGatedAllocator):
        def __init__(self, control, *args, **kwargs):
            # the v12 base validates against its own legacy control list;
            # register as no_maximin upstream and restore the label below
            super().__init__("no_maximin", *args, **kwargs)
            self.control = control

        def _relation_candidates(self, points, count, dim):
            return None

        def propose(self, archive, process, opt, problem, working_points=None):
            saved = self.control
            self.control = "no_maximin"
            try:
                return super().propose(
                    archive, process, opt, problem, working_points
                )
            finally:
                self.control = saved

        def record_outcome(self, *args, **kwargs):
            saved = self.control
            self.control = "no_terminal_memory"
            try:
                super().record_outcome(*args, **kwargs)
            finally:
                self.control = saved

    original_build = _rmc.build_search_components

    def build(params, **kwargs):
        control = str(params.get("causal_control", "full"))
        if control != "no_state":
            return original_build(params, **kwargs)
        (
            candidate_scale,
            gate_scale,
            _t,
            _k,
            _c,
            _s,
            relation_shell_scale,
        ) = _rmc._sensitivity_scales(params)
        seed = params.get("seed")
        guide_seed = None if seed is None else (int(seed) ^ 0x4D524D43) % (2**32)
        allocator = NoStateAllocator(
            "no_state",
            _np.random.default_rng(guide_seed),
            candidate_pool_scale=candidate_scale,
            sparse_gate_horizon_scale=gate_scale,
            relation_shell_scale=relation_shell_scale,
        )
        evidence_enabled = control == "ladder_trajectory"
        return allocator, _rmc._TrajectoryEvidence(evidence_enabled)

    _rmc.build_search_components = build



_LADDER_READY = False


def install_ladder_controls() -> None:
    """Additive-ladder intermediate controls between the no_state substrate
    and the full architecture (each adds exactly one component):

    - ladder_maximin:   uniform pool + maximin selection (no relations,
      no terminal memory, no trajectory evidence)
    - ladder_relations: + representative-relation candidates (still no
      terminal memory and no trajectory evidence)

    The remaining ladder levels reuse existing configurations:
    +terminal memory == RMC-NoTrajectoryEvidence (formal runs),
    +trajectory evidence == RMC-CMSA full (formal runs).
    """
    global _LADDER_READY
    if _LADDER_READY:
        return
    _LADDER_READY = True
    import numpy as _np
    import rmc_cmsa_final_four_component as _rmc

    ladder_names = {
        "ladder_maximin",
        "ladder_relations",
        "ladder_terminal",
        "ladder_trajectory",
    }
    _rmc._CONTROLS = frozenset(set(_rmc._CONTROLS) | ladder_names)

    class LadderAllocator(_rmc._TrustGatedAllocator):
        def __init__(self, control, *args, **kwargs):
            super().__init__("full", *args, **kwargs)
            self.control = control

        def _relation_candidates(self, points, count, dim):
            # The ladder isolates additions in a fixed order.  The first
            # level uses a uniform pool; relation-shell proposals are enabled
            # from +Relations onward, including +Terminal and +Trajectory.
            if self.control == "ladder_maximin":
                return None
            return super()._relation_candidates(points, count, dim)

        def record_outcome(self, *args, **kwargs):
            # ladder_terminal and ladder_trajectory keep the terminal
            # channels; the lower two levels disable them
            if self.control in {"ladder_terminal", "ladder_trajectory"}:
                return super().record_outcome(*args, **kwargs)
            saved = self.control
            self.control = "no_terminal_memory"
            try:
                super().record_outcome(*args, **kwargs)
            finally:
                self.control = saved

    previous_build = _rmc.build_search_components

    def build(params, **kwargs):
        control = str(params.get("causal_control", "full"))
        if control not in ladder_names:
            return previous_build(params, **kwargs)
        (
            candidate_scale,
            gate_scale,
            _t,
            _k,
            _c,
            _s,
            relation_shell_scale,
        ) = _rmc._sensitivity_scales(params)
        seed = params.get("seed")
        guide_seed = None if seed is None else (int(seed) ^ 0x4D524D43) % (2**32)
        allocator = LadderAllocator(
            control,
            _np.random.default_rng(guide_seed),
            normalized_projector=kwargs.get("normalized_projector"),
            candidate_pool_scale=candidate_scale,
            sparse_gate_horizon_scale=gate_scale,
            relation_shell_scale=relation_shell_scale,
        )
        evidence_type = kwargs.get("evidence_type") or _rmc._TrajectoryEvidence
        evidence_enabled = control == "ladder_trajectory"
        return allocator, evidence_type(evidence_enabled)

    _rmc.build_search_components = build

# ----------------------------------------------------------------------------
# Configuration registry
# ----------------------------------------------------------------------------

def build_configs() -> dict[str, dict]:
    """Named configurations; every entry is a kwargs dict for the frozen class."""
    configs: dict[str, dict] = {
        "RMC-CMSA": {"causal_control": "full"},
        # ablation controls
        "RMC-NoTrajectoryEvidence": {"causal_control": "no_trajectory_evidence"},
        "RMC-NoTerminalMemory": {"causal_control": "no_terminal_memory"},
        "RMC-NoRelations": {"causal_control": "no_relations"},
        "RMC-NoMaximin": {"causal_control": "no_maximin"},
        "RMC-NoState": {"causal_control": "no_state"},
        "RMC-LadderMaximin": {"causal_control": "ladder_maximin"},
        "RMC-LadderRelations": {"causal_control": "ladder_relations"},
        "RMC-LadderTerminal": {"causal_control": "ladder_terminal"},
        "RMC-LadderTrajectory": {"causal_control": "ladder_trajectory"},
    }
    # local-search depth coefficient s_r (stagnation window + convergence window)
    for value in (0.25, 0.5, 1.0, 2.0, 4.0):
        configs[f"RMC-Depth-{value:g}"] = {
            "causal_control": "full",
            "_depth_scale": value,
            "_depth_mode": "both",
        }
    # per-restart generation cap in units of the inherited stagnation scale s0
    for value in (1.0, 2.0, 4.0, 8.0, 16.0):
        configs[f"RMC-Cap-{value:g}"] = {
            "causal_control": "full",
            "_cap_mult": value,
        }
    # stagnation-window-only depth variant
    for value in (0.25, 0.5, 1.0, 2.0, 4.0):
        configs[f"RMC-Stag-{value:g}"] = {
            "causal_control": "full",
            "_depth_scale": value,
            "_depth_mode": "stag",
        }
    # representative-relation shell scale
    for value in (0.5, 1.0, 2.0):
        configs[f"RMC-RelShell-{value:g}"] = {
            "causal_control": "full",
            "relation_shell_scale": value,
        }
    # terminal-memory sparse-gate horizon scale
    for value in (0.5, 1.0, 2.0, 4.0):
        configs[f"RMC-Gate-{value:g}"] = {
            "causal_control": "full",
            "sparse_gate_horizon_scale": value,
        }
    # trajectory-evidence capacity scale
    for value in (1.0 / 3.0, 1.0, 2.0):
        configs[f"RMC-TrajCap-{value:g}"] = {
            "causal_control": "full",
            "trajectory_capacity_scale": value,
        }
    # trajectory-evidence separation scale
    for value in (0.5, 1.0, 2.0):
        configs[f"RMC-TrajSep-{value:g}"] = {
            "causal_control": "full",
            "trajectory_separation_scale": value,
        }
    return configs


FIELDS = [
    "config", "pid", "pin", "dim", "repeat", "seed", "budget_scale",
    "rpr", "f1", "score", "precision", "n_sol", "n_targets", "used_eval", "elapsed_sec",
    # Component-path diagnostics.  These are written for the rebuilt
    # ablation campaign so that a metric difference cannot be attributed to a
    # control unless the intended path was actually exercised.
    "control", "mrmc_relation_uses", "mrmc_maximin_selections",
    "mrmc_random_selections", "mrmc_terminal_memory_peak",
    "mrmc_persistent_guidance_uses", "mrmc_sparse_guidance_uses",
    "mrmc_working_memory_peak", "mrmc_evidence_observed",
    "mrmc_working_additions", "mrmc_working_promotions",
]


def run_one(job: dict) -> dict:
    install_no_state()
    install_ladder_controls()
    if job.pop("safeguard", False):
        install_rejection_safeguard()
    apply_patches(
        job.pop("_depth_scale", 1.0),
        job["budget_scale"],
        job.pop("_depth_mode", "both"),
        job.pop("_cap_mult", None),
    )
    from rmc_cmsa_final_four_component import ModularRMCCMSAFinal

    kwargs = {k: v for k, v in job["params"].items() if not k.startswith("_")}
    seed = derive_seed(job["seed_base"], job["pid"], job["pin"], job["dim"], job["repeat"])
    algo = ModularRMCCMSAFinal(seed=seed, **kwargs)
    started = time.perf_counter()
    result = algo.run(job["pid"], job["pin"], job["dim"])
    elapsed = time.perf_counter() - started
    stats = dict(getattr(algo, "mrmc_stats", {}) or {})
    control = str(kwargs.get("causal_control", "full"))
    return {
        "config": job["config"],
        "pid": job["pid"],
        "pin": job["pin"],
        "dim": job["dim"],
        "repeat": job["repeat"],
        "seed": seed,
        "budget_scale": job["budget_scale"],
        "rpr": result.rpr,
        "f1": result.f1,
        "score": result.score,
        "precision": result.precision,
        "n_sol": int(result.solutions.shape[0]),
        "n_targets": int(result.n_minima),
        "used_eval": int(result.used_eval),
        "elapsed_sec": round(elapsed, 3),
        "control": control,
        "mrmc_relation_uses": int(stats.get("mrmc_relation_uses", 0)),
        "mrmc_maximin_selections": int(stats.get("mrmc_maximin_selections", 0)),
        "mrmc_random_selections": int(stats.get("mrmc_random_selections", 0)),
        "mrmc_terminal_memory_peak": int(stats.get("mrmc_terminal_memory_peak", 0)),
        "mrmc_persistent_guidance_uses": int(stats.get("mrmc_persistent_guidance_uses", 0)),
        "mrmc_sparse_guidance_uses": int(stats.get("mrmc_sparse_guidance_uses", 0)),
        "mrmc_working_memory_peak": int(stats.get("mrmc_working_memory_peak", 0)),
        "mrmc_evidence_observed": int(stats.get("mrmc_evidence_observed", 0)),
        "mrmc_working_additions": int(stats.get("mrmc_working_additions", 0)),
        "mrmc_working_promotions": int(stats.get("mrmc_working_promotions", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--pids", type=int, nargs="+", required=True)
    parser.add_argument("--pin", type=int, default=1)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=20260808)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--safeguard", action="store_true")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    registry = build_configs()
    # The shared builder contains historical entries, but this supplementary
    # driver deliberately exposes only the four deleted variants below.
    expected_configs = (
        "RMC-NoMaximin",
        "RMC-NoRelations",
        "RMC-NoTerminalMemory",
        "RMC-NoTrajectoryEvidence",
    )
    if tuple(args.configs) != expected_configs:
        raise SystemExit("use exactly the four deleted configurations in this order: " + " ".join(expected_configs))
    unknown = [name for name in args.configs if name not in registry]
    if unknown:
        raise SystemExit(f"unknown configs: {unknown}; choices: {sorted(registry)}")

    jobs = []
    for name in args.configs:
        params = dict(registry[name])
        depth = params.pop("_depth_scale", 1.0)
        depth_mode = params.pop("_depth_mode", "both")
        cap_mult = params.pop("_cap_mult", None)
        for pid in args.pids:
            for repeat in range(1, args.repeats + 1):
                jobs.append(
                    {
                        "config": name,
                        "params": params,
                        "_depth_scale": depth,
                        "_depth_mode": depth_mode,
                        "_cap_mult": cap_mult,
                        "budget_scale": args.budget_scale,
                        "pid": pid,
                        "pin": args.pin,
                        "dim": args.dim,
                        "repeat": repeat,
                        "seed_base": args.seed_base,
                        "safeguard": args.safeguard,
                    }
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size > 0:
        with out.open() as handle:
            for row in csv.DictReader(handle):
                done.add((row["config"], int(row["pid"]), int(row["repeat"])))
    jobs = [j for j in jobs if (j["config"], j["pid"], j["repeat"]) not in done]
    print(f"jobs to run: {len(jobs)} (skipped {len(done)} completed)", flush=True)

    write_header = (not out.exists()) or out.stat().st_size == 0
    with out.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, job): job for job in jobs}
            finished = 0
            for future in as_completed(futures):
                row = future.result()
                writer.writerow(row)
                handle.flush()
                finished += 1
                if finished % 20 == 0 or finished == len(jobs):
                    print(f"progress {finished}/{len(jobs)}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
