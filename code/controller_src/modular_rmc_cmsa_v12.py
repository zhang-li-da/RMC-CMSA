"""Sparse-trajectory-evidence relational-memory CMSA candidate.

Version 7 separates sparse-archive coverage from mature relational allocation.
While fewer than two peaks are archived, completed terminal centers provide
coverage evidence.  If ``D`` consecutive searches fail to expand the archive,
the terminal model is disabled for the current sparse-archive epoch, including
competitive repeated-peak centers whose guidance has not produced progress.
Once two peaks are available, nearest-peak relations generate candidates and
terminal evidence no longer changes their ranking.  This prevents a long run
of low-quality terminals from becoming a permanent exclusion field.

Each CMSA trajectory also contributes a bounded pool of already evaluated,
high-quality samples.  Spatially separated samples enter the peak archive only
after the same finite hill-valley interface confirms a distinct domain.  The
evidence module therefore expands sparse archives without reporting raw
population members as optima.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

import modular_rmc_cmsa_v3 as _v3
from external_baselines import RSC_DIR, _RSCProblemAdapter, _baseline_import_path


_EPS = np.finfo(float).eps
_CONTROLS = frozenset(
    {
        "full",
        "no_trajectory_evidence",
        "no_terminal_memory",
        "no_persistent_terminal",
        "no_sparse_coverage",
        "no_relations",
        "raw_margin",
        "no_maximin",
        "no_repulsion",
    }
)


class _TrustGatedAllocator(_v3._RelationalMemoryAllocator):
    """Suspend the sparse terminal model after dimension-scaled failures."""

    def __init__(
        self,
        control: str,
        rng: np.random.Generator,
        normalized_projector: Callable[[np.ndarray, Any], np.ndarray] | None = None,
    ) -> None:
        if control not in _CONTROLS:
            choices = ", ".join(sorted(_CONTROLS))
            raise ValueError(f"unknown modular RMC control {control!r}; expected {choices}")
        self.control = control
        self.rng = rng
        self.normalized_projector = normalized_projector
        self.persistent_terminal_memory: list[np.ndarray] = []
        self.sparse_coverage_memory: list[np.ndarray] = []
        self.sparse_coverage_enabled = True
        self.no_growth_streak = 0
        self.calls = 0
        self.bootstrap_uses = 0
        self.relation_uses = 0
        self.relation_builds = 0
        self.candidate_sum = 0
        self.selected_score_sum = 0.0
        self.selected_r0_sum = 0.0
        self.persistent_records = 0
        self.sparse_coverage_records = 0
        self.productive_outcomes = 0
        self.evidence_gate_closures = 0
        self.projected_candidate_count = 0
        self.projected_candidate_displacement_sum = 0.0
        self.projected_terminal_count = 0
        self.projected_terminal_displacement_sum = 0.0

    def _project_normalized(
        self, points: np.ndarray, problem: Any, *, terminal: bool = False
    ) -> np.ndarray:
        source = np.atleast_2d(np.asarray(points, dtype=float))
        if self.normalized_projector is None:
            return source.copy()
        projected = np.asarray(
            self.normalized_projector(source.copy(), problem), dtype=float
        )
        if projected.shape != source.shape or not np.isfinite(projected).all():
            raise RuntimeError(
                "normalized projector must return a finite array with shape "
                f"{source.shape}, received {projected.shape}"
            )
        projected = np.clip(projected, 0.0, 1.0)
        displacement = np.sqrt(np.mean((projected - source) ** 2, axis=1))
        if terminal:
            self.projected_terminal_count += len(source)
            self.projected_terminal_displacement_sum += float(np.sum(displacement))
        else:
            self.projected_candidate_count += len(source)
            self.projected_candidate_displacement_sum += float(np.sum(displacement))
        return projected

    @staticmethod
    def _append_if_distinct(memory: list[np.ndarray], point: np.ndarray) -> bool:
        if any(np.array_equal(stored, point) for stored in memory):
            return False
        memory.append(point)
        return True

    def record_outcome(
        self,
        center: np.ndarray,
        problem: Any,
        *,
        archive_grew: bool,
        repeated_peak: bool,
        archive_size: int,
    ) -> None:
        """Update persistent evidence and the current sparse coverage epoch."""
        point = self._project_normalized(
            self._normalize(center, problem), problem, terminal=True
        )[0]

        if repeated_peak and self.control not in {
            "no_terminal_memory",
            "no_persistent_terminal",
        }:
            if self._append_if_distinct(self.persistent_terminal_memory, point):
                self.persistent_records += 1

        if archive_size >= 2:
            self.no_growth_streak = 0
            self.sparse_coverage_memory.clear()
            return

        if archive_grew:
            self.productive_outcomes += 1
            self.no_growth_streak = 0
            self.sparse_coverage_enabled = True
        else:
            self.no_growth_streak += 1

        if self.control not in {"no_terminal_memory", "no_sparse_coverage"} and self.sparse_coverage_enabled:
            self.sparse_coverage_memory.append(point)
            self.sparse_coverage_records += 1

        if self.no_growth_streak >= max(1, int(problem.dim)):
            if self.sparse_coverage_enabled:
                self.evidence_gate_closures += 1
            self.sparse_coverage_enabled = False
            self.sparse_coverage_memory.clear()

    def _bootstrap_scores(
        self, candidates: np.ndarray, archive_points: np.ndarray
    ) -> np.ndarray:
        references: list[np.ndarray] = []
        if len(archive_points):
            references.append(archive_points)
        if (
            self.control != "no_terminal_memory"
            and self.sparse_coverage_enabled
            and self.persistent_terminal_memory
        ):
            references.append(np.asarray(self.persistent_terminal_memory, dtype=float))
        if (
            self.control not in {"no_terminal_memory", "no_sparse_coverage"}
            and self.sparse_coverage_enabled
            and self.sparse_coverage_memory
        ):
            references.append(np.asarray(self.sparse_coverage_memory, dtype=float))
        if not references:
            return np.ones(len(candidates), dtype=float)
        reference = np.concatenate(references, axis=0)
        distance = np.linalg.norm(
            candidates[:, None, :] - reference[None, :, :], axis=2
        )
        return np.min(distance, axis=1)

    def propose(self, archive: Any, process: Any, opt: Any, problem: Any) -> _v3._Allocation:
        self.calls += 1
        dim = int(problem.dim)
        count = max(1, dim * dim)
        points = self._archive_points(archive, problem)

        candidates: np.ndarray | None = None
        source = "bootstrap"
        if self.control != "no_relations" and dim > 1 and len(points) >= 2:
            candidates = self._relation_candidates(points, count, dim)
            if candidates is not None:
                source = "relation"
        if candidates is None:
            candidates = self.rng.random((count, dim))
        candidates = self._project_normalized(candidates, problem)

        normalized_margin, raw_margin = self._archive_margins(
            candidates, archive, points
        )
        if source == "relation":
            scores = raw_margin if self.control == "raw_margin" else normalized_margin
        else:
            scores = self._bootstrap_scores(candidates, points)

        if self.control == "no_maximin" or np.ptp(scores) <= _EPS:
            selected = int(self.rng.integers(len(candidates)))
        else:
            selected = int(np.argmax(scores))

        r0 = float(process.iniR0)
        if len(points):
            reduction = float(opt.niching.redCoeff)
            if not 0.0 < reduction < 1.0:
                raise ValueError(
                    f"CMSA reduction coefficient must lie in (0,1), got {reduction}"
                )
            margin = float(normalized_margin[selected])
            while margin <= r0 and r0 > _EPS:
                r0 *= reduction
            r0 = max(r0, _EPS)

        if source == "relation":
            self.relation_uses += 1
        else:
            self.bootstrap_uses += 1
        self.candidate_sum += count
        self.selected_score_sum += float(scores[selected])
        self.selected_r0_sum += r0
        return _v3._Allocation(
            center=np.asarray(candidates[selected], dtype=float),
            r0=r0,
            source=source,
            score=float(scores[selected]),
        )

    def stats(self) -> dict[str, int | float | str]:
        denominator = max(self.calls, 1)
        persistent_size = len(self.persistent_terminal_memory)
        sparse_size = len(self.sparse_coverage_memory)
        return {
            "mrmc_control": self.control,
            "mrmc_calls": self.calls,
            "mrmc_bootstrap_uses": self.bootstrap_uses,
            "mrmc_relation_uses": self.relation_uses,
            "mrmc_relation_builds": self.relation_builds,
            "mrmc_terminal_memory_size": persistent_size + sparse_size,
            "mrmc_persistent_terminal_size": persistent_size,
            "mrmc_sparse_coverage_size": sparse_size,
            "mrmc_persistent_records": self.persistent_records,
            "mrmc_sparse_coverage_records": self.sparse_coverage_records,
            "mrmc_productive_outcomes": self.productive_outcomes,
            "mrmc_sparse_coverage_enabled": int(self.sparse_coverage_enabled),
            "mrmc_no_growth_streak": self.no_growth_streak,
            "mrmc_evidence_gate_closures": self.evidence_gate_closures,
            "mrmc_projected_candidate_count": self.projected_candidate_count,
            "mrmc_projected_candidate_displacement_mean": (
                self.projected_candidate_displacement_sum
                / max(self.projected_candidate_count, 1)
            ),
            "mrmc_projected_terminal_count": self.projected_terminal_count,
            "mrmc_projected_terminal_displacement_mean": (
                self.projected_terminal_displacement_sum
                / max(self.projected_terminal_count, 1)
            ),
            "mrmc_candidates_mean": self.candidate_sum / denominator,
            "mrmc_selected_score_mean": self.selected_score_sum / denominator,
            "mrmc_selected_r0_mean": self.selected_r0_sum / denominator,
        }


class _TrajectoryEvidence:
    """Consolidate diverse, evaluated CMSA samples into validated domains."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._points: list[np.ndarray] = []
        self._values: list[float] = []
        self.observed = 0
        self.validation_calls = 0
        self.validation_candidates = 0
        self.added_domains = 0
        self.budget_blocks = 0
        self.sufficient_archive_blocks = 0

    def begin_restart(self) -> None:
        self._points.clear()
        self._values.clear()

    def observe_generation(self, subpop: Any, problem: Any) -> None:
        if not self.enabled:
            return
        samples = getattr(subpop, "samples", None)
        if samples is None or not hasattr(samples, "X") or not hasattr(samples, "f"):
            return
        dim = int(problem.dim)
        points = np.asarray(samples.X, dtype=float).reshape(-1, dim)
        values = np.asarray(samples.f, dtype=float).reshape(-1)
        finite = np.flatnonzero(np.isfinite(values))
        if not len(finite):
            return
        keep_count = max(1, int(np.ceil(np.sqrt(dim))))
        keep = finite[np.argsort(values[finite])[:keep_count]]
        self._points.extend(points[int(index)].copy() for index in keep)
        self._values.extend(float(values[int(index)]) for index in keep)
        self.observed += len(keep)

        capacity = max(8, 3 * dim)
        if len(self._values) > capacity:
            order = np.argsort(np.asarray(self._values, dtype=float))[:capacity]
            self._points = [self._points[int(index)] for index in order]
            self._values = [self._values[int(index)] for index in order]

    @staticmethod
    def _normalized(points: np.ndarray, problem: Any) -> np.ndarray:
        low = np.asarray(problem.lowBound, dtype=float)
        span = np.maximum(np.asarray(problem.upBound, dtype=float) - low, _EPS)
        return np.clip((np.asarray(points, dtype=float) - low) / span, 0.0, 1.0)

    @staticmethod
    def _target_size(problem: Any) -> int:
        return max(3, int(np.ceil(2.0 * np.sqrt(int(problem.dim)))))

    @classmethod
    def _separation_scale(cls, archive: Any, problem: Any) -> float:
        dim = int(problem.dim)
        if int(archive.size) <= 1:
            return 1.0 / np.sqrt(max(1, dim))
        points = cls._normalized(
            np.asarray(archive.solution, dtype=float).reshape(-1, dim), problem
        )
        distance = np.linalg.norm(
            points[:, None, :] - points[None, :, :], axis=2
        )
        np.fill_diagonal(distance, np.inf)
        nearest = np.min(distance, axis=1)
        finite = nearest[np.isfinite(nearest)]
        if not len(finite):
            return 1.0 / np.sqrt(max(1, dim))
        return float(np.median(finite) + _EPS)

    def _diverse_candidates(
        self, archive: Any, problem: Any
    ) -> list[tuple[np.ndarray, float]]:
        if not self._points:
            return []
        dim = int(problem.dim)
        points = np.asarray(self._points, dtype=float).reshape(-1, dim)
        values = np.asarray(self._values, dtype=float)
        order = np.argsort(values)
        normalized = self._normalized(points, problem)
        if int(archive.size):
            archived = self._normalized(
                np.asarray(archive.solution, dtype=float).reshape(-1, dim), problem
            )
        else:
            archived = np.empty((0, dim), dtype=float)
        separation = self._separation_scale(archive, problem)
        target = self._target_size(problem)
        selected: list[int] = []
        for raw_index in order:
            index = int(raw_index)
            point = normalized[index]
            if len(archived) and float(
                np.min(np.linalg.norm(archived - point[None, :], axis=1))
            ) <= separation:
                continue
            if selected and float(
                np.min(
                    np.linalg.norm(
                        normalized[np.asarray(selected)] - point[None, :], axis=1
                    )
                )
            ) <= separation:
                continue
            selected.append(index)
            if len(selected) >= target:
                break
        return [(points[index].copy(), float(values[index])) for index in selected]

    def enrich(
        self,
        archive: Any,
        restart: Any,
        process: Any,
        opt: Any,
        problem: Any,
    ) -> int:
        if not self.enabled:
            return 0
        if int(archive.size) >= self._target_size(problem):
            self.sufficient_archive_blocks += 1
            return 0
        candidates = self._diverse_candidates(archive, problem)
        if not candidates:
            return 0
        self.validation_calls += 1
        added = 0
        for point, value in candidates:
            required = (
                min(int(archive.size), int(opt.archiving.neighborSize))
                * int(opt.archiving.hillVallBudget)
            )
            if int(problem.numCallF) + required > int(problem.maxEval):
                self.budget_blocks += 1
                break
            self.validation_candidates += 1
            is_new = True
            if int(archive.size):
                distance = np.linalg.norm(
                    np.asarray(archive.solution, dtype=float)
                    - point.reshape(1, -1),
                    axis=1,
                )
                limit = min(int(archive.size), int(opt.archiving.neighborSize))
                for index in np.argsort(distance)[:limit]:
                    if archive._same_domain(
                        point,
                        value,
                        int(index),
                        restart,
                        opt,
                        problem,
                        False,
                    ):
                        is_new = False
                        break
            if is_new:
                archive._append(point, value, process)
                added += 1
        self.added_domains += added
        return added

    def stats(self) -> dict[str, int]:
        return {
            "mrmc_evidence_observed": self.observed,
            "mrmc_evidence_validation_calls": self.validation_calls,
            "mrmc_evidence_candidates": self.validation_candidates,
            "mrmc_evidence_added_domains": self.added_domains,
            "mrmc_evidence_budget_blocks": self.budget_blocks,
            "mrmc_evidence_sufficient_archive_blocks": (
                self.sufficient_archive_blocks
            ),
        }


class ModularRMCCMSA(_v3.ModularRMCCMSA):
    """Independent RMC-CMSA with validated trajectory evidence."""

    @property
    def name(self) -> str:
        return "Modular-RMC-CMSA-v12"

    def _run(self, problem: Any) -> np.ndarray:
        seed = self.params.get("seed")
        if seed is not None:
            np.random.seed(int(seed) % (2**32 - 1))
        if not RSC_DIR.exists():
            raise RuntimeError(f"CMSA source directory not found: {RSC_DIR}")
        if not hasattr(np, "int"):
            np.int = int  # type: ignore[attr-defined]

        control = str(self.params.get("causal_control", "full"))
        guide_seed = None if seed is None else (int(seed) ^ 0x4D524D43) % (2**32)
        allocator = _TrustGatedAllocator(
            control, np.random.default_rng(guide_seed)
        )
        evidence = _TrajectoryEvidence(control != "no_trajectory_evidence")

        with _baseline_import_path():
            from OptimOption import OptimOption
            from OptimProcess import OptimProcess
            from Subpopulation import SubpopulationCMSA

            adapted = _RSCProblemAdapter(problem)
            opt = OptimOption(adapted)
            process = OptimProcess(opt, adapted)
            archive = _v3._PeakArchive(adapted)
            started = time.monotonic()
            equivalence_stops = 0
            completed_restarts = 0

            while int(adapted.numCallF) + int(process.subpopSize) <= int(
                adapted.maxEval
            ):
                restart = _v3._RestartState(process, opt, adapted)
                allocation = allocator.propose(archive, process, opt, adapted)
                restart.recIniR0 = float(allocation.r0)
                restart.source = allocation.source
                low = np.asarray(adapted.lowBound, dtype=float)
                span = np.asarray(adapted.upBound, dtype=float) - low
                sigma = min(
                    float(opt.coreSearch.maxIniSigma),
                    float(allocation.r0) * float(opt.coreSearch.iniSigCoeff),
                )
                subpop = SubpopulationCMSA(
                    low + allocation.center * span,
                    sigma,
                    span,
                    int(process.subpopSize),
                )
                evidence.begin_restart()

                budget_exhausted = False
                while int(adapted.numCallF) + int(process.subpopSize) <= int(
                    adapted.maxEval
                ):
                    restart.iterNo += 1
                    restart.stagSize = int(
                        opt.stopCr.stagPar[0]
                        + opt.stopCr.stagPar[1] * restart.iterNo
                        + opt.stopCr.stagPar[2]
                        * adapted.dim
                        / process.subpopSize
                    )
                    subpop.update_taboo_region(restart, archive, process, opt, adapted)
                    if control == "no_repulsion":
                        self._configure_repulsion(subpop, archive, opt, False)
                    subpop.update_merge_check(restart, archive, opt, adapted)
                    subpop.evolve(restart, archive, process, opt, adapted)
                    evidence.observe_generation(subpop, adapted)
                    subpop.update_term_flag(restart, archive, process, opt, adapted)
                    if int(subpop.terminationFlag) != 0:
                        if int(subpop.terminationFlag) == 3:
                            equivalence_stops += 1
                        break

                    reserve = (
                        min(archive.size, int(opt.archiving.neighborSize))
                        * int(opt.archiving.hillVallBudget)
                    )
                    if (
                        int(adapted.numCallF)
                        + int(process.subpopSize)
                        + reserve
                        > int(adapted.maxEval)
                    ):
                        budget_exhausted = True
                        break

                restart.subpopBestVal = float(subpop.bestVal)
                restart.subpopTermFlag = int(subpop.terminationFlag)
                if budget_exhausted:
                    restart.terminationFlag = -1
                    break
                archive_reserve = (
                    min(archive.size, int(opt.archiving.neighborSize))
                    * int(opt.archiving.hillVallBudget)
                )
                if int(adapted.numCallF) + archive_reserve > int(adapted.maxEval):
                    restart.terminationFlag = -1
                    break
                restart.terminationFlag = 1
                archive_grew = archive.update(subpop, restart, process, opt, adapted)
                evidence_added = evidence.enrich(
                    archive, restart, process, opt, adapted
                )
                archive_grew = bool(archive_grew or evidence_added)
                allocator.record_outcome(
                    np.asarray(subpop.center),
                    adapted,
                    archive_grew=archive_grew,
                    repeated_peak=bool(archive.evidence_eligible),
                    archive_size=int(archive.size),
                )
                process.update(restart, archive, opt, adapted)
                completed_restarts += 1

                if time.monotonic() - started > float(
                    self.params.get("walltime_limit_sec", np.inf)
                ):
                    break

            if archive.size:
                keep = archive.value < float(np.min(archive.value)) + float(
                    opt.archiving.tolFunArch
                )
                archive._prune(keep)
            self.mrmc_stats = allocator.stats() | archive.stats() | evidence.stats() | {
                "mrmc_completed_restarts": completed_restarts,
                "mrmc_early_stops": equivalence_stops,
                "mrmc_used_eval": int(adapted.numCallF),
            }
            self.vlarta_stats = self.mrmc_stats
            if archive.size:
                return np.asarray(archive.solution, dtype=float).reshape(
                    -1, problem.dim
                )
            return np.empty((0, problem.dim), dtype=float)


class TAGDE(ModularRMCCMSA):
    """Compatibility alias for the unified experiment runners."""
