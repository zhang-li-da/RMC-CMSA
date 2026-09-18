"""Modular Relational Memory Coverage CMSA (development candidate).

This module is an independent restart architecture around the published CMSA
sampling, selection, and recombination kernel.  It does not inherit from the
RS-CMSA-ESII ``Restart`` or ``Archive`` classes and never falls back to their
initializer.  Four factorized mechanisms address distinct multimodal-search
failure modes:

* terminal-memory bootstrap before a reliable peak relation exists;
* nearest-peak relational proposals after the peak archive becomes mature;
* taboo-normalized finite-candidate maximin allocation; and
* archive-aware rejection inside the local CMSA distribution.

The implementation is intentionally a development candidate.  It must pass
matched component gates before any mechanism or performance claim is made.
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple

import numpy as np
from scipy.special import ndtr

from base_algorithm import BaseAlgorithm
from external_baselines import RSC_DIR, _RSCProblemAdapter, _baseline_import_path


_EPS = np.finfo(float).eps
_CONTROLS = frozenset(
    {
        "full",
        "no_terminal_memory",
        "no_relations",
        "raw_margin",
        "no_maximin",
        "no_repulsion",
    }
)


class _Allocation(NamedTuple):
    center: np.ndarray
    r0: float
    source: str
    score: float


class _RelationalMemoryAllocator:
    """Allocate every restart without invoking a native RS initializer."""

    def __init__(self, control: str, rng: np.random.Generator) -> None:
        if control not in _CONTROLS:
            choices = ", ".join(sorted(_CONTROLS))
            raise ValueError(f"unknown modular RMC control {control!r}; expected {choices}")
        self.control = control
        self.rng = rng
        self.terminal_memory: list[np.ndarray] = []
        self.calls = 0
        self.bootstrap_uses = 0
        self.relation_uses = 0
        self.relation_builds = 0
        self.candidate_sum = 0
        self.selected_score_sum = 0.0
        self.selected_r0_sum = 0.0
        self.terminal_evidence_records = 0

    @staticmethod
    def _normalize(x: np.ndarray, problem: Any) -> np.ndarray:
        low = np.asarray(problem.lowBound, dtype=float)
        span = np.maximum(np.asarray(problem.upBound, dtype=float) - low, _EPS)
        return np.clip((np.asarray(x, dtype=float) - low) / span, 0.0, 1.0)

    def record_terminal(self, center: np.ndarray, problem: Any) -> None:
        """Retain only a terminal state that was competitive evidence.

        A poor terminal value does not identify an explored attraction domain;
        recording it would eventually repel all useful restart locations.  The
        caller therefore invokes this method only after the archive has marked
        the completed search as a competitive non-new landing.
        """
        point = self._normalize(center, problem)
        if self.terminal_memory:
            distance = np.linalg.norm(
                np.asarray(self.terminal_memory, dtype=float) - point[None, :],
                axis=1,
            )
            if float(np.min(distance)) <= 1e-12:
                return
        self.terminal_memory.append(point)
        self.terminal_evidence_records += 1

    @staticmethod
    def _archive_points(archive: Any, problem: Any) -> np.ndarray:
        if int(archive.size) == 0:
            return np.empty((0, int(problem.dim)), dtype=float)
        low = np.asarray(problem.lowBound, dtype=float)
        span = np.maximum(np.asarray(problem.upBound, dtype=float) - low, _EPS)
        solution = np.asarray(archive.solution, dtype=float).reshape(-1, int(problem.dim))
        return np.clip((solution - low) / span, 0.0, 1.0)

    @staticmethod
    def _nearest_relations(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(points) < 2:
            return np.empty((0, 2), dtype=int), np.empty(0, dtype=float)
        distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(distance, np.inf)
        edges: set[tuple[int, int]] = set()
        for index in range(len(points)):
            neighbor = int(np.argmin(distance[index]))
            edge = (min(index, neighbor), max(index, neighbor))
            edges.add(edge)
        edge_array = np.asarray(sorted(edges), dtype=int).reshape(-1, 2)
        length = np.linalg.norm(
            points[edge_array[:, 1]] - points[edge_array[:, 0]], axis=1
        )
        valid = np.isfinite(length) & (length > _EPS)
        return edge_array[valid], length[valid]

    def _relation_candidates(
        self, points: np.ndarray, count: int, dim: int
    ) -> np.ndarray | None:
        edge, length = self._nearest_relations(points)
        if not len(edge):
            return None
        self.relation_builds += 1
        picked = self.rng.integers(len(edge), size=count)
        side = self.rng.integers(2, size=count)
        origin = points[edge[picked, side]]
        direction = self.rng.normal(size=(count, dim))
        norm = np.linalg.norm(direction, axis=1)
        zero = norm <= _EPS
        while np.any(zero):
            direction[zero] = self.rng.normal(size=(int(np.sum(zero)), dim))
            norm[zero] = np.linalg.norm(direction[zero], axis=1)
            zero = norm <= _EPS
        direction /= norm[:, None]
        return np.clip(origin + length[picked, None] * direction, 0.0, 1.0)

    def _bootstrap_scores(
        self, candidates: np.ndarray, archive_points: np.ndarray
    ) -> np.ndarray:
        references: list[np.ndarray] = []
        if len(archive_points):
            references.append(archive_points)
        if self.control != "no_terminal_memory" and self.terminal_memory:
            references.append(np.asarray(self.terminal_memory, dtype=float))
        if not references:
            return np.ones(len(candidates), dtype=float)
        reference = np.concatenate(references, axis=0)
        distance = np.linalg.norm(
            candidates[:, None, :] - reference[None, :, :], axis=2
        )
        return np.min(distance, axis=1)

    def _terminal_scores(self, candidates: np.ndarray) -> np.ndarray:
        """Distance to competitive terminal evidence in normalized space."""
        if self.control == "no_terminal_memory" or not self.terminal_memory:
            return np.full(len(candidates), np.inf, dtype=float)
        reference = np.asarray(self.terminal_memory, dtype=float)
        distance = np.linalg.norm(
            candidates[:, None, :] - reference[None, :, :], axis=2
        )
        return np.min(distance, axis=1)

    @staticmethod
    def _archive_margins(
        candidates: np.ndarray, archive: Any, archive_points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not len(archive_points):
            ones = np.ones(len(candidates), dtype=float)
            return ones, ones
        distance = np.linalg.norm(
            candidates[:, None, :] - archive_points[None, :, :], axis=2
        )
        radius = np.maximum(
            np.asarray(archive.normTabDis, dtype=float).reshape(-1), _EPS
        )
        return np.min(distance / radius[None, :], axis=1), np.min(distance, axis=1)

    def propose(self, archive: Any, process: Any, opt: Any, problem: Any) -> _Allocation:
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

        normalized_margin, raw_margin = self._archive_margins(
            candidates, archive, points
        )
        if source == "relation":
            scores = raw_margin if self.control == "raw_margin" else normalized_margin
            # Relation shells are judged against both known peaks and
            # competitive terminal evidence.  This keeps the terminal-state
            # mechanism active after the peak archive becomes nonempty.
            scores = np.minimum(scores, self._terminal_scores(candidates))
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
                raise ValueError(f"CMSA reduction coefficient must lie in (0,1), got {reduction}")
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
        return _Allocation(
            center=np.asarray(candidates[selected], dtype=float),
            r0=r0,
            source=source,
            score=float(scores[selected]),
        )

    def stats(self) -> dict[str, int | float | str]:
        denominator = max(self.calls, 1)
        return {
            "mrmc_control": self.control,
            "mrmc_calls": self.calls,
            "mrmc_bootstrap_uses": self.bootstrap_uses,
            "mrmc_relation_uses": self.relation_uses,
            "mrmc_relation_builds": self.relation_builds,
            "mrmc_terminal_memory_size": len(self.terminal_memory),
            "mrmc_terminal_evidence_records": self.terminal_evidence_records,
            "mrmc_candidates_mean": self.candidate_sum / denominator,
            "mrmc_selected_score_mean": self.selected_score_sum / denominator,
            "mrmc_selected_r0_mean": self.selected_r0_sum / denominator,
        }


class _PeakArchive:
    """Peak archive with adaptive exclusion radii and hill-valley identity."""

    def __init__(self, problem: Any) -> None:
        self.problem = problem
        self.solution = np.empty((0, int(problem.dim)), dtype=float)
        self.value = np.empty(0, dtype=float)
        self.normTabDis = np.empty(0, dtype=float)
        self.hitTimesSoFar = np.empty(0, dtype=int)
        self.hitTimesThisRestart = np.empty(0, dtype=int)
        self.size = 0
        self.usedEval = 0
        self.new_peaks = 0
        self.repeated_peaks = 0
        self.hill_valley_evals = 0
        self.evidence_eligible = False

    def _append(
        self, x: np.ndarray, fx: float, process: Any
    ) -> None:
        point = np.asarray(x, dtype=float).reshape(1, -1)
        self.solution = np.concatenate((self.solution, point), axis=0)
        self.value = np.append(self.value, float(fx))
        self.normTabDis = np.append(
            self.normTabDis, max(float(process.defNormTabDis), _EPS)
        )
        self.hitTimesSoFar = np.append(self.hitTimesSoFar, 0)
        self.hitTimesThisRestart = np.append(self.hitTimesThisRestart, 0)
        self.size += 1
        self.new_peaks += 1

    def _prune(self, keep: np.ndarray) -> None:
        self.solution = self.solution[keep]
        self.value = self.value[keep]
        self.normTabDis = self.normTabDis[keep]
        self.hitTimesSoFar = self.hitTimesSoFar[keep]
        self.hitTimesThisRestart = self.hitTimesThisRestart[keep]
        self.size = int(len(self.value))

    def _same_domain(
        self,
        x: np.ndarray,
        fx: float,
        archive_index: int,
        restart: Any,
        opt: Any,
        problem: Any,
        online: bool,
    ) -> bool:
        probes = int(opt.archiving.hillVallBudget)
        for _ in range(probes):
            if int(problem.numCallF) >= int(problem.maxEval):
                return True
            fraction = 0.1 + 0.8 * float(np.random.random())
            point = self.solution[archive_index] + fraction * (
                np.asarray(x, dtype=float) - self.solution[archive_index]
            )
            value = float(problem.func_eval(point))
            self.hill_valley_evals += 1
            if online:
                restart.usedEvalMerge += 1
            else:
                self.usedEval += 1
            if value > max(float(fx), float(self.value[archive_index])) + float(
                opt.stopCr.tolHistFun
            ):
                return False
        return True

    def matches(
        self, x: np.ndarray, fx: float, restart: Any, opt: Any, problem: Any
    ) -> bool:
        if self.size == 0 or float(fx) > float(np.min(self.value)) + float(
            opt.archiving.tolFunArch
        ):
            return False
        distance = np.linalg.norm(
            self.solution - np.asarray(x, dtype=float).reshape(1, -1), axis=1
        )
        limit = min(self.size, int(opt.archiving.neighborSize))
        for index in np.argsort(distance)[:limit]:
            if self._same_domain(x, fx, int(index), restart, opt, problem, True):
                return True
        return False

    def update(
        self, subpop: Any, restart: Any, process: Any, opt: Any, problem: Any
    ) -> bool:
        self.usedEval = 0
        self.hitTimesThisRestart = np.zeros(self.size, dtype=int)
        self.evidence_eligible = False
        fx = float(subpop.bestVal)
        if not np.isfinite(fx):
            return False

        previous_best = min(
            float(process.bestValTillRestart),
            float(np.min(self.value)) if self.size else np.inf,
        )
        best = min(fx, previous_best)
        if self.size:
            keep = self.value < best + float(opt.archiving.tolFunArch)
            self._prune(keep)

        desirable = restart.iterNo > 1 and fx <= previous_best + float(
            opt.archiving.tolFunArch
        )
        is_new = False
        matched: int | None = None
        if desirable:
            if self.size == 0:
                is_new = True
            else:
                distance = np.linalg.norm(
                    self.solution - np.asarray(subpop.bestSol).reshape(1, -1), axis=1
                )
                limit = min(self.size, int(opt.archiving.neighborSize))
                is_new = True
                for index in np.argsort(distance)[:limit]:
                    if self._same_domain(
                        np.asarray(subpop.bestSol),
                        fx,
                        int(index),
                        restart,
                        opt,
                        problem,
                        False,
                    ):
                        is_new = False
                        matched = int(index)
                        break
            if is_new:
                self._append(np.asarray(subpop.bestSol), fx, process)
            elif matched is not None:
                # This is the only terminal state that supplies negative
                # coverage evidence: it was competitive but equivalent to an
                # already archived peak.
                self.evidence_eligible = True
                self.repeated_peaks += 1
                self.hitTimesThisRestart[matched] += 1
                self.hitTimesSoFar[matched] += 1
                if fx < self.value[matched] - float(opt.stopCr.tolHistFun):
                    self.solution[matched] = np.asarray(subpop.bestSol, dtype=float)
                    self.value[matched] = fx

        if self.size:
            learning = float(opt.archiving.tauNormTabDis)
            if not desirable:
                self.normTabDis *= np.exp(
                    -learning * float(opt.archiving.targetGlobFr) / self.size
                )
            elif matched is not None:
                if self.size == 1:
                    update = np.ones(1, dtype=float)
                else:
                    update = self.hitTimesThisRestart.astype(float)
                    zero = update == 0
                    update[zero] = -(
                        1.0 - float(opt.archiving.targetNewNicheFr)
                    ) / (self.size - 1)
                self.normTabDis *= np.exp(learning * update)
            self.normTabDis = np.maximum(self.normTabDis, _EPS)
        return bool(is_new)

    def stats(self) -> dict[str, int | float]:
        return {
            "mrmc_archive_size": self.size,
            "mrmc_new_peaks": self.new_peaks,
            "mrmc_repeated_peaks": self.repeated_peaks,
            "mrmc_hill_valley_evals": self.hill_valley_evals,
            "mrmc_evidence_eligible": int(self.evidence_eligible),
        }


class _RestartState:
    """Independent restart state expected by the CMSA sampling kernel."""

    def __init__(self, process: Any, opt: Any, problem: Any) -> None:
        self.stagSize = int(
            opt.stopCr.stagPar[0]
            + opt.stopCr.stagPar[2] * problem.dim / process.subpopSize
        )
        self.tolHistSize = int(10 + 30.0 * problem.dim / process.subpopSize)
        self.usedEvalEvolve = 0
        self.usedEvalMerge = 0
        self.usedEvalChangeDetect = 0
        self.iterNo = 0
        self.terminationFlag = 0
        self.subpopTermFlag = 0
        self.subpopBestVal = np.inf
        self.recIniR0: float | None = None
        self.source = ""


class ModularRMCCMSA(BaseAlgorithm):
    """Independent modular RMC architecture using CMSA as local optimizer."""

    @property
    def name(self) -> str:
        return "Modular-RMC-CMSA"

    @staticmethod
    def _configure_repulsion(
        subpop: Any, archive: _PeakArchive, opt: Any, enabled: bool
    ) -> None:
        if not enabled or archive.size == 0:
            subpop.tabooRegion.center = np.empty((0, subpop.center.size), dtype=float)
            subpop.tabooRegion.normTabDis = np.empty(0, dtype=float)
            subpop.tabooRegion.criticality = np.empty(0, dtype=float)
            subpop.tabooRegion.criticInd = np.empty(0, dtype=int)
            return

        eligible = archive.value < float(subpop.bestVal)
        center = archive.solution[eligible]
        radius = archive.normTabDis[eligible]
        if not len(center):
            subpop.tabooRegion.center = np.empty((0, subpop.center.size), dtype=float)
            subpop.tabooRegion.normTabDis = np.empty(0, dtype=float)
            subpop.tabooRegion.criticality = np.empty(0, dtype=float)
            subpop.tabooRegion.criticInd = np.empty(0, dtype=int)
            return

        distance = np.asarray(
            [
                float(subpop.calc_norm_dis(point, subpop.center, "Mahalanobis"))
                for point in center
            ],
            dtype=float,
        )
        criticality = ndtr(distance + radius) - ndtr(distance - radius)
        order = np.argsort(-criticality)
        active = order[criticality[order] > float(opt.niching.criticTabooThresh)]
        subpop.tabooRegion.center = np.asarray(center, dtype=float)
        subpop.tabooRegion.normTabDis = np.asarray(radius, dtype=float)
        subpop.tabooRegion.criticality = np.asarray(criticality, dtype=float)
        subpop.tabooRegion.criticInd = np.asarray(active, dtype=int)

    @staticmethod
    def _terminated(subpop: Any, restart: _RestartState, opt: Any) -> bool:
        stretch = np.asarray(subpop.mutProfile.stretch, dtype=float)
        condition = (
            np.max(stretch) / max(np.min(stretch), np.finfo(float).tiny)
        ) ** 2
        if condition > float(opt.stopCr.maxCondC):
            return True
        if np.max(stretch) * float(subpop.mutProfile.smean) < float(opt.stopCr.tolX):
            return True
        if subpop.iterNo >= restart.tolHistSize:
            history = subpop.bestValNonEliteHist[-restart.tolHistSize :]
            if len(history) == restart.tolHistSize and np.ptp(history) < float(
                opt.stopCr.tolHistFun
            ):
                return True
        if subpop.iterNo >= restart.stagSize and len(subpop.bestValNonEliteHist) >= 40:
            history = subpop.bestValNonEliteHist
            if np.median(history[-20:]) >= np.median(history[:20]):
                return True
        return False

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
        allocator = _RelationalMemoryAllocator(
            control, np.random.default_rng(guide_seed)
        )

        with _baseline_import_path():
            from OptimOption import OptimOption
            from OptimProcess import OptimProcess
            from Subpopulation import SubpopulationCMSA

            adapted = _RSCProblemAdapter(problem)
            opt = OptimOption(adapted)
            process = OptimProcess(opt, adapted)
            archive = _PeakArchive(adapted)
            started = time.monotonic()
            equivalence_stops = 0
            completed_restarts = 0

            while int(adapted.numCallF) + int(process.subpopSize) <= int(
                adapted.maxEval
            ):
                restart = _RestartState(process, opt, adapted)
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
                    # Keep the published CMSA evolution and its fixed stopping
                    # interface intact while retaining ownership of allocation,
                    # memory, archiving, and the outer restart lifecycle here.
                    subpop.update_taboo_region(restart, archive, process, opt, adapted)
                    if control == "no_repulsion":
                        self._configure_repulsion(subpop, archive, opt, False)
                    subpop.update_merge_check(restart, archive, opt, adapted)
                    subpop.evolve(restart, archive, process, opt, adapted)
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
                restart.terminationFlag = 1
                archive.update(subpop, restart, process, opt, adapted)
                if archive.evidence_eligible:
                    allocator.record_terminal(np.asarray(subpop.center), adapted)
                process.update(restart, archive, opt, adapted)
                completed_restarts += 1

                if time.monotonic() - started > float(
                    self.params.get("walltime_limit_sec", np.inf)
                ):
                    break

            self.mrmc_stats = allocator.stats() | archive.stats() | {
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
