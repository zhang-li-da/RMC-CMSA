"""Final four-component relational-memory CMSA.

The algorithm separates trajectory-domain evidence from the competitive peak
archive.  The working-domain memory guides subsequent restart allocation, but
only evidence satisfying the current competitive-value condition is promoted
to the peak archive and returned as a solution.  The two terminal channels are
jointly bounded and directly observable.  Four matched controls isolate the
trajectory-evidence, cumulative-memory, relation-proposal, and finite-pool
maximin components.
Published CMSA, archive, and hill-valley settings remain frozen and are not
reclassified as parameters of the proposed architecture.

The sparse failure-isolation gate closes the entire terminal model: after the
no-growth horizon it clears both terminal channels and blocks further records
until working-domain growth reopens the sparse epoch.  Both terminal channels
share a fixed ``D^2`` resource bound, so removing the expiry gate does not also
introduce unbounded storage.  Competitive repeated-domain terminals persist
across archive revisions; other realized outcomes remain sparse-epoch evidence.

Trajectory samples admitted after spatial prefiltering and hill-valley
validation are stored in a separate working-domain memory.  A per-call target
scale limits the number of candidates validated in one enrichment call; the
capacity scale bounds the cumulative working memory.  A competitive sample is
promoted to the peak archive; a noncompetitive sample remains only a coverage
representative and cannot affect peak identity, taboo regions, or the reported
solution set.

The default dimension-derived rules are ``D^2`` candidates per proposal
source, a
``D``-restart sparse-evidence horizon, ``ceil(sqrt(D))`` retained samples per
generation, a ``3D`` evidence capacity, ``ceil(sqrt(D))`` validated trajectory
representatives, the unscaled trajectory-separation distance, and relation
proposals at one nearest-representative edge length.  Candidate ranking uses
only normalized coverage margins, and every restart passes the inherited
CMSA initial radius unchanged.  Raw-distance ranking and restart-radius
adaptation are not part of this implementation.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import numpy as np

import modular_rmc_cmsa_v12 as _v12


_EPS = np.finfo(float).eps
_DEFAULT_CANDIDATE_POOL_SCALE = 1.0
_DEFAULT_SPARSE_GATE_HORIZON_SCALE = 1.0
_DEFAULT_TRAJECTORY_TARGET_SCALE = 1.0
_DEFAULT_TRAJECTORY_KEEP_SCALE = 1.0
_DEFAULT_TRAJECTORY_CAPACITY_SCALE = 1.0
_DEFAULT_TRAJECTORY_SEPARATION_SCALE = 1.0
_DEFAULT_RELATION_SHELL_SCALE = 1.0
_CANDIDATE_RULES = frozenset({"d", "dlogd", "d2", "2d2"})
_CONTROLS = frozenset(
    {
        "full",
        "no_trajectory_evidence",
        "no_terminal_memory",
        "no_relations",
        "no_maximin",
    }
)


def _positive_scale(value: Any, name: str) -> float:
    """Return a finite positive scale or fail before a search is started."""
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return scale


def _sensitivity_scales(
    params: Mapping[str, Any],
) -> tuple[float, float, float, float, float, float, float]:
    """Read all numerical scales introduced by the MRMC architecture."""
    return (
        _positive_scale(
            params.get("candidate_pool_scale", _DEFAULT_CANDIDATE_POOL_SCALE),
            "candidate_pool_scale",
        ),
        _positive_scale(
            params.get(
                "sparse_gate_horizon_scale",
                _DEFAULT_SPARSE_GATE_HORIZON_SCALE,
            ),
            "sparse_gate_horizon_scale",
        ),
        _positive_scale(
            params.get(
                "trajectory_target_scale", _DEFAULT_TRAJECTORY_TARGET_SCALE
            ),
            "trajectory_target_scale",
        ),
        _positive_scale(
            params.get("trajectory_keep_scale", _DEFAULT_TRAJECTORY_KEEP_SCALE),
            "trajectory_keep_scale",
        ),
        _positive_scale(
            params.get(
                "trajectory_capacity_scale",
                _DEFAULT_TRAJECTORY_CAPACITY_SCALE,
            ),
            "trajectory_capacity_scale",
        ),
        _positive_scale(
            params.get(
                "trajectory_separation_scale",
                _DEFAULT_TRAJECTORY_SEPARATION_SCALE,
            ),
            "trajectory_separation_scale",
        ),
        _positive_scale(
            params.get("relation_shell_scale", _DEFAULT_RELATION_SHELL_SCALE),
            "relation_shell_scale",
        ),
    )


class _TrustGatedAllocator(_v12._TrustGatedAllocator):
    """Relational allocator with factorized controls and working-domain edges."""

    def __init__(
        self,
        control: str,
        rng: np.random.Generator,
        normalized_projector: Callable[[np.ndarray, Any], np.ndarray] | None = None,
        *,
        candidate_pool_scale: float = _DEFAULT_CANDIDATE_POOL_SCALE,
        candidate_pool_rule: str | None = None,
        sparse_gate_horizon_scale: float = _DEFAULT_SPARSE_GATE_HORIZON_SCALE,
        relation_shell_scale: float = _DEFAULT_RELATION_SHELL_SCALE,
    ) -> None:
        if control not in _CONTROLS:
            choices = ", ".join(sorted(_CONTROLS))
            raise ValueError(
                f"unknown modular RMC control {control!r}; expected {choices}"
            )
        super().__init__(control, rng, normalized_projector)
        self.control = control
        self.candidate_pool_scale = _positive_scale(
            candidate_pool_scale, "candidate_pool_scale"
        )
        if candidate_pool_rule is not None and candidate_pool_rule not in _CANDIDATE_RULES:
            choices = ", ".join(sorted(_CANDIDATE_RULES))
            raise ValueError(
                f"unknown candidate_pool_rule {candidate_pool_rule!r}; expected {choices}"
            )
        self.candidate_pool_rule = candidate_pool_rule
        self.sparse_gate_horizon_scale = _positive_scale(
            sparse_gate_horizon_scale, "sparse_gate_horizon_scale"
        )
        self.relation_shell_scale = _positive_scale(
            relation_shell_scale, "relation_shell_scale"
        )
        self.last_candidate_count = 0
        self.last_uniform_candidate_count = 0
        self.last_relation_candidate_count = 0
        self.last_gate_horizon = 0
        self.mature_uniform_uses = 0
        self.expired_persistent_records = 0
        self.expired_sparse_records = 0
        self.gate_record_blocks = 0
        self.gate_reopens = 0
        self.last_terminal_capacity = 0
        self.terminal_capacity_blocks = 0
        self.terminal_memory_peak = 0
        self.persistent_memory_peak = 0
        self.sparse_memory_peak = 0
        self.persistent_guidance_uses = 0
        self.sparse_guidance_uses = 0
        self.working_guidance_uses = 0
        self.mature_outcomes = 0
        self.working_domain_outcomes = 0
        self.maximin_selections = 0
        self.random_selections = 0
        self.normalized_ranking_uses = 0
        self.relation_working_uses = 0

    def _candidate_count(self, problem: Any) -> int:
        dim = int(problem.dim)
        if self.candidate_pool_rule == "d":
            count = dim
        elif self.candidate_pool_rule == "dlogd":
            count = int(np.ceil(dim * np.log2(dim + 1.0)))
        elif self.candidate_pool_rule == "d2":
            count = dim * dim
        elif self.candidate_pool_rule == "2d2":
            count = 2 * dim * dim
        else:
            count = int(np.ceil(self.candidate_pool_scale * float(dim * dim)))
        count = max(1, count)
        self.last_candidate_count = count
        return count

    def _gate_horizon(self, problem: Any) -> int:
        horizon = max(
            1,
            int(np.ceil(self.sparse_gate_horizon_scale * float(int(problem.dim)))),
        )
        self.last_gate_horizon = horizon
        return horizon

    def _terminal_capacity(self, problem: Any) -> int:
        capacity = max(1, int(problem.dim) * int(problem.dim))
        self.last_terminal_capacity = capacity
        return capacity

    def _record_terminal(
        self, point: np.ndarray, problem: Any, *, persistent: bool
    ) -> None:
        memories = (
            self.persistent_terminal_memory,
            self.sparse_coverage_memory,
        )
        if any(
            np.array_equal(stored, point)
            for memory in memories
            for stored in memory
        ):
            return
        if sum(len(memory) for memory in memories) >= self._terminal_capacity(problem):
            self.terminal_capacity_blocks += 1
            return
        if persistent:
            self.persistent_terminal_memory.append(point)
            self.persistent_records += 1
        else:
            self.sparse_coverage_memory.append(point)
            self.sparse_coverage_records += 1
        self.persistent_memory_peak = max(
            self.persistent_memory_peak, len(self.persistent_terminal_memory)
        )
        self.sparse_memory_peak = max(
            self.sparse_memory_peak, len(self.sparse_coverage_memory)
        )
        self.terminal_memory_peak = max(
            self.terminal_memory_peak,
            len(self.persistent_terminal_memory) + len(self.sparse_coverage_memory),
        )

    def record_outcome(
        self,
        center: np.ndarray,
        problem: Any,
        *,
        archive_grew: bool,
        repeated_peak: bool,
        archive_size: int,
        working_domain_grew: bool = False,
    ) -> None:
        """Apply the version-12 state transition with a scaled gate horizon."""
        point = self._project_normalized(
            self._normalize(center, problem), problem, terminal=True
        )[0]

        domain_grew = bool(archive_grew or working_domain_grew)
        if working_domain_grew:
            self.working_domain_outcomes += 1

        if archive_size >= 2:
            self.mature_outcomes += 1
            if not self.sparse_coverage_enabled:
                self.gate_reopens += 1
            self.sparse_coverage_enabled = True
            if repeated_peak and self.control != "no_terminal_memory":
                self._record_terminal(point, problem, persistent=True)
            self.no_growth_streak = 0
            self.sparse_coverage_memory.clear()
            return

        if domain_grew:
            self.productive_outcomes += 1
            self.no_growth_streak = 0
            if not self.sparse_coverage_enabled:
                self.gate_reopens += 1
            self.sparse_coverage_enabled = True
        else:
            self.no_growth_streak += 1

        if self.sparse_coverage_enabled:
            if repeated_peak and self.control != "no_terminal_memory":
                self._record_terminal(point, problem, persistent=True)
            elif not repeated_peak and self.control != "no_terminal_memory":
                self._record_terminal(point, problem, persistent=False)
        elif self.control != "no_terminal_memory":
            self.gate_record_blocks += 1

        horizon = self._gate_horizon(problem)
        if self.no_growth_streak >= horizon:
            if self.sparse_coverage_enabled:
                self.evidence_gate_closures += 1
            self.sparse_coverage_enabled = False
            self.expired_persistent_records += len(
                self.persistent_terminal_memory
            )
            self.expired_sparse_records += len(self.sparse_coverage_memory)
            self.persistent_terminal_memory.clear()
            self.sparse_coverage_memory.clear()

    def _relation_candidates(
        self, points: np.ndarray, count: int, dim: int
    ) -> np.ndarray | None:
        edge, length = self._nearest_relations(points)
        if not len(edge):
            return None
        self.relation_builds += 1
        picked = self.rng.integers(len(edge), size=count)
        side = self.rng.integers(2, size=count)
        origin_index = edge[picked, side]
        origin = points[origin_index]
        direction = self.rng.normal(size=(count, dim))
        norm = np.linalg.norm(direction, axis=1)
        zero = norm <= _EPS
        while np.any(zero):
            direction[zero] = self.rng.normal(size=(int(np.sum(zero)), dim))
            norm[zero] = np.linalg.norm(direction[zero], axis=1)
            zero = norm <= _EPS
        direction /= norm[:, None]
        shell = self.relation_shell_scale * length[picked, None]
        return np.clip(origin + shell * direction, 0.0, 1.0)

    def _bootstrap_scores(
        self,
        candidates: np.ndarray,
        archive_points: np.ndarray,
        problem: Any,
        working_points: np.ndarray | None = None,
    ) -> np.ndarray:
        references: list[np.ndarray] = []
        if len(archive_points):
            references.append(archive_points)
        if (
            self.control != "no_terminal_memory"
            and self.sparse_coverage_enabled
            and self.persistent_terminal_memory
        ):
            self.persistent_guidance_uses += 1
            references.append(
                np.asarray(self.persistent_terminal_memory, dtype=float)
            )
        if (
            self.control != "no_terminal_memory"
            and self.sparse_coverage_enabled
            and self.sparse_coverage_memory
        ):
            self.sparse_guidance_uses += 1
            references.append(np.asarray(self.sparse_coverage_memory, dtype=float))

        if not references:
            scores = np.ones(len(candidates), dtype=float)
        else:
            reference = np.concatenate(references, axis=0)
            distance = np.linalg.norm(
                candidates[:, None, :] - reference[None, :, :], axis=2
            )
            scores = np.min(distance, axis=1)
        if working_points is not None and len(working_points):
            self.working_guidance_uses += 1
            distance = np.linalg.norm(
                candidates[:, None, :] - working_points[None, :, :], axis=2
            )
            scores = np.minimum(scores, np.min(distance, axis=1))
        return scores

    def _working_clearance(
        self, candidates: np.ndarray, working_points: np.ndarray | None
    ) -> np.ndarray | None:
        if working_points is None or not len(working_points):
            return None
        self.working_guidance_uses += 1
        distance = np.linalg.norm(
            candidates[:, None, :] - working_points[None, :, :], axis=2
        )
        return np.min(distance, axis=1)

    def _terminal_margin(
        self, candidates: np.ndarray, process: Any
    ) -> np.ndarray | None:
        """Return clearance from cumulative terminal evidence on one fixed scale."""
        if self.control == "no_terminal_memory" or not self.sparse_coverage_enabled:
            return None
        references: list[np.ndarray] = []
        if self.persistent_terminal_memory:
            self.persistent_guidance_uses += 1
            references.append(np.asarray(self.persistent_terminal_memory, dtype=float))
        if self.sparse_coverage_memory:
            self.sparse_guidance_uses += 1
            references.append(np.asarray(self.sparse_coverage_memory, dtype=float))
        if not references:
            return None
        terminal = np.concatenate(references, axis=0)
        distance = np.linalg.norm(
            candidates[:, None, :] - terminal[None, :, :], axis=2
        )
        fixed_scale = max(
            float(getattr(process, "defNormTabDis", process.iniR0)), _EPS
        )
        return np.min(distance, axis=1) / fixed_scale

    @staticmethod
    def _normalized_archive_margin(
        candidates: np.ndarray,
        archive: Any,
        archive_points: np.ndarray,
    ) -> np.ndarray:
        """Return each candidate's minimum radius-normalized clearance."""
        if not len(archive_points):
            return np.ones(len(candidates), dtype=float)
        distance = np.linalg.norm(
            candidates[:, None, :] - archive_points[None, :, :], axis=2
        )
        radius = np.maximum(
            np.asarray(archive.normTabDis, dtype=float).reshape(-1), _EPS
        )
        return np.min(distance / radius[None, :], axis=1)

    def propose(
        self,
        archive: Any,
        process: Any,
        opt: Any,
        problem: Any,
        working_points: np.ndarray | None = None,
    ) -> _v12._v3._Allocation:
        """Apply the version-12 proposal rule to a scaled finite pool."""
        self.calls += 1
        dim = int(problem.dim)
        count = self._candidate_count(problem)
        points = self._archive_points(archive, problem)
        working_in_view = int(getattr(archive, "working_size", 0)) > 0

        uniform_candidates = self.rng.random((count, dim))
        self.last_uniform_candidate_count = count
        self.last_relation_candidate_count = 0
        candidates = uniform_candidates
        source = "bootstrap"
        mature = bool(dim > 1 and len(points) >= 2)
        if mature:
            source = "mature_uniform"
            if self.control != "no_relations":
                relation_points = points
                if working_points is not None and len(working_points):
                    relation_points = np.vstack((points, working_points))
                    self.relation_working_uses += 1
                elif working_in_view:
                    self.relation_working_uses += 1
                relation_candidates = self._relation_candidates(
                    relation_points, count, dim
                )
                if relation_candidates is not None:
                    candidates = np.vstack((uniform_candidates, relation_candidates))
                    self.last_relation_candidate_count = len(relation_candidates)
                    source = "relation"
            else:
                candidates = np.vstack(
                    (uniform_candidates, self.rng.random((count, dim)))
                )
                self.last_uniform_candidate_count = len(candidates)
        candidates = self._project_normalized(candidates, problem)
        self.last_candidate_count = len(candidates)

        normalized_margin = self._normalized_archive_margin(
            candidates, archive, points
        )
        if mature:
            scores = normalized_margin
            terminal_margin = self._terminal_margin(candidates, process)
            if terminal_margin is not None:
                scores = np.minimum(scores, terminal_margin)
            self.normalized_ranking_uses += 1
        else:
            scores = self._bootstrap_scores(
                candidates, points, problem, working_points
            )

        if self.control == "no_maximin" or np.ptp(scores) <= _EPS:
            selected = int(self.rng.integers(len(candidates)))
            self.random_selections += 1
        else:
            selected = int(np.argmax(scores))
            self.maximin_selections += 1

        r0 = float(process.iniR0)

        if source == "relation":
            self.relation_uses += 1
        elif source == "mature_uniform":
            self.mature_uniform_uses += 1
        else:
            self.bootstrap_uses += 1
        self.candidate_sum += len(candidates)
        self.selected_score_sum += float(scores[selected])
        self.selected_r0_sum += r0
        return _v12._v3._Allocation(
            center=np.asarray(candidates[selected], dtype=float),
            r0=r0,
            source=source,
            score=float(scores[selected]),
        )

    def stats(self) -> dict[str, int | float | str]:
        return super().stats() | {
            "mrmc_candidate_pool_scale": self.candidate_pool_scale,
            "mrmc_candidate_pool_rule": self.candidate_pool_rule or "scaled_d2",
            "mrmc_candidate_pool_size": self.last_candidate_count,
            "mrmc_uniform_candidate_size": self.last_uniform_candidate_count,
            "mrmc_relation_candidate_size": self.last_relation_candidate_count,
            "mrmc_relation_shell_scale": self.relation_shell_scale,
            "mrmc_relation_working_uses": self.relation_working_uses,
            "mrmc_mature_uniform_uses": self.mature_uniform_uses,
            "mrmc_sparse_gate_horizon_scale": self.sparse_gate_horizon_scale,
            "mrmc_sparse_gate_horizon": self.last_gate_horizon,
            "mrmc_sparse_expiry_enabled": 1,
            "mrmc_expired_persistent_records": self.expired_persistent_records,
            "mrmc_expired_sparse_records": self.expired_sparse_records,
            "mrmc_gate_record_blocks": self.gate_record_blocks,
            "mrmc_gate_reopens": self.gate_reopens,
            "mrmc_terminal_capacity": self.last_terminal_capacity,
            "mrmc_terminal_capacity_blocks": self.terminal_capacity_blocks,
            "mrmc_terminal_memory_peak": self.terminal_memory_peak,
            "mrmc_persistent_memory_peak": self.persistent_memory_peak,
            "mrmc_sparse_memory_peak": self.sparse_memory_peak,
            "mrmc_persistent_guidance_uses": self.persistent_guidance_uses,
            "mrmc_sparse_guidance_uses": self.sparse_guidance_uses,
            "mrmc_working_guidance_uses": self.working_guidance_uses,
            "mrmc_mature_outcomes": self.mature_outcomes,
            "mrmc_working_domain_outcomes": self.working_domain_outcomes,
            "mrmc_maximin_selections": self.maximin_selections,
            "mrmc_random_selections": self.random_selections,
            "mrmc_normalized_ranking_uses": self.normalized_ranking_uses,
        }


class _SearchArchiveView:
    """Expose working domains to CMSA without changing the peak archive."""

    def __init__(
        self,
        peak_archive: Any,
        evidence: "_TrajectoryEvidence",
        problem: Any,
        default_radius: float,
    ) -> None:
        self._peak_archive = peak_archive
        self._evidence = evidence
        self._problem = problem
        self._default_radius = max(float(default_radius), _EPS)
        self.refresh()

    def refresh(self) -> None:
        dimension = int(self._problem.dim)
        peak_solution = np.asarray(self._peak_archive.solution, dtype=float).reshape(
            -1, dimension
        )
        peak_value = np.asarray(self._peak_archive.value, dtype=float).reshape(-1)
        peak_radius = np.asarray(
            self._peak_archive.normTabDis, dtype=float
        ).reshape(-1)
        working = self._evidence.working_points
        if len(working):
            low = np.asarray(self._problem.lowBound, dtype=float)
            span = np.maximum(
                np.asarray(self._problem.upBound, dtype=float) - low, _EPS
            )
            working_solution = low + working * span
            working_value = self._evidence.working_values
            self.solution = np.vstack((peak_solution, working_solution))
            self.value = np.concatenate((peak_value, working_value))
            self.normTabDis = np.concatenate(
                (
                    peak_radius,
                    np.full(len(working_solution), self._default_radius),
                )
            )
        else:
            self.solution = peak_solution.copy()
            self.value = peak_value.copy()
            self.normTabDis = peak_radius.copy()
        self.peak_size = int(len(peak_solution))
        self.working_size = int(len(self.solution) - self.peak_size)
        self.size = int(len(self.solution))

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
        index = int(archive_index)
        if index < self.peak_size:
            return self._peak_archive._same_domain(
                x, fx, index, restart, opt, problem, online
            )
        base = self.solution[index]
        base_value = float(self.value[index])
        for _ in range(int(opt.archiving.hillVallBudget)):
            if int(problem.numCallF) >= int(problem.maxEval):
                return True
            fraction = 0.1 + 0.8 * float(np.random.random())
            point = base + fraction * (np.asarray(x, dtype=float) - base)
            value = float(problem.func_eval(point))
            self._peak_archive.hill_valley_evals += 1
            if online:
                restart.usedEvalMerge += 1
            else:
                self._peak_archive.usedEval += 1
            if value > max(float(fx), base_value) + float(
                opt.stopCr.tolHistFun
            ):
                return False
        return True

    @property
    def evidence_eligible(self) -> bool:
        return bool(self._peak_archive.evidence_eligible)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._peak_archive, name)


class _TrajectoryEvidence(_v12._TrajectoryEvidence):
    """Separate working-domain evidence with competitive peak promotion."""

    def __init__(
        self,
        enabled: bool,
        trajectory_target_scale: float = _DEFAULT_TRAJECTORY_TARGET_SCALE,
        trajectory_keep_scale: float = _DEFAULT_TRAJECTORY_KEEP_SCALE,
        trajectory_capacity_scale: float = _DEFAULT_TRAJECTORY_CAPACITY_SCALE,
        trajectory_separation_scale: float = _DEFAULT_TRAJECTORY_SEPARATION_SCALE,
        separation_enabled: bool = True,
    ) -> None:
        super().__init__(enabled)
        self.trajectory_target_scale = _positive_scale(
            trajectory_target_scale, "trajectory_target_scale"
        )
        self.trajectory_keep_scale = _positive_scale(
            trajectory_keep_scale, "trajectory_keep_scale"
        )
        self.trajectory_capacity_scale = _positive_scale(
            trajectory_capacity_scale, "trajectory_capacity_scale"
        )
        self.trajectory_separation_scale = _positive_scale(
            trajectory_separation_scale, "trajectory_separation_scale"
        )
        self.separation_enabled = bool(separation_enabled)
        self.last_target_size = 0
        self.last_keep_count = 0
        self.last_capacity = 0
        self.last_separation = 0.0
        self._working_points: list[np.ndarray] = []
        self._working_values: list[float] = []
        self.working_additions = 0
        self.working_replacements = 0
        self.working_duplicate_blocks = 0
        self.working_memory_peak = 0
        self.working_promotions = 0
        self.working_noncompetitive = 0
        self.working_empty_archive_additions = 0
        self.working_same_domain_blocks = 0

    @property
    def working_points(self) -> np.ndarray:
        if not self._working_points:
            return np.empty((0, 0), dtype=float)
        return np.asarray(self._working_points, dtype=float)

    @property
    def working_values(self) -> np.ndarray:
        return np.asarray(self._working_values, dtype=float)

    def _target_size(self, problem: Any) -> int:
        target = max(
            3,
            int(
                np.ceil(
                    self.trajectory_target_scale
                    * np.sqrt(float(int(problem.dim)))
                )
            ),
        )
        self.last_target_size = target
        return target

    def _keep_count(self, problem: Any) -> int:
        count = max(
            1,
            int(
                np.ceil(
                    self.trajectory_keep_scale
                    * np.sqrt(float(int(problem.dim)))
                )
            ),
        )
        self.last_keep_count = count
        return count

    def _capacity(self, problem: Any) -> int:
        capacity = max(
            8,
            int(
                np.ceil(
                    self.trajectory_capacity_scale
                    * 3.0
                    * float(int(problem.dim))
                )
            ),
        )
        self.last_capacity = capacity
        return capacity

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
        keep = finite[np.argsort(values[finite])[: self._keep_count(problem)]]
        self._points.extend(points[int(index)].copy() for index in keep)
        self._values.extend(float(values[int(index)]) for index in keep)
        self.observed += len(keep)

        capacity = self._capacity(problem)
        if len(self._values) > capacity:
            order = np.argsort(np.asarray(self._values, dtype=float))[:capacity]
            self._points = [self._points[int(index)] for index in order]
            self._values = [self._values[int(index)] for index in order]

    def _separation_scale(self, archive: Any, problem: Any) -> float:
        base = float(super()._separation_scale(archive, problem))
        separation = (
            self.trajectory_separation_scale * base
            if self.separation_enabled
            else 0.0
        )
        self.last_separation = separation
        return separation

    def _append_working(
        self,
        point: np.ndarray,
        value: float,
        archive: Any,
        problem: Any,
    ) -> bool:
        normalized = self._normalized(point.reshape(1, -1), problem)[0]
        separation = self._separation_scale(archive, problem)
        if self._working_points and separation > 0.0:
            distance = np.linalg.norm(
                np.asarray(self._working_points, dtype=float)
                - normalized.reshape(1, -1),
                axis=1,
            )
            if float(np.min(distance)) <= separation:
                self.working_duplicate_blocks += 1
                return False

        capacity = self._capacity(problem)
        if len(self._working_values) < capacity:
            self._working_points.append(normalized)
            self._working_values.append(float(value))
            self.working_additions += 1
            self.working_memory_peak = max(
                self.working_memory_peak, len(self._working_values)
            )
            return True

        worst = int(np.argmax(np.asarray(self._working_values, dtype=float)))
        if float(value) >= self._working_values[worst]:
            self.working_duplicate_blocks += 1
            return False
        self._working_points[worst] = normalized
        self._working_values[worst] = float(value)
        self.working_replacements += 1
        return True

    @staticmethod
    def _competitive_best(archive: Any, process: Any) -> float:
        values: list[float] = []
        process_best = np.asarray(
            getattr(process, "bestValTillRestart", np.inf), dtype=float
        ).reshape(-1)
        values.extend(float(value) for value in process_best if np.isfinite(value))
        if int(getattr(archive, "size", 0)):
            archive_values = np.asarray(archive.value, dtype=float).reshape(-1)
            values.extend(
                float(value) for value in archive_values if np.isfinite(value)
            )
        return min(values) if values else np.inf

    def enrich(
        self,
        archive: Any,
        restart: Any,
        process: Any,
        opt: Any,
        problem: Any,
    ) -> int:
        """Validate domains, promote competitive points, and retain the rest."""
        if not self.enabled:
            return 0
        candidates = self._diverse_candidates(archive, problem)
        if not candidates:
            return 0

        self.validation_calls += 1
        promoted = 0
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
                limit = min(int(archive.size), int(opt.archiving.neighborSize))
                # The fixed hill--valley interface is queried in archive order.
                # Restart allocation does not use an additional distance-ranked
                # neighbour list; the benchmark archive interface remains the
                # sole equivalence test.
                for index in range(limit):
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
            if not is_new:
                self.working_same_domain_blocks += 1
                continue

            best = self._competitive_best(archive, process)
            competitive = np.isfinite(best) and float(value) <= best + float(
                opt.archiving.tolFunArch
            )
            if competitive:
                archive._append(point, value, process)
                promoted += 1
                self.working_promotions += 1
                continue

            self.working_noncompetitive += 1
            if int(archive.size) == 0:
                self.working_empty_archive_additions += 1
            self._append_working(point, value, archive, problem)

        self.added_domains += promoted
        return promoted

    def stats(self) -> dict[str, int | float]:
        return super().stats() | {
            "mrmc_evidence_target_scale": self.trajectory_target_scale,
            "mrmc_evidence_target_size": self.last_target_size,
            "mrmc_evidence_keep_scale": self.trajectory_keep_scale,
            "mrmc_evidence_keep_count": self.last_keep_count,
            "mrmc_evidence_capacity_scale": self.trajectory_capacity_scale,
            "mrmc_evidence_capacity": self.last_capacity,
            "mrmc_evidence_separation_scale": self.trajectory_separation_scale,
            "mrmc_evidence_separation": self.last_separation,
            "mrmc_evidence_prefilter_enabled": int(self.separation_enabled),
            "mrmc_working_memory_size": len(self._working_values),
            "mrmc_working_memory_peak": self.working_memory_peak,
            "mrmc_working_additions": self.working_additions,
            "mrmc_working_replacements": self.working_replacements,
            "mrmc_working_duplicate_blocks": self.working_duplicate_blocks,
            "mrmc_working_promotions": self.working_promotions,
            "mrmc_working_noncompetitive": self.working_noncompetitive,
            "mrmc_working_empty_archive_additions": (
                self.working_empty_archive_additions
            ),
            "mrmc_working_same_domain_blocks": self.working_same_domain_blocks,
        }


def build_search_components(
    params: Mapping[str, Any],
    *,
    normalized_projector: Callable[[np.ndarray, Any], np.ndarray] | None = None,
    evidence_type: type[_TrajectoryEvidence] = _TrajectoryEvidence,
    evidence_kwargs: Mapping[str, Any] | None = None,
) -> tuple[_TrustGatedAllocator, _TrajectoryEvidence]:
    """Build the audited four-component search state for one run.

    Domain adapters may supply a feasibility projector and an evidence subclass,
    while the control validation, sensitivity scales, and guide-seed derivation
    remain identical to the benchmark implementation.
    """
    (
        candidate_scale,
        gate_scale,
        target_scale,
        keep_scale,
        capacity_scale,
        separation_scale,
        relation_shell_scale,
    ) = _sensitivity_scales(params)
    control = str(params.get("causal_control", "full"))
    if control not in _CONTROLS:
        choices = ", ".join(sorted(_CONTROLS))
        raise ValueError(
            f"unknown final RMC-CMSA control {control!r}; expected {choices}"
        )
    seed = params.get("seed")
    guide_seed = None if seed is None else (int(seed) ^ 0x4D524D43) % (2**32)
    allocator = _TrustGatedAllocator(
        control,
        np.random.default_rng(guide_seed),
        normalized_projector=normalized_projector,
        candidate_pool_scale=candidate_scale,
        candidate_pool_rule=(
            None
            if params.get("candidate_pool_rule") in (None, "")
            else str(params["candidate_pool_rule"])
        ),
        sparse_gate_horizon_scale=gate_scale,
        relation_shell_scale=relation_shell_scale,
    )
    extra = dict(evidence_kwargs or {})
    reserved = {
        "enabled",
        "trajectory_target_scale",
        "trajectory_keep_scale",
        "trajectory_capacity_scale",
        "trajectory_separation_scale",
        "separation_enabled",
    }
    overlap = reserved.intersection(extra)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"evidence_kwargs cannot override audited settings: {names}")
    evidence = evidence_type(
        control != "no_trajectory_evidence",
        trajectory_target_scale=target_scale,
        trajectory_keep_scale=keep_scale,
        trajectory_capacity_scale=capacity_scale,
        trajectory_separation_scale=separation_scale,
        separation_enabled=True,
        **extra,
    )
    if not isinstance(evidence, _TrajectoryEvidence):
        raise TypeError("evidence_type must construct a final _TrajectoryEvidence")
    return allocator, evidence


class ModularRMCCMSAFinal(_v12.ModularRMCCMSA):
    """Four-component RMC-CMSA with matched ablation controls."""

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        control = str(kwargs.get("causal_control", "full"))
        if control not in _CONTROLS:
            choices = ", ".join(sorted(_CONTROLS))
            raise ValueError(
                f"unknown final RMC-CMSA control {control!r}; expected {choices}"
            )
        super().__init__(seed=seed, **kwargs)

    @property
    def name(self) -> str:
        return "RMC-CMSA"

    def _run(self, problem: Any) -> np.ndarray:
        seed = self.params.get("seed")
        if seed is not None:
            np.random.seed(int(seed) % (2**32 - 1))
        if not _v12.RSC_DIR.exists():
            raise RuntimeError(f"CMSA source directory not found: {_v12.RSC_DIR}")
        if not hasattr(np, "int"):
            np.int = int  # type: ignore[attr-defined]

        allocator, evidence = build_search_components(self.params)

        with _v12._baseline_import_path():
            from OptimOption import OptimOption
            from OptimProcess import OptimProcess
            from Subpopulation import SubpopulationCMSA

            adapted = _v12._RSCProblemAdapter(problem)
            opt = OptimOption(adapted)
            process = OptimProcess(opt, adapted)
            archive = _v12._v3._PeakArchive(adapted)
            started = time.monotonic()
            equivalence_stops = 0
            completed_restarts = 0
            working_search_view_restarts = 0
            working_search_view_peak = 0

            while int(adapted.numCallF) + int(process.subpopSize) <= int(
                adapted.maxEval
            ):
                restart = _v12._v3._RestartState(process, opt, adapted)
                search_archive = _SearchArchiveView(
                    archive,
                    evidence,
                    adapted,
                    process.defNormTabDis,
                )
                if search_archive.working_size:
                    working_search_view_restarts += 1
                    working_search_view_peak = max(
                        working_search_view_peak,
                        search_archive.working_size,
                    )
                allocation = allocator.propose(
                    search_archive,
                    process,
                    opt,
                    adapted,
                )
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
                    subpop.update_taboo_region(
                        restart, search_archive, process, opt, adapted
                    )
                    subpop.update_merge_check(
                        restart, search_archive, opt, adapted
                    )
                    subpop.evolve(
                        restart, search_archive, process, opt, adapted
                    )
                    evidence.observe_generation(subpop, adapted)
                    subpop.update_term_flag(
                        restart, search_archive, process, opt, adapted
                    )
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
                working_changes_before = (
                    evidence.working_additions + evidence.working_replacements
                )
                archive_grew = archive.update(
                    subpop, restart, process, opt, adapted
                )
                evidence_promoted = evidence.enrich(
                    archive, restart, process, opt, adapted
                )
                archive_grew = bool(archive_grew or evidence_promoted)
                working_changes = (
                    evidence.working_additions
                    + evidence.working_replacements
                    - working_changes_before
                )
                search_archive.refresh()
                allocator.record_outcome(
                    np.asarray(subpop.center),
                    adapted,
                    archive_grew=archive_grew,
                    repeated_peak=bool(archive.evidence_eligible),
                    archive_size=int(search_archive.size),
                    working_domain_grew=working_changes > 0,
                )
                process.update(restart, search_archive, opt, adapted)
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
            self.mrmc_stats = (
                allocator.stats()
                | archive.stats()
                | evidence.stats()
                | {
                    "mrmc_completed_restarts": completed_restarts,
                    "mrmc_early_stops": equivalence_stops,
                    "mrmc_used_eval": int(adapted.numCallF),
                    "mrmc_working_search_view_restarts": (
                        working_search_view_restarts
                    ),
                    "mrmc_working_search_view_peak": working_search_view_peak,
                }
            )
            self.vlarta_stats = self.mrmc_stats
            if archive.size:
                return np.asarray(archive.solution, dtype=float).reshape(
                    -1, problem.dim
                )
            return np.empty((0, problem.dim), dtype=float)


class ModularRMCCMSA(ModularRMCCMSAFinal):
    """Compatibility name for existing experiment scripts."""


class TAGDE(ModularRMCCMSAFinal):
    """Compatibility alias for the unified experiment runners."""


__all__ = [
    "ModularRMCCMSA",
    "ModularRMCCMSAFinal",
    "TAGDE",
    "_CONTROLS",
    "_SearchArchiveView",
    "_TrajectoryEvidence",
    "_TrustGatedAllocator",
    "build_search_components",
]
