"""Unified base class for all multimodal optimization algorithms."""

import numpy as np
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CMMOP.ProblemMM import ProblemMM
from CMMOP.UtilityMethod import UtilityMethod


@dataclass
class RunResult:
    """Stores all information from a single optimization run."""
    # Problem parameters
    pid: int
    pin: int
    dim: int
    n_minima: int
    max_eval: int

    # Final solution set (shape: [n_sol, dim+1], last col = fitness)
    solutions: np.ndarray = field(default_factory=lambda: np.empty((0,)))

    # Performance metrics
    rpr: float = 0.0
    f1: float = 0.0
    score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    # Runtime
    elapsed: float = 0.0
    used_eval: int = 0

    # Convergence history: list of (eval_count, best_f, mean_f, rpr_so_far)
    history: list = field(default_factory=list)


class BaseAlgorithm(ABC):
    """
    Abstract base class every algorithm must inherit from.

    Subclasses implement `_run(problem)` and return a solution matrix
    with shape (n_sol, dim), where each row is a reported global optimum.
    The number of reported solutions should ideally equal problem.n_minima.
    """

    FTOL = np.array([1e-5, 1.0])  # [tight, loose] tolerance

    def __init__(self, **kwargs):
        self.params = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'NCDE', 'LIPS'."""

    @abstractmethod
    def _run(self, problem: ProblemMM) -> np.ndarray:
        """
        Run the algorithm on *problem*.

        Returns
        -------
        np.ndarray, shape (n_sol, dim)
            Reported solutions (candidate global optima). Fitness will be
            re-evaluated here for fair comparison.
        """

    def run(self, pid: int, pin: int, dim: int) -> RunResult:
        """Public entry point: create problem, run, evaluate, return RunResult."""
        problem = ProblemMM(pid, pin, dim)
        problem.form()

        t0 = time.perf_counter()
        sol = self._run(problem)
        elapsed = time.perf_counter() - t0

        sol = np.atleast_2d(sol)
        # clip to bounds
        sol = np.clip(sol, problem.low_bound, problem.up_bound)

        # Final reporting is re-evaluated for metric consistency, but it is
        # evaluator-side accounting rather than search budget consumed by the
        # algorithm. Store the search count before this post-run evaluation.
        search_used_eval = problem.used_eval
        f = problem.func_eval(sol)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rpr, i_pr = UtilityMethod.calc_robust_peak_ratio(
                sol, f, problem.minima.X, problem.minima.f, self.FTOL
            )

        n_gm = problem.n_minima
        n_sol = sol.shape[0]
        precision = rpr * n_gm / n_sol if n_sol > 0 else 0.0
        recall = rpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        score = (rpr + f1) / 2.0

        solutions_with_f = np.column_stack([sol, f])

        result = RunResult(
            pid=pid, pin=pin, dim=dim,
            n_minima=n_gm,
            max_eval=problem.max_eval,
            solutions=solutions_with_f,
            rpr=rpr, f1=f1, score=score,
            precision=precision, recall=recall,
            elapsed=elapsed,
            used_eval=search_used_eval,
            history=list(getattr(self, "history", [])),
        )
        return result

    # ------------------------------------------------------------------
    # Helpers shared by multiple algorithms
    # ------------------------------------------------------------------

    @staticmethod
    def random_init(n: int, lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Latin-Hypercube-like uniform initialization."""
        dim = lb.size
        pop = np.zeros((n, dim))
        for d in range(dim):
            cuts = np.linspace(lb[d], ub[d], n + 1)
            pop[:, d] = rng.uniform(cuts[:n], cuts[1:])
            rng.shuffle(pop[:, d])
        return pop

    @staticmethod
    def clip_to_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        return np.clip(x, lb, ub)

    @staticmethod
    def reflect_to_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Reflect out-of-bound values back in."""
        x = x.copy()
        range_ = ub - lb
        lo = x < lb
        hi = x > ub
        x[lo] = lb[lo] + np.abs(x[lo] - lb[lo]) % range_[lo]
        x[hi] = ub[hi] - np.abs(x[hi] - ub[hi]) % range_[hi]
        return np.clip(x, lb, ub)
