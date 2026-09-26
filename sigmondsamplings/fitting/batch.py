"""Portable pieces of a resampled fit.

The fitting driver can execute a central fit and arbitrary resampling-index
batches in different processes or scheduler jobs. These two data-only objects
are the hand-off format; orchestration and scheduler policy deliberately live
outside this package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

__all__ = ["CentralFit", "ResampleBatch"]


@dataclass(slots=True)
class CentralFit:
    """Serializable full-sample result used to initialize resampling batches."""

    params: dict[str, float]
    chi2: float
    success: bool
    message: str
    n_free: int
    theory: np.ndarray | None = None
    trace_path: str | None = None
    used_openmp: bool = False
    optimize_result: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def from_result(
        cls,
        result: Any,
        names: list[str],
        *,
        theory: np.ndarray | None = None,
        trace_path: str | None = None,
        used_openmp: bool = False,
        n_free: int | None = None,
    ) -> CentralFit:
        return cls(
            params={name: float(result.params[name]) for name in names},
            chi2=float(result.fun),
            success=bool(result.success),
            message=str(result.message),
            n_free=int(n_free if n_free is not None else getattr(result, "n_free", len(names))),
            theory=None if theory is None else np.asarray(theory, dtype=float),
            trace_path=trace_path,
            used_openmp=used_openmp,
            optimize_result=result,
        )

    def as_optimize_result(self) -> Any:
        """Native result when local, or the minimal equivalent after reload."""
        if self.optimize_result is not None:
            return self.optimize_result
        return SimpleNamespace(
            params=dict(self.params),
            fun=float(self.chi2),
            success=bool(self.success),
            message=str(self.message),
            n_free=int(self.n_free),
            convergence_report=None,
        )

    def write(self, path: str | Path) -> None:
        """Write portable JSON. Atomic publication is the caller's policy."""
        document = {
            "format": "sigmondsamplings-central-fit-v1",
            "params": self.params,
            "chi2": self.chi2,
            "success": self.success,
            "message": self.message,
            "n_free": self.n_free,
            "theory": None if self.theory is None else self.theory.tolist(),
            "trace_path": self.trace_path,
            "used_openmp": self.used_openmp,
        }
        Path(path).write_text(json.dumps(document, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> CentralFit:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("format") != "sigmondsamplings-central-fit-v1":
            raise ValueError(f"unsupported central-fit format in {path}")
        theory = document.get("theory")
        return cls(
            params={str(k): float(v) for k, v in document["params"].items()},
            chi2=float(document["chi2"]),
            success=bool(document["success"]),
            message=str(document["message"]),
            n_free=int(document["n_free"]),
            theory=None if theory is None else np.asarray(theory, dtype=float),
            trace_path=document.get("trace_path"),
            used_openmp=bool(document.get("used_openmp", False)),
        )


@dataclass(slots=True)
class ResampleBatch:
    """Results for an arbitrary, unique set of positive resampling indices."""

    indices: np.ndarray
    names: tuple[str, ...]
    params: np.ndarray
    chi2: np.ndarray
    success: np.ndarray
    messages: tuple[str, ...]
    n_iterations: np.ndarray
    theory: np.ndarray | None = None
    optimize_results: dict[int, Any] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=np.int64)
        self.params = np.asarray(self.params, dtype=float)
        self.chi2 = np.asarray(self.chi2, dtype=float)
        self.success = np.asarray(self.success, dtype=bool)
        self.n_iterations = np.asarray(self.n_iterations, dtype=np.int64)
        if self.theory is not None:
            self.theory = np.asarray(self.theory, dtype=float)
        n = self.indices.size
        if self.indices.ndim != 1 or np.any(self.indices < 1):
            raise ValueError("resampling indices must be a one-dimensional array of positive ints")
        if len(set(self.indices.tolist())) != n:
            raise ValueError("resampling indices in a batch must be unique")
        if self.params.shape != (len(self.names), n):
            raise ValueError(
                f"params must have shape ({len(self.names)}, {n}), got {self.params.shape}"
            )
        for label, values in (
            ("chi2", self.chi2),
            ("success", self.success),
            ("n_iterations", self.n_iterations),
        ):
            if values.shape != (n,):
                raise ValueError(f"{label} must have shape ({n},), got {values.shape}")
        if len(self.messages) != n:
            raise ValueError(f"messages must contain {n} values")
        if self.theory is not None and self.theory.shape[1:] != (n,):
            raise ValueError(f"theory must have {n} columns, got {self.theory.shape}")

    def write(self, path: str | Path) -> None:
        """Write an NPZ checkpoint without pickled Python objects."""
        if self.optimize_results:
            raise ValueError("a batch with keep='full' native results is not portable")
        with open(path, "wb") as handle:
            np.savez_compressed(
                handle,
                format=np.asarray(["sigmondsamplings-resample-batch-v1"]),
                indices=self.indices,
                names=np.asarray(self.names, dtype=str),
                params=self.params,
                chi2=self.chi2,
                success=self.success,
                messages=np.asarray(self.messages, dtype=str),
                n_iterations=self.n_iterations,
                has_theory=np.asarray([self.theory is not None], dtype=bool),
                theory=(
                    self.theory
                    if self.theory is not None
                    else np.empty((0, self.indices.size), dtype=float)
                ),
            )

    @classmethod
    def read(cls, path: str | Path) -> ResampleBatch:
        with np.load(path, allow_pickle=False) as data:
            fmt = str(data["format"][0])
            if fmt != "sigmondsamplings-resample-batch-v1":
                raise ValueError(f"unsupported resample-batch format in {path}")
            theory = np.asarray(data["theory"], dtype=float) if bool(data["has_theory"][0]) else None
            return cls(
                indices=np.asarray(data["indices"], dtype=np.int64),
                names=tuple(str(v) for v in data["names"]),
                params=np.asarray(data["params"], dtype=float),
                chi2=np.asarray(data["chi2"], dtype=float),
                success=np.asarray(data["success"], dtype=bool),
                messages=tuple(str(v) for v in data["messages"]),
                n_iterations=np.asarray(data["n_iterations"], dtype=np.int64),
                theory=theory,
            )
