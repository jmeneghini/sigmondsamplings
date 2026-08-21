"""Execution helpers for the resampling fit driver.

Owns the three orthogonal concerns of running many small fits:

* **Thread budget** — native thread limits via :mod:`threadpoolctl` rather than
  environment variables, so already-imported libraries are throttled dynamically
  and per-region. BLAS and OpenMP are budgeted *separately*: an objective whose
  inner loop is itself an OpenMP parallel region needs a wide OpenMP pool and a
  narrow BLAS one, and a single combined limit cannot express that. Four levels
  compete for the machine — outer worker pool, optimagic's per-fit cores, the
  objective's OpenMP region, and BLAS — and :func:`plan_thread_budget` divides
  the available CPUs among them.
* **Workers** — SLURM-/affinity-aware CPU detection and pool construction
  (serial, thread, or process), with both thread and process workers throttled
  in their initializer.
* **Progress** — a tqdm/marimo-aware bar driven from the completion loop.

The module is intentionally free of any fitting logic; the driver in
``fitting.fit`` supplies the per-index callable.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import (
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

FitBackend = Literal["serial", "thread", "process"]
ErrorPolicy = Literal["record", "raise"]
ProgressKind = bool | str  # True/False/"auto"/"notebook"/"terminal"/"marimo"

#: A thread-count setting: an explicit count, ``"auto"`` (let
#: :func:`plan_thread_budget` choose), or ``None`` (leave the current limit
#: untouched).
ThreadSetting = int | Literal["auto"] | None

__all__ = [
    "ErrorPolicy",
    "FitBackend",
    "ProgressKind",
    "ThreadBudget",
    "ThreadSetting",
    "available_cpus",
    "default_num_workers",
    "executor_context",
    "format_observed_threads",
    "limit_native_threads",
    "observed_native_threads",
    "plan_thread_budget",
    "run_jobs",
]


# ---------------------------------------------------------------------------
# native thread limits (threadpoolctl)
# ---------------------------------------------------------------------------


def _thread_limit_stack(blas: int | None, omp: int | None) -> ExitStack | None:
    """An entered :class:`ExitStack` holding the requested per-API limits.

    ``None`` for either API leaves that API's current limit untouched; ``None``
    for both yields ``None`` (nothing to restore). OpenMP is limited *before*
    BLAS so that a BLAS library whose own threading layer is OpenMP — a common
    build, where both appear as separate ``threadpoolctl`` modules backed by the
    same runtime — ends up honouring its BLAS-specific setting.
    """
    if blas is None and omp is None:
        return None
    import threadpoolctl

    stack = ExitStack()
    try:
        if omp is not None:
            stack.enter_context(
                threadpoolctl.threadpool_limits(limits=int(omp), user_api="openmp")
            )
        if blas is not None:
            stack.enter_context(
                threadpoolctl.threadpool_limits(limits=int(blas), user_api="blas")
            )
    except BaseException:
        stack.close()
        raise
    return stack


@contextmanager
def limit_native_threads(
    limits: int | None = None,
    *,
    blas: int | None = None,
    omp: int | None = None,
):
    """Limit native threads of loaded libraries within the block, per API.

    ``blas`` and ``omp`` are budgeted independently: an objective that is itself
    an OpenMP parallel region wants ``blas=1, omp=<wide>``, which a single
    combined limit cannot express. ``None`` leaves that API's limits untouched.
    The positional ``limits`` is the legacy spelling for "the same limit on both
    APIs" and is only used for an API whose keyword was not given.

    Use this around an in-process (serial-backed) fit loop, or around a single
    fit. Pool workers are throttled in their initializer instead, since the
    OpenMP thread count is a *per-thread* setting that a limit applied on the
    calling thread does not propagate to pool workers (see
    :func:`executor_context`).
    """
    if limits is not None:
        if blas is None:
            blas = limits
        if omp is None:
            omp = limits
    stack = _thread_limit_stack(blas, omp)
    if stack is None:
        yield
        return
    with stack:
        yield


#: Reference kept alive in process workers so the limit persists for the
#: worker's lifetime (threadpoolctl restores limits when this is collected).
_WORKER_THREAD_LIMITS: Any = None

#: Same, for thread workers — one entry per worker thread, since the OpenMP
#: thread count is a per-thread setting.
_THREAD_WORKER_STATE = threading.local()


def _process_worker_init(
    blas_threads: int | None,
    omp_threads: int | None,
    user_initializer: Callable | None,
    user_initargs: tuple,
) -> None:
    """Initializer run once per process worker: pin native threads, then chain."""
    global _WORKER_THREAD_LIMITS
    _WORKER_THREAD_LIMITS = _thread_limit_stack(blas_threads, omp_threads)
    logger.debug(
        "process worker %d native threads: %s", os.getpid(), format_observed_threads()
    )
    if user_initializer is not None:
        user_initializer(*user_initargs)


def _thread_worker_init(
    blas_threads: int | None,
    omp_threads: int | None,
    user_initializer: Callable | None,
    user_initargs: tuple,
) -> None:
    """Initializer run once per worker *thread*.

    Thread workers cannot inherit an OpenMP limit from the submitting thread:
    ``omp_set_num_threads`` writes a per-thread internal control variable, so a
    limit applied around the loop in the main thread leaves pool workers at the
    runtime default and the machine oversubscribed by ``workers x omp``. Pinning
    from inside the worker is the only placement that takes effect.
    """
    _THREAD_WORKER_STATE.limits = _thread_limit_stack(blas_threads, omp_threads)
    logger.debug(
        "thread worker %s native threads: %s",
        threading.current_thread().name,
        format_observed_threads(),
    )
    if user_initializer is not None:
        user_initializer(*user_initargs)


# ---------------------------------------------------------------------------
# worker-count budgeting
# ---------------------------------------------------------------------------


def available_cpus() -> int:
    """Usable CPU count: SLURM allocation, then affinity mask, then cpu_count."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        return max(1, int(slurm_cpus))
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            return max(1, len(sched_getaffinity(0)))
        except OSError:
            pass
    return os.cpu_count() or 1


def _positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def default_num_workers(
    inner_cores: int | None = 1,
    blas_threads: int | None = 1,
    omp_threads: int | None = 1,
) -> int:
    """Outer worker count given per-fit inner cores and native thread counts.

    Returns ``available_cpus() // (inner_cores * blas_threads * omp_threads)``
    (floor 1). ``inner_cores`` is the *total* cores a single fit consumes —
    optimagic's parallel numerical-difference cores plus any algorithm-internal
    or multistart parallelism — which the caller specifies rather than the driver
    inferring.
    """
    inner = _positive_int(inner_cores, "inner_cores") or 1
    blas = _positive_int(blas_threads, "blas_threads") or 1
    omp = _positive_int(omp_threads, "omp_threads") or 1
    return max(1, available_cpus() // (inner * blas * omp))


def resolve_num_workers(
    num_workers: int | Literal["auto"] | None,
    *,
    inner_cores: int | None = 1,
    blas_threads: int | None = 1,
    omp_threads: int | None = 1,
) -> int:
    """Resolve ``"auto"``/``None`` to :func:`default_num_workers`, else validate."""
    if num_workers in (None, "auto"):
        return default_num_workers(
            inner_cores=inner_cores, blas_threads=blas_threads, omp_threads=omp_threads
        )
    resolved = _positive_int(num_workers, "num_workers")
    assert resolved is not None
    return resolved


@dataclass(frozen=True, slots=True)
class ThreadBudget:
    """How the machine is divided between the four levels of parallelism.

    ``workers`` outer jobs run at once; each consumes ``inner_cores`` optimagic
    cores, and each of those runs an objective whose OpenMP region uses
    ``omp_threads``, each of which may call BLAS with ``blas_threads``. A
    ``None`` thread count means "leave that API's current limit untouched".
    """

    workers: int
    omp_threads: int | None
    blas_threads: int | None
    inner_cores: int = 1
    cpus: int = 1
    omp_width: int | None = None

    @property
    def total_threads(self) -> int:
        """Peak native threads implied by the budget (``None`` counted as 1)."""
        return self.workers * self.inner_cores * (self.omp_threads or 1) * (
            self.blas_threads or 1
        )

    @staticmethod
    def _fmt(value: int | None) -> str:
        return "unpinned" if value is None else str(value)

    def summary(self) -> str:
        """Compact one-liner, for embedding in another message."""
        return (
            f"{self.workers}w x {self.inner_cores}opt x "
            f"{self._fmt(self.omp_threads)}omp x {self._fmt(self.blas_threads)}blas "
            f"= {self.total_threads}/{self.cpus} cpus"
        )

    def describe(self) -> str:
        """Aligned block of the materialized settings, for the driver's logs."""
        width = "" if self.omp_width is None else f"   (block loop, width {self.omp_width})"
        return (
            f"  cpus        : {self.cpus}\n"
            f"  workers     : {self.workers}\n"
            f"  opt cores   : {self.inner_cores}   (per fit)\n"
            f"  omp threads : {self._fmt(self.omp_threads)}{width}\n"
            f"  blas threads: {self._fmt(self.blas_threads)}\n"
            f"  total       : {self.total_threads}/{self.cpus} cpus"
        )


def _resolve_setting(value: ThreadSetting, auto: int, name: str) -> int | None:
    """An explicit count, the computed ``auto`` value, or ``None`` (untouched)."""
    if value is None:
        return None
    if value == "auto":
        return max(1, auto)
    return _positive_int(value, name)


def plan_thread_budget(
    n_jobs: int,
    *,
    cpus: int | None = None,
    inner_cores: int | None = 1,
    omp_width: int | None = None,
    num_workers: int | Literal["auto"] | None = "auto",
    omp_threads: ThreadSetting = "auto",
    blas_threads: ThreadSetting = "auto",
) -> ThreadBudget:
    """Divide ``cpus`` among outer workers, OpenMP, and BLAS.

    Cores are spent in the order they scale. Outer workers come first — one
    whole job per core beats splitting one job, and there is no fork/join cost
    per objective evaluation — but never more workers than there are jobs, which
    is what leaves a remainder to spend inward. That remainder goes to OpenMP up
    to ``omp_width``, the natural width of the objective's inner parallel region
    (for a spectrum fit, its number of momentum blocks): widening past that adds
    threads with no iterations to run. Whatever is left over after OpenMP goes to
    BLAS, so each OpenMP worker gets an equal share and the cores handed to one
    job stay a whole multiple of ``omp_width``.

    With ``n_jobs=1`` this is the single-fit case — no outer pool to compete
    with, so the whole machine goes to OpenMP and BLAS.

    Any of ``num_workers``/``omp_threads``/``blas_threads`` may be given
    explicitly to pin that level and budget the rest around it; ``None`` opts a
    native API out of pinning entirely.
    """
    cpus = int(cpus) if cpus is not None else available_cpus()
    cpus = max(1, cpus)
    inner = _positive_int(inner_cores, "inner_cores") or 1
    n_jobs = max(1, int(n_jobs))

    if num_workers in (None, "auto"):
        workers = max(1, min(n_jobs, cpus // inner))
    else:
        workers = _positive_int(num_workers, "num_workers") or 1

    # Cores one job may spend inward, once the outer pool has taken its share.
    per_job = max(1, cpus // (workers * inner))

    width = per_job if omp_width is None else max(1, int(omp_width))
    omp = _resolve_setting(omp_threads, min(width, per_job), "omp_threads")
    blas = _resolve_setting(blas_threads, per_job // (omp or 1), "blas_threads")

    return ThreadBudget(
        workers=workers,
        omp_threads=omp,
        blas_threads=blas,
        inner_cores=inner,
        cpus=cpus,
        omp_width=None if omp_width is None else max(1, int(omp_width)),
    )


def observed_native_threads() -> dict[str, int]:
    """What the loaded native runtimes report *right now*, per user API.

    The budget is a plan; this is the result. Worth logging next to the plan
    because the two can disagree for reasons no amount of arithmetic reveals: an
    extension built without OpenMP, a runtime :mod:`threadpoolctl` cannot see, or
    a limit that a library silently refused. Returns the maximum reported by any
    module of each API, or an empty mapping when nothing is loaded or
    ``threadpoolctl`` is unavailable.
    """
    try:
        import threadpoolctl
    except ImportError:
        return {}
    observed: dict[str, int] = {}
    for info in threadpoolctl.threadpool_info():
        api = str(info.get("user_api", "?"))
        threads = int(info.get("num_threads", 0))
        observed[api] = max(observed.get(api, 0), threads)
    return observed


def format_observed_threads() -> str:
    """``observed_native_threads`` as a log-friendly ``blas=1, openmp=6``."""
    observed = observed_native_threads()
    if not observed:
        return "no native thread pools detected"
    return ", ".join(f"{api}={n}" for api, n in sorted(observed.items()))


# ---------------------------------------------------------------------------
# executors
# ---------------------------------------------------------------------------


def validate_backend(backend: str) -> FitBackend:
    if backend not in ("serial", "thread", "process"):
        raise ValueError("backend must be one of 'serial', 'thread', or 'process'")
    return backend  # type: ignore[return-value]


def resolve_mp_context(
    mp_context: str | Any | None,
    *,
    uses_openmp: bool = False,
) -> Any:
    """The multiprocessing context for the process backend.

    ``"auto"``/``None`` picks ``"spawn"`` when the workers will run an OpenMP
    objective, and the platform default otherwise. Forking a process that has
    already run an OpenMP parallel region — which the full-sample fit does before
    the resample pool is built — leaves the child holding a thread pool it did
    not create, a long-standing source of hangs. ``spawn`` starts each worker
    from a fresh interpreter, so nothing is inherited; the cost is that the
    objective must be picklable (it already must be, to be sent at all) and that
    a *script* entry point needs the usual ``if __name__ == "__main__":`` guard.

    Pass an explicit ``"fork"`` to opt out, or a context object to use verbatim.
    """
    if mp_context is not None and not isinstance(mp_context, str):
        return mp_context
    if mp_context in (None, "auto"):
        if not uses_openmp:
            return None  # platform default
        method = "spawn"
    else:
        method = mp_context
    return multiprocessing.get_context(method)


@contextmanager
def executor_context(
    backend: FitBackend,
    *,
    max_workers: int,
    worker_initializer: Callable | None = None,
    worker_initargs: tuple = (),
    blas_threads: int | None = None,
    omp_threads: int | None = None,
    mp_context: str | Any | None = "auto",
) -> Iterator[Executor | None]:
    """Yield an :class:`~concurrent.futures.Executor` (or ``None`` for serial).

    Both pooled backends pin native threads in their worker initializer, which
    for OpenMP is the only placement that works: its thread count is a per-thread
    setting, so a limit applied by the submitting thread does not reach the
    workers. The user's own initializer is chained after the pinning.
    """
    if backend == "serial":
        yield None
        return

    if backend == "thread":
        executor: Executor = ThreadPoolExecutor(
            max_workers=max_workers,
            initializer=_thread_worker_init,
            initargs=(blas_threads, omp_threads, worker_initializer, worker_initargs),
        )
    else:
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_process_worker_init,
            initargs=(blas_threads, omp_threads, worker_initializer, worker_initargs),
            mp_context=resolve_mp_context(
                mp_context, uses_openmp=(omp_threads or 1) > 1
            ),
        )
    try:
        yield executor
    finally:
        executor.shutdown()


# ---------------------------------------------------------------------------
# job loop
# ---------------------------------------------------------------------------


def run_jobs(
    fn: Callable[[int], Any],
    indices: Iterable[int],
    *,
    executor: Executor | None,
    progress: ProgressKind = False,
    error_policy: ErrorPolicy = "record",
    desc: str = "Resample fits",
) -> Iterator[tuple[int, Any, str | None]]:
    """Run ``fn(index)`` over ``indices`` yielding ``(index, value, error)``.

    ``value`` is the function result, or ``None`` with ``error`` set to the
    exception repr when the call raised and ``error_policy`` is ``"record"``;
    ``"raise"`` re-raises, tagged with the index it came from. Results arrive
    in completion order for pooled backends, declared order when serial.
    """
    indices = list(indices)
    bar = _maybe_tqdm(progress, total=len(indices), desc=desc)
    try:
        if executor is None:
            for i in indices:
                yield _run_one(fn, i, error_policy, desc)
                if bar is not None:
                    bar.update(1)
            return

        futures = {executor.submit(fn, i): i for i in indices}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                yield i, fut.result(), None
            except Exception as exc:  # noqa: BLE001 - policy decides re-raise
                if error_policy == "raise":
                    _tag_index(exc, i, desc)
                    raise
                yield i, None, repr(exc)
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()


def _tag_index(exc: BaseException, i: int, desc: str) -> None:
    """Record which job raised *exc*, so a fatal abort names the index.

    A resample fit that dies takes the whole run with it under
    ``error_policy="raise"``, and the propagating exception otherwise says
    nothing about *which* of the thousand samples failed — the one fact needed
    to reproduce it. ``add_note`` (PEP 678) carries that along without
    substituting a wrapper type for the original exception, which callers may
    be catching; it is a no-op before Python 3.11.
    """
    add_note = getattr(exc, "add_note", None)
    if add_note is not None:
        add_note(f"{desc}: raised at index {i}")


def _run_one(
    fn: Callable[[int], Any], i: int, error_policy: ErrorPolicy, desc: str
) -> tuple[int, Any, str | None]:
    try:
        return i, fn(i), None
    except Exception as exc:  # noqa: BLE001 - policy decides re-raise
        if error_policy == "raise":
            _tag_index(exc, i, desc)
            raise
        return i, None, repr(exc)


# ---------------------------------------------------------------------------
# progress bars (tqdm / marimo)
# ---------------------------------------------------------------------------


_PROGRESS_ALIASES = {
    True: "auto",
    "auto": "auto",
    "notebook": "notebook",
    "nb": "notebook",
    "terminal": "terminal",
    "text": "terminal",
    "std": "terminal",
    "marimo": "marimo",
    "mo": "marimo",
}


def _maybe_tqdm(progress: ProgressKind, *, total: int, desc: str):
    """Build a progress bar exposing ``.update(n)``/``.close()`` or ``None``."""
    if not progress:
        return None
    kind = _PROGRESS_ALIASES.get(progress)
    if kind is None:
        valid = sorted(k for k in _PROGRESS_ALIASES if isinstance(k, str))
        raise ValueError(f"Unknown progress kind {progress!r}; expected one of {valid} or a bool.")

    if kind == "auto":
        if _running_in_marimo():
            return _build_tqdm_bar("notebook", total=total, desc=desc)
        return _build_tqdm_bar("auto", total=total, desc=desc)
    if kind == "marimo":
        if _running_in_marimo():
            return _build_tqdm_bar("notebook", total=total, desc=desc)
        bar = _build_marimo_bar(total=total, desc=desc)
        return bar if bar is not None else _build_tqdm_bar("auto", total=total, desc=desc)
    return _build_tqdm_bar(kind, total=total, desc=desc)


def _build_tqdm_bar(kind: str, *, total: int, desc: str):
    try:
        if kind == "notebook":
            from tqdm.notebook import tqdm
        elif kind == "terminal":
            from tqdm.std import tqdm
        else:
            from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc)
    except ImportError:
        return None


def _running_in_marimo() -> bool:
    try:
        import marimo as mo
    except ImportError:
        return False
    try:
        return bool(mo.running_in_notebook())
    except Exception:
        return False


class _MarimoBar:
    """tqdm-style ``update``/``close`` adapter over ``marimo.status.progress_bar``."""

    def __init__(self, ctx):
        self._ctx = ctx
        self._bar = ctx.__enter__()

    def update(self, n: int = 1) -> None:
        try:
            self._bar.update(increment=n)
        except TypeError:
            self._bar.update(n)

    def close(self) -> None:
        try:
            self._ctx.__exit__(None, None, None)
        except Exception:
            pass


def _build_marimo_bar(*, total: int, desc: str):
    try:
        import marimo as mo
    except ImportError:
        return None
    try:
        return _MarimoBar(mo.status.progress_bar(total=total, title=desc))
    except Exception:
        return None
