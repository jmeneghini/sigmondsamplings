"""Thread-budget planning and per-API native thread limits.

Covers the four-level budget (outer workers x optimagic cores x OpenMP x BLAS)
that ``fitting._execution`` divides the machine into, and the two placement
rules that make it real: BLAS and OpenMP are limited independently, and pool
workers are pinned from *inside* the worker.
"""

import multiprocessing

import numpy  # noqa: F401 - loads the BLAS/OpenMP runtimes the limit tests inspect
import pytest

from sigmondsamplings.fitting._execution import (
    ThreadBudget,
    executor_context,
    limit_native_threads,
    plan_thread_budget,
    resolve_mp_context,
)

# ---------------------------------------------------------------------------
# plan_thread_budget
# ---------------------------------------------------------------------------


def test_single_fit_spends_whole_machine_in_multiples_of_the_omp_width():
    """One fit, no outer pool: OpenMP takes the block count, BLAS the rest.

    This is the ``full_sample`` case. With 6 blocks on 24 cores each OpenMP
    worker gets 4 BLAS threads, so the cores handed to the fit stay a whole
    multiple of the block count.
    """
    budget = plan_thread_budget(1, cpus=24, omp_width=6)

    assert budget.workers == 1
    assert budget.omp_threads == 6
    assert budget.blas_threads == 4
    assert budget.total_threads == 24
    assert budget.total_threads % 6 == 0


def test_many_jobs_fill_the_outer_pool_first():
    """More resamples than cores: workers take everything, inner levels get 1."""
    budget = plan_thread_budget(200, cpus=24, omp_width=6)

    assert budget.workers == 24
    assert budget.omp_threads == 1
    assert budget.blas_threads == 1
    assert budget.total_threads == 24


def test_leftover_after_the_pool_goes_inward():
    """Fewer jobs than cores: the remainder is spent on OpenMP, then BLAS."""
    budget = plan_thread_budget(4, cpus=24, omp_width=6)

    assert budget.workers == 4  # never more workers than jobs
    assert budget.omp_threads == 6  # 24/4 = 6 cores per job, all to OpenMP
    assert budget.blas_threads == 1
    assert budget.total_threads == 24


def test_omp_never_widens_past_the_parallel_region():
    """Extra cores go to BLAS, not to OpenMP threads with no block to run."""
    budget = plan_thread_budget(1, cpus=32, omp_width=2)

    assert budget.omp_threads == 2
    assert budget.blas_threads == 16


def test_optimagic_cores_are_taken_off_the_top():
    budget = plan_thread_budget(100, cpus=24, inner_cores=4, omp_width=6)

    assert budget.workers == 6
    assert budget.total_threads <= 24


def test_explicit_settings_are_honoured_and_the_rest_budgeted_around_them():
    budget = plan_thread_budget(100, cpus=24, omp_width=6, num_workers=2, omp_threads=3)

    assert budget.workers == 2
    assert budget.omp_threads == 3
    assert budget.blas_threads == 4  # (24/2) / 3


def test_none_opts_an_api_out_of_pinning():
    budget = plan_thread_budget(100, cpus=24, omp_threads=None, blas_threads=None)

    assert budget.omp_threads is None
    assert budget.blas_threads is None
    assert budget.total_threads == budget.workers  # None counted as 1


def test_budget_never_returns_zero_threads():
    """A machine smaller than the requested inner width still gets a valid plan."""
    budget = plan_thread_budget(100, cpus=1, inner_cores=4, omp_width=6)

    assert budget.workers >= 1
    assert budget.omp_threads >= 1
    assert budget.blas_threads >= 1


def test_rejects_nonsense_thread_counts():
    with pytest.raises(ValueError):
        plan_thread_budget(10, cpus=8, omp_threads=0)


# ---------------------------------------------------------------------------
# limit_native_threads
# ---------------------------------------------------------------------------


def _limits_by_api() -> dict[str, int]:
    threadpoolctl = pytest.importorskip("threadpoolctl")
    return {
        info["user_api"]: info["num_threads"] for info in threadpoolctl.threadpool_info()
    }


def test_blas_and_omp_are_limited_independently():
    """The whole point of the split: a wide OpenMP pool over narrow BLAS calls.

    A single combined limit cannot express this, and pinning BLAS to 1 the old
    way silently serialized any OpenMP region in the objective.
    """
    before = _limits_by_api()
    if "openmp" not in before or "blas" not in before:
        pytest.skip("needs both a BLAS and an OpenMP runtime loaded")

    with limit_native_threads(blas=1, omp=2):
        during = _limits_by_api()
        assert during["blas"] == 1
        assert during["openmp"] == 2

    assert _limits_by_api() == before


def test_untouched_api_keeps_its_limit():
    before = _limits_by_api()
    if "openmp" not in before:
        pytest.skip("needs an OpenMP runtime loaded")

    with limit_native_threads(blas=1):
        assert _limits_by_api()["openmp"] == before["openmp"]


def test_positional_limit_still_sets_both():
    """Legacy spelling used by the chi-square scan path."""
    before = _limits_by_api()
    if not before:
        pytest.skip("no native thread pools loaded")

    with limit_native_threads(1):
        during = _limits_by_api()
        assert all(n == 1 for n in during.values())


def test_no_arguments_is_a_no_op():
    before = _limits_by_api()
    with limit_native_threads():
        assert _limits_by_api() == before


# ---------------------------------------------------------------------------
# worker pinning
# ---------------------------------------------------------------------------


def _omp_limit_in_worker(_: int) -> int | None:
    import threadpoolctl

    for info in threadpoolctl.threadpool_info():
        if info["user_api"] == "openmp":
            return info["num_threads"]
    return None


def test_thread_workers_are_pinned_from_inside_the_worker():
    """``omp_set_num_threads`` is per-thread, so pool workers need their own call.

    Limiting around the loop on the submitting thread leaves workers at the
    runtime default — the machine oversubscribed by ``workers x omp``.
    """
    pytest.importorskip("threadpoolctl")
    if _omp_limit_in_worker(0) is None:
        pytest.skip("needs an OpenMP runtime loaded")

    with executor_context("thread", max_workers=2, omp_threads=1, blas_threads=1) as ex:
        assert list(ex.map(_omp_limit_in_worker, range(4))) == [1, 1, 1, 1]


# ---------------------------------------------------------------------------
# multiprocessing start method
# ---------------------------------------------------------------------------


def test_openmp_workers_avoid_fork():
    """Forking after the prologue has run an OpenMP region is a known hang."""
    ctx = resolve_mp_context("auto", uses_openmp=True)

    assert ctx is not None
    assert ctx.get_start_method() == "spawn"


def test_platform_default_kept_when_no_openmp_is_involved():
    assert resolve_mp_context("auto", uses_openmp=False) is None


def test_explicit_start_method_wins():
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("no fork start method on this platform")

    ctx = resolve_mp_context("fork", uses_openmp=True)

    assert ctx.get_start_method() == "fork"


def test_context_object_passes_through():
    ctx = multiprocessing.get_context("spawn")

    assert resolve_mp_context(ctx, uses_openmp=False) is ctx


# ---------------------------------------------------------------------------
# ThreadBudget
# ---------------------------------------------------------------------------


def test_describe_reports_every_materialized_setting():
    budget = ThreadBudget(
        workers=4, omp_threads=6, blas_threads=1, inner_cores=2, cpus=48, omp_width=6
    )

    text = budget.describe()

    assert "workers     : 4" in text
    assert "opt cores   : 2" in text
    assert "omp threads : 6" in text
    assert "width 6" in text  # why omp stopped where it did
    assert "blas threads: 1" in text
    assert "total       : 48/48 cpus" in text


def test_describe_names_unpinned_apis_rather_than_implying_one_thread():
    text = ThreadBudget(
        workers=4, omp_threads=None, blas_threads=None, cpus=8
    ).describe()

    assert "omp threads : unpinned" in text
    assert "blas threads: unpinned" in text


def test_summary_is_a_single_line():
    summary = ThreadBudget(
        workers=4, omp_threads=6, blas_threads=1, inner_cores=1, cpus=24
    ).summary()

    assert "\n" not in summary
    assert "24/24 cpus" in summary


def test_observed_threads_render_per_api():
    from sigmondsamplings.fitting._execution import format_observed_threads

    with limit_native_threads(blas=1, omp=1):
        text = format_observed_threads()

    # Either a real reading or the honest "nothing loaded" message, never blank.
    assert text
    if "=" in text:
        assert all("=" in part for part in text.split(", "))
