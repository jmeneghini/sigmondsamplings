"""Regression tests for ObservableCollection query and export behavior."""

from __future__ import annotations

import pytest

from sigmondsamplings.info import SamplingInfo


def test_filtered_update_invalidates_parent_shared_attr_cache(observables):
    assert observables.shared_attr("index") == 0

    observables.filter(name="junk_obs").obs.update(index=7)

    assert observables.shared_attr("index", default=None) is None


def test_filtered_view_cache_does_not_contaminate_parent(observables):
    filtered = observables.filter(name="junk_obs")
    assert filtered.shared_attr("name") == "junk_obs"

    assert observables.shared_attr("name", default=None) is None


def test_dataframe_uses_correct_confidence_interval_columns(observables):
    lower, upper = observables[0].confidence_interval()

    row = observables.to_dataframe().iloc[0]

    assert row["CI_lower"] == pytest.approx(lower)
    assert row["CI_upper"] == pytest.approx(upper)


def test_dataframe_allows_jackknife_samplings(observables):
    jackknife = observables.copy()
    sampling_info = SamplingInfo("jackknife", 20)
    for sampling in jackknife:
        sampling.sampling_info = sampling_info

    frame = jackknife.to_dataframe()

    assert frame["CI_lower"].isna().all()
    assert frame["CI_upper"].isna().all()


@pytest.mark.parametrize("nulls_last", [False, True])
def test_descending_sort_preserves_requested_null_placement(observables, nulls_last):
    collection = observables[:2]
    collection.obs.update(sort_marker=[None, 1])

    result = collection.sort("sort_marker", reverse=True, nulls_last=nulls_last)

    expected = [collection[1], collection[0]] if nulls_last else [collection[0], collection[1]]
    assert result.to_list() == expected


def test_filter_data_rejects_unknown_comparison_operator(observables):
    with pytest.raises(ValueError, match="Unknown comparison operator.*gte"):
        observables.filter_data(gte=1.0)


def test_mapping_values_align_by_observable_info(observables):
    collection = observables[:2]
    values = {
        collection[1].observable_info: 0,
        collection[0].observable_info: 1,
    }

    result = collection.sort(values=values)

    assert result.to_list() == [collection[1], collection[0]]


def test_mapping_values_require_all_observables(observables):
    collection = observables[:2]

    with pytest.raises(ValueError, match="every collection observable_info"):
        collection.sort(values={collection[0].observable_info: 1})
