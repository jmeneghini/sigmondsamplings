"""Renderer-backed coverage for empty cells and interactive legend groups.

Kaleido and its headless browser are optional; unavailable browsers skip these
checks explicitly. Once a browser starts, rendering or interaction failures fail.
"""

from __future__ import annotations

import asyncio
import time
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from test_plotting_backends import stats
from test_plotting_correlators import _stack, fit_result

import slat.plotting as slp
from sigmondsamplings.plotting import FitOverlay, SamplingPlotter, SeriesPlotter, SeriesStyle

__all__ = ["stats", "fit_result"]


@pytest.fixture(scope="module")
def renderer():
    pytest.importorskip("kaleido")
    pytest.importorskip("choreographer")
    import plotly.graph_objects as go
    from choreographer.browsers import BrowserDepsError, BrowserFailedError, ChromeNotFoundError

    try:
        go.Figure(go.Scatter(x=[0], y=[0])).full_figure_for_development(warn=False)
    except (BrowserDepsError, BrowserFailedError, ChromeNotFoundError) as exc:
        pytest.skip(f"Kaleido's headless browser is unavailable: {exc}")


@pytest.mark.parametrize("kind", ["title", "text", "shape", "nan", "filtered"])
@pytest.mark.parametrize("shared", [False, True])
def test_empty_cell_has_its_own_rendered_axes_and_frame(renderer, tmp_path, kind, shared):
    fig = slp.figure(2, 1, sharex=shared, backend="plotly")
    try:
        fig.axes().line([2, 8], [3, 9])
        fig.axes().set(xlim=(2, 8), ylim=(3, 9))
        axes = fig.axes(1)
        if kind == "title":
            axes.set(title="empty row")
        elif kind == "text":
            axes.text(0.5, 0.5, "empty cell", coords="axes")
        elif kind == "shape":
            axes.axhline(4)
        elif kind == "nan":
            axes.line([np.nan], [np.nan])
        else:
            data = _stack().isel(operator=0)
            SeriesPlotter(data, style=SeriesStyle(max_rel_error=0)).plot(axes=axes)
        # Explicit ranges on both shared members avoid relying on the separate
        # range/matches follow-up in Tier 10.2.
        axes.set(xlim=(2, 8), ylim=(3, 9))
        original = tuple(fig.native.data)
        fig.save(tmp_path / "empty.svg")
        full = fig.native.full_figure_for_development(warn=False)
        assert full.layout.xaxis.range == (2, 8)
        assert full.layout.yaxis.range == (3, 9)
        assert full.layout.xaxis2.range == (2, 8)
        assert full.layout.yaxis2.range == (3, 9)
        assert full.layout.yaxis.domain[0] > full.layout.yaxis2.domain[1]
        assert all(actual is expected for actual, expected in zip(fig.native.data, original))
        root = ET.fromstring((tmp_path / "empty.svg").read_text())
        classes = [node.get("class", "").split() for node in root.iter()]
        assert any("xy" in parts and "subplot" in parts for parts in classes)
        assert any("x2y2" in parts and "subplot" in parts for parts in classes)
        # Each visible cell's frame is a separate rendered shape path.
        shape_paths = [node for node in root.iter() if node.tag.endswith("path")
                       and node.get("data-index") is not None]
        assert len(shape_paths) >= 2
        for trace in fig.native.data[len(original):]:
            assert not trace.showlegend and trace.hoverinfo == "skip"
    finally:
        fig.close()


async def _check_legends(paths_and_expectations):
    import choreographer
    from choreographer.browsers import BrowserDepsError, BrowserFailedError, ChromeNotFoundError

    try:
        browser = choreographer.Browser()
        await browser.open()
    except (BrowserDepsError, BrowserFailedError, ChromeNotFoundError) as exc:
        pytest.skip(f"Headless browser is unavailable: {exc}")
    try:
        for path, checks in paths_and_expectations:
            tab = await browser.create_tab(path.as_uri())
            await tab.send_command("Emulation.setDeviceMetricsOverride", params={
                "width": 1600, "height": 1400, "deviceScaleFactor": 1, "mobile": False,
            })

            async def evaluate(expression):
                response = await tab.send_command("Runtime.evaluate", params={
                    "expression": expression, "returnByValue": True, "awaitPromise": True,
                })
                result = response["result"]
                assert "exceptionDetails" not in result, result
                return result["result"].get("value")

            graph = "document.querySelector('.plotly-graph-div')"
            toggles = "document.querySelectorAll('rect[class^=legend][class$=toggle]')"
            required = max((check[0] for check in checks), default=-1) + 1
            deadline = time.monotonic() + 15
            while not await evaluate(f"Boolean({graph}?._fullLayout && "
                                    f"document.querySelector('.subplot') && {toggles}.length >= {required})"):
                assert time.monotonic() < deadline, "Plotly did not finish rendering"
                await asyncio.sleep(0.05)
            if not checks:
                assert await evaluate(f"Boolean({graph}._fullLayout._plots.x2y2)")
            for check in checks:
                legend_index, hidden_indices = check[:2]
                shape_indices = check[2] if len(check) == 3 else set()
                selector = f"{toggles}[{legend_index}]"
                point = await evaluate(f"(() => {{const r={selector}.getBoundingClientRect();"
                                       "return {x:r.x+r.width/2,y:r.y+r.height/2};})()")
                assert 0 < point["x"] < 1600 and 0 < point["y"] < 1400
                for event in ("mouseMoved", "mousePressed", "mouseReleased"):
                    await tab.send_command("Input.dispatchMouseEvent", params={
                        "type": event, **point, "button": "left", "clickCount": 1,
                    })
                deadline = time.monotonic() + 3
                expected = [i in hidden_indices for i in range(await evaluate(f"{graph}.data.length"))]
                while await evaluate(f"{graph}.data.map(t => t.visible === 'legendonly')") != expected:
                    assert time.monotonic() < deadline, "legend click did not toggle the whole item"
                    await asyncio.sleep(0.05)
                if shape_indices:
                    assert await evaluate(f"{graph}.layout.shapes.map(s => s.visible === 'legendonly')") == [
                        i in shape_indices for i in range(await evaluate(f"{graph}.layout.shapes.length"))
                    ]
                # Restore it with a second click, separated from double-click timing.
                await asyncio.sleep(0.4)
                for event in ("mousePressed", "mouseReleased"):
                    await tab.send_command("Input.dispatchMouseEvent", params={
                        "type": event, **point, "button": "left", "clickCount": 1,
                    })
                deadline = time.monotonic() + 3
                while await evaluate(f"{graph}.data.some(t => t.visible === 'legendonly')"):
                    assert time.monotonic() < deadline, "legend click did not restore the item"
                    await asyncio.sleep(0.05)
                await asyncio.sleep(0.4)
            await tab.close()
    finally:
        await browser.close()


def test_exported_html_toggles_intervals_split_series_and_fits(renderer, tmp_path, monkeypatch,
                                                            stats, fit_result):
    # Use the production save path and post-script, with inline Plotly and no
    # external MathJax request: these plain-text fixtures work offline.
    import plotly.graph_objects as go

    original_write = go.Figure.write_html

    def write_html(self, *args, **kwargs):
        kwargs.update(include_plotlyjs=True, include_mathjax=False)
        return original_write(self, *args, **kwargs)

    monkeypatch.setattr(go.Figure, "write_html", write_html)
    paths_and_expectations = []
    figures = []
    try:
        intervals = SamplingPlotter(stats).plot_bootstrap_intervals(backend="plotly").figure
        figures.append(intervals)
        path = tmp_path / "intervals.html"
        intervals.save(path)
        paths_and_expectations.append((path, [(1, {1, 2, 3}), (2, {4, 5, 6})]))

        data = _stack().isel(operator=0)
        split = slp.figure(2, 1, backend="plotly")
        figures.append(split)
        for axes in split.flat:
            SeriesPlotter(data, labels=["same"],
                          hollow=[np.arange(data.shape[0]) % 2 == 0]).plot(axes=axes)
            axes.legend()
        path = tmp_path / "split.html"
        split.save(path)
        paths_and_expectations.append((path, [(0, {0, 1}), (1, {2, 3})]))

        fits = SeriesPlotter(data).plot(backend="plotly",
                                       fits=[FitOverlay(fit_result, label="fit")]).figure
        figures.append(fits)
        path = tmp_path / "fit.html"
        fits.save(path)
        paths_and_expectations.append((path, [(1, {1, 2})]))
        shape = slp.figure(backend="plotly")
        figures.append(shape)
        shape.axes().line([0, 1], [0, 1], label="data")
        shape.axes().axhline(0.5, label="reference", hover="reference hover")
        shape.axes().legend()
        path = tmp_path / "shape.html"
        shape.save(path)
        paths_and_expectations.append((path, [(1, {1}, {1})]))

        empty = slp.figure(2, 1, backend="plotly")
        figures.append(empty)
        empty.axes().line([0, 1], [0, 1])
        empty.axes(1).set(title="empty row")
        path = tmp_path / "empty.html"
        empty.save(path)
        paths_and_expectations.append((path, []))
        asyncio.run(_check_legends(paths_and_expectations))
    finally:
        for fig in figures:
            fig.close()
