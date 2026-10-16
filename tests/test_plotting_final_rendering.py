"""Rendered contour, native-layout and browser zoom integration contracts."""

import asyncio
import re
import time
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from test_plotting_tier9_rendering import renderer

import slat.plotting as slp

__all__ = ["renderer"]


def _elements(root, css_class):
    return [node for node in root.iter() if css_class in node.get("class", "").split()]


def _path_xs(path):
    # Plotly's contour paths use M/L/Q/C/Z, each with coordinate pairs.
    assert not re.search(r"[AaHhVvSsTt]", path)
    values = [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", path)]
    return np.asarray(values[::2])


@pytest.mark.parametrize("filled", [False, True])
def test_irregular_contours_have_correct_rendered_boundaries(renderer, tmp_path, filled):
    fig = slp.figure(backend="plotly")
    try:
        x = np.linspace(0, 4, 17)
        levels = (0.5, 1, 3)
        fig.axes().set(xlim=(0, 4), ylim=(0, 4))
        fig.axes().contour(x, x, np.tile(x, (len(x), 1)), levels=levels, filled=filled,
                           labels=True, colors=("red", "blue", "green"))
        path = tmp_path / "contours.svg"
        fig.save(path)
        full = fig.native.full_figure_for_development(warn=False)
        root = ET.fromstring(path.read_text())
        clip = _elements(root, "plotclip")[0]
        width = float(next(iter(clip)).get("width"))
        groups = _elements(root, "contourfill" if filled else "contourlines")
        assert len(groups) == (2 if filled else 3)
        for i, group in enumerate(groups):
            xs = np.concatenate([_path_xs(node.get("d")) for node in group.iter()
                                 if node.tag.endswith("path") and node.get("d")])
            wanted = (levels[i], levels[i + 1]) if filled else (levels[i], levels[i])
            np.testing.assert_allclose([xs.min(), xs.max()], np.array(wanted) / 4 * width, atol=0.02)
            if filled:
                assert list(full.data[i].contours.value) == list(wanted)
            else:
                assert full.data[i].contours.start == full.data[i].contours.end == levels[i]
        texts = [node.text for group in _elements(root, "contourlabels") for node in group.iter()
                 if node.tag.endswith("text")]
        if filled:
            assert not texts
        else:
            assert {float(t) for t in texts} == set(levels)
    finally:
        fig.close()


def test_wrapped_empty_native_grid_renders_and_point_size_uses_inner_extent(renderer, tmp_path):
    from plotly.subplots import make_subplots

    native = make_subplots(2, 1, shared_xaxes=True)
    native.update_layout(width=600, height=500, margin={"l": 80, "r": 80, "t": 100, "b": 100},
                         template="simple_white")
    fig = slp.wrap(native)
    try:
        fig.axes(1).set(xlim=(2, 8), ylim=(1, 100), yscale="log")
        fig.axes().text(0.5, 0.5, "empty native cell", coords="axes")
        fig.axes(1).line([2, 8], [1, 100])
        fig.save(tmp_path / "native.svg")
        root = ET.fromstring((tmp_path / "native.svg").read_text())
        assert _elements(root, "xy") and _elements(root, "x2y2")
        full = native.full_figure_for_development(warn=False)
        assert full.layout.xaxis.range == full.layout.xaxis2.range == (2, 8)
        clips = _elements(root, "plotclip")
        rendered = float(next(iter(clips[1])).get("height"))
        assert fig.axes(1).point_size(6, units="axes") == pytest.approx((6 * 100 / 72) / rendered)
    finally:
        fig.close()


async def _zoom(path):
    choreographer = pytest.importorskip("choreographer")
    from choreographer.browsers import BrowserDepsError, BrowserFailedError, ChromeNotFoundError

    try:
        browser = choreographer.Browser()
        await browser.open()
    except (BrowserDepsError, BrowserFailedError, ChromeNotFoundError) as exc:
        pytest.skip(f"Headless browser is unavailable: {exc}")
    try:
        tab = await browser.create_tab(path.as_uri())
        await tab.send_command("Emulation.setDeviceMetricsOverride", params={
            "width": 1600, "height": 1400, "deviceScaleFactor": 1, "mobile": False,
        })
        async def evaluate(expression):
            response = await tab.send_command("Runtime.evaluate", params={
                "expression": expression, "returnByValue": True, "awaitPromise": True,
            })
            assert "exceptionDetails" not in response["result"], response
            return response["result"]["result"].get("value")
        graph = "document.querySelector('.plotly-graph-div')"
        deadline = time.monotonic() + 15
        while not await evaluate(f"Boolean({graph}?._fullLayout && {graph}.data[1].x.length > 10)"):
            assert time.monotonic() < deadline, "Plotly did not finish rendering hover targets"
            await asyncio.sleep(0.05)
        initial = await evaluate(f"{graph}.data.map(t=>({{type:t.type,group:t.legendgroup}}))")
        assert {t["type"] for t in initial} == {"scatter", "bar", "heatmap", "contour"}
        assert await evaluate(f"{graph}._fullData.map(t=>t.zorder)") == [20, 20, 0, -10, 10, 10, 30, 30, 30]
        # Browser rendering also works with the minimum supported Plotly,
        # independently of Kaleido's stricter Python-package requirement.
        fills = await evaluate(f"(() => {{const g={graph};return {{width:g._fullLayout.xaxis._length,"
                               "bands:[...g.querySelectorAll('.contourfill path')].map(p=>{const b=p.getBBox();"
                               "return [b.x,b.x+b.width];})};})()")
        assert len(fills["bands"]) == 2
        np.testing.assert_allclose(fills["bands"], np.array([[2, 4], [4, 9]]) / 10 * fills["width"], atol=0.03)
        labels = await evaluate(f"[...{graph}.querySelectorAll('.contourlabels text')].map(t=>t.textContent)")
        assert {float(label) for label in labels} == {2, 4, 9}
        size = await evaluate(f"{graph}.data[0].marker.size")
        # Exercise a real user drag on the plot, which emits plotly_relayout.
        rect = await evaluate(f"(() => {{const r={graph}.querySelector('.nsewdrag').getBoundingClientRect();"
                              "return {x:r.x,y:r.y,w:r.width,h:r.height};})()")
        for event, x, y in (("mouseMoved", 0.25, 0.25), ("mousePressed", 0.25, 0.25),
                            ("mouseMoved", 0.75, 0.75), ("mouseReleased", 0.75, 0.75)):
            await tab.send_command("Input.dispatchMouseEvent", params={
                "type": event, "x": rect["x"] + x * rect["w"], "y": rect["y"] + y * rect["h"],
                "button": "left", "buttons": 1 if event != "mouseReleased" else 0, "clickCount": 1,
            })
        deadline = time.monotonic() + 5
        while not await evaluate(f"{graph}.data[0].marker.size > {size}"):
            assert time.monotonic() < deadline, "drag zoom did not run the marker callback"
            await asyncio.sleep(0.05)
        assert await evaluate(f"{graph}.data.map(t=>({{type:t.type,group:t.legendgroup}}))") == initial
        assert size < await evaluate(f"{graph}.data[0].marker.size") <= size * 1.5
        state = await evaluate(f"(() => {{const g={graph},a=g._fullLayout.xaxis;"
                               "return {range:a.range,width:a._length,x:g.data[1].x,text:g.data[1].hovertext};})()")
        lo, hi = sorted(state["range"])
        interior = sorted({x for x in state["x"] if x is not None and lo <= x <= hi})
        assert len(interior) > 10
        assert max(np.diff(interior)) / (hi - lo) * state["width"] <= 12.01
        assert set(state["text"]) == {"segment hover"}
        # Relayout to a deeper zoom and then reset to check the cap and absence
        # of cumulative scaling; these run through the same registered event.
        await evaluate(f"Plotly.relayout({graph},{{'xaxis.range':[4.99,5.01]}}).then(()=>true)")
        await asyncio.sleep(0.1)
        assert await evaluate(f"{graph}.data[0].marker.size") == pytest.approx(size * 1.5)
        await evaluate(f"Plotly.relayout({graph},{{'xaxis.range':[0,10],'yaxis.range':[0,10]}}).then(()=>true)")
        await asyncio.sleep(0.1)
        assert await evaluate(f"{graph}.data[0].marker.size") == pytest.approx(size)
    finally:
        await browser.close()


def test_exported_html_zoom_callbacks_keep_groups_indices_and_dense_hover(tmp_path, monkeypatch):
    import plotly.graph_objects as go

    original = go.Figure.write_html
    def write_html(self, *args, **kwargs):
        kwargs.update(include_plotlyjs=True, include_mathjax=False)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(go.Figure, "write_html", write_html)
    fig = slp.figure(backend="plotly")
    try:
        ax = fig.axes()
        ax.set(xlim=(0, 10), ylim=(0, 10))
        with ax.legend_group():
            ax.points([2, 5, 8], [2, 5, 8], label="points", markersize=6)
            ax.hlines(5, 0, 10, hover="segment hover", linestyle="--")
        x = np.linspace(0, 10, 21)
        ax.bar([1], [1], native={"zorder": 0})
        ax.heatmap(np.tile(x, (len(x), 1)), x=x, y=x, colorbar=False,
                   native={"zorder": -10})
        ax.contour(x, x, np.tile(x, (len(x), 1)), levels=[2, 4, 9], filled=True,
                   colors=["red", "blue"])
        ax.contour(x, x, np.tile(x, (len(x), 1)), levels=[2, 4, 9], labels=True,
                   native={"zorder": 30})
        ax.legend()
        path = tmp_path / "zoom.html"
        fig.save(path)
        asyncio.run(_zoom(path))
    finally:
        fig.close()
