"""Exercise exported hover callbacks without requiring a browser."""

import shutil
import subprocess

import pytest

from slat.plotting._plotly._figure import _POST_SCRIPT


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript checks need Node")
def test_line_hover_covers_visible_span_after_zoom_and_pan(tmp_path):
    setup = """
const assert = require('node:assert/strict');
let callback;
const graph = {
    data: [{type:'scatter', mode:'lines', line:{dash:'dash'},
            meta:{slat_hover_segments:[[2, 2, 8, 'NI pair']]}}],
    _fullData: [{mode:'lines'}],
    _fullLayout: {xaxis:{type:'linear', range:[0,10], _length:1000}},
    on: (event, cb) => {callback=cb;}
};
const document = {getElementById: () => graph};
const Plotly = {restyle: (g, update, indices) => {
    indices.forEach((index, i) => {
        for (const [key, values] of Object.entries(update)) g.data[index][key]=values[i];
    });
}};
"""
    checks = """
function checkCoverage(low, high) {
    const axis=graph._fullLayout.xaxis, trace=graph.data[0];
    const visible=trace.x.filter(x => x !== null && x>=low && x<=high);
    assert.ok(visible.length>1);
    assert.ok(Math.abs(visible[0]-low)<1e-10);
    assert.ok(Math.abs(visible.at(-1)-high)<1e-10);
    for(let i=1;i<visible.length;i++) {
        const span=axis.type==='log'
            ? Math.log10(visible[i])-Math.log10(visible[i-1])
            : visible[i]-visible[i-1];
        assert.ok(span/(axis.range[1]-axis.range[0])*axis._length<=12.000001);
    }
    assert.equal(trace.x[0],2); assert.equal(trace.x.at(-2),8);
    assert.equal(trace.x.length,trace.y.length);
    assert.equal(trace.x.length,trace.hovertext.length);
    assert.ok(trace.hovertext.every(t => t==='NI pair'));
    const [dash,gap]=trace['line.dash'].split(',').map(parseFloat);
    const fullPixels=(axis.type==='log' ? Math.log10(8)-Math.log10(2) : 6)
        /(axis.range[1]-axis.range[0])*axis._length;
    const gaps=Math.max(0,Math.round((fullPixels-6)/10));
    assert.ok(Math.abs((gaps+1)*dash+gaps*gap-fullPixels)<1e-8);
}
checkCoverage(2,8);
graph._fullLayout.xaxis.range=[4.1,4.3];
callback({'xaxis.range[0]':4.1}); checkCoverage(4.1,4.3);
graph._fullLayout.xaxis.range=[7.9,8.1];
callback({'xaxis.range[0]':7.9}); checkCoverage(7.9,8);
graph._fullLayout.xaxis.range=[0,10];
callback({'xaxis.autorange':true}); checkCoverage(2,8);
graph._fullLayout.xaxis._length=2000;
callback({width:2000}); checkCoverage(2,8);
graph._fullLayout.xaxis.type='log';
graph._fullLayout.xaxis.range=[Math.log10(4.1),Math.log10(4.3)];
callback({'xaxis.range[0]':Math.log10(4.1)}); checkCoverage(4.1,4.3);
graph._fullLayout.xaxis.type='linear';
graph._fullLayout.xaxis.range=[9,10];
callback({'xaxis.range[0]':9});
assert.deepEqual(graph.data[0].x,[2,8,null]);
"""
    script = tmp_path / "hover.cjs"
    script.write_text(setup + _POST_SCRIPT + checks)
    subprocess.run([shutil.which("node"), str(script)], check=True, capture_output=True, text=True)
