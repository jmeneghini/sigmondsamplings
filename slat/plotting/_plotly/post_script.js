
(function () {
    const graph = document.getElementById('{plot_id}');
    const axisKey = ref => ref[0] + 'axis' + ref.slice(1);
    const span = key => {
        const axis = graph._fullLayout[key];
        if (!axis || !axis.range) return null;
        const value = Math.abs(axis.range[1] - axis.range[0]);
        return Number.isFinite(value) && value > 0 ? value : null;
    };
    const markers = graph.data.flatMap((trace, index) => {
        if (!['scatter', 'scattergl'].includes(trace.type) ||
            !(trace.mode || graph._fullData[index].mode || '').includes('markers')) {
            return [];
        }
        const x = axisKey(trace.xaxis || 'x');
        const y = axisKey(trace.yaxis || 'y');
        const size = trace.marker && trace.marker.size != null
            ? trace.marker.size : graph._fullData[index].marker.size;
        return [{index, x, y, xspan: span(x), yspan: span(y),
                 size: Array.isArray(size) || ArrayBuffer.isView(size)
                    ? Array.from(size) : size}];
    });
    const segments = graph.data.flatMap((trace, index) => {
        const lines = trace.meta && trace.meta.slat_hover_segments;
        return Array.isArray(lines) ? [{index, axis: axisKey(trace.xaxis || 'x'), lines,
            dash: (trace.line && trace.line.dash) || 'solid'}] : [];
    });
    function updateLineHoverTargets() {
        if (!segments.length) return;
        const xs = [], ys = [], texts = [], dashes = [];
        for (const trace of segments) {
            const axis = graph._fullLayout[trace.axis];
            const log = axis.type === 'log';
            const linear = value => log ? Math.log10(value) : value;
            const data = value => log ? Math.pow(10, value) : value;
            const [low, high] = [...axis.range].sort((a, b) => a - b);
            const range = high - low;
            let dash = trace.dash;
            if (dash === 'dash' && trace.lines.length === 1 && range > 0) {
                const [, start, end] = trace.lines[0];
                const pixels = Math.abs(linear(end) - linear(start)) / range *
                    (axis._length || 1000);
                if (Number.isFinite(pixels) && pixels > 0) {
                    // Fit complete end dashes around 4px gaps, rather than
                    // letting SVG's fixed dash pattern finish in a gap.
                    const gaps = Math.max(0, Math.round((pixels - 6) / 10));
                    const length = (pixels - gaps * 4) / (gaps + 1);
                    dash = `${length}px,4px`;
                }
            }
            dashes.push(dash);
            const x = [], y = [], text = [];
            for (const [height, start, end, label] of trace.lines) {
                const a = linear(start), b = linear(end);
                const left = Math.max(Math.min(a, b), low);
                const right = Math.min(Math.max(a, b), high);
                const points = [start];
                if (Number.isFinite(range) && range > 0 && right >= left) {
                    // At most 12 screen pixels between hover targets, even
                    // when the original segment endpoints are off screen.
                    const pixels = (right - left) / range * (axis._length || 1000);
                    const count = Math.min(4096, Math.max(1, Math.ceil(pixels / 12)));
                    const interior = Array.from({length: count + 1}, (_, i) =>
                        data(left + (right - left) * i / count));
                    if (a > b) interior.reverse();
                    points.push(...interior);
                }
                points.push(end);
                x.push(...points, null);
                y.push(...points.map(() => height), null);
                text.push(...points.map(() => label), label);
            }
            xs.push(x); ys.push(y); texts.push(text);
        }
        return Plotly.restyle(graph, {x: xs, y: ys, hovertext: texts, 'line.dash': dashes},
            segments.map(trace => trace.index));
    }
    updateLineHoverTargets();
    graph.on('plotly_relayout', function (event) {
        if (!Object.keys(event).some(key =>
            /^[xy]axis\d*\.(range|autorange)/.test(key) ||
            ['width', 'height', 'autosize'].includes(key))) {
            return;
        }
        const sizes = markers.map(marker => {
            const xspan = span(marker.x), yspan = span(marker.y);
            const zoom = Math.max(
                marker.xspan && xspan ? marker.xspan / xspan : 1,
                marker.yspan && yspan ? marker.yspan / yspan : 1
            );
            const scale = Math.min(1.5, Math.max(1, Math.pow(zoom, 0.2)));
            return Array.isArray(marker.size)
                ? marker.size.map(size => size * scale) : marker.size * scale;
        });
        if (markers.length) {
            Promise.resolve(Plotly.restyle(graph, {'marker.size': sizes},
                markers.map(marker => marker.index))).then(updateLineHoverTargets);
        } else {
            updateLineHoverTargets();
        }
    });
})();
