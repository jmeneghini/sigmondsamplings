"""Process-isolated Sigmond reference harness (never a production dependency)."""
from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


def unavailable(reason):
    if os.environ.get("SS_REQUIRE_SIGMOND") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def bindings():
    try:
        module = importlib.import_module("sigmond")
        try:
            module.EffectiveEnergyCalculator
        except ModuleNotFoundError as exc:
            # scikit-build's editable mapping can select the stubs directory
            # instead of the sibling extension. Resolve the INSTALLED extension
            # from distribution metadata; never assume a local build path.
            if exc.name != "sigmond._core.tasks":
                raise
            import sys
            from importlib.metadata import distribution
            from importlib.util import module_from_spec, spec_from_file_location
            dist = distribution("sigmond")
            matches = [f for f in dist.files or () if "/_core." in str(f) and
                       str(f).endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))]
            if len(matches) != 1:
                raise
            spec = spec_from_file_location("sigmond._core", dist.locate_file(matches[0]))
            core = module_from_spec(spec)
            sys.modules["sigmond._core"] = core
            spec.loader.exec_module(core)
            module.EffectiveEnergyCalculator
    except (ImportError, AttributeError) as exc:
        unavailable(f"Sigmond bindings unavailable: {exc}")
    return module


def binary(numeric_type="complex"):
    name = (os.environ.get("SS_SIGMOND_BATCH_REAL") if numeric_type == "real" else
            os.environ.get("SS_SIGMOND_BATCH", "sigmond_batch"))
    path = shutil.which(name) if name else None
    if path is None:
        unavailable(f"Sigmond {numeric_type} batch binary unavailable")
    return path


def inventory(numeric_type="complex"):
    path = binary(numeric_type)
    return {"binary": path, "binary_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def parse_log(path):
    root = ET.parse(path).getroot()
    errors = [ET.tostring(e, encoding="unicode") for e in root.iter()
              if e.tag.lower() in ("error", "errormessage", "taskerror") and e.text and e.text.strip()]
    errors += [e.text for e in root.iter("Status") if e.text != "Success"]
    return root, errors


def rotate(matrix, directory, *, norm_time=1, metric_time=1, diagonalize_time=3,
           cutoff=1e-8, rotate_mode="samplings_unsubt", check_metric=True,
           input_matrix=None, expect_empty=False, set_imaginary_parts_zero=False, zero_elements=(),
           check_null=False, numeric_type="complex", timings=None, timeout=60):
    """Write SS keys, run the real task, and require successful grouped output."""
    started = time.perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data, log, pivot, output = [directory / f for f in
                              ("input.hdf5", "log.xml", "pivot.hdf5", "rotated.hdf5")]
    from sigmondsamplings.data import _kind_cls
    def complex_buffer(original):
        if original is None:
            return None
        return _kind_cls(original.sampling_info)._from_parts(
            original.data.astype(complex), original.observable_info, original.sampling_info,
            True, original.dims, original.coords, original.infos)
    # CN expects explicit Im keys even when the buffer is real valued.
    source = matrix if input_matrix is None else input_matrix
    exported = (source._replace(values=complex_buffer(source.values), vevs=complex_buffer(source.vevs))
                if numeric_type == "complex" else source)
    exported.write(data, group="input", overwrite=True)
    root = ET.Element("SigMonD")
    init = ET.SubElement(root, "Initialize")
    ET.SubElement(init, "ProjectName").text = "ss_pivot_oracle"
    ET.SubElement(init, "LogFile").text = str(log)
    from sigmondsamplings.io.writer import SigmondWriter
    writer = SigmondWriter()
    header = (writer._generate_bins_header_xml(matrix.values.ensemble_info) if matrix.is_bins
              else writer._generate_header_xml(matrix.values.ensemble_info, matrix.sampling_info))
    # Reuse the same ensemble/bin/sampling header as the input writer.
    header_root = ET.fromstring(header)
    for tag in ("MCBinsInfo", "MCSamplingInfo"):
        node = header_root.find(f".//{tag}")
        if node is not None:
            init.append(node)
    obs = ET.SubElement(init, "MCObservables")
    files = ET.SubElement(obs, "BinData" if source.is_bins else "SamplingData")
    ET.SubElement(files, "FileName").text = f"{data}[input]"
    task = ET.SubElement(ET.SubElement(root, "TaskSequence"), "Task")
    for tag, value in {"Action": "DoCorrMatrixRotation", "MinTimeSep": min(matrix.times),
                       "MaxTimeSep": max(matrix.times), "RotateMode": rotate_mode,
                       "Type": "SinglePivot"}.items():
        ET.SubElement(task, tag).text = str(value)
    create = ET.SubElement(task, "SinglePivotInitiate")
    from sigmondsamplings.correlators import GenIrrepOperator
    label = GenIrrepOperator.from_channel(matrix.info.channel, name="ROT", index=0)
    ET.SubElement(create, "RotatedCorrelator").append(label.to_xml())
    create.append(ET.fromstring(matrix.info.to_sigmond_xml()))
    if input_matrix is not None and matrix.transformation is not None:
        import numpy as np
        trans = matrix.transformation
        definitions = ET.SubElement(create, "ImprovedOperators")
        for col, op in enumerate(matrix.operators):
            terms = trans.coefficients[:, col]
            if op in trans.source_operators:
                unit = np.zeros(len(terms))
                unit[trans.source_operators.index(op)] = 1
                if np.array_equal(unit, terms):
                    continue
            definition = ET.SubElement(definitions, "ImprovedOperator")
            ET.SubElement(definition, "OpName").append(op.to_xml())
            for original, coefficient in zip(trans.source_operators, terms):
                if coefficient == 0:
                    continue
                term = ET.SubElement(definition, "OpTerm")
                term.append(original.to_xml())
                ET.SubElement(term, "Coefficient").text = (str(coefficient.real) if numeric_type == "real"
                                                          else f"({coefficient.real},{coefficient.imag})")
    for tag, value in {"NormTime": norm_time, "MetricTime": metric_time,
                       "DiagonalizeTime": diagonalize_time,
                       "MinimumInverseConditionNumber": cutoff}.items():
        ET.SubElement(create, tag).text = str(value)
    if set_imaginary_parts_zero:
        ET.SubElement(create, "SetImaginaryPartsZero")
    if zero_elements:
        from sigmondsamplings.correlators import CorrelatorInfo
        for sink, source in zero_elements:
            ET.SubElement(create, "SetToZero").append(ET.fromstring(CorrelatorInfo(sink, source).to_sigmond()))
    ET.SubElement(create, "PrintTransformationMatrix")
    if check_metric:
        ET.SubElement(create, "CheckMetricErrors")
    if check_null:
        ET.SubElement(create, "CheckCommonMetricMatrixNullSpace")
    write = ET.SubElement(create, "WritePivotToFile")
    ET.SubElement(write, "PivotFileName").text = f"{pivot}[pivot]"
    ET.SubElement(write, "Overwrite")
    write = ET.SubElement(task, "WriteRotatedCorrToFile")
    ET.SubElement(write, "RotatedCorrFileName").text = f"{output}[rotated]"
    ET.SubElement(write, "WriteMode").text = "overwrite"
    xml = directory / "task.xml"
    ET.ElementTree(root).write(xml)
    prepared = time.perf_counter()
    proc = subprocess.run([binary(numeric_type), str(xml)], capture_output=True, text=True, timeout=timeout)
    finished = time.perf_counter()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert log.exists(), proc.stdout + proc.stderr
    parsed, errors = parse_log(log)
    assert not errors, "\n".join(errors)
    import h5py
    with h5py.File(output) as file:
        assert "rotated/Values" in file, ET.tostring(parsed).decode()
        assert bool(len(file["rotated/Values"])) != expect_empty, ET.tostring(parsed).decode()
    if timings is not None:
        timings.update(input_preparation=prepared - started,
                       batch_process=finished - prepared,
                       output_validation=time.perf_counter() - finished)
    return parsed, pivot, output



def read_pivot(path, *, group="pivot"):
    """Decode RN/CN ArrayBuf storage, retaining header-defined row order.

    ArrayBuf writes interleaved complex scalars in column-major order into an
    HDF5 array whose dimensions alone do not describe that scalar layout.
    """
    import h5py

    from sigmondsamplings.correlators import operator_from_xml
    with h5py.File(path) as file:
        identifier = file["Info/FIdentifier"][()].decode()
        assert identifier in ("Sigmond--SinglePivotFile-CN", "Sigmond--SinglePivotFile-RN")
        cn = identifier.endswith("CN")
        header = ET.fromstring(file[f"{group}/Header"][()].decode())
        def labels(node):
            return tuple(operator_from_xml(e) for e in node if e.tag in
                         ("GIOperatorString", "BLOperatorString", "GIOperator", "BLOperator"))
        pivot_labels = labels(header.find("MatrixDefinition/CorrelatorMatrixInfo"))
        original = header.find("OriginalMatrix/CorrelatorMatrixInfo")
        source_labels = pivot_labels if original is None else labels(original)
        def decode(key):
            raw = file[f"{group}/Values/UIntKey_{key}"][...]
            if cn:
                assert raw.ndim == 3 and raw.shape[0] == 2
                values = raw.ravel().reshape(-1, 2)
                values = values[:, 0] + 1j * values[:, 1]
            else:
                assert raw.ndim == 2
                values = raw.ravel()
            rows, cols = raw.shape[-2:]
            return values.reshape(cols, rows).T
        R, Z = decode(0), decode(1)
    assert R.shape[0] == len(source_labels) and Z.shape[0] == len(pivot_labels)
    return dict(numeric_type="complex" if cn else "real", source_labels=source_labels,
                pivot_labels=pivot_labels, transformation=R, zmatrix=Z)


def transformation_from_log(root, source_labels):
    import numpy as np

    from sigmondsamplings.correlators import operator_from_xml
    levels = root.findall(".//TransformationMatrix/ImprovedOperators/ImprovedOperator")
    R = np.zeros((len(source_labels), len(levels)), dtype=complex)
    for col, level in enumerate(levels):
        for term in level.findall("OpTerm"):
            op = operator_from_xml(term[0])
            value = term.findtext("Coefficient")
            if value.startswith("("):
                re, im = value.strip("()").split(",")
                coefficient = complex(float(re), float(im))
            else:
                coefficient = float(value)
            R[source_labels.index(op), col] = coefficient
    return R
