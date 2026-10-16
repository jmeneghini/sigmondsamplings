"""Regenerate small pinned fixtures; execute explicitly, never during tests.

    python tests/generate_pivot_goldens.py
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from pivot_data import REFERENCE_CASES, reference_fixture
from sigmond_oracle import inventory, read_pivot, rotate

DEST = Path(__file__).with_name('fixtures') / 'pivot'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--numeric-type', choices=('complex', 'real', 'both'), default='complex')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    runtime = {}
    for numeric_type in ('complex', 'real'):
        if args.numeric_type in (numeric_type, 'both'):
            runtime[numeric_type] = inventory(numeric_type)
    commit = subprocess.check_output(
        ['git', '-C', str(Path(__file__).parents[2] / 'sigmond'), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == '926e10ee52a2becef96c2f8ec96a729487089d55'
    for name in REFERENCE_CASES:
        numeric_type = 'real' if name.startswith('rn_') else 'complex'
        if numeric_type not in runtime:
            continue
        with tempfile.TemporaryDirectory() as temp:
            matrix, _, options, task = reference_fixture(name)
            root, path, output = rotate(matrix, temp, norm_time=options['norm_time'],
                                       metric_time=options['metric_time'], diagonalize_time=options['diagonalize_time'],
                                       cutoff=options['min_inv_condition'], **task)
            ref = read_pivot(path)
            target = DEST / name
            target.mkdir(exist_ok=True)
            for filename in ('task.xml', 'input.hdf5', 'log.xml', 'pivot.hdf5', 'rotated.hdf5'):
                shutil.copy(Path(temp) / filename, target / filename)
                if filename == 'task.xml':
                    # Run this self-contained input from the fixture directory.
                    text = (target / filename).read_text().replace(str(Path(temp)) + '/', '')
                    (target / filename).write_text(text)
            np.savez(target / 'arrays.npz', transformation=ref['transformation'], zmatrix=ref['zmatrix'],
                     source_labels=np.array([op.coordinate for op in ref['source_labels']]),
                     pivot_labels=np.array([op.coordinate for op in ref['pivot_labels']]))
            metadata = dict(runtime[numeric_type], sigmond_commit=commit,
                            generator_version=3, fixture_seed=271, numeric_type=ref['numeric_type'],
                            case=name,
                            generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in target.iterdir() if p.name != 'metadata.json'})
            (target / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
