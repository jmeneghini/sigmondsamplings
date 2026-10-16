"""Isolated binding oracle for unnormalized overlap output (run as a process)."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from sigmond_oracle import binary, bindings, parse_log


def main(directory):
    sigmond = bindings()
    directory = Path(directory)
    root = ET.parse(directory / 'task.xml').getroot()
    obs = root.find('Initialize/MCObservables/SamplingData')
    ET.SubElement(obs, 'FileName').text = f'{directory}/fit.hdf5[fit]'
    xml = sigmond.XMLHandler()
    xml.set_from_string(ET.tostring(root, encoding='unicode'))
    handler = sigmond.TaskHandler(xml)
    create = root.find('TaskSequence/Task/SinglePivotInitiate')
    pivot_xml = sigmond.XMLHandler()
    pivot_xml.set_from_string(ET.tostring(create, encoding='unicode'))
    args = sigmond.ArgsHandler(pivot_xml)
    pivot = sigmond.Pivot()
    pivot.setType('SinglePivot')
    pivot.initiatePivotPython(handler.getMCObsHandler(), args, sigmond.LogHelper())
    for level in range(pivot.getNumberOfLevels()):
        pivot.insertAmplitudeFitInfo(level, sigmond.MCObsInfo('oracle_amplitude', level))
        pivot.insertEnergyFitInfo(level, sigmond.MCObsInfo('oracle_energy', level))
    log = sigmond.LogHelper()
    pivot.reorderLevelsByFitEnergy(log)
    (directory / 'reorder.xml').write_text(log.output())
    pivot.computeZMagnitudesSquaredPython(f'{directory}/overlaps.hdf5[overlaps]',
                                          sigmond.WriteMode.Protect, 'H', 'Z2oracle')
    # Exercise the actual normalized task table independently of the output-file API.
    import subprocess
    root.find('Initialize/LogFile').text = str(directory / 'overlap_table.xml')
    ET.SubElement(create, 'AssignName').text = 'oracle_pivot'
    sequence = root.find('TaskSequence')
    for action in ('DoRotCorrMatInsertFitInfos', 'DoCorrMatrixZMagSquares'):
        task = ET.SubElement(sequence, 'Task')
        ET.SubElement(task, 'Action').text = action
        ET.SubElement(task, 'Type').text = 'SinglePivot'
        memory = ET.SubElement(ET.SubElement(task, 'SinglePivotInitiate'), 'GetFromMemory')
        ET.SubElement(memory, 'IDName').text = 'oracle_pivot'
        if action == 'DoRotCorrMatInsertFitInfos':
            ET.SubElement(task, 'EnergyFitCommonName').text = 'oracle_energy'
            ET.SubElement(task, 'RotatedAmplitudeCommonName').text = 'oracle_amplitude'
            ET.SubElement(task, 'ReorderByFitEnergy')
    table_input = directory / 'overlap_task.xml'
    ET.ElementTree(root).write(table_input)
    result = subprocess.run([binary(), str(table_input)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    _, errors = parse_log(directory / 'overlap_table.xml')
    assert not errors, errors



if __name__ == '__main__':
    main(sys.argv[1])
