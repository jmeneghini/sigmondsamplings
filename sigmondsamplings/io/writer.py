"""
Writer module for Sigmond samplings and bins files.

HDF5 is the only output format (the legacy fstream writer is not supported).
fstream inputs are read via the loader and re-emitted as HDF5; HDF5 inputs are
mutated in place. All write paths build dataset keys through
``_dataset_key_for_observable`` so CorrT/raw-XML observable names round-trip.
"""

import logging
import os
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import h5py
import numpy as np

from ..bins import SigmondBins
from ..sampling import EnsembleInfo, ObservableInfo, SamplingInfo, SigmondSampling
from . import obsmeta
from .loader import (
    DEFAULT_GROUP,
    SigmondLoader,
    clean_group,
    is_hdf5_file,
    verify_sigmond_hdf5,
)

logger = logging.getLogger(__name__)

HDF5_EXTENSIONS = {".h5", ".hdf5"}
WriteMode = Literal["a", "w"]
FileKind = Literal["samplings", "bins"]
DatasetRow = tuple[str, np.ndarray, dict | None]


@dataclass(frozen=True)
class _HDF5WritePayload:
    group: str
    kind: FileKind
    observables: tuple[SigmondSampling | SigmondBins, ...]
    header_xml: str
    dataset_rows: tuple[tuple[DatasetRow, ...], ...]
    component_keys: tuple[frozenset[str], ...]


class SigmondWriter:
    """
    Writer for Sigmond samplings and bins files, in HDF5 format.

    Writing is always HDF5. ``write_hdf5``/``write_bins_hdf5`` support new-file
    (``mode="w"``) and append (``mode="a"``) behavior, while ``append_to_file``
    and ``modify_observable`` provide explicit in-place editing operations.
    Optional numbered backups protect files before they are overwritten or modified.
    """

    def __init__(self, create_backups: bool = True):
        """
        Args:
            create_backups: Create numbered backups before modifying existing files.
        """
        self.create_backups = create_backups

    # ──────────────────────────────────────────────────────────────────────
    # Backups
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _backup_path(filename: str | Path, n: int) -> Path:
        """Path of the n-th numbered backup for ``filename`` (e.g. ``f.hdf5.backup_001``)."""
        return Path(f"{filename}.backup_{n:03d}")

    def _iter_backups(self, filename: str | Path) -> Iterator[Path]:
        """Yield existing backup paths for ``filename`` in ascending order."""
        n = 1
        while (path := self._backup_path(filename, n)).exists():
            yield path
            n += 1

    def _create_numbered_backup(self, filename: str | Path) -> Path | None:
        """
        Copy ``filename`` to the next free numbered backup slot.

        Returns the backup path, or None if backups are disabled or the file
        does not exist.
        """
        filepath = Path(filename)
        if not self.create_backups or not filepath.exists():
            return None

        next_n = sum(1 for _ in self._iter_backups(filename)) + 1
        backup_path = self._backup_path(filename, next_n)
        try:
            shutil.copy2(filepath, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"Failed to create backup of {filepath}: {e}")
            return None

    def restore_from_backup(self, filename: str, backup_number: int | None = None) -> None:
        """
        Restore ``filename`` from a numbered backup (latest if ``backup_number`` is None).
        """
        if backup_number is not None:
            backup_path = self._backup_path(filename, backup_number)
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file {backup_path} does not exist")
        else:
            backups = list(self._iter_backups(filename))
            if not backups:
                raise FileNotFoundError(f"No backups found for file {filename}")
            backup_path = backups[-1]

        shutil.copy2(backup_path, filename)
        logger.info(f"Restored {filename} from backup {backup_path}")

    def delete_backups(self, filename: str, backup_numbers: list[int] | int | None = None) -> None:
        """
        Delete backups of ``filename`` (all of them if ``backup_numbers`` is None).
        """
        if backup_numbers is None:
            targets = list(self._iter_backups(filename))
            if not targets:
                logger.info(f"No backups found for file {filename}")
                return
        else:
            if isinstance(backup_numbers, int):
                backup_numbers = [backup_numbers]
            targets = [self._backup_path(filename, n) for n in backup_numbers]

        for path in targets:
            if path.exists():
                os.remove(path)
                logger.info(f"Deleted backup: {path}")
            else:
                logger.warning(f"Backup file {path} does not exist")

    # ──────────────────────────────────────────────────────────────────────
    # HDF5 / XML building blocks
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_hdf5_format(self, filename: str, group: str | None = None) -> tuple[Path, str]:
        """
        Resolve the HDF5 file that in-place mutating operations should act on.

        - HDF5 input is returned unchanged so the caller mutates it in place.
        - fstream (.smp) input cannot be written back in its native format, so it
          is converted once to a sibling ``file.hdf5`` (the original is left
          untouched) and that path is returned.

        Reading legacy fstream still goes through ``sigmond_query``; HDF5 reads
        use the fast h5py path.

        Returns:
            (hdf5_filename, group) ready for in-place operations.
        """
        filepath = Path(filename)

        # Already a valid Sigmond HDF5 file -> operate on it directly.
        if is_hdf5_file(str(filepath)):
            is_valid, _, available_groups = verify_sigmond_hdf5(str(filepath))
            if not is_valid:
                raise ValueError(f"File {filepath} is not a valid Sigmond HDF5 file")
            if group is None:
                if not available_groups:
                    raise ValueError(f"No data groups found in HDF5 file {filepath}")
                group = available_groups[0]
            return filepath, clean_group(group)

        # fstream input: convert once to a sibling .hdf5, preserving the original.
        hdf5_filename = filepath.with_suffix(".hdf5")
        if hdf5_filename == filepath:
            raise ValueError(f"Cannot derive an .hdf5 sibling for {filepath}")

        loader = SigmondLoader()
        group = clean_group(group or DEFAULT_GROUP)

        logger.info(f"Converting fstream {filepath} -> {hdf5_filename} for in-place processing...")
        loader.load_file(str(filepath), group=group)
        observables = list(loader.observables)
        if not observables:
            raise ValueError(f"No observables found in {filepath}")

        # Preserve bins vs. samplings so the converted file keeps its native kind.
        if loader.file_kind == "bins" or all(isinstance(o, SigmondBins) for o in observables):
            self.write_bins_hdf5(str(hdf5_filename), observables, group, overwrite=True)
        else:
            self.write_hdf5(str(hdf5_filename), observables, group, overwrite=True)

        logger.info(f"Conversion complete. Original fstream {filepath} left unchanged.")
        return hdf5_filename, group

    def _append_mcbins_info(self, root: ET.Element, ensemble_info: EnsembleInfo) -> None:
        """Append the shared ``<MCBinsInfo>`` block (ensemble + optional tweaks) to ``root``."""
        bins_elem = ET.SubElement(root, "MCBinsInfo")
        # Keep lattice geometry self-contained when it is known.  Writing only
        # ``name`` would discard the extents and make a file that loaded without
        # a known-ensembles database impossible to reconstruct faithfully.
        ET.SubElement(bins_elem, "MCEnsembleInfo").text = ensemble_info.short_str
        ET.SubElement(bins_elem, "NumberOfMeasurements").text = str(ensemble_info.num_measurements)
        ET.SubElement(bins_elem, "NumberOfBins").text = str(ensemble_info.num_bins)
        if ensemble_info.tweak_info:
            tweak_elem = ET.SubElement(bins_elem, "TweakEnsemble")
            for key, value in ensemble_info.tweak_info.items():
                ET.SubElement(tweak_elem, key).text = str(value)

    def _generate_header_xml(self, ensemble_info: EnsembleInfo, sampling_info: SamplingInfo) -> str:
        """Generate the compact ``SigmondSamplingsFile`` header XML."""
        root = ET.Element("SigmondSamplingsFile")
        self._append_mcbins_info(root, ensemble_info)

        sampling_elem = ET.SubElement(root, "MCSamplingInfo")
        if sampling_info.method == "bootstrap":
            bootstrap_elem = ET.SubElement(sampling_elem, "Bootstrapper")
            ET.SubElement(bootstrap_elem, "NumberResamplings").text = str(
                sampling_info.num_resamplings
            )
            ET.SubElement(bootstrap_elem, "Seed").text = str(sampling_info.seed)
            ET.SubElement(bootstrap_elem, "BootSkip").text = str(sampling_info.boot_skip)
        elif sampling_info.method == "jackknife":
            if sampling_info.num_resamplings == ensemble_info.num_bins:
                ET.SubElement(sampling_elem, "Jackknife")  # simple jackknife (self-closing)
            else:
                jackknife_elem = ET.SubElement(sampling_elem, "Jackkniffer")
                ET.SubElement(jackknife_elem, "NumberResamplings").text = str(
                    sampling_info.num_resamplings
                )
        else:
            raise ValueError(f"Unsupported sampling method: {sampling_info.method}")

        return ET.tostring(root, encoding="unicode")

    def _generate_bins_header_xml(self, ensemble_info: EnsembleInfo) -> str:
        """Generate the compact ``SigmondBinsFile`` header XML (no sampling info)."""
        root = ET.Element("SigmondBinsFile")
        self._append_mcbins_info(root, ensemble_info)
        return ET.tostring(root, encoding="unicode")

    def _generate_observable_key_xml(self, observable_info: ObservableInfo) -> str:
        """Build the flat ``<MCObservable><Info>name index op_type re_im</Info></MCObservable>`` key."""
        root = ET.Element("MCObservable")
        ET.SubElement(root, "Info").text = (
            f"{observable_info.name} {observable_info.index} "
            f"{observable_info.op_type} {observable_info.re_im}"
        )
        return ET.tostring(root, encoding="unicode")

    def _dataset_key_for_observable(self, observable_info: ObservableInfo, re_im: str) -> str:
        """
        Return the HDF5-safe dataset key for one Re/Im component of an observable.

        * If ``observable_info.name`` begins with ``<`` it is raw XML (e.g. a
          ``<CorrT>`` fragment); it is wrapped as
          ``<MCObservable>{name}<Arg>Re|Im</Arg></MCObservable>``.
        * Otherwise the flat ``<Info>`` form is emitted.

        The result is passed through ``_make_hdf5_safe_key`` so it can be used as
        an HDF5 dataset name.
        """
        name = observable_info.name
        arg_value = "Im" if re_im.lower().startswith("im") else "Re"

        if isinstance(name, str) and name.startswith("<"):
            xml_key = f"<MCObservable>{name}<Arg>{arg_value}</Arg></MCObservable>"
        else:
            cloned = ObservableInfo(
                name,
                observable_info.index,
                observable_info.op_type,
                re_im,
                observable_info.ensemble_info,
            )
            xml_key = self._generate_observable_key_xml(cloned)
        return self._make_hdf5_safe_key(xml_key)

    def _make_hdf5_safe_key(self, xml_key: str) -> str:
        """
        Escape an XML key so it is usable as an HDF5 dataset name.

        ``/`` is special in HDF5 (path separator), so every ``/`` becomes ``|``
        (this also turns closing tags ``</`` into ``<|``). The loader reverses
        this on read.
        """
        return xml_key.replace("</", "<|").replace("/", "|")

    @staticmethod
    def _obs_attrs(observable_info: ObservableInfo) -> dict | None:
        """
        Self-describing dataset attrs for an observable, or None for plain ones.

        Duck-typed: any ObservableInfo exposing ``to_attrs`` (currently the energy
        types) gets its metadata persisted; explicit display labels are preserved
        via ``latex_str``, except for types that regenerate their own label from
        those attrs (``_reconstructs_own_label``), where storing it is redundant.
        """
        attrs = {}
        to_attrs = getattr(observable_info, "to_attrs", None)
        if to_attrs:
            attrs.update(to_attrs())

        latex_str = getattr(observable_info, "_latex_str", None)
        if latex_str and not getattr(observable_info, "_reconstructs_own_label", False):
            attrs["latex_str"] = latex_str

        return attrs or None

    def _observable_datasets(
        self, observable_info: ObservableInfo, data: np.ndarray, is_complex: bool
    ) -> Iterator[tuple[str, np.ndarray, dict | None]]:
        """
        Yield ``(hdf5_safe_key, float64_array, attrs_or_None)`` for each component.

        Complex observables yield two datasets (Re then Im); real observables one.
        Self-describing attrs are attached to the Re/sole component only — the
        loader fuses Re/Im onto the Re part, so a single annotation suffices.
        """
        arr = np.asarray(data)
        attrs = self._obs_attrs(observable_info)
        if is_complex:
            re_key = self._dataset_key_for_observable(observable_info, "re")
            im_key = self._dataset_key_for_observable(observable_info, "im")
            if re_key == im_key:
                raise ValueError(
                    f"Could not construct distinct Re/Im dataset keys for "
                    f"observable {observable_info}"
                )
            yield re_key, np.real(arr).astype(np.float64), attrs
            yield im_key, np.imag(arr).astype(np.float64), None
        else:
            key = self._dataset_key_for_observable(observable_info, observable_info.re_im)
            yield key, arr.astype(np.float64), attrs

    def _fixed_str_dataset(self, group: h5py.Group, name: str, value: str) -> None:
        """Create a fixed-length, null-terminated UTF-8 string dataset (matches real files)."""
        dtype = h5py.string_dtype(encoding="utf-8", length=len(value.encode("utf-8")) + 1)
        group.create_dataset(name, data=value, dtype=dtype)

    def _write_sigmond_skeleton(
        self, hdf5_file: h5py.File, group_name: str, fidentifier: str, header_xml: str
    ) -> h5py.Group:
        """
        Write the common Sigmond HDF5 scaffolding and return the empty ``Values`` group.

        Layout: ``/Info`` (FIdentifier + Endianness) and ``/<group_name>``
        (Header + IncludeCKS + Values). The Header lives only in the data group,
        never in ``Info`` — matching real Sigmond files.
        """
        info_group = hdf5_file.create_group("Info")
        self._fixed_str_dataset(info_group, "FIdentifier", fidentifier)
        self._fixed_str_dataset(info_group, "Endianness", "L")

        return self._write_data_group_skeleton(hdf5_file, group_name, header_xml)

    def _write_data_group_skeleton(
        self, hdf5_file: h5py.File, group_name: str, header_xml: str
    ) -> h5py.Group:
        """Create one Sigmond data group and return its empty ``Values`` group."""

        data_group = hdf5_file.create_group(group_name)
        self._fixed_str_dataset(data_group, "Header", header_xml)
        self._fixed_str_dataset(data_group, "IncludeCKS", "N")
        return data_group.create_group("Values")

    def _prepare_output(self, filename: str, overwrite: bool) -> None:
        """Guard an output path: raise if it exists and ``overwrite`` is False, else back it up."""
        if Path(filename).exists():
            if not overwrite:
                raise FileExistsError(
                    f"File {filename} already exists. Use overwrite=True to overwrite."
                )
            self._create_numbered_backup(filename)

    @staticmethod
    def _validate_write_mode(mode: str) -> WriteMode:
        if mode not in {"a", "w"}:
            raise ValueError(f"Unsupported write mode {mode!r}; expected 'a' or 'w'")
        return cast(WriteMode, mode)

    def _component_keys_for_observable(self, observable_info: ObservableInfo) -> frozenset[str]:
        """All possible component keys for one logical observable."""
        return frozenset(
            {
                self._dataset_key_for_observable(observable_info, "re"),
                self._dataset_key_for_observable(observable_info, "im"),
            }
        )

    def _prepare_payload(
        self,
        observables: Sequence[SigmondSampling] | Sequence[SigmondBins],
        group: str,
        kind: FileKind,
    ) -> _HDF5WritePayload:
        """Validate and normalize one data group before touching its output file."""
        if not observables:
            noun = "samplings" if kind == "samplings" else "bins"
            raise ValueError(f"No {noun} provided")

        group_name = clean_group(group)
        if not group_name:
            raise ValueError("HDF5 group must not be empty")
        first = observables[0]
        ensemble_info = first.observable_info.ensemble_info

        if kind == "samplings":
            if not isinstance(first, SigmondSampling):
                raise TypeError(f"Item 0 is a {type(first).__name__}, expected SigmondSampling")
            sampling_info = first.sampling_info
            header_xml = self._generate_header_xml(ensemble_info, sampling_info)
        else:
            if not isinstance(first, SigmondBins):
                raise TypeError(f"Item 0 is a {type(first).__name__}, expected SigmondBins")
            num_bins = first.num_bins
            header_xml = self._generate_bins_header_xml(ensemble_info)

        dataset_rows: list[tuple[DatasetRow, ...]] = []
        component_keys: list[frozenset[str]] = []
        incoming_component_keys: set[str] = set()
        normalized: list[SigmondSampling | SigmondBins] = []

        for i, observable in enumerate(observables):
            expected_type = SigmondSampling if kind == "samplings" else SigmondBins
            if not isinstance(observable, expected_type):
                raise TypeError(
                    f"Item {i} is a {type(observable).__name__}, "
                    f"expected {expected_type.__name__}"
                )
            if observable.observable_info.ensemble_info != ensemble_info:
                raise ValueError(f"Item {i} has incompatible ensemble info")
            if kind == "samplings" and observable.sampling_info != sampling_info:
                raise ValueError(f"Sampling {i} has incompatible sampling info")
            if kind == "bins" and observable.num_bins != num_bins:
                raise ValueError(f"Bins {i} has {observable.num_bins} bins; expected {num_bins}")

            observable_component_keys = self._component_keys_for_observable(
                observable.observable_info
            )
            if incoming_component_keys.intersection(observable_component_keys):
                raise ValueError(
                    f"Duplicate logical observable {observable.observable_info.name!r} "
                    "in write batch"
                )
            incoming_component_keys.update(observable_component_keys)
            component_keys.append(observable_component_keys)
            data = observable.data if kind == "samplings" else observable._as_numpy()
            dataset_rows.append(
                tuple(
                    self._observable_datasets(
                        observable.observable_info, data, observable.is_complex
                    )
                )
            )
            normalized.append(observable)

        return _HDF5WritePayload(
            group_name,
            kind,
            tuple(normalized),
            header_xml,
            tuple(dataset_rows),
            tuple(component_keys),
        )

    @staticmethod
    def _fidentifier_for_kind(kind: FileKind) -> str:
        return "Sigmond--SamplingsFile" if kind == "samplings" else "Sigmond--BinsFile"

    def _validate_existing_group(self, filename: str, payload: _HDF5WritePayload) -> None:
        existing = SigmondLoader(filename=filename, group=payload.group).observables
        first = payload.observables[0]
        if payload.kind == "samplings":
            if existing.sampling_info != first.sampling_info:
                raise ValueError("New samplings have incompatible sampling info with file")
            if existing.ensemble_info != first.observable_info.ensemble_info:
                raise ValueError("New samplings have incompatible ensemble info with file")
        else:
            existing_first = existing[0]
            if existing_first.observable_info.ensemble_info != first.observable_info.ensemble_info:
                raise ValueError("New bins have incompatible ensemble info with file")
            if existing_first.num_bins != first.num_bins:
                raise ValueError("New bins have an incompatible number of bins with file")

    def _preflight_append(
        self, filename: str, payloads: Sequence[_HDF5WritePayload], overwrite: bool
    ) -> set[str]:
        """Validate an append completely and return the existing data-group paths."""
        is_valid, file_kind, available_groups = verify_sigmond_hdf5(filename)
        if not is_valid:
            raise ValueError(f"File {filename} is not a valid Sigmond HDF5 file")
        expected_kind = payloads[0].kind
        if file_kind != expected_kind:
            raise ValueError(f"Cannot append {expected_kind} to a {file_kind} file")

        existing_groups = set(available_groups or [])
        with h5py.File(filename, "r") as hdf5_file:
            for payload in payloads:
                if payload.group in hdf5_file and payload.group not in existing_groups:
                    raise ValueError(f"Values group not found in root group {payload.group}")
                if payload.group not in existing_groups:
                    continue
                if overwrite:
                    continue
                values_group = hdf5_file[payload.group]["Values"]
                for keys in payload.component_keys:
                    collisions = keys.intersection(values_group.keys())
                    if collisions:
                        raise FileExistsError(
                            f"Observable dataset {next(iter(collisions))!r} already exists. "
                            "Use overwrite=True to replace it."
                        )
        for payload in payloads:
            if payload.group in existing_groups:
                self._validate_existing_group(filename, payload)
        return existing_groups

    def _write_payload_to_group(
        self,
        hdf5_file: h5py.File,
        payload: _HDF5WritePayload,
        *,
        group_exists: bool,
        overwrite: bool,
    ) -> None:
        if group_exists:
            data_group = hdf5_file[payload.group]
            values_group = data_group["Values"]
            meta = obsmeta.read(data_group)
        else:
            values_group = self._write_data_group_skeleton(
                hdf5_file, payload.group, payload.header_xml
            )
            data_group = values_group.parent
            meta = {}

        for keys, rows in zip(payload.component_keys, payload.dataset_rows, strict=True):
            if overwrite:
                for key in keys:
                    if key in values_group:
                        del values_group[key]
                    meta.pop(key, None)
            for key, arr, attrs in rows:
                values_group.create_dataset(key, data=arr)
                meta[key] = obsmeta.fields_for(arr, attrs)
        obsmeta.write(data_group, meta)

    def _write_payloads(
        self,
        filename: str,
        payloads: Sequence[_HDF5WritePayload],
        *,
        overwrite: bool,
        mode: WriteMode,
    ) -> None:
        """Commit one or more prevalidated groups with one backup and file open."""
        if not payloads:
            raise ValueError("No HDF5 groups provided")
        kinds = {payload.kind for payload in payloads}
        if len(kinds) != 1:
            raise ValueError("Cannot mix bins and samplings in one HDF5 file")
        groups = [payload.group for payload in payloads]
        if len(groups) != len(set(groups)):
            raise ValueError("Duplicate HDF5 group in write batch")

        append_existing = mode == "a" and Path(filename).exists()
        if append_existing:
            existing_groups = self._preflight_append(filename, payloads, overwrite)
            self._create_numbered_backup(filename)
            with h5py.File(filename, "r+") as hdf5_file:
                for payload in payloads:
                    self._write_payload_to_group(
                        hdf5_file,
                        payload,
                        group_exists=payload.group in existing_groups,
                        overwrite=overwrite,
                    )
            return

        self._prepare_output(filename, overwrite)
        with h5py.File(filename, "w") as hdf5_file:
            first, *remaining = payloads
            self._write_sigmond_skeleton(
                hdf5_file,
                first.group,
                self._fidentifier_for_kind(first.kind),
                first.header_xml,
            )
            self._write_payload_to_group(
                hdf5_file, first, group_exists=True, overwrite=overwrite
            )
            for payload in remaining:
                self._write_payload_to_group(
                    hdf5_file, payload, group_exists=False, overwrite=overwrite
                )

    @staticmethod
    def hdf5_output_path(input_filename: str | Path, output_filename: str | Path) -> Path:
        """
        Resolve an output filename for HDF5 writes.

        A requested ``.h5``/``.hdf5`` suffix is preserved. If the output path has
        no HDF5 suffix, use the input file's HDF5 suffix when it has one, falling
        back to ``.hdf5`` for legacy fstream inputs.
        """
        outpath = Path(output_filename)
        if outpath.suffix.lower() in HDF5_EXTENSIONS:
            return outpath

        input_suffix = Path(input_filename).suffix
        if input_suffix.lower() in HDF5_EXTENSIONS:
            return outpath.with_suffix(input_suffix)

        return outpath.with_suffix(".hdf5")

    # ──────────────────────────────────────────────────────────────────────
    # Public write API
    # ──────────────────────────────────────────────────────────────────────

    def write_file(
        self,
        filename: str,
        samplings: list[SigmondSampling],
        group: str = DEFAULT_GROUP,
        overwrite: bool = False,
        mode: WriteMode = "w",
    ) -> Path:
        """
        Write samplings to an HDF5 file.

        Returns the path actually written.
        """
        outpath = self.hdf5_output_path(filename, filename)
        self.write_hdf5(str(outpath), samplings, group, overwrite, mode)
        return outpath

    def write_hdf5(
        self,
        filename: str,
        samplings: list[SigmondSampling],
        group: str = DEFAULT_GROUP,
        overwrite: bool = False,
        mode: WriteMode = "w",
    ) -> None:
        """
        Write samplings to an HDF5 samplings file in Sigmond format.

        Args:
            filename: Output file path.
            samplings: SigmondSampling objects (all sharing ensemble/sampling info).
            group: HDF5 root group; may be nested (e.g. "isotriplet/P0A1g").
            overwrite: In ``"w"`` mode, replace an existing file. In ``"a"``
                mode, replace colliding observable datasets.
            mode: ``"w"`` creates a new file; ``"a"`` preserves an existing
                file and appends to (or creates) ``group``.
        """
        mode = self._validate_write_mode(mode)
        payload = self._prepare_payload(samplings, group, "samplings")
        self._write_payloads(filename, [payload], overwrite=overwrite, mode=mode)
        logger.info(
            f"Wrote {len(samplings)} samplings to {filename} at path '/{payload.group}/'"
        )

    def write_bins_hdf5(
        self,
        filename: str,
        bins_list: list[SigmondBins],
        group: str = DEFAULT_GROUP,
        overwrite: bool = False,
        mode: WriteMode = "w",
    ) -> None:
        """
        Write a collection of ``SigmondBins`` to an HDF5 bins file.

        The output mirrors a real Sigmond bins file: FIdentifier
        ``Sigmond--BinsFile``, a ``SigmondBinsFile`` header, and one dataset per
        observable component under ``<group>/Values``. Complex bins are split
        into Re/Im datasets. CorrT/raw-XML names round-trip verbatim because keys
        are built from the observable name (see ``_dataset_key_for_observable``).
        """
        mode = self._validate_write_mode(mode)
        payload = self._prepare_payload(bins_list, group, "bins")
        self._write_payloads(filename, [payload], overwrite=overwrite, mode=mode)
        logger.info(
            f"Wrote {len(bins_list)} bins observables to {filename} "
            f"at path '/{payload.group}/'"
        )

    def write_groups_hdf5(
        self,
        filename: str,
        groups: Mapping[
            str, Sequence[SigmondSampling] | Sequence[SigmondBins]
        ],
        overwrite: bool = False,
        mode: WriteMode = "w",
    ) -> None:
        """Write multiple HDF5 data groups as one validated operation."""
        mode = self._validate_write_mode(mode)
        payloads: list[_HDF5WritePayload] = []
        for group, observables in groups.items():
            if not observables:
                raise ValueError(f"No observables provided for group {group!r}")
            if all(isinstance(observable, SigmondSampling) for observable in observables):
                kind: FileKind = "samplings"
            elif all(isinstance(observable, SigmondBins) for observable in observables):
                kind = "bins"
            else:
                raise TypeError(f"Group {group!r} contains mixed observable types")
            payloads.append(self._prepare_payload(observables, group, kind))
        self._write_payloads(filename, payloads, overwrite=overwrite, mode=mode)

    # ──────────────────────────────────────────────────────────────────────
    # In-place editing
    # ──────────────────────────────────────────────────────────────────────

    def append_to_file(
        self,
        filename: str,
        new_samplings: list[SigmondSampling],
        overwrite: bool = True,
        group: str | None = None,
    ) -> Path:
        """
        Append new samplings to an existing file.

        HDF5 files are appended in place. fstream (.smp) files are converted once
        to a sibling ``file.hdf5`` (the original .smp is left unchanged) which is
        then appended to.

        Args:
            filename: Path to the existing file.
            new_samplings: Samplings to add.
            overwrite: Replace observables whose keys already exist.
            group: HDF5 root group (None = auto-detect).

        Returns:
            Path to the modified HDF5 file (the original for HDF5 input, or the
            converted ``file.hdf5`` sibling for fstream input).
        """
        if not Path(filename).exists():
            raise FileNotFoundError(f"File {filename} does not exist")

        hdf5_filename, group = self._ensure_hdf5_format(filename, group)
        self.write_hdf5(
            str(hdf5_filename),
            new_samplings,
            group=group,
            overwrite=overwrite,
            mode="a",
        )
        return Path(hdf5_filename)

    def modify_observable(
        self,
        filename: str,
        observable_name: str,
        observable_index: int,
        new_data: np.ndarray,
        group: str | None = None,
    ) -> Path:
        """
        Replace the data of one observable in an existing file.

        HDF5 files are modified in place. fstream (.smp) files are converted once
        to a sibling ``file.hdf5`` (the original .smp is left unchanged) which is
        then modified.

        Only the target observable's datasets are rewritten; other observables
        and other root groups in the file are left untouched.

        Returns the path to the modified HDF5 file.
        """
        if not Path(filename).exists():
            raise FileNotFoundError(f"File {filename} does not exist")

        hdf5_filename, group = self._ensure_hdf5_format(filename, group)
        loader = SigmondLoader()
        loader.load_file(str(hdf5_filename), group=group)

        original = loader.observables.find(name=observable_name, index=observable_index)
        if original is None:
            available = [
                f"{s.observable_info.name} {s.observable_info.index}"
                for s in list(loader.observables)[:5]
            ]
            raise ValueError(
                f"Observable {observable_name} {observable_index} not found in file. "
                f"Available observables: {available}..."
            )

        # Replace just this observable's dataset(s) in place, preserving sibling groups.
        modified = SigmondSampling(
            new_data, original.observable_info, original.sampling_info, original.is_complex
        )
        self.write_hdf5(
            str(hdf5_filename), [modified], overwrite=True, group=group, mode="a"
        )
        return Path(hdf5_filename)

    def convert_format(
        self,
        input_filename: str,
        output_filename: str,
        output_format: str = "hdf5",
        group: str = DEFAULT_GROUP,
        overwrite: bool = False,
    ) -> Path:
        """
        Convert a Sigmond file to HDF5, preserving whether it stores samplings or bins.

        ``output_format`` must be ``"hdf5"`` (the only supported output). Returns
        the path to the converted file.
        """
        if output_format != "hdf5":
            raise ValueError(f"Unsupported output format: {output_format}")

        outpath = self.hdf5_output_path(input_filename, output_filename)

        # Raw bins carry no SamplingInfo, so they must use the bins writer.
        loader = SigmondLoader(filename=input_filename)
        observables = list(loader.observables)
        if not observables:
            raise ValueError(f"No observables found in {input_filename}")

        if loader.file_kind == "bins" or all(isinstance(o, SigmondBins) for o in observables):
            self.write_bins_hdf5(str(outpath), observables, group, overwrite=overwrite)
        elif loader.file_kind == "samplings" or all(
            isinstance(o, SigmondSampling) for o in observables
        ):
            self.write_hdf5(str(outpath), observables, group, overwrite=overwrite)
        else:
            raise TypeError(
                "Loaded observables are a mixed or unsupported set of types and "
                "cannot be converted to HDF5."
            )

        return outpath
