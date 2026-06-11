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
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np

from .loader import (
    DEFAULT_ROOT_PATH,
    SigmondLoader,
    clean_hdf5_path,
    is_hdf5_file,
    verify_sigmond_hdf5,
)
from ..bins import SigmondBins
from ..sampling import EnsembleInfo, ObservableInfo, SamplingInfo, SigmondSampling

logger = logging.getLogger(__name__)


class SigmondWriter:
    """
    Writer for Sigmond samplings and bins files, in HDF5 format.

    Writing is always HDF5: ``write_hdf5``/``write_bins_hdf5`` create new files,
    while ``append_to_file`` and ``modify_observable`` edit an existing HDF5 file
    in place (fstream input is first converted to a sibling ``.hdf5``). Optional
    numbered backups protect files before they are overwritten or modified.
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

    def _ensure_hdf5_format(
        self, filename: str, hdf5_root_path: str | None = None
    ) -> tuple[Path, str]:
        """
        Resolve the HDF5 file that in-place mutating operations should act on.

        - HDF5 input is returned unchanged so the caller mutates it in place.
        - fstream (.smp) input cannot be written back in its native format, so it
          is converted once to a sibling ``file.hdf5`` (the original is left
          untouched) and that path is returned.

        Reading legacy fstream still goes through ``sigmond_query``; HDF5 reads
        use the fast h5py path.

        Returns:
            (hdf5_filename, root_path) ready for in-place operations.
        """
        filepath = Path(filename)

        # Already a valid Sigmond HDF5 file -> operate on it directly.
        if is_hdf5_file(str(filepath)):
            is_valid, _, available_paths = verify_sigmond_hdf5(str(filepath))
            if not is_valid:
                raise ValueError(f"File {filepath} is not a valid Sigmond HDF5 file")
            root_path = hdf5_root_path
            if root_path is None:
                if not available_paths:
                    raise ValueError(f"No data paths found in HDF5 file {filepath}")
                root_path = available_paths[0]
            return filepath, clean_hdf5_path(root_path)

        # fstream input: convert once to a sibling .hdf5, preserving the original.
        hdf5_filename = filepath.with_suffix(".hdf5")
        if hdf5_filename == filepath:
            raise ValueError(f"Cannot derive an .hdf5 sibling for {filepath}")

        loader = SigmondLoader()
        root_path = clean_hdf5_path(hdf5_root_path or DEFAULT_ROOT_PATH)

        logger.info(f"Converting fstream {filepath} -> {hdf5_filename} for in-place processing...")
        loader.load_file(str(filepath), hdf5_path=root_path)
        observables = list(loader.observables)
        if not observables:
            raise ValueError(f"No observables found in {filepath}")

        # Preserve bins vs. samplings so the converted file keeps its native kind.
        if loader.file_kind == "bins" or all(isinstance(o, SigmondBins) for o in observables):
            self.write_bins_hdf5(str(hdf5_filename), observables, root_path, overwrite=True)
        else:
            self.write_hdf5(str(hdf5_filename), observables, root_path, overwrite=True)

        logger.info(f"Conversion complete. Original fstream {filepath} left unchanged.")
        return hdf5_filename, root_path

    def _append_mcbins_info(self, root: ET.Element, ensemble_info: EnsembleInfo) -> None:
        """Append the shared ``<MCBinsInfo>`` block (ensemble + optional tweaks) to ``root``."""
        bins_elem = ET.SubElement(root, "MCBinsInfo")
        ET.SubElement(bins_elem, "MCEnsembleInfo").text = ensemble_info.name
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
        types) gets its metadata persisted; the loader reverses this on read.
        """
        to_attrs = getattr(observable_info, "to_attrs", None)
        return to_attrs() if to_attrs else None

    @staticmethod
    def _write_attrs(dataset: h5py.Dataset, attrs: dict | None) -> None:
        """Attach ``attrs`` to ``dataset`` (list values become vlen UTF-8 arrays)."""
        if not attrs:
            return
        for key, value in attrs.items():
            if isinstance(value, list):
                dataset.attrs.create(key, value, dtype=h5py.string_dtype("utf-8"))
            else:
                dataset.attrs[key] = value

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

    # ──────────────────────────────────────────────────────────────────────
    # Public write API
    # ──────────────────────────────────────────────────────────────────────

    def write_file(
        self,
        filename: str,
        samplings: list[SigmondSampling],
        root_path: str = DEFAULT_ROOT_PATH,
        overwrite: bool = False,
    ) -> Path:
        """
        Write samplings to an HDF5 file (the ``.hdf5`` extension is enforced).

        Returns the path actually written.
        """
        outpath = Path(filename).with_suffix(".hdf5")
        self.write_hdf5(str(outpath), samplings, root_path, overwrite)
        return outpath

    def write_hdf5(
        self,
        filename: str,
        samplings: list[SigmondSampling],
        root_path: str = DEFAULT_ROOT_PATH,
        overwrite: bool = False,
    ) -> None:
        """
        Write samplings to an HDF5 samplings file in Sigmond format.

        Args:
            filename: Output file path.
            samplings: SigmondSampling objects (all sharing ensemble/sampling info).
            root_path: HDF5 data group path; may be nested (e.g. "isotriplet/P0A1g").
            overwrite: Overwrite (and back up) an existing file.
        """
        self._prepare_output(filename, overwrite)
        if not samplings:
            raise ValueError("No samplings provided")

        group_name = clean_hdf5_path(root_path)
        ensemble_info = samplings[0].observable_info.ensemble_info
        sampling_info = samplings[0].sampling_info

        for i, sampling in enumerate(samplings):
            if sampling.sampling_info != sampling_info:
                raise ValueError(f"Sampling {i} has incompatible sampling info")
            if sampling.observable_info.ensemble_info != ensemble_info:
                raise ValueError(f"Sampling {i} has incompatible ensemble info")

        header_xml = self._generate_header_xml(ensemble_info, sampling_info)
        with h5py.File(filename, "w") as hdf5_file:
            values_group = self._write_sigmond_skeleton(
                hdf5_file, group_name, "Sigmond--SamplingsFile", header_xml
            )
            for sampling in samplings:
                for key, arr, attrs in self._observable_datasets(
                    sampling.observable_info, sampling.data, sampling.is_complex
                ):
                    self._write_attrs(values_group.create_dataset(key, data=arr), attrs)

        logger.info(f"Wrote {len(samplings)} samplings to {filename} at path '/{group_name}/'")

    def write_bins_hdf5(
        self,
        filename: str,
        bins_list: list[SigmondBins],
        root_path: str = DEFAULT_ROOT_PATH,
        overwrite: bool = False,
    ) -> None:
        """
        Write a collection of ``SigmondBins`` to an HDF5 bins file.

        The output mirrors a real Sigmond bins file: FIdentifier
        ``Sigmond--BinsFile``, a ``SigmondBinsFile`` header, and one dataset per
        observable component under ``<root_path>/Values``. Complex bins are split
        into Re/Im datasets. CorrT/raw-XML names round-trip verbatim because keys
        are built from the observable name (see ``_dataset_key_for_observable``).
        """
        self._prepare_output(filename, overwrite)
        if not bins_list:
            raise ValueError("No bins provided")

        group_name = clean_hdf5_path(root_path)
        ensemble_info = bins_list[0].observable_info.ensemble_info
        num_bins = bins_list[0].num_bins

        for i, b in enumerate(bins_list):
            if not isinstance(b, SigmondBins):
                raise TypeError(f"Item {i} is a {type(b).__name__}, expected SigmondBins")
            if b.observable_info.ensemble_info != ensemble_info:
                raise ValueError(f"Bins {i} has incompatible ensemble info")
            if b.num_bins != num_bins:
                raise ValueError(f"Bins {i} has {b.num_bins} bins; expected {num_bins}")

        header_xml = self._generate_bins_header_xml(ensemble_info)
        with h5py.File(filename, "w") as hdf5_file:
            values_group = self._write_sigmond_skeleton(
                hdf5_file, group_name, "Sigmond--BinsFile", header_xml
            )
            for b in bins_list:
                for key, arr, attrs in self._observable_datasets(
                    b.observable_info, b._as_numpy(), b.is_complex
                ):
                    self._write_attrs(values_group.create_dataset(key, data=arr), attrs)

        logger.info(
            f"Wrote {len(bins_list)} bins observables to {filename} at path '/{group_name}/'"
        )

    # ──────────────────────────────────────────────────────────────────────
    # In-place editing
    # ──────────────────────────────────────────────────────────────────────

    def append_to_file(
        self,
        filename: str,
        new_samplings: list[SigmondSampling],
        overwrite: bool = True,
        hdf5_root_path: str | None = None,
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
            hdf5_root_path: HDF5 root path (None = auto-detect).

        Returns:
            Path to the modified HDF5 file (the original for HDF5 input, or the
            converted ``file.hdf5`` sibling for fstream input).
        """
        if not Path(filename).exists():
            raise FileNotFoundError(f"File {filename} does not exist")

        hdf5_filename, root_path = self._ensure_hdf5_format(filename, hdf5_root_path)
        self._create_numbered_backup(hdf5_filename)
        self._append_to_hdf5(hdf5_filename, new_samplings, overwrite, root_path)
        return Path(hdf5_filename)

    def _append_to_hdf5(
        self,
        filename: str | Path,
        new_samplings: list[SigmondSampling],
        overwrite: bool,
        root_path: str | None = None,
    ) -> None:
        if not new_samplings:
            raise ValueError("No samplings provided")

        # Resolve and validate the target data group (read-only pass).
        with h5py.File(filename, "r") as f:
            if root_path is None:
                data_groups = [key for key in f.keys() if key != "Info"]
                if not data_groups:
                    raise ValueError("No data groups found in HDF5 file")
                root_path = data_groups[0]
            if root_path not in f:
                raise ValueError(f"Data group {root_path} not found in file")
            if "Values" not in f[root_path]:
                raise ValueError(f"Values group not found in data group {root_path}")

        # Compatibility check runs with the file closed to avoid locking issues.
        self._validate_samplings_compatibility(str(filename), new_samplings, root_path)

        with h5py.File(filename, "r+") as f:
            values_group = f[root_path]["Values"]
            for sampling in new_samplings:
                oi = sampling.observable_info
                for key, arr, attrs in self._observable_datasets(
                    oi, sampling.data, sampling.is_complex
                ):
                    if key in values_group:
                        if not overwrite:
                            raise FileExistsError(
                                f"Observable {oi.name}[{oi.index}] already exists. "
                                f"Use overwrite=True to replace it."
                            )
                        del values_group[key]
                    self._write_attrs(values_group.create_dataset(key, data=arr), attrs)

    def _validate_samplings_compatibility(
        self, filename: str, new_samplings: list[SigmondSampling], root_path: str | None = None
    ) -> None:
        """
        Ensure ``new_samplings`` share the existing file's ensemble and sampling info.

        Raises ValueError on any mismatch (with the file or among themselves).
        """
        if not new_samplings:
            return

        loader = SigmondLoader(filename=filename, hdf5_path=root_path)
        existing = loader.observables
        if not existing:
            raise ValueError(f"No existing samplings found in {filename}")

        try:
            ref_ensemble = existing.ensemble_info
            ref_sampling_info = existing.sampling_info
        except Exception as e:
            raise ValueError(f"Failed to read existing file for compatibility check: {e}")

        first_new = new_samplings[0]
        for i, samp in enumerate(new_samplings):
            if samp.sampling_info != ref_sampling_info:
                raise ValueError(f"New sampling {i} has incompatible sampling info with file")
            if samp.observable_info.ensemble_info != ref_ensemble:
                raise ValueError(f"New sampling {i} has incompatible ensemble info with file")
            if samp.sampling_info != first_new.sampling_info:
                raise ValueError(f"New sampling {i} has inconsistent sampling info")
            if samp.observable_info.ensemble_info != first_new.observable_info.ensemble_info:
                raise ValueError(f"New sampling {i} has inconsistent ensemble info")

    def modify_observable(
        self,
        filename: str,
        observable_name: str,
        observable_index: int,
        new_data: np.ndarray,
        hdf5_root_path: str | None = None,
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

        hdf5_filename, root_path = self._ensure_hdf5_format(filename, hdf5_root_path)
        self._create_numbered_backup(str(hdf5_filename))

        loader = SigmondLoader()
        loader.load_file(str(hdf5_filename), hdf5_path=root_path)

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

        # Replace just this observable's dataset(s) in place (overwrite=True), so
        # write_hdf5's truncating "w" mode never wipes sibling root groups.
        modified = SigmondSampling(
            new_data, original.observable_info, original.sampling_info, original.is_complex
        )
        self._append_to_hdf5(hdf5_filename, [modified], overwrite=True, root_path=root_path)
        return Path(hdf5_filename)

    def convert_format(
        self,
        input_filename: str,
        output_filename: str,
        output_format: str = "hdf5",
        hdf5_root_path: str = DEFAULT_ROOT_PATH,
        overwrite: bool = False,
    ) -> Path:
        """
        Convert a Sigmond file to HDF5, preserving whether it stores samplings or bins.

        ``output_format`` must be ``"hdf5"`` (the only supported output). Returns
        the path to the converted file.
        """
        if output_format != "hdf5":
            raise ValueError(f"Unsupported output format: {output_format}")

        outpath = Path(output_filename).with_suffix(".hdf5")

        # Raw bins carry no SamplingInfo, so they must use the bins writer.
        loader = SigmondLoader(filename=input_filename)
        observables = list(loader.observables)
        if not observables:
            raise ValueError(f"No observables found in {input_filename}")

        if loader.file_kind == "bins" or all(isinstance(o, SigmondBins) for o in observables):
            self.write_bins_hdf5(str(outpath), observables, hdf5_root_path, overwrite=overwrite)
        elif loader.file_kind == "samplings" or all(
            isinstance(o, SigmondSampling) for o in observables
        ):
            self.write_hdf5(str(outpath), observables, hdf5_root_path, overwrite=overwrite)
        else:
            raise TypeError(
                "Loaded observables are a mixed or unsupported set of types and "
                "cannot be converted to HDF5."
            )

        return outpath
