"""
Loader module for Sigmond samplings files.
"""

import logging
import os
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypeAlias, cast

if TYPE_CHECKING:
    from .spec import SamplingDataSourceResolved, SamplingDataResolved

import h5py
import numpy as np

from ..bins import SigmondBins
from ..ensemble_collection import MultiEnsembleCollection, SingleEnsembleCollection
from ..info import EnsembleInfo, KnownEnsembles, ObservableInfo, SamplingInfo, obs_kind_class
from ..lazy import (
    HDF5ObservableRecord,
    LazySigmondBins,
    LazySigmondSampling,
    _FileRef,
)
from ..sampling import SigmondSampling
from . import obsmeta

logger = logging.getLogger(__name__)
SIGMOND_QUERY_CMD = "sigmond_query"

# Canonical default HDF5 root group, shared by the loader and writer so a single
# convention is used everywhere a group is not given explicitly.
DEFAULT_GROUP = "data"

# Maps the global /Info/FIdentifier string to the kind of file it denotes.
_HDF5_FIDENTIFIERS = {
    "Sigmond--SamplingsFile": "samplings",
    "Sigmond--BinsFile": "bins",
}


def clean_group(group: str) -> str:
    """Normalize an HDF5 root group name by stripping leading/trailing slashes."""
    return group.strip("/")


def is_hdf5_file(filename: str) -> bool:
    """Return True if ``filename`` can be opened as an HDF5 file."""
    try:
        with h5py.File(filename, "r"):
            return True
    except OSError:
        return False


def discover_root_groups(h5: h5py.File) -> list[str]:
    """
    Full paths of every Sigmond root group in an open HDF5 file.

    A root group is any group containing a ``Values`` subgroup, found at *any*
    depth, so nested paths such as ``isotriplet/P0A1g`` (the form used in the
    Sigmond HDF5 spec) are discovered. The global ``Info`` group is excluded
    automatically since it has no ``Values`` child.
    """
    found: list[str] = []

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Group) and isinstance(obj.get("Values"), h5py.Group):
            found.append(name)

    h5.visititems(visit)
    return found


def verify_sigmond_hdf5(filename: str) -> tuple[bool, str | None, list[str] | None]:
    """
    Verify an HDF5 file is a valid Sigmond samplings or bins file.

    Returns ``(is_valid, file_kind, available_groups)`` where ``file_kind`` is
    ``'samplings'`` or ``'bins'`` and ``available_groups`` are the full root group
    names (see :func:`discover_root_groups`). Returns ``(False, None, None)`` if
    the file is not a recognized Sigmond HDF5 file.
    """
    try:
        with h5py.File(filename, "r") as f:
            if "Info" not in f or "FIdentifier" not in f["Info"]:
                return False, None, None
            fid = f["Info"]["FIdentifier"][()].decode("utf-8")
            file_kind = _HDF5_FIDENTIFIERS.get(fid)
            if file_kind is None:
                return False, None, None
            return True, file_kind, discover_root_groups(f)
    except Exception as e:
        logger.debug(f"HDF5 verification failed: {e}")
        return False, None, None


class SigmondLoader:
    """
    Loader for Sigmond samplings files.

    Automatically detects file format (HDF5 or fstream) and uses the appropriate method:
    - HDF5 files: Direct h5py reading (fast)
    - Fstream files: sigmond_query tool (slower but supports legacy format)

    For HDF5 files with a single root group, the group is auto-detected.
    For HDF5 files with multiple groups, you must specify the group parameter.

    Provides queryable SingleEnsembleCollection of all loaded observables:
        loader.observables - All loaded samplings
        loader.group - The HDF5 root group used (None for fstream files)

    Use SingleEnsembleCollection filtering methods to query:
        loader.observables.filter(index=0)
        loader.observables.find(lambda obs: 'PSQ' in obs.name)
        loader.observables.find(lambda obs: re.search(r'PSQ.*', obs.name))
    """

    def __init__(
        self,
        filename: str | None = None,
        group: str | None = None,
        lazy: bool = False,
    ):
        """
        Initialize the loader.

        Args:
            filename: Path to the samplings or bins file to load upon construction (optional)
            group: For HDF5 files, the root group to read (default: None = auto-detect)
                      If None and file has a single root group, it will be used automatically.
                      If None and file has multiple root groups, an error will be raised.
            lazy: If True, only read header + dataset names up front and defer reading
                  each observable's sample array until its data is first accessed.
                  Supported for HDF5 files only; fstream files raise NotImplementedError.
        """
        # Single source of truth for data
        self._filename = None
        self._group = group
        self._lazy = lazy
        self._file_kind: str | None = None  # "samplings" or "bins"
        self._all_samplings = SingleEnsembleCollection([])

        # Load file if provided
        if filename:
            self.load_file(str(filename), self._group, lazy=lazy)

    @classmethod
    def from_resolved(
        cls,
        resolved: "SamplingDataSourceResolved",
        lazy: bool = False,
    ) -> Self:
        return cls(filename=str(resolved.file), group=resolved.group, lazy=lazy)

    @property
    def observables(self) -> SingleEnsembleCollection:
        """Access the loaded samplings collection."""
        return self._all_samplings

    def energy_observables(
        self,
        *,
        skip_missing_particles: bool = True,
        return_type: str | None = None,
    ):
        """
        Return loaded observables as an energy-level collection.

        This is an explicit semantic view over ``observables``; raw loader output
        remains unchanged.
        """
        return self.observables.as_energy_levels(
            skip_missing_particles=skip_missing_particles,
            return_type=return_type,
        )

    @property
    def group(self) -> str | None:
        """Get the HDF5 root group used for loading (None for fstream files)."""
        return self._group

    @property
    def file_kind(self) -> str | None:
        """Return 'samplings' or 'bins' for the most recently loaded file."""
        return self._file_kind

    def _load_from_hdf5(self, filename: str, group: str = "samplings") -> SingleEnsembleCollection:
        """
        Load samplings directly from HDF5 file using h5py.

        Args:
            filename: Path to the HDF5 file
            group: Root group in HDF5 file (default: "samplings")

        Returns:
            SingleEnsembleCollection of loaded samplings
        """
        ensemble_info, sampling_info, parsed = self._read_hdf5_values(filename, group)
        self._check_file_kind(filename, "samplings", sampling_info)
        samplings_list = self._build_samplings_list(
            [p[0] for p in parsed], [p[1] for p in parsed], sampling_info
        )
        return SingleEnsembleCollection(samplings_list)

    def _load_bins_from_hdf5(self, filename: str, group: str) -> SingleEnsembleCollection:
        """
        Load Sigmond bins directly from HDF5 file using h5py.

        Returns:
            SingleEnsembleCollection of SigmondBins objects.
        """
        ensemble_info, sampling_info, parsed = self._read_hdf5_values(filename, group)
        self._check_file_kind(filename, "bins", sampling_info)
        bins_list = self._build_bins_list([p[0] for p in parsed], [p[1] for p in parsed])
        return SingleEnsembleCollection(bins_list)

    @staticmethod
    def _check_file_kind(
        filename: str, expected_kind: str, sampling_info: SamplingInfo | None
    ) -> None:
        """
        Reject a file whose contents disagree with its declared kind.

        A samplings file must carry sampling info; a bins file must not. Shared by
        the eager and lazy HDF5 loaders so the diagnostic is worded identically.
        """
        if expected_kind == "samplings" and sampling_info is None:
            raise ValueError(f"File {filename} appears to be a bins file, not a samplings file.")
        if expected_kind == "bins" and sampling_info is not None:
            raise ValueError(f"File {filename} appears to be a samplings file, not a bins file.")

    def _iter_hdf5_values(
        self, filename: str, group: str, extract
    ) -> tuple[EnsembleInfo, SamplingInfo | None, list]:
        """
        Open a Sigmond HDF5 file, validate its structure, and map ``extract`` over each
        observable dataset in the ``Values`` group.

        ``extract(obs_info, dataset_name, values_group, fields)`` is invoked once per
        parseable dataset and its return value collected, where ``fields`` is the
        observable's :mod:`obsmeta` row (``None`` for legacy files). Datasets whose
        key cannot be parsed are skipped. This is the single read/validate path
        shared by the eager (full-array) and lazy (name+shape) loaders.

        Observable annotations come from the consolidated ``ObsMeta`` table in one
        read; datasets absent from it (legacy or real Sigmond files) fall back to
        their per-dataset HDF5 attributes.

        Returns:
            (ensemble_info, sampling_info_or_None, [extract(...), ...])
        """
        with h5py.File(filename, "r") as f:
            if group not in f:
                available_groups = [k for k in f.keys() if k != "Info"]
                raise ValueError(
                    f"Group '{group}' not found in HDF5 file. Available groups: {available_groups}"
                )

            root = f[group]
            if "Header" not in root:
                raise ValueError(f"No Header dataset found in {group}")

            header_xml = root["Header"][()].decode("utf-8")
            ensemble_info, sampling_info = self._parse_header_xml(header_xml)

            if "Values" not in root:
                raise ValueError(f"No Values group found in {group}")

            values_group = root["Values"]
            meta = obsmeta.read(root)

            results: list = []
            for dataset_name in values_group.keys():
                try:
                    obs_info = self._parse_observable_key(dataset_name, ensemble_info)
                except (ValueError, NotImplementedError) as e:
                    logger.debug(f"Skipping dataset {dataset_name}: {e}")
                    continue
                fields = meta.get(dataset_name)
                attrs = fields if fields is not None else values_group[dataset_name].attrs
                obs_info = self._apply_obs_attrs(obs_info, attrs)
                results.append(extract(obs_info, dataset_name, values_group, fields))

            if not results:
                raise ValueError("No valid observable data found in file")

        return ensemble_info, sampling_info, results

    @staticmethod
    def _apply_obs_attrs(obs_info: ObservableInfo, attrs) -> ObservableInfo:
        """
        Upgrade a name-parsed ObservableInfo using its dataset attrs, if present.

        Datasets the writer tagged with an ``obs_kind`` are rebuilt as the concrete
        registered type (including non-interacting pairs) so reads are deterministic
        rather than relying on name heuristics. Untagged datasets (old files, real
        Sigmond files, fstream) pass through unchanged, keeping only an explicit
        ``latex_str``.
        """
        cls = obs_kind_class(attrs.get("obs_kind"))
        if cls is not None:
            return cls.from_attrs(obs_info, attrs)
        latex_str = attrs.get("latex_str")
        if latex_str:
            obs_info = obs_info.copy()
            obs_info.latex_str = latex_str
        return obs_info

    def _read_hdf5_values(
        self, filename: str, group: str
    ) -> tuple[EnsembleInfo, SamplingInfo | None, list[tuple[ObservableInfo, np.ndarray]]]:
        """
        Eager read of a Sigmond HDF5 file (samplings or bins): full sample arrays.

        Returns:
            (ensemble_info, sampling_info_or_None, [(obs_info, data), ...])
        """
        return self._iter_hdf5_values(
            filename, group, lambda obs_info, name, vg, fields: (obs_info, vg[name][:])
        )

    def _read_hdf5_index(
        self, filename: str, group: str
    ) -> tuple[
        EnsembleInfo, SamplingInfo | None, list[tuple[ObservableInfo, str, tuple[int, ...]]]
    ]:
        """
        Index phase for lazy loading: read header + ``Values`` dataset names and shapes only.

        No sample arrays are read. Shapes come from the consolidated ``ObsMeta``
        table (so no dataset is opened at all), falling back to the HDF5 dataset
        header for legacy files; either way lazy bins report ``num_bins`` without
        materializing.

        Returns:
            (ensemble_info, sampling_info_or_None, [(obs_info, dataset_name, shape), ...])
        """
        return self._iter_hdf5_values(
            filename,
            group,
            lambda obs_info, name, vg, fields: (
                obs_info,
                name,
                obsmeta.shape_of(fields) or tuple(vg[name].shape),
            ),
        )

    def _group_records(
        self,
        file_ref: _FileRef,
        named_infos: list[tuple[ObservableInfo, str, tuple[int, ...]]],
        sampling_info: SamplingInfo | None,
        file_kind: str,
    ) -> list[HDF5ObservableRecord]:
        """
        Group indexed datasets by (name, index), fusing Re/Im parts into one record.

        Mirrors :meth:`_fuse_re_im`'s grouping but operates purely on dataset names
        and shapes (no array reads). A group with both a Re and an Im part becomes a
        single complex record; a lone part stays real.
        """
        grouped: dict[tuple, dict[str, tuple[ObservableInfo, str, tuple[int, ...]]]] = {}
        for obs_info, dataset_name, shape in named_infos:
            grouped.setdefault((obs_info.name, obs_info.index), {})[obs_info.re_im] = (
                obs_info,
                dataset_name,
                shape,
            )

        records: list[HDF5ObservableRecord] = []
        for parts in grouped.values():
            re_info, re_name, re_shape = parts.get("re", (None, None, None))
            im_info, im_name, im_shape = parts.get("im", (None, None, None))
            obs_info = re_info or im_info
            records.append(
                HDF5ObservableRecord(
                    file=file_ref,
                    observable_info=obs_info,
                    sampling_info=sampling_info,
                    file_kind=file_kind,
                    real_name=re_name,
                    imag_name=im_name,
                    real_shape=re_shape,
                    imag_shape=im_shape,
                )
            )
        return records

    def _load_lazy_from_hdf5(
        self, filename: str, group: str, file_kind: str
    ) -> SingleEnsembleCollection:
        """
        Build a collection of lazy observables from an HDF5 file's index.

        ``file_kind`` is the canonical kind from the file's FIdentifier; it is
        cross-checked against the header so a mismatched file is rejected up front,
        mirroring the eager loaders.
        """
        ensemble_info, sampling_info, named_infos = self._read_hdf5_index(filename, group)
        self._check_file_kind(filename, file_kind, sampling_info)

        file_ref = _FileRef(os.path.abspath(filename), f"{group}/Values")
        records = self._group_records(file_ref, named_infos, sampling_info, file_kind)

        factory = LazySigmondBins if file_kind == "bins" else LazySigmondSampling
        return SingleEnsembleCollection([factory(r) for r in records])

    def _check_sigmond_query(self):
        """Check if sigmond_query is available."""
        try:
            result = subprocess.run(
                [SIGMOND_QUERY_CMD, "-h"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"sigmond_query command failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"sigmond_query not found or not working: {e}")

    def _run_sigmond_query(self, filename: str, options: str) -> str:
        """Run sigmond_query with given options."""
        cmd = [SIGMOND_QUERY_CMD] + options.split() + [filename]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                # For HDF5 files, useful info might be in stdout even with non-zero exit code
                if "bad root path" in result.stdout:
                    raise RuntimeError(f"sigmond_query failed: {result.stdout}")
                else:
                    error_msg = result.stderr if result.stderr else result.stdout
                    raise RuntimeError(f"sigmond_query failed: {error_msg}")
            return result.stdout
        except subprocess.TimeoutExpired:
            raise RuntimeError("sigmond_query timed out")
        except Exception as e:
            raise RuntimeError(f"sigmond_query error with cmd '{' '.join(cmd)}': {e}")

    def check_file_validity(self, filename: str) -> tuple[bool, str | None, list[str] | None]:
        """
        Check if a file is a valid Sigmond samplings file.

        Returns:
            (is_valid, file_type, groups)
            - is_valid: True if file is valid
            - file_type: 'fstream', 'hdf5', or None
            - groups: List of available root groups for HDF5 files, None otherwise
        """
        try:
            output = self._run_sigmond_query(filename, "-i")

            if "This is a Sigmond samplings file in fstreams format" in output:
                return True, "fstream", None
            elif "This is a Sigmond samplings file in HDF5 format" in output:
                return True, "hdf5", None
            elif "This is a Sigmond bins file in fstreams format" in output:
                return True, "fstream", None
            elif "This is a Sigmond bins file in HDF5 format" in output:
                return True, "hdf5", None
            else:
                return False, None, None

        except RuntimeError as e:
            error_msg = str(e)
            if "bad root path" in error_msg:
                # Extract available paths from error message
                paths = []
                lines = error_msg.split("\n")
                for line in lines:
                    if line.strip().startswith("/") and line.strip().endswith("/"):
                        paths.append(line.strip())
                return True, "hdf5", paths
            elif "This file type is not known to Sigmond" in error_msg:
                return False, None, None
            return False, None, None

    def load_file(self, filename: str, group: str = None, lazy: bool = False) -> None:
        """
        Load all data from a file.

        Args:
            filename: Path to the samplings file
            group: For HDF5 files, the root group to read (default: None = auto-detect)
                      If None and file has a single root group, it will be used automatically.
                      If None and file has multiple root groups, an error will be raised.
            lazy: If True, defer reading each observable's sample array until first
                  accessed (HDF5 only). See :meth:`__init__`.
        """
        # Load samplings
        try:
            self._all_samplings = self._load_samplings_impl(filename, group, lazy=lazy)
            logger.info(f"Successfully loaded {len(self._all_samplings)} samplings from {filename}")
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            raise

        self._filename = filename
        self._lazy = lazy

    def _load_samplings_impl(
        self, filename: str, group: str = None, lazy: bool = False
    ) -> SingleEnsembleCollection:
        """
        Load all samplings from a file.

        Args:
            filename: Path to the file
            group: For HDF5 files, the root group to read (None = auto-detect)
            lazy: If True, defer reading sample arrays (HDF5 only).

        Returns:
            SingleEnsembleCollection of loaded samplings
        """
        # Auto-detect file type
        if is_hdf5_file(filename):
            # Verify it's a valid Sigmond HDF5 file
            is_valid, file_kind, available_groups = verify_sigmond_hdf5(filename)
            if not is_valid:
                raise ValueError(f"File {filename} is not a valid Sigmond HDF5 file")

            # Handle root group selection
            if group is None:
                if len(available_groups) == 1:
                    group = available_groups[0]
                    logger.info(f"Auto-detected single group: '{group}'")
                elif len(available_groups) > 1:
                    raise ValueError(
                        f"Multiple groups found in HDF5 file. Please specify the group parameter.\n"
                        f"Available groups: {available_groups}"
                    )
                else:
                    raise ValueError(f"No data groups found in HDF5 file {filename}")
            else:
                group = clean_group(group)
                if group not in available_groups:
                    raise ValueError(
                        f"Group '{group}' not found in HDF5 file. "
                        f"Available groups: {available_groups}"
                    )
                logger.info(f"Using specified group: '{group}'")

            self._group = group
            self._file_kind = file_kind

            if lazy:
                logger.info(
                    f"Lazily loading HDF5 {file_kind} file {filename} using group '{group}'"
                )
                return self._load_lazy_from_hdf5(filename, group, file_kind)

            logger.info(f"Loading HDF5 {file_kind} file {filename} using group '{group}'")
            if file_kind == "bins":
                return self._load_bins_from_hdf5(filename, group)
            return self._load_from_hdf5(filename, group)
        else:
            if lazy:
                raise NotImplementedError(
                    "lazy=True is supported for HDF5 files only; fstream files load "
                    "eagerly via sigmond_query."
                )
            # Use sigmond_query for fstream files
            logger.info(f"Loading fstream file {filename} using sigmond_query")
            return self._load_from_sigmond_query(filename)

    def _load_from_sigmond_query(self, filename: str) -> SingleEnsembleCollection:
        """Load samplings or bins using sigmond_query (for fstream files)."""
        # Get header info
        header_output = self._run_sigmond_query(filename, "-i")
        ensemble_info, sampling_info = self._parse_header_xml(header_output)

        # Record file kind based on the parsed header
        self._file_kind = "samplings" if sampling_info is not None else "bins"

        # Get and parse keys
        keys_output = self._run_sigmond_query(filename, "-k")
        observable_infos = self._parse_keys_from_output(keys_output, ensemble_info)
        if not observable_infos:
            raise ValueError("No observable keys found in the file")

        # Get all values
        values_output = self._run_sigmond_query(filename, "-v")
        all_data = self._parse_all_values(values_output)

        if sampling_info is None:
            # Bins file: no resampling metadata, raw bins per record
            bins_list = self._build_bins_list(observable_infos, all_data)
            return SingleEnsembleCollection(bins_list)

        samplings_list = self._build_samplings_list(observable_infos, all_data, sampling_info)
        return SingleEnsembleCollection(samplings_list)

    @staticmethod
    def _build_ensemble_info(
        ensemble_element: ET.Element,
        num_measurements: int,
        num_bins: int,
        tweak_info: dict,
    ) -> EnsembleInfo:
        """
        Build an :class:`EnsembleInfo` from a header's ``<MCEnsembleInfo>``.

        Three forms are accepted:

        1. A bare ensemble id, ``<MCEnsembleInfo>clover_s24_t128_ud840_s743</...>``,
           which is looked up in the configured known-ensembles XML file (falling
           back to the header's own measurement count when unavailable).
        2. A packed header string, ``<MCEnsembleInfo>id|Nmeas|Nstreams|Nx|Ny|Nz|Nt</...>``.
        3. An expanded element carrying ``<Id>``, ``<NMeas>``, ``<NStreams>``,
           ``<NSpace>``, ``<NTime>`` (and their synonyms); ``<Weighted/>`` is ignored.

        Forms 2 and 3 are self-describing, so no database lookup is attempted.
        """
        text = (ensemble_element.text or "").strip()
        if len(ensemble_element) > 0 or "|" in text:
            return EnsembleInfo.from_xml_element(
                ensemble_element,
                num_bins=num_bins,
                num_measurements=num_measurements,
                tweak_info=tweak_info,
            )

        # Bare ensemble id: look it up in the config's ensembles XML file.
        try:
            return KnownEnsembles().get(
                name=text, num_bins=num_bins, tweak_info=tweak_info
            )
        except (ValueError, KeyError, FileNotFoundError) as e:
            logger.info(f"Ensemble '{text}' not resolved ({e}). Using basic ensemble info.")
            return EnsembleInfo(
                name=text,
                num_bins=num_bins,
                num_measurements=num_measurements,
                tweak_info=tweak_info,
            )

    def _parse_header_xml(self, xml_string: str) -> tuple[EnsembleInfo, SamplingInfo | None]:
        """
        Parse the header XML to extract ensemble (and sampling) info.

        Samplings files (``SigmondSamplingsFile`` root) return
        ``(ensemble_info, sampling_info)``.

        Bins files (``SigmondBinsFile`` root) have no MCSamplingInfo section
        and return ``(ensemble_info, None)``.
        """
        try:
            root = ET.fromstring(xml_string.strip())
        except ET.ParseError:
            # Try to find the XML part in the string
            xml_start = xml_string.find("<")
            xml_end = xml_string.rfind(">") + 1
            if xml_start >= 0 and xml_end > xml_start:
                xml_part = xml_string[xml_start:xml_end]
                root = ET.fromstring(xml_part)
            else:
                raise ValueError("Could not parse header XML")

        # Extract ensemble info
        bins_info = root.find(".//MCBinsInfo")
        if bins_info is None:
            raise ValueError("MCBinsInfo not found in header")

        ensemble_element = bins_info.find("MCEnsembleInfo")
        if ensemble_element is None:
            raise ValueError("MCEnsembleInfo not found in header")
        num_measurements = int(bins_info.find("NumberOfMeasurements").text)
        num_bins = int(bins_info.find("NumberOfBins").text)

        # Extract tweak info if present
        tweak_info = {}
        tweak_element = bins_info.find("TweakEnsemble")
        if tweak_element is not None:
            for child in tweak_element:
                tweak_info[child.tag] = child.text

        ensemble_info = self._build_ensemble_info(
            ensemble_element, num_measurements, num_bins, tweak_info
        )

        # Extract sampling info. Bins files don't have an MCSamplingInfo section;
        # return None in that case so the caller can dispatch accordingly.
        sampling_element = root.find(".//MCSamplingInfo")
        if sampling_element is None:
            return ensemble_info, None

        # Check for Bootstrap or Jackknife
        bootstrap = sampling_element.find(".//Bootstrapper")
        jackknife = sampling_element.find(".//Jackkniffer")  # Note: might be misspelled in XML
        jackknife_simple = sampling_element.find(".//Jackknife")  # Simple self-closing tag

        if bootstrap is not None:
            method = "bootstrap"
            num_resamplings = int(bootstrap.find("NumberResamplings").text)
            seed = int(bootstrap.find("Seed").text)
            boot_skip = int(bootstrap.find("BootSkip").text)
            sampling_info = SamplingInfo(method, num_resamplings, seed, boot_skip)
        elif jackknife is not None:
            method = "jackknife"
            num_resamplings = int(jackknife.find("NumberResamplings").text)
            sampling_info = SamplingInfo(method, num_resamplings)
        elif jackknife_simple is not None:
            method = "jackknife"
            # For simple jackknife, num_resamplings equals num_bins
            num_resamplings = num_bins
            sampling_info = SamplingInfo(method, num_resamplings)
        else:
            raise ValueError("No Bootstrap or Jackknife info found")

        return ensemble_info, sampling_info

    def _parse_observable_key(self, key_xml: str, ensemble_info: EnsembleInfo) -> ObservableInfo:
        """
        Parse an observable key from XML.

        Supports two forms of ``<MCObservable>``:

        1. ``<MCObservable><Info>name index op_type re_im</Info></MCObservable>`` —
           the simple form produced by the sampling/bin writer for scalar observables.
        2. ``<MCObservable><CorrT>...</CorrT><Arg>Re|Im</Arg></MCObservable>`` (or any
           other non-Info child) — typical of correlator-matrix observables in bins files.
           In this case the inner XML (with the ``<Arg>`` element removed) is stored
           verbatim as the ``ObservableInfo.name`` so the observable can be round-tripped
           without collapsing the CorrT metadata into a flat form. On export the
           writer regenerates the dataset key from this name.
        """
        # Reverse the HDF5-safe encoding applied by the writer's _make_hdf5_safe_key:
        #   "</" -> "<|"  (closing tags)   and   "/" -> "|"  (slashes in names)
        # First restore closing tags, then any remaining "|" are escaped slashes.
        # fstream keys are never escaped, so these replacements are no-ops there.
        try:
            standard_xml = key_xml.replace("<|", "</").replace("|", "/")
            root = ET.fromstring(standard_xml.strip())
        except ET.ParseError as e:
            raise ValueError(f"Could not parse key XML: {e}")

        # Case 1: simple <Info>name index op_type re_im</Info>
        info_element = root.find(".//Info")
        if info_element is not None and info_element.text is not None:
            info_text = info_element.text.strip()
            try:
                return ObservableInfo.from_string(info_text, ensemble_info)
            except ValueError:
                pass  # fall through to generic parsing

        # Case 2: generic MCObservable (e.g. CorrT/Arg). Extract Re/Im from <Arg>,
        # strip it from the root, and use the remaining inner XML as the observable name.
        re_im = "re"
        arg_element = root.find("./Arg")
        if arg_element is not None and arg_element.text is not None:
            arg_text = arg_element.text.strip().lower()
            if arg_text.startswith("im"):
                re_im = "im"
            root.remove(arg_element)

        # Serialize remaining children as the observable's identifying name.
        child_xml_parts = [ET.tostring(child, encoding="unicode").strip() for child in list(root)]
        if not child_xml_parts:
            raise ValueError(f"Empty MCObservable element in key: {key_xml!r}")
        name = "".join(child_xml_parts)

        obs_info = ObservableInfo(
            name=name,
            index=0,
            op_type="n",
            re_im=re_im,
            ensemble_info=ensemble_info,
        )
        return obs_info

    def _parse_keys_from_output(
        self, keys_output: str, ensemble_info: EnsembleInfo
    ) -> list[ObservableInfo]:
        """Parse observable keys from sigmond_query output."""
        observable_infos = []
        lines = keys_output.split("\n")
        current_key_lines = []
        in_key = False

        for line in lines:
            if line.startswith("Record ") and ":" in line:
                if current_key_lines:
                    # Process previous key
                    key_xml = "\n".join(current_key_lines)
                    try:
                        obs_info = self._parse_observable_key(key_xml, ensemble_info)
                        observable_infos.append(obs_info)
                    except (ValueError, NotImplementedError):
                        pass  # Skip problematic keys
                current_key_lines = []
                in_key = True
            elif in_key and line.strip():
                current_key_lines.append(line)
            elif in_key and not line.strip() and current_key_lines:
                # Process current key when we hit empty line
                key_xml = "\n".join(current_key_lines)
                try:
                    obs_info = self._parse_observable_key(key_xml, ensemble_info)
                    observable_infos.append(obs_info)
                except (ValueError, NotImplementedError):
                    pass  # Skip problematic keys
                current_key_lines = []
                in_key = False

        # Process last key
        if current_key_lines:
            key_xml = "\n".join(current_key_lines)
            try:
                obs_info = self._parse_observable_key(key_xml, ensemble_info)
                observable_infos.append(obs_info)
            except (ValueError, NotImplementedError):
                pass

        return observable_infos

    def _parse_all_values(self, values_output: str) -> list[np.ndarray]:
        """Parse the output of 'sigmond_query -v' into a list of numpy arrays."""
        all_records_values = []
        lines = values_output.split("\n")
        current_values = []

        in_record = False
        for line in lines:
            if line.startswith("Record ") and ":" in line:
                if current_values:
                    all_records_values.append(np.array(current_values))
                current_values = []
                in_record = True
            elif in_record and "[" in line and "]" in line and "=" in line:
                value_str = line.split("=")[1].strip()
                try:
                    value = float(value_str)
                    current_values.append(value)
                except ValueError:
                    try:
                        value = complex(value_str)
                        current_values.append(value)
                    except ValueError:
                        pass

        if current_values:
            all_records_values.append(np.array(current_values))

        return all_records_values

    def _fuse_re_im(
        self,
        observable_infos: list[ObservableInfo],
        all_data: list[np.ndarray],
        factory,
    ) -> list:
        """
        Group parsed records by (name, index) and build one observable per group.

        Records split into separate Re/Im parts are fused into a single complex
        observable; a lone part stays as-is. ``factory(data, obs_info, is_complex)``
        constructs the concrete object (SigmondSampling or SigmondBins).
        """
        if len(all_data) != len(observable_infos):
            raise ValueError("Mismatch between number of observables in header and data records.")

        grouped: dict[tuple, dict[str, tuple[ObservableInfo, int]]] = {}
        for i, obs_info in enumerate(observable_infos):
            grouped.setdefault((obs_info.name, obs_info.index), {})[obs_info.re_im] = (obs_info, i)

        result = []
        for parts in grouped.values():
            if "re" in parts and "im" in parts:
                obs_info, re_idx = parts["re"]
                _, im_idx = parts["im"]
                data = np.asarray(all_data[re_idx]) + 1j * np.asarray(all_data[im_idx])
                is_complex = True
            else:
                obs_info, idx = parts.get("re") or parts["im"]
                data = np.asarray(all_data[idx])
                is_complex = np.iscomplexobj(data)

            result.append(factory(data, obs_info, is_complex))

        return result

    def _build_bins_list(
        self,
        observable_infos: list[ObservableInfo],
        all_data: list[np.ndarray],
    ) -> list[SigmondBins]:
        """Build ``SigmondBins`` from parsed records, fusing Re/Im pairs into complex bins."""
        return self._fuse_re_im(
            observable_infos,
            all_data,
            lambda data, info, is_complex: SigmondBins(data, info, is_complex=is_complex),
        )

    def _build_samplings_list(
        self,
        observable_infos: list[ObservableInfo],
        all_data: list[np.ndarray],
        sampling_info: SamplingInfo,
    ) -> list[SigmondSampling]:
        """Build ``SigmondSampling`` from parsed records, fusing Re/Im pairs into complex samplings."""
        return self._fuse_re_im(
            observable_infos,
            all_data,
            lambda data, info, is_complex: SigmondSampling(
                data, info, sampling_info, is_complex=is_complex
            ),
        )

    def get_file_info(
        self, filename: str = None
    ) -> tuple[EnsembleInfo, SamplingInfo | None, list[ObservableInfo]]:
        """
        Get header info and list of observable keys from a file.

        Returns ``(ensemble_info, sampling_info, observable_infos)`` derived from
        the loaded collection. ``sampling_info`` is ``None`` for bins files.
        """
        if filename is not None:
            self.load_file(filename)
        elif self._filename is None:
            raise ValueError("No file loaded. Please provide a filename or call load_file() first.")

        if not self._all_samplings:
            raise ValueError("No observables loaded.")

        ensemble_info = self._all_samplings.ensemble_info
        sampling_info = (
            self._all_samplings.sampling_info if self._file_kind == "samplings" else None
        )
        observable_infos = [obs.observable_info for obs in self._all_samplings]
        return ensemble_info, sampling_info, observable_infos


@dataclass(frozen=True)
class SigmondFile:
    """A Sigmond file together with the HDF5 root group to read from it.

    The pieces live in different namespaces and are kept apart on purpose:

    * ``path`` is a filesystem path (coerced to :class:`pathlib.Path`).
    * ``group`` is an *in-file* HDF5 root group — one ensemble per group (see
      ``docs/sigmond_hdf5_structure.md``). ``None`` auto-detects, which only
      succeeds when the file holds a single root group. Leading/trailing slashes
      are stripped so ``"/samplings/"`` and ``"samplings"`` are equivalent.
    * ``name`` is an optional, caller-chosen label used purely for dict-style
      lookup in :class:`MultiSigmondLoader` (e.g. ``loader["c103"]``). It has no
      effect on how the file is read.

    Construct directly, or via :meth:`coerce` from a bare path or a
    ``(path, group)`` pair.
    """

    path: Path
    group: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.group is not None:
            object.__setattr__(self, "group", clean_group(self.group))

    @classmethod
    def coerce(cls, spec: "SigmondFileSpec", *, name: str | None = None) -> "SigmondFile":
        """Coerce a :class:`SigmondFile`, bare path, or ``(path, group)`` pair.

        ``name`` attaches (or overrides) the lookup label on the result.
        """
        if isinstance(spec, cls):
            return spec if name is None else replace(spec, name=name)
        if isinstance(spec, (str, os.PathLike)):
            return cls(spec, name=name)
        path, group = spec
        return cls(path, group, name=name)


# Anything that names a Sigmond file to load: a SigmondFile, a bare path (root
# group auto-detected), or a ``(path, group)`` pair (``group=None`` auto-detects).
SigmondFileSpec: TypeAlias = "SigmondFile | str | os.PathLike | tuple[str, str | None]"


class MultiSigmondLoader:
    """Load several Sigmond files into one multi-ensemble collection.

    Each file is read by its own :class:`SigmondLoader` (one ensemble per root
    group), and the results are concatenated into a
    :class:`MultiEnsembleCollection`. Every file must share the same
    ``sampling_info``; this is checked at construction, so a mismatched
    bootstrap/jackknife configuration is rejected up front rather than on first
    access to :attr:`observables`.

    Files may be named for dict-style access. Either pass a mapping (keys become
    names), or set ``name`` on individual :class:`SigmondFile` specs::

        loader = MultiSigmondLoader({
            "c103": SigmondFile("cls21_c103.h5", group="samplings"),
            "d200": ("cls21_d200.h5", "samplings"),
        })
        c103 = loader["c103"]                    # the per-file SigmondLoader
        "d200" in loader                         # True
        for name, single in loader.loaders_by_name.items():
            ...

    Example::

        loader = MultiSigmondLoader([
            SigmondFile("cls21_c103.h5", group="samplings"),
            SigmondFile("cls21_d200.h5", group="samplings"),
        ])
        multi = loader.observables               # MultiEnsembleCollection
        energies = loader.energy_observables()   # MultiEnsembleEnergyCollection
        for ens, single in multi.by_ensemble.items():
            ...
    """

    def __init__(
        self,
        files: Sequence[SigmondFileSpec] | Mapping[str, SigmondFileSpec],
        *,
        lazy: bool = True,
    ):
        """Build one :class:`SigmondLoader` per file.

        Args:
            files: Either a sequence of :class:`SigmondFile` specs (or anything
                :meth:`SigmondFile.coerce` accepts: a bare path, or a
                ``(path, group)`` pair), or a mapping of ``name -> spec`` where
                each key becomes the file's lookup label. A file with
                ``group=None`` auto-detects its HDF5 root group, which only
                succeeds when the file holds a single root group.
            lazy: Defer reading each observable's sample array until first access
                (HDF5 only). See :meth:`SigmondLoader.__init__`.
        """
        if not files:
            raise ValueError("MultiSigmondLoader requires at least one file")
        self._lazy = lazy
        if isinstance(files, Mapping):
            named = cast(Mapping[str, SigmondFileSpec], files)
            self._files = [SigmondFile.coerce(spec, name=name) for name, spec in named.items()]
        else:
            self._files = [SigmondFile.coerce(entry) for entry in files]
        self._loaders = [SigmondLoader(str(f.path), group=f.group, lazy=lazy) for f in self._files]
        self._by_name: dict[str, SigmondLoader] = {}
        for f, loader in zip(self._files, self._loaders):
            if f.name is None:
                continue
            if f.name in self._by_name:
                raise ValueError(f"Duplicate file name: {f.name!r}")
            self._by_name[f.name] = loader

        # Reject mismatched sampling configurations up front, before any
        # observables are concatenated into a MultiEnsembleCollection.
        sampling_infos = [loader.observables.sampling_info for loader in self._loaders]
        if len(set(sampling_infos)) > 1:
            found = sorted({str(s) for s in sampling_infos})
            raise ValueError(
                "All files in a MultiSigmondLoader must share the same "
                f"sampling_info; found {len(found)}: {found}"
            )
        self._sampling_info: SamplingInfo = sampling_infos[0]
        
    @classmethod
    def from_resolved(
        cls,
        resolved: "SamplingDataResolved",
        lazy: bool = True,
    ) -> Self:
        return cls(files={id: (str(src.file), src.group) for id, src in resolved.sources.items()},
                   lazy=lazy)

    @property
    def files(self) -> list[SigmondFile]:
        """The resolved :class:`SigmondFile` specs, in input order."""
        return list(self._files)

    @property
    def loaders(self) -> list[SigmondLoader]:
        """The per-file :class:`SigmondLoader` instances, in input order."""
        return list(self._loaders)

    @property
    def names(self) -> list[str | None]:
        """The lookup label of each file (``None`` if unnamed), in input order."""
        return [f.name for f in self._files]

    @property
    def sampling_info(self) -> SamplingInfo:
        """The ``sampling_info`` shared by every loaded file."""
        return self._sampling_info

    @property
    def ensembles(self) -> list[EnsembleInfo]:
        """Distinct ensembles across all loaded files, in input order."""
        return list(dict.fromkeys(loader.observables.ensemble_info for loader in self._loaders))

    @property
    def loaders_by_name(self) -> dict[str, SigmondLoader]:
        """Named :class:`SigmondLoader` instances keyed by file name."""
        return dict(self._by_name)

    @property
    def files_by_name(self) -> dict[str, SigmondFile]:
        """Named :class:`SigmondFile` specs keyed by file name."""
        return {f.name: f for f in self._files if f.name is not None}

    def __getitem__(self, name: str) -> SigmondLoader:
        """Return the :class:`SigmondLoader` for the file named ``name``."""
        try:
            return self._by_name[name]
        except KeyError:
            available = sorted(self._by_name)
            raise KeyError(f"No file named {name!r}; available names: {available}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def get(self, name: str, default: SigmondLoader | None = None):
        """Return the loader named ``name``, or ``default`` if there is none."""
        return self._by_name.get(name, default)

    @property
    def file_kinds(self) -> list[str | None]:
        """``'samplings'``/``'bins'`` per loaded file, in input order."""
        return [loader.file_kind for loader in self._loaders]

    @property
    def observables(self) -> MultiEnsembleCollection:
        """All observables across files as one :class:`MultiEnsembleCollection`."""
        collection = MultiEnsembleCollection([])
        for loader in self._loaders:
            collection += loader.observables
        return collection

    def energy_observables(
        self,
        *,
        skip_missing_particles: bool = True,
        return_type: str | None = None,
    ):
        """Return the loaded observables as a multi-ensemble energy collection.

        Mirrors :meth:`SigmondLoader.energy_observables`; an explicit semantic
        view over :attr:`observables`. Returns a ``MultiEnsembleEnergyCollection``.
        """
        return self.observables.as_energy_levels(
            skip_missing_particles=skip_missing_particles,
            return_type=return_type,
        )
