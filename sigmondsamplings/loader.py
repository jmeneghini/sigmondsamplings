"""
Loader module for Sigmond samplings files.
"""

import logging
import re
import subprocess
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .bins import SigmondBins
from .ensemble_collection import SingleEnsembleCollection
from .info import EnsembleInfo, KnownEnsembles, ObservableInfo, SamplingInfo
from .sampling import SigmondSampling

logger = logging.getLogger(__name__)
SIGMOND_QUERY_CMD = "sigmond_query"


class SigmondLoader:
    """
    Loader for Sigmond samplings files.

    Automatically detects file format (HDF5 or fstream) and uses the appropriate method:
    - HDF5 files: Direct h5py reading (fast)
    - Fstream files: sigmond_query tool (slower but supports legacy format)

    For HDF5 files with a single data path, the path is auto-detected.
    For HDF5 files with multiple paths, you must specify hdf5_path parameter.

    Provides queryable SingleEnsembleCollection of all loaded observables:
        loader.observables - All loaded samplings
        loader.hdf5_path - The HDF5 path used (None for fstream files)

    Use SingleEnsembleCollection filtering methods to query:
        loader.observables.filter(index=0)
        loader.observables.find(lambda obs: 'PSQ' in obs.name)
        loader.observables.find(lambda obs: re.search(r'PSQ.*', obs.name))
    """

    # TODO: some major issues with appending. Creates a 'working' file, still uses sigmond_query, and doesn't replace the original file with the working file.
    def __init__(
        self,
        filename: str = None,
        hdf5_path: str = None,
    ):
        """
        Initialize the loader.

        Args:
            filename: Path to the samplings or bins file to load upon construction (optional)
            hdf5_path: For HDF5 files, the root path to use (default: None = auto-detect)
                      If None and file has a single path, it will be used automatically.
                      If None and file has multiple paths, an error will be raised.
        """
        # Single source of truth for data
        self._filename = None
        self._hdf5_path = hdf5_path
        self._file_kind: str | None = None  # "samplings" or "bins"
        self._all_samplings = SingleEnsembleCollection([])

        # Load file if provided
        if filename:
            self.load_file(filename, self._hdf5_path)

    @property
    def observables(self) -> SingleEnsembleCollection:
        """Access the loaded samplings collection."""
        return self._all_samplings

    @property
    def hdf5_path(self) -> str | None:
        """Get the HDF5 path used for loading (None for fstream files)."""
        return self._hdf5_path

    @property
    def file_kind(self) -> str | None:
        """Return 'samplings' or 'bins' for the most recently loaded file."""
        return self._file_kind

    def _clean_hdf5_path(self, path: str) -> str:
        """Clean HDF5 path by removing leading/trailing slashes."""
        return path.strip("/")

    def _is_hdf5_file(self, filename: str) -> bool:
        """Check if a file is an HDF5 file."""
        try:
            with h5py.File(filename, "r"):
                return True
        except OSError:
            return False

    _HDF5_FIDENTIFIERS = {
        "Sigmond--SamplingsFile": "samplings",
        "Sigmond--BinsFile": "bins",
    }

    def _verify_hdf5_sigmond_file(
        self, filename: str
    ) -> tuple[bool, str | None, list[str] | None]:
        """
        Verify that an HDF5 file is a valid Sigmond samplings or bins file.

        Returns:
            (is_valid, file_kind, available_paths)
            - is_valid: True if file has correct Sigmond structure
            - file_kind: 'samplings' or 'bins' (None on failure)
            - available_paths: List of available data paths at root (excluding "Info")
        """
        try:
            with h5py.File(filename, "r") as f:
                if "Info" not in f or "FIdentifier" not in f["Info"]:
                    return False, None, None

                fid = f["Info"]["FIdentifier"][()].decode("utf-8")
                file_kind = self._HDF5_FIDENTIFIERS.get(fid)
                if file_kind is None:
                    return False, None, None

                available_paths = [
                    key for key in f.keys()
                    if key != "Info" and isinstance(f[key], h5py.Group)
                ]
                return True, file_kind, available_paths
        except Exception as e:
            logger.debug(f"HDF5 verification failed: {e}")
            return False, None, None

    def _load_from_hdf5(self, filename: str, path: str = "samplings") -> SingleEnsembleCollection:
        """
        Load samplings directly from HDF5 file using h5py.

        Args:
            filename: Path to the HDF5 file
            path: Root path in HDF5 file (default: "samplings")

        Returns:
            SingleEnsembleCollection of loaded samplings
        """
        ensemble_info, sampling_info, parsed = self._read_hdf5_values(filename, path)
        if sampling_info is None:
            raise ValueError(
                f"File {filename} appears to be a bins file, not a samplings file."
            )
        samplings_list = self._build_samplings_list(
            [p[0] for p in parsed], [p[1] for p in parsed], sampling_info
        )
        return SingleEnsembleCollection(samplings_list)

    def _load_bins_from_hdf5(self, filename: str, path: str) -> SingleEnsembleCollection:
        """
        Load Sigmond bins directly from HDF5 file using h5py.

        Returns:
            SingleEnsembleCollection of SigmondBins objects.
        """
        ensemble_info, sampling_info, parsed = self._read_hdf5_values(filename, path)
        if sampling_info is not None:
            raise ValueError(
                f"File {filename} appears to be a samplings file, not a bins file."
            )
        bins_list = self._build_bins_list(
            [p[0] for p in parsed], [p[1] for p in parsed], [p[2] for p in parsed]
        )
        return SingleEnsembleCollection(bins_list)

    def _read_hdf5_values(
        self, filename: str, path: str
    ) -> tuple[EnsembleInfo, SamplingInfo | None, list[tuple[ObservableInfo, np.ndarray, str]]]:
        """
        Shared helper to read a Sigmond HDF5 file (samplings or bins).

        Returns:
            (ensemble_info, sampling_info_or_None, [(obs_info, data, raw_key), ...])
        """
        with h5py.File(filename, "r") as f:
            if path not in f:
                available_paths = [k for k in f.keys() if k != "Info"]
                raise ValueError(
                    f"Path '{path}' not found in HDF5 file. Available paths: {available_paths}"
                )

            group = f[path]
            if "Header" not in group:
                raise ValueError(f"No Header dataset found in {path}")

            header_xml = group["Header"][()].decode("utf-8")
            ensemble_info, sampling_info = self._parse_header_xml(header_xml)

            if "Values" not in group:
                raise ValueError(f"No Values group found in {path}")

            values_group = group["Values"]

            parsed: list[tuple[ObservableInfo, np.ndarray, str]] = []
            for dataset_name in values_group.keys():
                try:
                    obs_info = self._parse_observable_key(dataset_name, ensemble_info)
                    data = values_group[dataset_name][:]
                    parsed.append((obs_info, data, dataset_name))
                except (ValueError, NotImplementedError) as e:
                    logger.debug(f"Skipping dataset {dataset_name}: {e}")
                    continue

            if not parsed:
                raise ValueError("No valid observable data found in file")

            return ensemble_info, sampling_info, parsed

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
            (is_valid, file_type, hdf5_paths)
            - is_valid: True if file is valid
            - file_type: 'fstream', 'hdf5', or None
            - hdf5_paths: List of available paths for HDF5 files, None otherwise
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

    def load_file(self, filename: str, hdf5_path: str = None) -> None:
        """
        Load all data from a file.

        Args:
            filename: Path to the samplings file
            hdf5_path: For HDF5 files, the root path to use (default: None = auto-detect)
                      If None and file has a single path, it will be used automatically.
                      If None and file has multiple paths, an error will be raised.
        """
        # Load samplings
        try:
            self._all_samplings = self._load_samplings_impl(filename, hdf5_path)
            logger.info(f"Successfully loaded {len(self._all_samplings)} samplings from {filename}")
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            raise

        self._filename = filename

    def _load_samplings_impl(
        self, filename: str, hdf5_path: str = None
    ) -> SingleEnsembleCollection:
        """
        Load all samplings from a file.

        Args:
            filename: Path to the file
            hdf5_path: For HDF5 files, the root path to use (None = auto-detect)

        Returns:
            SingleEnsembleCollection of loaded samplings
        """
        # Auto-detect file type
        if self._is_hdf5_file(filename):
            # Verify it's a valid Sigmond HDF5 file
            is_valid, file_kind, available_paths = self._verify_hdf5_sigmond_file(filename)
            if not is_valid:
                raise ValueError(f"File {filename} is not a valid Sigmond HDF5 file")

            # Handle path selection
            if hdf5_path is None:
                if len(available_paths) == 1:
                    hdf5_path = available_paths[0]
                    logger.info(f"Auto-detected single path: '{hdf5_path}'")
                elif len(available_paths) > 1:
                    raise ValueError(
                        f"Multiple paths found in HDF5 file. Please specify hdf5_path parameter.\n"
                        f"Available paths: {available_paths}"
                    )
                else:
                    raise ValueError(f"No data paths found in HDF5 file {filename}")
            else:
                hdf5_path = self._clean_hdf5_path(hdf5_path)
                if hdf5_path not in available_paths:
                    raise ValueError(
                        f"Path '{hdf5_path}' not found in HDF5 file. "
                        f"Available paths: {available_paths}"
                    )
                logger.info(f"Using specified path: '{hdf5_path}'")

            self._hdf5_path = hdf5_path
            self._file_kind = file_kind

            logger.info(
                f"Loading HDF5 {file_kind} file {filename} using path '{hdf5_path}'"
            )
            if file_kind == "bins":
                return self._load_bins_from_hdf5(filename, hdf5_path)
            return self._load_from_hdf5(filename, hdf5_path)
        else:
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
            bins_list = self._build_bins_list(observable_infos, all_data, raw_keys=None)
            return SingleEnsembleCollection(bins_list)

        samplings_list = self._build_samplings_list(observable_infos, all_data, sampling_info)
        return SingleEnsembleCollection(samplings_list)

    def _parse_header_xml(
        self, xml_string: str
    ) -> tuple[EnsembleInfo, SamplingInfo | None]:
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

        ensemble_name = bins_info.find("MCEnsembleInfo").text
        num_measurements = int(bins_info.find("NumberOfMeasurements").text)
        num_bins = int(bins_info.find("NumberOfBins").text)

        # Extract tweak info if present
        tweak_info = {}
        tweak_element = bins_info.find("TweakEnsemble")
        if tweak_element is not None:
            for child in tweak_element:
                tweak_info[child.tag] = child.text

        # Check for ensemble name in config's ensembles XML file
        known_ensembles = KnownEnsembles()
        try:
            ensemble_info = known_ensembles.get(
                name=ensemble_name, num_bins=num_bins, tweak_info=tweak_info
            )
        except (ValueError, KeyError) as e:
            # Fallback to basic ensemble info
            ensemble_info = EnsembleInfo(
                name=ensemble_name,
                num_bins=num_bins,
                num_measurements=num_measurements,
                tweak_info=tweak_info,
            )
            if e is ValueError:
                logger.info(f"KnownEnsembles not configured: {e}. Using basic ensemble info.")

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
           without collapsing the CorrT metadata into a flat form.

        The HDF5-safe dataset key is stored on the returned ObservableInfo as ``_raw_key``
        so the writer can re-emit the exact same dataset name on export.
        """
        try:
            standard_xml = key_xml.replace("<|", "</")
            root = ET.fromstring(standard_xml.strip())
        except ET.ParseError as e:
            raise ValueError(f"Could not parse key XML: {e}")

        # Case 1: simple <Info>name index op_type re_im</Info>
        info_element = root.find(".//Info")
        if info_element is not None and info_element.text is not None:
            info_text = info_element.text.strip()
            try:
                obs_info = ObservableInfo.from_string(info_text, ensemble_info)
                obs_info._raw_key = key_xml
                return obs_info
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
        child_xml_parts = [
            ET.tostring(child, encoding="unicode").strip() for child in list(root)
        ]
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
        obs_info._raw_key = key_xml
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

    def _build_bins_list(
        self,
        observable_infos: list[ObservableInfo],
        all_data: list[np.ndarray],
        raw_keys: list[str] | None = None,
    ) -> list[SigmondBins]:
        """
        Build a list of ``SigmondBins`` from parsed ObservableInfos and raw bin arrays.

        Complex observables (split into Re/Im parts across datasets / records) are
        regrouped into a single complex ``SigmondBins`` per (name, index).
        """
        if len(all_data) != len(observable_infos):
            raise ValueError(
                "Mismatch between number of observables in header and data records."
            )

        # Index observables by (name, index) so Re/Im pairs can be fused.
        grouped: dict[tuple, dict[str, tuple[ObservableInfo, int]]] = {}
        for i, obs_info in enumerate(observable_infos):
            key = (obs_info.name, obs_info.index)
            grouped.setdefault(key, {})[obs_info.re_im] = (obs_info, i)

        result: list[SigmondBins] = []
        for key, parts in grouped.items():
            if "re" in parts and "im" in parts:
                re_info, re_idx = parts["re"]
                _, im_idx = parts["im"]
                combined = np.asarray(all_data[re_idx]) + 1j * np.asarray(all_data[im_idx])
                bins = SigmondBins(combined, re_info, is_complex=True)
                # Preserve the real part's raw key so round-tripping produces matching keys.
                if raw_keys is not None:
                    re_info._raw_key = raw_keys[re_idx]
                result.append(bins)
            elif "re" in parts:
                re_info, re_idx = parts["re"]
                data = np.asarray(all_data[re_idx])
                bins = SigmondBins(data, re_info, is_complex=np.iscomplexobj(data))
                if raw_keys is not None:
                    re_info._raw_key = raw_keys[re_idx]
                result.append(bins)
            elif "im" in parts:
                im_info, im_idx = parts["im"]
                data = np.asarray(all_data[im_idx])
                bins = SigmondBins(data, im_info, is_complex=np.iscomplexobj(data))
                if raw_keys is not None:
                    im_info._raw_key = raw_keys[im_idx]
                result.append(bins)

        return result

    def _build_samplings_list(
        self,
        observable_infos: list[ObservableInfo],
        all_data: list[np.ndarray],
        sampling_info: SamplingInfo,
    ) -> list[SigmondSampling]:
        """Build the samplings list from parsed data."""
        if len(all_data) != len(observable_infos):
            raise ValueError("Mismatch between number of observables in header and data records.")

        # Group observables by name and index to find complex pairs
        grouped_observables = {}
        for i, obs_info in enumerate(observable_infos):
            key = (obs_info.name, obs_info.index)
            if key not in grouped_observables:
                grouped_observables[key] = {}
            grouped_observables[key][obs_info.re_im] = (obs_info, i)

        result = []
        for key, parts in grouped_observables.items():
            if "re" in parts and "im" in parts:
                re_info, re_idx = parts["re"]
                im_info, im_idx = parts["im"]
                re_data = all_data[re_idx]
                im_data = all_data[im_idx]
                complex_data = re_data + 1j * im_data
                sampling = SigmondSampling(complex_data, re_info, sampling_info, is_complex=True)
                result.append(sampling)
            elif "re" in parts:
                re_info, re_idx = parts["re"]
                re_data = all_data[re_idx]
                sampling = SigmondSampling(
                    re_data, re_info, sampling_info, is_complex=np.iscomplexobj(re_data)
                )
                result.append(sampling)
            elif "im" in parts:
                im_info, im_idx = parts["im"]
                im_data = all_data[im_idx]
                sampling = SigmondSampling(
                    im_data, im_info, sampling_info, is_complex=np.iscomplexobj(im_data)
                )
                result.append(sampling)

        return result

    # Utility methods for backward compatibility
    def get_file_info(
        self, filename: str = None
    ) -> tuple[EnsembleInfo, SamplingInfo, list[ObservableInfo]]:
        """Get header info and list of observable keys from a file."""
        if filename is not None:
            self.load_file(filename)
        elif self._filename is None:
            raise ValueError("No file loaded. Please provide a filename or call load_file() first.")

        return self._ensemble_info, self._sampling_info, self._observable_infos

    @staticmethod
    def get_name_and_index_from_dict_key(key: str) -> tuple[str, int]:
        """Extract the (name, index) pair from keys like 'observable[3]'."""
        m = re.match(r"^(.*)\[(\d+)\]$", key)
        if not m:
            raise ValueError(
                f"Key '{key}' does not end with an integer in brackets (e.g. 'name[0]')."
            )
        name, index_str = m.groups()
        return name, int(index_str)
