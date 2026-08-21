"""
Metadata classes for Sigmond samplings.

This module contains the core metadata classes that describe ensembles,
sampling methods, and observables.
"""

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Accepted spellings for the fields of an ensemble description, whether it
# appears as ``<EnsembleInfo>`` in a known-ensembles XML file or as an expanded
# ``<MCEnsembleInfo>`` in a file header. Matching is case-insensitive.
# ``<Weighted/>`` (CLS reweighting flag) is deliberately not parsed.
_ENSEMBLE_TAG_SYNONYMS: dict[str, tuple[str, ...]] = {
    "name": ("id", "name", "ensembleid", "mcensembleinfo"),
    "num_measurements": ("nmeas", "nmeasurements", "numberofmeasurements"),
    "num_streams": ("nstreams", "nstream", "numberofstreams"),
    "spatial_extent": ("nspace", "nspatial", "nx", "ns", "lx", "ls"),
    "temporal_extent": ("ntime", "ntemporal", "nt", "lt"),
}

# Spatial extents given per-direction; all three must agree.
_SPATIAL_DIRECTION_TAGS = ("nx", "ny", "nz")


def parse_ensemble_element(element: ET.Element) -> dict[str, Any]:
    """
    Parse an ensemble description element into a field dict.

    Recognizes the tag synonyms in :data:`_ENSEMBLE_TAG_SYNONYMS`, e.g.::

        <EnsembleInfo>
          <Id>clover_s24_t128_ud840_s743</Id>
          <NStreams>4</NStreams>
          <NMeas>551</NMeas>
          <NSpace>24</NSpace>
          <NTime>128</NTime>
          <Weighted/>            <!-- ignored -->
        </EnsembleInfo>

    Returns a dict with any of ``name``, ``num_measurements``, ``num_streams``,
    ``spatial_extent`` and ``temporal_extent`` that were found. Unknown children
    (including ``<Weighted/>``) are ignored.
    """
    found: dict[str, str] = {}
    directions: dict[str, int] = {}

    for child in element:
        tag = child.tag.strip().lower()
        text = (child.text or "").strip()
        if not text:
            continue  # e.g. <Weighted/>, or an empty element

        if tag in _SPATIAL_DIRECTION_TAGS:
            directions[tag] = int(text)

        for field, synonyms in _ENSEMBLE_TAG_SYNONYMS.items():
            if tag in synonyms and field not in found:
                found[field] = text
                break

    if len(directions) > 1:
        if len(set(directions.values())) > 1:
            raise ValueError(
                f"Anisotropic spatial extents are not supported: {directions}"
            )
        found["spatial_extent"] = str(next(iter(directions.values())))

    parsed: dict[str, Any] = {}
    if "name" in found:
        parsed["name"] = found["name"]
    for field in ("num_measurements", "num_streams", "spatial_extent", "temporal_extent"):
        if field in found:
            parsed[field] = int(found[field])
    return parsed


class KnownEnsembles:
    """
    Helper class for loading ensemble information from XML files.

    This class manages a database of known ensembles, allowing users to create
    EnsembleInfo objects by name without specifying all parameters manually.

    The XML file path is stored in :mod:`sigmondsamplings.rcparams` under the
    ``ensembles.xml_file`` key. When you construct a ``KnownEnsembles`` with
    an explicit ``xml_file``, that path is written back to the rc system and
    persisted via :func:`sigmondsamplings.rc_save` so later sessions pick it up
    automatically (the same role the old ``~/.sigmondsamplings/config`` file
    played, but now through a single TOML config).

    Example:
        # First time setup
        >>> known = KnownEnsembles('/path/to/ensembles.xml')

        # Later sessions (path is remembered)
        >>> known = KnownEnsembles()
        >>> ensemble = known.get('cls21_n203')
    """

    def __init__(self, xml_file: str | None = None):
        """
        Initialize KnownEnsembles.

        Args:
            xml_file: Path to ensembles.xml file. If not provided, falls back
                to ``rc["ensembles.xml_file"]``. If provided, sets that rc key
                and persists the config so future sessions reuse it.
        """
        # Imported here to avoid a circular import at module load.
        from . import rcparams as _rc

        self._ensembles: dict[str, dict[str, Any]] = {}

        if xml_file is not None:
            self._xml_file = Path(xml_file).expanduser()
            _rc.rc["ensembles.xml_file"] = str(self._xml_file.absolute())
            _rc.rc_save()
        else:
            configured = _rc.rc.get("ensembles.xml_file")
            self._xml_file = Path(configured).expanduser() if configured else None
            if self._xml_file is not None and not self._xml_file.exists():
                raise FileNotFoundError(
                    f"Previously configured ensemble file not found: {self._xml_file}\n"
                    f"Please provide a new path when initializing KnownEnsembles."
                )

        if self._xml_file is not None:
            self._load_ensembles()

    def _load_ensembles(self):
        """Parse the XML file and load ensemble information."""
        if not self._xml_file.exists():
            raise FileNotFoundError(f"Ensemble XML file not found: {self._xml_file}")

        tree = ET.parse(self._xml_file)
        root = tree.getroot()

        # Find the <Infos> section
        infos = root.find("Infos")
        if infos is None:
            raise ValueError("No <Infos> section found in XML file")

        # Parse each EnsembleInfo entry
        for ensemble_elem in infos.findall("EnsembleInfo"):
            fields = parse_ensemble_element(ensemble_elem)

            if "name" not in fields or "num_measurements" not in fields:
                continue  # Skip incomplete entries

            name = fields["name"]
            self._ensembles[name] = {
                "num_measurements": fields["num_measurements"],
                "num_streams": fields.get("num_streams"),
                "spatial_extent": fields.get("spatial_extent"),
                "temporal_extent": fields.get("temporal_extent"),
            }

    def get(
        self,
        name: str,
        num_bins: int | None = None,
        rebin_size: int | None = None,
        tweak_info: dict[str, Any] | None = None,
    ) -> "EnsembleInfo":
        """
        Create an EnsembleInfo object for a known ensemble.

        Args:
            name: Name/ID of the ensemble to look up
            num_bins: Optional number of bins for rebinning
            rebin_size: Optional rebinning factor
            tweak_info: Optional additional tweak information

        Returns:
            EnsembleInfo object with parameters loaded from XML

        Raises:
            KeyError: If name is not found in the database
            ValueError: If no XML file has been configured
        """
        if not self._ensembles:
            if self._xml_file is None:
                raise ValueError(
                    "No ensemble database loaded. Please provide an xml_file "
                    "when initializing KnownEnsembles."
                )
            else:
                raise ValueError(f"Ensemble database is empty. Check XML file: {self._xml_file}")

        if name not in self._ensembles:
            raise KeyError(
                f"Ensemble '{name}' not found in database. "
                f"Available ensembles: {', '.join(sorted(self._ensembles.keys())[:10])}..."
            )
        ensemble_data = self._ensembles[name]

        return EnsembleInfo(
            name=name,
            num_measurements=ensemble_data["num_measurements"],
            spatial_extent=ensemble_data["spatial_extent"],
            temporal_extent=ensemble_data["temporal_extent"],
            num_streams=ensemble_data.get("num_streams"),
            num_bins=num_bins,
            rebin_size=rebin_size,
            tweak_info=tweak_info,
        )

    def list_ensembles(self) -> list:
        """Return a list of all available ensemble names."""
        return sorted(self._ensembles.keys())

    def __contains__(self, name: str) -> bool:
        """Check if an ensemble exists in the database."""
        return name in self._ensembles

    def __len__(self) -> int:
        """Return the number of ensembles in the database."""
        return len(self._ensembles)

    def __repr__(self) -> str:
        return f"KnownEnsembles(xml_file='{self._xml_file}', n_ensembles={len(self._ensembles)})"


class EnsembleInfo:
    """Information about the Monte Carlo ensemble."""

    def __init__(
        self,
        name: str,
        num_measurements: int,
        spatial_extent: int | None = None,
        temporal_extent: int | None = None,
        num_streams: int | None = None,
        num_bins: int | None = None,
        rebin_size: int | None = None,
        tweak_info: dict[str, Any] | None = None,
    ):
        """
        Initialize EnsembleInfo.

        Args:
            name: Name of the ensemble
            num_measurements: Total number of measurements
            spatial_extent: Spatial lattice extent (optional)
            temporal_extent: Temporal lattice extent (optional)
            num_streams: Number of Monte Carlo streams (optional)
            num_bins: Target number of bins after rebinning (optional).
                     If provided, rebin_size will be calculated automatically.
            rebin_size: Rebinning factor (optional). Alternative to num_bins.
            tweak_info: Additional tweak information

        Note:
            - If both num_bins and rebin_size are None: no rebinning
            - If num_bins is provided: rebin_size will be calculated
            - If rebin_size is provided via tweak_info['rebin']: use that
            - Cannot specify both num_bins and rebin_size explicitly
        """
        if num_bins is not None and rebin_size is not None:
            # verify consistency
            if num_measurements // rebin_size != num_bins:
                raise ValueError("Inconsistent num_bins and rebin_size provided.")

        self.name = name
        self.num_measurements = num_measurements
        self.spatial_extent = spatial_extent
        self.temporal_extent = temporal_extent
        self.num_streams = num_streams
        self.tweak_info = tweak_info or {}

        # Ensure all string keys in tweak_info are lowercase
        self.tweak_info = {
            k.lower() if isinstance(k, str) else k: v for k, v in self.tweak_info.items()
        }
        # If a value is a string key, convert to lowercase as well. Try to convert to int if possible.
        for k, v in self.tweak_info.items():
            if isinstance(v, str):
                v_lower = v.lower()
                try:
                    v_converted = int(v_lower)
                except ValueError:
                    v_converted = v_lower
                self.tweak_info[k] = v_converted

        # Calculate num_bins and rebin_size based on what's provided
        if rebin_size is not None:
            # Given rebin_size, calculate num_bins
            self.tweak_info["rebin"] = rebin_size
            self.num_bins = num_measurements // rebin_size
        elif num_bins is not None:
            # Given num_bins, calculate rebin_size
            self.num_bins = num_bins
            self.tweak_info["rebin"] = num_measurements // num_bins
        elif "rebin" in self.tweak_info:
            # rebin_size in tweak_info, calculate num_bins
            self.num_bins = num_measurements // self.tweak_info["rebin"]
        else:
            # No rebinning
            self.num_bins = num_measurements
            self.tweak_info["rebin"] = 1
            
    @classmethod
    def from_short_str(
        cls,
        short_str: str,
        num_bins: int = None,
        rebin_size: int = None,
        tweak_info: dict = None,
    ) -> "EnsembleInfo":
        """
        Create an EnsembleInfo from sigmond's packed ensemble string.

        Args:
            short_str: String of the form ``"id|Nmeas|Nstreams|Nx|Ny|Nz|Nt"``,
                e.g. ``"clover_s24_t36_ud840_s743|800|1|24|24|24|36"``. This form
                is self-describing, so the measurement count is taken from it.
            num_bins: Optional number of bins after rebinning.
            rebin_size: Optional rebinning factor (alternative to ``num_bins``).
            tweak_info: Optional additional tweak information.

        Returns:
            EnsembleInfo object with parameters parsed from the string.

        Raises:
            ValueError: If the string is malformed, or if it is a bare ensemble
                id — such a name carries no lattice information and must be
                resolved through :class:`KnownEnsembles` instead.
        """
        parts = [part.strip() for part in short_str.split("|")]
        name = parts[0]
        if not name:
            raise ValueError(f"Invalid ensemble short string: {short_str!r}")

        if len(parts) == 1:
            raise ValueError(
                f"{short_str!r} is a bare ensemble id, not a short string; resolve "
                "it with KnownEnsembles.get() or construct EnsembleInfo directly."
            )

        if len(parts) != 7:
            raise ValueError(
                f"Invalid ensemble short string: {short_str!r} (expected "
                "'id|Nmeas|Nstreams|Nx|Ny|Nz|Nt')"
            )

        num_measurements, num_streams, n_x, n_y, n_z, n_time = (
            int(part) for part in parts[1:]
        )

        if len({n_x, n_y, n_z}) > 1:
            raise ValueError(f"Anisotropic spatial extents are not supported: {short_str!r}")

        return cls(
            name=name,
            num_measurements=num_measurements,
            spatial_extent=n_x,
            temporal_extent=n_time,
            num_streams=num_streams,
            num_bins=num_bins,
            rebin_size=rebin_size,
            tweak_info=tweak_info,
        )

    @property
    def short_str(self) -> str:
        """
        Sigmond's packed ``"id|Nmeas|Nstreams|Nx|Ny|Nz|Nt"`` form.

        Falls back to the bare ensemble name when the lattice extents are
        unknown, since those cannot be invented. ``num_streams`` defaults to 1.
        """
        if self.spatial_extent is None or self.temporal_extent is None:
            return self.name

        return "|".join(
            str(field)
            for field in (
                self.name,
                self.num_measurements,
                self.num_streams if self.num_streams is not None else 1,
                self.spatial_extent,
                self.spatial_extent,
                self.spatial_extent,
                self.temporal_extent,
            )
        )

    @classmethod
    def from_xml_element(
        cls,
        element: ET.Element,
        num_bins: int = None,
        num_measurements: int = None,
        tweak_info: dict = None,
    ) -> "EnsembleInfo":
        """
        Create an EnsembleInfo from an expanded ensemble XML element.

        Accepts either an element with child tags (``<Id>``, ``<NMeas>``,
        ``<NStreams>``, ``<NSpace>``/``<NX>``…, ``<NTime>``, and their synonyms;
        ``<Weighted/>`` is ignored) or a leaf element whose text is a packed
        ``"id|Nmeas|Nstreams|Nx|Ny|Nz|Nt"`` short string.

        ``num_measurements`` is only a fallback for expanded elements that omit
        a measurement count; a short string always supplies its own.
        """
        if len(element) == 0:
            return cls.from_short_str(
                (element.text or "").strip(),
                num_bins=num_bins,
                tweak_info=tweak_info,
            )

        fields = parse_ensemble_element(element)
        if "name" not in fields:
            raise ValueError(f"No ensemble id found in <{element.tag}>")

        nmeas = fields.pop("num_measurements", None)
        if nmeas is None:
            nmeas = num_measurements
        if nmeas is None:
            raise ValueError(f"No measurement count found for ensemble '{fields['name']}'")

        return cls(
            num_measurements=nmeas,
            num_bins=num_bins,
            tweak_info=tweak_info,
            **fields,
        )

    @property
    def slug(self) -> str:
        """
        Returns a file-safe version of the ensemble name.
        Replaces non-alphanumeric characters with underscores.
        Example: 'Ensemble A/1' -> 'Ensemble_A_1'
        """
        # Keep letters, numbers, hyphens; replace everything else with '_'
        # Also strip leading/trailing underscores for cleanliness
        safe_name = re.sub(r"[^a-zA-Z0-9\-]", "_", self.name)
        return safe_name.strip("_")

    def __eq__(self, other):
        if not isinstance(other, EnsembleInfo):
            return False
        return (
            self.name == other.name
            and self.num_measurements == other.num_measurements
            and self.num_bins == other.num_bins
            and self.tweak_info == other.tweak_info
        )

    def __hash__(self):
        """Make EnsembleInfo hashable for use as dictionary keys."""
        # Convert tweak_info dict to frozenset of items for hashing
        # Only hash hashable values; skip unhashable ones
        hashable_tweaks = []
        for k, v in self.tweak_info.items():
            try:
                hash(v)  # Check if value is hashable
                hashable_tweaks.append((k, v))
            except TypeError:
                # Skip unhashable values (e.g., lists, dicts)
                pass

        return hash(
            (
                self.name,
                self.num_measurements,
                self.num_bins,
                self.spatial_extent,
                self.temporal_extent,
                self.num_streams,
                frozenset(hashable_tweaks),
            )
        )

    def __repr__(self):
        return f"EnsembleInfo('{self.name}', Nmeas = {self.num_measurements}, Nbins = {self.num_bins}, L = {self.spatial_extent}, T = {self.temporal_extent})"


# Default ensemble for general use - accessible to users
INDEP_ENSEMBLE = EnsembleInfo("indep", 1, 1)


class SamplingInfo:
    """Information about the sampling method (Bootstrap/Jackknife)."""

    def __init__(
        self,
        method: str,
        num_resamplings: int,
        seed: int = 0,
        boot_skip: int = 0,
        **kwargs,
    ):
        self.method = method.lower()
        self.num_resamplings = num_resamplings
        self.seed = seed
        self.boot_skip = boot_skip
        self.extra_params = kwargs

    @property
    def slug(self):
        """Fixed-width identifier for filenames."""
        return f"{self.method[:4]}_n{self.num_resamplings:04d}_s{self.seed:03d}"

    def __eq__(self, other):
        if not isinstance(other, SamplingInfo):
            return False
        return (
            self.method == other.method
            and self.num_resamplings == other.num_resamplings
            and self.seed == other.seed
            and self.boot_skip == other.boot_skip
            and self.extra_params == other.extra_params
        )

    def __hash__(self):
        """Make SamplingInfo hashable."""
        # Convert extra_params dict to tuple of items for hashing
        extra_items = tuple(sorted(self.extra_params.items())) if self.extra_params else ()
        return hash((self.method, self.num_resamplings, self.seed, self.boot_skip, extra_items))

    def __repr__(self):
        return f"SamplingInfo('{self.method}', n={self.num_resamplings}, seed={self.seed}, boot_skip={self.boot_skip})"


# ---------------------------------------------------------------------------
# Observable kind registry
# ---------------------------------------------------------------------------
# ``obs_kind`` is the discriminator persisted to HDF5 (in the ``ObsMeta`` table)
# so a read can rebuild the concrete ObservableInfo subclass rather than guess
# it from the dataset name. Subclasses opt in with :func:`register_obs_kind`,
# which records the canonical tag plus any legacy/short aliases; the loader then
# dispatches through :func:`obs_kind_class` with no knowledge of the subclasses.
#
# Query filters use the boolean facets (``is_energy``, ``is_single_hadron``)
# instead, since those compose across the hierarchy - an SH energy is an energy.
_OBS_KIND_REGISTRY: dict[str, type["ObservableInfo"]] = {}
_OBS_KIND_ALIASES: dict[str, str] = {}


def register_obs_kind(kind: str, *aliases: str):
    """
    Class decorator registering an ObservableInfo subclass under a kind tag.

    Parameters
    ----------
    kind : str
        Canonical tag; becomes the class's ``obs_kind`` and is what gets written
        to HDF5.
    *aliases : str
        Additional tags accepted on read and in query filters, for legacy files
        and short spellings. They are never written.

    Returns
    -------
    callable
        The decorator, which sets ``obs_kind`` on the class and returns it.
    """

    def decorate(cls):
        cls.obs_kind = kind
        for tag in (kind, *aliases):
            _OBS_KIND_REGISTRY[tag] = cls
            _OBS_KIND_ALIASES[tag] = kind
        return cls

    return decorate


# Modules whose import populates the registry. Imported on first lookup rather than
# at module scope: they import this one, and the package exposes them lazily, so a
# caller may reach the registry before anything has pulled them in.
_OBS_KIND_SOURCES = ("sigmondsamplings.energy_levels",)


def _ensure_obs_kinds_registered() -> None:
    """Import the modules that register obs kinds (a no-op once they are loaded)."""
    import importlib

    for module in _OBS_KIND_SOURCES:
        importlib.import_module(module)


def obs_kind_class(kind: Any) -> "type[ObservableInfo] | None":
    """Class registered for an ``obs_kind`` tag or alias, or ``None`` if unknown."""
    if kind is None:
        return None
    _ensure_obs_kinds_registered()
    return _OBS_KIND_REGISTRY.get(str(kind))


def canonical_obs_kind(kind: Any) -> Any:
    """Resolve an ``obs_kind`` tag or alias to its canonical tag, unchanged if unknown."""
    _ensure_obs_kinds_registered()
    return _OBS_KIND_ALIASES.get(str(kind), kind)


def obs_kinds() -> list[str]:
    """Every registered ``obs_kind`` tag, aliases included."""
    _ensure_obs_kinds_registered()
    return sorted(_OBS_KIND_REGISTRY)


class ObservableInfo:
    """Information about a specific observable."""

    obs_kind = "observable"
    """str: Discriminator persisted to HDF5 so the loader can rebuild this type."""

    is_energy = False
    """bool: Whether this observable is an energy level."""

    is_single_hadron = False
    """bool: Whether this observable is a single hadron energy level."""

    _reconstructs_own_label = False
    """bool: Whether :attr:`latex_str` is regenerated from attrs, so need not be stored."""

    def __init__(
        self,
        name: str,
        index: int = 0,
        op_type: str = "n",
        re_im: str = "re",
        ensemble_info: EnsembleInfo = INDEP_ENSEMBLE,
        latex_str: str = None,
    ):
        self.name = name
        self._index = int(index)
        self.op_type = op_type
        self.re_im = re_im
        self.ensemble_info = ensemble_info
        self._latex_str = latex_str  # used for plotting

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = int(value)

    @property
    def latex_str(self) -> str:
        """LaTeX representation for plotting."""
        if self._latex_str:
            return self._latex_str
        else:
            name = str(self.name).replace("_", r"\_")
            return rf"\text{{{name}}}"

    @latex_str.setter
    def latex_str(self, value: str):
        """Set LaTeX representation for plotting."""
        self._latex_str = value

    def copy(self):
        """Return a shallow copy, bypassing __init__ validation."""
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    @classmethod
    def from_string(
        cls, obs_string: str, ensemble_info: EnsembleInfo = INDEP_ENSEMBLE
    ) -> "ObservableInfo":
        """
        Parse observable info from string format.

        Expected format: "name index op_type re_im"
        """
        parts = obs_string.strip().split()
        if len(parts) != 4:
            raise ValueError(f"Invalid observable string format: {obs_string}")

        name, index_str, op_type, re_im = parts
        try:
            index = int(index_str)
        except ValueError:
            raise ValueError(f"Invalid index in observable string: {index_str}")

        return cls(name, index, op_type, re_im, ensemble_info)

    def __eq__(self, other):
        if not isinstance(other, ObservableInfo):
            return False
        return (
            self.name == other.name
            and self.index == other.index
            and self.op_type == other.op_type
            and self.re_im == other.re_im
            and self.ensemble_info == other.ensemble_info
        )

    def __hash__(self):
        """Make ObservableInfo hashable for use as dictionary keys."""
        return hash(
            (
                self.name,
                self.index,
                self.op_type,
                self.re_im,
                self.ensemble_info if self.ensemble_info else None,
            )
        )

    def _repr_latex__(self):
        """LaTeX representation for Jupyter notebooks."""
        if self.latex_str:
            return f"${self.latex_str}$"
        else:
            return self.__str__()

    def __repr__(self):
        return f"ObservableInfo(name='{self.name}', index={self.index}, ensemble='{self.ensemble_info}')"

    def __str__(self):
        return f"{self.name} {self.index}"  # Simple MCObs string format


def _fmt_halfint(twoX: int | None) -> str | None:
    if twoX is None:
        return None
    # exact half-integer printing
    if twoX % 2 == 0:
        return str(twoX // 2)
    return f"{twoX}/2"


@dataclass(frozen=True, slots=True)
class SectorInfo:
    """
    Physical sector information for classifying observables.

    - Hashable and usable as a dict key (hash uses all fields as stored).
    - `compatible_with(other)` compares only fields that are not None in either object
      (i.e., None means "unspecified / don't care").
    """

    # Charges
    B: int | None = None  # Baryon number
    Q: int | None = None  # Electric charge
    S: int | None = None  # Strangeness
    C: int | None = None  # Charm

    # Flavor symmetry
    twoI: int | None = None  # 2 * isospin
    twoI3: int | None = None  # 2 * isospin third component
    Gpar: int | None = None  # G-parity (±1)
    Cpar: int | None = None  # C-parity (±1)

    # Continuum spin/parity
    twoJ: int | None = None  # 2 * total angular momentum
    par: int | None = None  # parity (±1)

    @classmethod
    def from_sigmond_str(cls, s: str) -> "SectorInfo":
        """
        Parse SectorInfo from Sigmond string format.

        Currently just parses isospin and strangeness, of the form:
        isosinglet, isodoublet, isotriplet
        S=-1, S=0, etc.
        """
        res_dict = {}
        twoI_str_map = {
            "isosinglet": 0,
            "isodoublet": 1,
            "isotriplet": 2,
        }
        # search for isospin substring
        pattern = "(" + "|".join(twoI_str_map.keys()) + ")"
        match = re.search(pattern, s.lower())
        if match:
            res_dict["twoI"] = twoI_str_map[match.group(0)]

        # search for strangeness S=...
        match = re.search(r"S=([-+]?\d+)", s)
        if match:
            res_dict["S"] = int(match.group(1))

        return cls(**res_dict)

    # Hash / equality
    def __hash__(self) -> int:
        """
        Default dataclass hash is fine, but we define it explicitly to make the
        "used as a key" intent obvious and stable.

        IMPORTANT: hash/equality use the literal stored values, including None.
        If you want "None is wildcard" semantics, use compatible_with().
        """
        return hash(
            (
                self.B,
                self.Q,
                self.S,
                self.C,
                self.twoI,
                self.twoI3,
                self.Gpar,
                self.Cpar,
                self.twoJ,
                self.par,
            )
        )

    def compatible_with(self, other: "SectorInfo") -> bool:
        """
        Two SectorInfo objects are compatible if they do not disagree on any field
        where BOTH have a specified (non-None) value.
        """
        for field in (
            "B",
            "Q",
            "S",
            "C",
            "twoI",
            "twoI3",
            "Gpar",
            "Cpar",
            "twoJ",
            "par",
        ):
            a = getattr(self, field)
            b = getattr(other, field)
            if a is not None and b is not None and a != b:
                return False
        return True

    # String / reprs
    def __str__(self) -> str:
        parts = []

        # Charges
        for name in ("B", "Q", "S", "C"):
            val = getattr(self, name)
            if val is not None:
                parts.append(f"{name}={val}")

        # Isospin
        isospin = _fmt_halfint(self.twoI)
        I3 = _fmt_halfint(self.twoI3)
        if isospin is not None:
            parts.append(f"I={isospin}")
        if I3 is not None:
            parts.append(f"I3={I3}")

        # Discrete parities
        if self.Cpar is not None:
            parts.append(f"C={self.Cpar:+d}")
        if self.Gpar is not None:
            parts.append(f"G={self.Gpar:+d}")

        # Spin/parity
        J = _fmt_halfint(self.twoJ)
        if J is not None:
            parts.append(f"J={J}")
        if self.par is not None:
            parts.append(f"P={self.par:+d}")

        return "SectorInfo(" + (", ".join(parts) if parts else "unspecified") + ")"

    def __repr__(self) -> str:
        # Explicit, unambiguous representation (good for logs/debug)
        return (
            "SectorInfo("
            f"B={self.B!r}, Q={self.Q!r}, S={self.S!r}, C={self.C!r}, "
            f"twoI={self.twoI!r}, twoI3={self.twoI3!r}, Gpar={self.Gpar!r}, Cpar={self.Cpar!r}, "
            f"twoJ={self.twoJ!r}, par={self.par!r}"
            ")"
        )

    def _repr_latex_(self) -> str:
        """
        Jupyter rich display: returns a LaTeX math string.

        Example output:
          $B=0,\ Q=0,\ S=-1,\ I=\frac{1}{2},\ I_3=\frac{1}{2},\ J=\frac{3}{2},\ P=+$
        """
        items = []

        # Charges
        for sym, val in (("B", self.B), ("Q", self.Q), ("S", self.S), ("C", self.C)):
            if val is not None:
                items.append(rf"{sym}={val}")

        # Isospin
        if self.twoI is not None:
            if self.twoI % 2 == 0:
                items.append(rf"I={self.twoI // 2}")
            else:
                items.append(rf"I=\frac{{{self.twoI}}}{{2}}")
        if self.twoI3 is not None:
            if self.twoI3 % 2 == 0:
                items.append(rf"I_3={self.twoI3 // 2}")
            else:
                items.append(rf"I_3=\frac{{{self.twoI3}}}{{2}}")

        # C/G parity
        if self.Cpar is not None:
            items.append(rf"C={self.Cpar:+d}")
        if self.Gpar is not None:
            items.append(rf"G={self.Gpar:+d}")

        # J^P
        if self.twoJ is not None:
            if self.twoJ % 2 == 0:
                items.append(rf"J={self.twoJ // 2}")
            else:
                items.append(rf"J=\frac{{{self.twoJ}}}{{2}}")
        if self.par is not None:
            items.append(rf"P={'+ ' if self.par > 0 else '-'}".replace(" ", ""))

        if not items:
            return r"$\mathrm{SectorInfo}:\ \text{unspecified}$"

        return r"$" + r",\ ".join(items) + r"$"
