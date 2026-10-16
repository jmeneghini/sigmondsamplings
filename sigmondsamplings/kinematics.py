"""
Multi-particle kinematics on a finite lattice.

All masses and energies are expressed in reference mass units, i.e. the
inputs are ``mref * L`` (dimensionless box size) and ``m_i / mref``. Each input
may be a plain float, a numpy array, or a
:class:`~sigmondsamplings.sampling.SigmondSampling`; the arithmetic is written
with numpy ufuncs so errors propagate through resamplings automatically.

:class:`MultiParticleKinem` handles any number of particles and enumerates
momentum *vectors*; :class:`TwoParticleKinem` is its two-particle
specialization with the original ``(d1_sqr, d2_sqr)`` interface.
"""

from collections.abc import Sequence
from functools import lru_cache
from itertools import product
from math import isqrt

import numpy as np

from .energy_levels import EnergyObsInfo, Particle
from .sampling import SigmondSampling

# Any value the kinematic formulas accept: scalar, numpy array, or sampling.
KinemValue = float | np.ndarray | SigmondSampling

# Explicit identical-particle groups: each inner sequence holds particle indices.
IdenticalGroups = Sequence[Sequence[int]]


class MultiParticleKinem:
    """
    Kinematics for an n-particle system in a periodic box.

    Args:
        mref_L: Reference mass times box size, ``mref * L`` (dimensionless).
        masses: Particle masses in reference mass units, one per particle.
        dvec: Optional total momentum direction, an integer 3-vector ``d``
            such that ``P = (2 pi / L) d``. Required only by the methods
            that depend on the total momentum (e.g. ``dsq``, ``configurations``).
    """

    def __init__(
        self,
        mref_L: KinemValue,
        masses: Sequence[KinemValue],
        dvec: np.ndarray | list[int] | tuple[int, int, int] | None = None,
    ):
        if len(masses) == 0:
            raise ValueError("at least one mass is required")
        self.mref_L = mref_L
        self.masses = list(masses)
        self.dvec = dvec

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return len(self.masses)

    @property
    def dvec(self) -> np.ndarray | None:
        """Total momentum direction as an integer 3-vector, or None if unset."""
        return self._dvec

    @dvec.setter
    def dvec(self, value: np.ndarray | list[int] | tuple[int, int, int] | None):
        if value is None:
            self._dvec = None
            return
        self._dvec = _int_vector(value, "dvec")

    def _require_dvec(self, method_name: str) -> np.ndarray:
        if self._dvec is None:
            raise ValueError(
                f"{method_name} requires dvec; set it in the constructor or via the dvec property"
            )
        return self._dvec

    @property
    def dsq(self) -> int:
        """Integer momentum squared ``d . d`` of the total momentum vector."""
        dvec = self._require_dvec("dsq")
        return int(np.dot(dvec, dvec))

    @property
    def momentum_factor_sqr(self) -> KinemValue:
        """Squared lattice momentum unit ``(2 pi / (mref L))^2`` in reference mass units."""
        return (2.0 * np.pi / self.mref_L) ** 2

    @property
    def momentum_sqr(self) -> KinemValue:
        """Total momentum squared ``(2 pi / (mref L))^2 * dsq`` in reference mass units."""
        return self.momentum_factor_sqr * self.dsq

    def ni_energy(self, momenta) -> KinemValue:
        """
        Non-interacting (NI) lab-frame energy ``sum_i sqrt(m_i^2 + (2 pi / (mref L))^2 d_i^2)``.

        When the result is a :class:`SigmondSampling` and every particle name can
        be resolved from the mass observables (e.g. masses from single-hadron
        fits carrying :class:`SHEnergyObsInfo`), the result is tagged with an
        :class:`EnergyObsInfo` whose ``particles`` carry the momenta.

        Args:
            momenta: Either ``n`` integer momenta squared ``d_i^2``, or an
                ``(n, 3)`` array of integer momentum vectors. Vectors are checked
                to sum to ``dvec`` when ``dvec`` is set.

        Returns:
            NI energy in reference mass units.
        """
        return self._ni_energy(self._momentum_squares(momenta))

    def _ni_energy(self, dsqs: Sequence[int]) -> KinemValue:
        pfac_sqr = self.momentum_factor_sqr
        result = None
        for mass, d_sqr in zip(self.masses, dsqs):
            term = np.sqrt(mass**2 + pfac_sqr * d_sqr)
            result = term if result is None else result + term

        if isinstance(result, SigmondSampling):
            obs_info = self._ni_obs_info(dsqs, result.observable_info)
            if obs_info is not None:
                result.observable_info = obs_info
        return result

    def cm_energy(self, momenta) -> KinemValue:
        """
        NI center-of-mass energy ``sqrt(E_lab^2 - (2 pi / (mref L))^2 d^2)`` (requires ``dvec``).

        Args:
            momenta: As for :meth:`ni_energy`.
        """
        e_lab = self._ni_energy(self._momentum_squares(momenta))
        return np.sqrt(e_lab**2 - self.momentum_sqr)

    def configurations(
        self,
        emax: KinemValue,
        emin: KinemValue = 0.0,
        identical: bool | IdenticalGroups | None = None,
    ) -> list[tuple[np.ndarray, float]]:
        """
        All momentum-vector configurations with central NI energy in ``[emin, emax]``.

        Enumerates integer vectors ``d_1 + ... + d_n = dvec`` (requires ``dvec``),
        pruning each particle's search ball by the energy left once every later
        particle is given at least its mass. Configurations differing only by a
        permutation within an identical-particle group count once; the
        representative has that group's vectors in ascending lexicographic order.

        Interval filtering uses central values: the full-sample value for
        samplings, the mean for arrays, the value itself for floats.

        Args:
            emax: Upper edge of the energy interval (reference mass units).
            emin: Lower edge of the energy interval (reference mass units).
            identical: Identical-particle groups as sequences of particle
                indices; ``True`` (all particles identical) or ``False`` (all
                distinct). Default (None) groups particles with equal resolved
                names, or equal central masses when names are unavailable.

        Returns:
            List of ``(momenta, central_energy)`` sorted by central energy,
            where ``momenta`` is an ``(n, 3)`` integer array.
        """
        dvec = self._require_dvec("configurations")
        groups = self._identical_groups(identical)
        c_masses = [self._central(m) for m in self.masses]
        c_pfac = self._central(self.momentum_factor_sqr)
        c_emin = self._central(emin)
        c_emax = self._central(emax)

        # Minimal energy still needed by particles i, i+1, ..., n-1.
        tail_mass = np.cumsum(c_masses[::-1])[::-1].tolist() + [0.0]
        n = self.n_particles
        found: dict[tuple, tuple[np.ndarray, float]] = {}

        def recurse(i: int, chosen: list[tuple[int, int, int]], energy: float, total: np.ndarray):
            if i == n - 1:
                last = tuple(int(c) for c in dvec - total)
                e_last = np.sqrt(c_masses[i] ** 2 + c_pfac * _sqr(last))
                e_tot = energy + e_last
                if c_emin <= e_tot <= c_emax:
                    config = chosen + [last]
                    key = _canonical_key(config, groups)
                    if key not in found:
                        found[key] = (np.array(key, dtype=int), float(e_tot))
                return
            budget = c_emax - energy - tail_mass[i + 1]
            if budget < c_masses[i]:
                return
            dsq_max = int((budget**2 - c_masses[i] ** 2) / c_pfac + 1e-9)
            for vec in _ball(dsq_max):
                e_i = np.sqrt(c_masses[i] ** 2 + c_pfac * _sqr(vec))
                recurse(i + 1, chosen + [vec], energy + e_i, total + vec)

        if c_emax >= tail_mass[0]:
            recurse(0, [], 0.0, np.zeros(3, dtype=int))
        return sorted(found.values(), key=lambda item: (item[1], item[0].tolist()))

    def ni_energies_in_interval(
        self,
        emin: KinemValue,
        emax: KinemValue,
        identical: bool | IdenticalGroups | None = None,
    ) -> list[tuple[tuple[int, ...], KinemValue]]:
        """
        All NI energy levels with central value in ``[emin, emax]``, keyed by momenta squared.

        Since the energy depends only on the ``d_i^2``, the vector configurations
        of :meth:`configurations` are reduced to unique tuples of squares (sorted
        within each identical group) and only those are evaluated, so the
        (possibly resampling-sized) arithmetic runs once per level.

        Returns:
            List of ``(dsqs, energy)`` sorted by central energy, with each
            energy computed via :meth:`ni_energy`.
        """
        groups = self._identical_groups(identical)
        kept: dict[tuple[int, ...], float] = {}
        for momenta, e_central in self.configurations(emax, emin, groups):
            dsqs = [_sqr(p) for p in momenta]
            for group in groups:
                for idx, value in zip(group, sorted(dsqs[j] for j in group)):
                    dsqs[idx] = value
            kept.setdefault(tuple(dsqs), e_central)
        ordered = sorted(kept.items(), key=lambda item: (item[1], item[0]))
        return [(dsqs, self._ni_energy(dsqs)) for dsqs, _ in ordered]

    def _momentum_squares(self, momenta) -> list[int]:
        arr = np.asarray(momenta)
        if arr.ndim == 1:
            if len(arr) != self.n_particles:
                raise ValueError(f"expected {self.n_particles} momenta squared, got {len(arr)}")
            return [int(v) for v in arr]
        if arr.shape != (self.n_particles, 3):
            raise ValueError(f"expected momenta of shape ({self.n_particles}, 3), got {arr.shape}")
        vectors = [_int_vector(p, "momentum") for p in arr]
        if self._dvec is not None and not np.array_equal(sum(vectors), self._dvec):
            raise ValueError(f"momenta sum to {tuple(sum(vectors))}, not dvec {tuple(self._dvec)}")
        return [_sqr(p) for p in vectors]

    def _identical_groups(self, identical: bool | IdenticalGroups | None) -> list[list[int]]:
        """Identical-particle index groups with at least two members."""
        n = self.n_particles
        if identical is True:
            groups = [list(range(n))]
        elif identical is False:
            groups = []
        elif identical is not None:
            groups = [sorted(int(i) for i in group) for group in identical]
            flat = [i for group in groups for i in group]
            if len(flat) != len(set(flat)) or any(not 0 <= i < n for i in flat):
                raise ValueError(
                    f"identical groups must be disjoint indices below {n}: {identical}"
                )
        else:
            names = [self._particle_name(m) for m in self.masses]
            centrals = [self._central(m) for m in self.masses]
            groups = []
            assigned = [False] * n
            for i in range(n):
                if assigned[i]:
                    continue
                group = [i]
                for j in range(i + 1, n):
                    if assigned[j]:
                        continue
                    if names[i] is not None and names[j] is not None:
                        same = names[i] == names[j]
                    else:
                        same = self.masses[i] is self.masses[j] or centrals[i] == centrals[j]
                    if same:
                        group.append(j)
                        assigned[j] = True
                groups.append(group)
        return [group for group in groups if len(group) > 1]

    @staticmethod
    def _central(value: KinemValue) -> float:
        """Central value: full sample for samplings, mean for arrays, else the value."""
        if isinstance(value, SigmondSampling):
            return float(value.full_sample_value)
        arr = np.asarray(value)
        if arr.ndim == 0:
            return float(arr)
        return float(arr.mean())

    @staticmethod
    def _particle_name(mass: KinemValue) -> str | None:
        """Resolve the particle name from a mass sampling's observable metadata."""
        if isinstance(mass, SigmondSampling):
            info = mass.observable_info
            if isinstance(info, EnergyObsInfo) and info.is_single_hadron:
                return info.hadron
        return None

    def _ni_obs_info(self, dsqs: Sequence[int], base_info) -> EnergyObsInfo | None:
        """
        Build the EnergyObsInfo for an NI energy sampling, or None if the
        particle names cannot be resolved from the mass observables.
        """
        names = [self._particle_name(m) for m in self.masses]
        if any(name is None for name in names):
            return None

        particles = [Particle(name, psq=int(d_sqr)) for name, d_sqr in zip(names, dsqs)]

        # Propagate reference mode only when all masses agree on it.
        refs = {m.observable_info.ref_particle for m in self.masses}
        ref_particle = refs.pop() if len(refs) == 1 else None

        psq = self.dsq if self._dvec is not None else None
        return EnergyObsInfo(
            ensemble_info=base_info.ensemble_info,
            psq=psq,
            energy_type="elab",
            ni_particles=particles,
            non_interacting=True,
            ref_particle=ref_particle,
        )

    def __repr__(self):
        dvec = None if self._dvec is None else tuple(self._dvec)
        return f"MultiParticleKinem(mref_L={self.mref_L!r}, masses={self.masses!r}, dvec={dvec})"


class TwoParticleKinem(MultiParticleKinem):
    """
    Kinematics for a two-particle system in a periodic box.

    Args:
        mref_L: Reference mass times box size, ``mref * L`` (dimensionless).
        m1: Mass of particle 1 in reference mass units.
        m2: Mass of particle 2 in reference mass units.
        dvec: Optional total momentum direction, an integer 3-vector ``d``
            such that ``P = (2 pi / L) d``. Required only by the methods
            that depend on the total momentum (e.g. ``dsq``, ``momentum_sqr``).
    """

    def __init__(
        self,
        mref_L: KinemValue,
        m1: KinemValue,
        m2: KinemValue,
        dvec: np.ndarray | list[int] | tuple[int, int, int] | None = None,
    ):
        super().__init__(mref_L, [m1, m2], dvec)

    @property
    def m1(self) -> KinemValue:
        return self.masses[0]

    @m1.setter
    def m1(self, value: KinemValue):
        self.masses[0] = value

    @property
    def m2(self) -> KinemValue:
        return self.masses[1]

    @m2.setter
    def m2(self, value: KinemValue):
        self.masses[1] = value

    def ni_energy(self, d1_sqr: int, d2_sqr: int) -> KinemValue:
        """
        Non-interacting (NI) two-particle lab-frame energy.

        ``E_NI = sqrt(m1^2 + (2 pi / (mref L))^2 d1^2) + sqrt(m2^2 + (2 pi / (mref L))^2 d2^2)``

        When the result is a :class:`SigmondSampling` and both particle names can
        be resolved from the mass observables (e.g. masses from single-hadron
        fits carrying :class:`SHEnergyObsInfo`), the result is tagged with an
        :class:`EnergyObsInfo` whose ``particles`` carry the momenta
        ``(Particle(p1, psq=d1_sqr), Particle(p2, psq=d2_sqr))``.

        Args:
            d1_sqr: Integer momentum squared of particle 1.
            d2_sqr: Integer momentum squared of particle 2.

        Returns:
            NI energy in reference mass units.
        """
        return self._ni_energy([d1_sqr, d2_sqr])

    def ni_energies_in_interval(
        self,
        emin: KinemValue,
        emax: KinemValue,
        identical: bool | None = None,
    ) -> list[tuple[int, int, KinemValue]]:
        """
        All NI energy levels with central value in ``[emin, emax]``.

        Enumerates integer momentum pairs ``d1 + d2 = dvec`` (requires ``dvec``).
        Since the energy depends only on ``(d1^2, d2^2)``, vectors are first
        reduced to unique squared pairs and only those are evaluated, so the
        (possibly resampling-sized) arithmetic runs once per level. The search
        ball is bounded by ``sqrt(m1^2 + pfac^2 d1^2) <= emax - m2``.

        Interval filtering uses central values: the full-sample value for
        samplings, the mean for arrays, the value itself for floats.

        Args:
            emin: Lower edge of the energy interval (reference mass units).
            emax: Upper edge of the energy interval (reference mass units).
            identical: Treat the two particles as identical so that
                ``(d1_sqr, d2_sqr)`` and ``(d2_sqr, d1_sqr)`` count as one
                level. Default (None) auto-detects: equal resolved particle
                names, or equal central masses when names are unavailable.

        Returns:
            List of ``(d1_sqr, d2_sqr, energy)`` sorted by central energy,
            with each energy computed via :meth:`ni_energy` (so sampling
            results carry the tagged EnergyObsInfo metadata).
        """
        self._require_dvec("ni_energies_in_interval")
        levels = super().ni_energies_in_interval(emin, emax, identical)
        return [(d1_sqr, d2_sqr, energy) for (d1_sqr, d2_sqr), energy in levels]

    def __repr__(self):
        dvec = None if self._dvec is None else tuple(self._dvec)
        return (
            f"TwoParticleKinem(mref_L={self.mref_L!r}, m1={self.m1!r}, m2={self.m2!r}, dvec={dvec})"
        )


def _int_vector(value, what: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (3,):
        raise ValueError(f"{what} must be a 3-vector, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(arr == np.round(arr)):
            raise ValueError(f"{what} must have integer components, got {value}")
        arr = arr.astype(int)
    return arr


def _sqr(vec) -> int:
    return int(sum(int(c) * int(c) for c in vec))


@lru_cache(maxsize=64)
def _ball(dsq_max: int) -> tuple[tuple[int, int, int], ...]:
    """All integer 3-vectors with squared norm at most ``dsq_max``."""
    if dsq_max < 0:
        return ()
    radius = isqrt(dsq_max)
    rng = range(-radius, radius + 1)
    return tuple(v for v in product(rng, rng, rng) if _sqr(v) <= dsq_max)


def _canonical_key(config: list[tuple[int, int, int]], groups: list[list[int]]) -> tuple:
    """Sort the vectors within each identical group so permuted configurations coincide."""
    config = list(config)
    for group in groups:
        for idx, vec in zip(group, sorted(config[j] for j in group)):
            config[idx] = vec
    return tuple(config)
