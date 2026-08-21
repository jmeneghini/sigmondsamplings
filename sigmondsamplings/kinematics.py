"""
Two-particle kinematics on a finite lattice.

All masses and energies are expressed in reference mass units, i.e. the
inputs are ``mref * L`` (dimensionless box size), ``m1 / mref`` and
``m2 / mref``. Each input may be a plain float, a numpy array, or a
:class:`~sigmondsamplings.sampling.SigmondSampling`; the arithmetic is written
with numpy ufuncs so errors propagate through resamplings automatically.
"""

from math import isqrt

import numpy as np

from .energy_levels import EnergyObsInfo, Particle
from .sampling import SigmondSampling

# Any value the kinematic formulas accept: scalar, numpy array, or sampling.
KinemValue = float | np.ndarray | SigmondSampling


class TwoParticleKinem:
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
        self.mref_L = mref_L
        self.m1 = m1
        self.m2 = m2
        self.dvec = dvec

    @property
    def dvec(self) -> np.ndarray | None:
        """Total momentum direction as an integer 3-vector, or None if unset."""
        return self._dvec

    @dvec.setter
    def dvec(self, value: np.ndarray | list[int] | tuple[int, int, int] | None):
        if value is None:
            self._dvec = None
            return

        arr = np.asarray(value)
        if arr.shape != (3,):
            raise ValueError(f"dvec must be a 3-vector, got shape {arr.shape}")
        if not np.issubdtype(arr.dtype, np.integer):
            if not np.all(arr == np.round(arr)):
                raise ValueError(f"dvec must have integer components, got {value}")
            arr = arr.astype(int)
        self._dvec = arr

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
        pfac_sqr = self.momentum_factor_sqr
        e1 = np.sqrt(self.m1**2 + pfac_sqr * d1_sqr)
        e2 = np.sqrt(self.m2**2 + pfac_sqr * d2_sqr)
        result = e1 + e2

        if isinstance(result, SigmondSampling):
            obs_info = self._ni_obs_info(d1_sqr, d2_sqr, result.observable_info)
            if obs_info is not None:
                result.observable_info = obs_info
        return result

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
        dvec = self._require_dvec("ni_energies_in_interval")

        c_m1 = self._central(self.m1)
        c_m2 = self._central(self.m2)
        c_pfac = self._central(self.momentum_factor_sqr)
        c_emin = self._central(emin)
        c_emax = self._central(emax)

        if identical is None:
            name1 = self._particle_name(self.m1)
            name2 = self._particle_name(self.m2)
            if name1 is not None and name2 is not None:
                identical = name1 == name2
            else:
                identical = self.m1 is self.m2 or c_m1 == c_m2

        e1_max = c_emax - c_m2
        if e1_max < c_m1:
            return []
        d1sq_max = int((e1_max**2 - c_m1**2) / c_pfac)

        dx0, dy0, dz0 = (int(c) for c in dvec)
        pairs: set[tuple[int, int]] = set()
        radius = isqrt(d1sq_max)
        for x in range(-radius, radius + 1):
            rem_x = d1sq_max - x * x
            ry = isqrt(rem_x)
            for y in range(-ry, ry + 1):
                rem_y = rem_x - y * y
                rz = isqrt(rem_y)
                for z in range(-rz, rz + 1):
                    d1_sqr = x * x + y * y + z * z
                    ex, ey, ez = dx0 - x, dy0 - y, dz0 - z
                    d2_sqr = ex * ex + ey * ey + ez * ez
                    pair = (d1_sqr, d2_sqr)
                    if identical and d2_sqr < d1_sqr:
                        pair = (d2_sqr, d1_sqr)
                    pairs.add(pair)

        kept = []
        for d1_sqr, d2_sqr in pairs:
            e_central = np.sqrt(c_m1**2 + c_pfac * d1_sqr) + np.sqrt(c_m2**2 + c_pfac * d2_sqr)
            if c_emin <= e_central <= c_emax:
                kept.append((e_central, d1_sqr, d2_sqr))
        kept.sort()

        return [(d1_sqr, d2_sqr, self.ni_energy(d1_sqr, d2_sqr)) for _, d1_sqr, d2_sqr in kept]

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
            if isinstance(info, EnergyObsInfo) and len(info.particles) == 1:
                return info.particles[0].name
        return None

    def _ni_obs_info(self, d1_sqr: int, d2_sqr: int, base_info) -> EnergyObsInfo | None:
        """
        Build the EnergyObsInfo for an NI energy sampling, or None if the
        particle names cannot be resolved from the mass observables.
        """
        name1 = self._particle_name(self.m1)
        name2 = self._particle_name(self.m2)
        if name1 is None or name2 is None:
            return None

        particles = [Particle(name1, psq=int(d1_sqr)), Particle(name2, psq=int(d2_sqr))]

        # Propagate reference mode only when both masses agree on it.
        ref1 = self.m1.observable_info.ref_particle
        ref2 = self.m2.observable_info.ref_particle
        ref_particle = ref1 if ref1 == ref2 else None

        psq = self.dsq if self._dvec is not None else None
        parts = [] if psq is None else [f"PSQ{psq}"]
        parts += ["elab", "ni", str(particles[0]), str(particles[1])]
        if ref_particle is not None:
            parts.append("ref")

        return EnergyObsInfo(
            name="_".join(parts),
            ensemble_info=base_info.ensemble_info,
            psq=psq,
            energy_type="elab",
            particles=particles,
            ref_particle=ref_particle,
        )

    def __repr__(self):
        dvec = None if self._dvec is None else tuple(self._dvec)
        return (
            f"TwoParticleKinem(mref_L={self.mref_L!r}, m1={self.m1!r}, "
            f"m2={self.m2!r}, dvec={dvec})"
        )
