"""Apply an edit program to an observable set.

The executor behind ``ss edit``. :func:`apply_edits` takes an already-loaded
collection so notebooks and other front-ends can drive the same ops without going
through a file; :func:`edit_file` wraps it with load and write.

Ops mutate in place wherever they can. ``ObservableCollection.filter`` returns a
collection over the *same* observable objects rather than copies, so an op applied to
a filtered scope is visible in the collection it was filtered from - no split and
rejoin, and observable order in the output is preserved. The three places that break
that aliasing get explicit handling here:

* ``obs.replace`` copies, so scoped attribute writes use ``obs.update`` instead.
* ``create_ref`` appends to the scope's own list, so it returns what it created and
  :func:`_add_ref` extends the working collection with it.
* in-place writes leave cached shared attributes stale, so the cache is cleared.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..edit_spec import (
    AddRefOp,
    DropOp,
    EditResolved,
    KeepOp,
    ScopedOp,
    SetOp,
    SetRefOp,
    TagEnergyOp,
)
from ..energy_level_collection import SingleEnsembleEnergyCollection
from ..energy_levels import EnergyObsInfo
from ..ensemble_collection import SingleEnsembleCollection
from ..selection import obs_attrs, unknown_attr_message
from .loader import DEFAULT_GROUP, SigmondLoader
from .writer import SigmondWriter

logger = logging.getLogger(__name__)

__all__ = ["apply_edits", "edit_file"]


def apply_edits(collection, spec: EditResolved):
    """Apply every op in *spec*, in order, and return the resulting collection.

    The input collection may be mutated in place; the returned collection is what
    should be written, and differs in identity from the input only when an op changed
    the observable set (``keep``, ``drop``, ``add-ref``, ``tag-energy``).

    Args:
        collection: Any single-ensemble observable collection.
        spec: Resolved edit program, from :meth:`EditSpec.resolve`.

    Returns:
        The edited collection.

    Raises:
        ValueError: For an op that cannot apply to this collection - an energy-aware
            op with no energy levels present, or a reference particle with no (or
            more than one) matching single-hadron observable.
    """
    for op in spec.edit:
        collection = _APPLY[type(op)](collection, op)
        collection.clear_shared_attr_cache()
    return collection


def edit_file(
    input_file: str,
    output_file: str,
    spec: EditResolved,
    in_group: str | None = None,
    out_group: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Load *input_file*, apply *spec*, and write the result to *output_file*.

    Args:
        input_file: Input Sigmond file (.smp, .bins, .fstream, .hdf5).
        output_file: Output path. A ``.h5``/``.hdf5`` suffix is preserved; if omitted,
            the input HDF5 suffix is used, with ``.hdf5`` as fallback.
        spec: Resolved edit program.
        in_group: Root group to read from a multi-group HDF5 input (None = auto-detect).
        out_group: Root group for the output (default: input group or DEFAULT_GROUP).
        overwrite: Overwrite (and back up) an existing output file.

    Returns:
        Path to the written HDF5 file.
    """
    loader = SigmondLoader(filename=input_file, group=in_group)
    collection = loader.observables
    if not len(collection):
        raise ValueError(f"No observables found in {input_file}")

    collection = apply_edits(collection, spec)
    if not len(collection):
        raise ValueError("The edit removed every observable; nothing to write.")

    group = out_group or loader.group or DEFAULT_GROUP
    out_path = SigmondWriter.hdf5_output_path(input_file, output_file)
    collection.to_hdf5(str(out_path), overwrite=overwrite, group=group, mode="w")
    return out_path


# ---------------------------------------------------------------------------
# Scope helpers
# ---------------------------------------------------------------------------


def _is_energy(sampling) -> bool:
    """Whether this observable carries energy-level metadata (SH levels included)."""
    return isinstance(sampling.observable_info, EnergyObsInfo)


def _energy_view(collection, *, op_name: str) -> SingleEnsembleEnergyCollection:
    """An energy-level collection over the energy-typed observables of *collection*.

    Shares the observable objects, so mutating the view mutates *collection*. Plain
    observables are simply absent from the view and are left untouched by energy ops.
    """
    if isinstance(collection, SingleEnsembleEnergyCollection):
        return collection
    energy = [sampling for sampling in collection if _is_energy(sampling)]
    if not energy:
        raise ValueError(
            f"'{op_name}' needs energy levels, but none of the {len(collection)} "
            "observables carry energy metadata. Run a 'tag-energy' op first."
        )
    return SingleEnsembleEnergyCollection(energy, return_type="list")


def _scope(collection, op: ScopedOp, *, energy: bool = False):
    """The subset *op* applies to.

    Resolved against the energy view when the op is energy-aware or selects by
    spectrum spec (``filter_from_toml`` is an energy-collection method), and against
    the whole collection otherwise.
    """
    base = _energy_view(collection, op_name=op.op) if (energy or op.spec) else collection
    return op.scope(base)


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------


def _tag_energy(collection, op: TagEnergyOp):
    """Interpret observables as energy levels, keeping the rest untouched."""
    energy, plain = [], []
    for sampling in collection:
        # Already interpreted: keep it as is. as_energy_level re-parses the name, and
        # the name cannot carry ref_particle or ni_pairs, so re-tagging an edited file
        # would silently drop them.
        if _is_energy(sampling):
            energy.append(sampling)
            continue
        try:
            energy.append(sampling.as_energy_level())
        except (ValueError, AttributeError) as exc:
            logger.debug(
                f"Keeping {sampling.observable_info.name} as a plain observable: {exc}"
            )
            plain.append(sampling)

    if op.skip_missing_particles:
        energy, skipped = _drop_missing_particles(energy)
        plain.extend(skipped)

    if not energy:
        if op.ni_yml:
            logger.warning("NI pair YAML given but no energy levels found; ignoring.")
        return SingleEnsembleCollection(plain)

    energy_coll = SingleEnsembleEnergyCollection(energy)
    if op.ni_yml:
        energy_coll.set_shift_particles_from_pycalq_yml(op.ni_yml)

    # "other wins" on collision, and the two parts are disjoint by construction.
    return SingleEnsembleCollection(plain) + energy_coll if plain else energy_coll


def _drop_missing_particles(energy):
    """Split off single-hadron levels whose particle name did not resolve."""
    from ..energy_levels import SHEnergyObsInfo

    kept, skipped = [], []
    for sampling in energy:
        obs_info = sampling.observable_info
        if isinstance(obs_info, SHEnergyObsInfo) and obs_info.particle is None:
            logger.warning(f"Single-hadron level missing a particle name: {obs_info}. Skipping.")
            skipped.append(sampling)
        else:
            kept.append(sampling)
    return kept, skipped


def _set(collection, op: SetOp):
    """Set attributes on the scope, in place, optionally resyncing canonical names."""
    scope = _scope(collection, op)
    if not len(scope):
        logger.warning("'set' matched no observables; nothing changed.")
        return collection

    # Attribute names cannot be checked when the flags are parsed - there is no
    # collection yet - so a typo would otherwise silently create a new attribute
    # that nothing reads and that the writer would drop.
    _check_settable(scope, op.attrs)
    try:
        scope.obs.update(**op.attrs)
    except AttributeError as exc:
        raise ValueError(f"Cannot set that attribute: {exc}") from exc

    if op.rename:
        # update_name is an energy-level method; plain observables have no canonical
        # form to resync to. strict=False keeps levels too incomplete to name.
        renamable = scope.filter(predicate=lambda obs_info: hasattr(obs_info, "update_name"))
        if len(renamable):
            renamable.obs.update_name(strict=False)
    return collection


def _check_settable(scope, attrs: dict) -> None:
    """Reject a 'set' naming an attribute the scope's observables do not have."""
    available = obs_attrs(scope)
    for attr in attrs:
        if attr not in available:
            raise ValueError(unknown_attr_message(attr, available, label="--set"))


def _set_ref(collection, op: SetRefOp):
    """Tag levels already in reference mode with the reference particle's name."""
    _scope(collection, op, energy=True).set_ref(op.particle)
    return collection


def _add_ref(collection, op: AddRefOp):
    """Divide each level in scope by a single-hadron mass, appending the ratios."""
    energy = _energy_view(collection, op_name=op.op)
    # Resolved against the whole collection, not the scope, so a scope can name the
    # levels to divide without having to include the reference particle itself.
    ref_sampling = _reference_sampling(energy, op)

    created = op.scope(energy).create_ref(ref_sampling)
    if not created:
        logger.info(f"Reference levels for {op.particle!r} already present; nothing added.")
        return collection
    return collection + type(collection)(created)


def _reference_sampling(energy: SingleEnsembleEnergyCollection, op: AddRefOp):
    """The single-hadron observable supplying the reference mass for *op*."""
    # is_ref=False: a previous add-ref leaves e.g. PSQ0_N_ref alongside PSQ0_N, and
    # that ratio is itself a single-hadron observable for the same particle. Excluding
    # reference levels is what keeps a second run of the same edit unambiguous.
    single_hadrons = energy.single_hadron_spectra.filter(is_ref=False)
    matches = single_hadrons.filter(particle=op.particle, psq=op.psq)
    if len(matches) == 1:
        return matches[0]

    if not len(matches):
        available = sorted(
            {
                f"{samp.observable_info.particle}(psq={samp.observable_info.psq})"
                for samp in single_hadrons
            }
        )
        raise ValueError(
            f"No single-hadron observable for particle {op.particle!r} at psq={op.psq}. "
            f"Available: {', '.join(available) or 'none'}."
        )

    raise ValueError(
        f"{len(matches)} single-hadron observables match particle {op.particle!r} at "
        f"psq={op.psq}; expected exactly one, so the reference mass is ambiguous."
    )


def _keep(collection, op: KeepOp):
    """Write only the scope."""
    return _scope(collection, op)


def _drop(collection, op: DropOp):
    """Write everything except the scope."""
    return collection - _scope(collection, op)


_APPLY = {
    TagEnergyOp: _tag_energy,
    SetOp: _set,
    SetRefOp: _set_ref,
    AddRefOp: _add_ref,
    KeepOp: _keep,
    DropOp: _drop,
}
