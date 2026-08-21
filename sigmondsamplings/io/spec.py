from __future__ import annotations

from pathlib import Path

from pydantic import Field
from slat import StrictModel


def _normalize_hdf5_group(group: str) -> str:
    """Return an absolute-style HDF5 group path."""
    return group if group.startswith("/") else f"/{group}"


class SamplingDataSourceResolved(StrictModel):
    """Fully resolved sigmond-style HDF5 sampling-data location."""

    file: Path = Field(
        description="Absolute path to the validated HDF5 sampling file.",
    )
    group: str = Field(
        min_length=1,
        description="Resolved absolute-style HDF5 group containing sampling data.",
        examples=["/samplings", "/C103"],
    )


class SamplingDataResolved(StrictModel):
    """
    Resolved sampling-data sources available to the project.

    Mapping keys retain the stable user-defined identifiers from the spec.
    """

    sources: dict[str, SamplingDataSourceResolved] = Field(
        min_length=1,
        description=(
            "Mapping from project-local sampling-data source identifiers "
            "to fully resolved HDF5 locations."
        ),
    )


class SamplingDataSourceSpec(StrictModel):
    """Reference to a sigmond-style HDF5 sampling data source."""

    file: str = Field(
        min_length=1,
        description=(
            "Absolute or base-directory-relative path to a sigmond-style "
            "HDF5 sampling file."
        ),
    )
    group: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional HDF5 group containing sampling data. If omitted, "
            "resolution selects the group only when the file contains "
            "exactly one recognized sampling group."
        ),
        examples=["/samplings", "/C103"],
    )

    def resolve(self, *, base_dir: Path) -> SamplingDataSourceResolved:
        """Validate and resolve this source against ``base_dir``."""
        file = Path(self.file)
        if not file.is_absolute():
            file = base_dir / file

        file = file.resolve()

        if not file.is_file():
            raise FileNotFoundError(
                f"Sampling HDF5 file does not exist: {file}"
            )

        from .loader import verify_sigmond_hdf5

        valid, _file_kind, groups = verify_sigmond_hdf5(str(file))

        if not valid:
            raise ValueError(
                "File is not a recognized sigmond-style sampling HDF5 file: "
                f"{file}"
            )

        available_groups = tuple(
            _normalize_hdf5_group(group)
            for group in groups
        )

        if self.group is None:
            match available_groups:
                case (resolved_group,):
                    pass
                case ():
                    raise ValueError(
                        f"No sampling groups were found in HDF5 file: {file}"
                    )
                case _:
                    raise ValueError(
                        f"Multiple sampling groups were found in {file}: "
                        f"{available_groups}. Specify `group` explicitly."
                    )
        else:
            resolved_group = _normalize_hdf5_group(self.group)

            if resolved_group not in available_groups:
                raise ValueError(
                    f"Sampling group {resolved_group!r} was not found in {file}. "
                    f"Available groups: {available_groups}."
                )

        return SamplingDataSourceResolved(
            file=file,
            group=resolved_group,
        )


class SamplingDataSpec(StrictModel):
    """
    Named sampling-data sources available to the project.

    Mapping keys are arbitrary stable identifiers chosen by the user. They may
    describe an ensemble, file variant, preprocessing stage, or another
    project-local concept.
    """

    sources: dict[str, SamplingDataSourceSpec] = Field(
        min_length=1,
        description=(
            "Mapping from user-defined sampling-data source identifiers "
            "to HDF5 file and group specifications."
        ),
        examples=[
            {
                "c103_raw": {
                    "file": "data/c103_samplings.h5",
                    "group": "/samplings",
                },
                "hdibaryon_levels": {
                    "file": "levels/hdibaryon_levels_for_luscher.hdf5",
                },
            }
        ],
    )

    def resolve(self, *, base_dir: Path) -> SamplingDataResolved:
        """Resolve every named sampling-data source against ``base_dir``."""
        return SamplingDataResolved(
            sources={
                name: source.resolve(base_dir=base_dir)
                for name, source in self.sources.items()
            }
        )