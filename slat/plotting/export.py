"""Export metadata; dependency detection is advisory, never an export guarantee."""

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class ExportCapability:
    format: str
    dependency: str | None = None

    @property
    def available(self) -> bool:
        return self.dependency is None or find_spec(self.dependency) is not None


@dataclass(frozen=True)
class BackendExports:
    formats: tuple[ExportCapability, ...]
    defaults: tuple[str, ...]

    @property
    def accepted(self) -> tuple[str, ...]:
        return tuple(item.format for item in self.formats)


def export_capabilities(backend: str) -> BackendExports:
    if backend == "matplotlib":
        return BackendExports(
            tuple(
                ExportCapability(f)
                for f in (
                    "png",
                    "pdf",
                    "svg",
                    "jpg",
                    "jpeg",
                    "eps",
                    "pgf",
                    "ps",
                "raw",
                "rgba",
                "svgz",
                    "tif",
                    "tiff",
                    "webp",
                )
            ),
            ("png", "pdf"),
        )
    if backend == "plotly":
        return BackendExports(
            (
                ExportCapability("html"),
                ExportCapability("htm"),
                *(
                    ExportCapability(f, "kaleido")
                    for f in ("png", "pdf", "svg", "jpg", "jpeg", "webp")
                ),
            ),
            ("html",),
        )
    if backend == "recording":
        return BackendExports((), ())
    raise ValueError(f"unknown export backend {backend!r}")
