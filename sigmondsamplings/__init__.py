"""
sigmondsamplings: A Python package for handling Sigmond samplings files.

This package provides tools to load, manipulate, and analyze Sigmond samplings files
in both fstream and HDF5 formats using the sigmond_query tool.

Public names are attached lazily (see ``__init__.pyi``) via ``lazy_loader`` so
that importing the package -- or any single submodule, e.g. ``loader`` for the
``ss-query`` CLI -- does not eagerly drag in heavy optional dependencies
(scipy, matplotlib, dask, ...). Each name imports its submodule on first access.
"""

import lazy_loader as lazy

# Load rc first so any sub-module that reads from `rc` at import time sees the
# user's persisted settings (e.g. KnownEnsembles' ensembles.xml_file). This is
# cheap (pure Python) and preserves the previous eager-import ordering guarantee.
from . import rcparams as rcparams

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

# ``rcparams`` is imported eagerly above (not via the stub), so add it back to
# the public surface that ``attach_stub`` derived from ``__init__.pyi``.
__all__ = [*__all__, "rcparams"]

__version__ = "0.1.0"
