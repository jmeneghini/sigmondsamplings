"""optimagic-native fitting layer for SigmondSamplings.

Layers:

* parameter description — authoring specs (:class:`ParamSpec`,
  :class:`ConstraintSpec`, :class:`ParamSetSpec`) resolved via
  :meth:`ParamSetSpec.resolve` into the canonical, optimagic-facing
  :class:`ParamSetResolved` (dict-PyTree params, bounds, constraints).
* minimizer selection — :class:`MinimizerConfig` (a TOML-round-trippable wrapper
  over an optimagic algorithm and its option groups).
* model & objective — :class:`Model`/:class:`ResampledModel`/:class:`CallableModel`
  (flat theory vector) and :class:`LeastSquaresObjective`/:class:`CallableObjective`
  (whitened residuals marked for optimagic least-squares).
* driver & result — :class:`SamplingFit` (``fit_one``/``fit_full_sample``/
  ``fit_resampled``) producing a :class:`FitResult` that keeps both the native
  optimagic result and the framework SigmondSampling outputs.

* post-fit propagation & plotting — :class:`SigmondModelFunc` (error propagation
  through fitted parameter samplings) and the plotting helpers
  (:func:`plot_fit_result`, :func:`plot_chi2_1d`/``2d``, …) with their style
  dataclasses.

Public names are attached lazily (see ``__init__.pyi``) via ``lazy_loader`` so
that importing this subpackage -- e.g. via ``sigmondsamplings.fitting`` or a
top-level lazy name -- does not eagerly drag in heavy optional dependencies
(optimagic, scipy, matplotlib, ...). Each name imports its submodule on first
access.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
