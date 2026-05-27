"""nir.akida — BrainChip Akida backend for NIR.

This package provides:

- :class:`~nir.akida.profiles.AkidaTargetProfile` — hardware target definitions
- :class:`~nir.akida.metadata.QuantizationMetadata` — quantization schema
- :class:`~nir.akida.validator.AkidaValidator` — graph validation
- :class:`~nir.akida.diagnostics.ValidationReport` — structured diagnostics
- :func:`~nir.akida.export_keras.export_keras` — NIR → Keras export
  (requires ``keras`` or ``tensorflow``)

Toolchain orchestration (Phase 4) is provided in :mod:`nir.akida.toolchain`.

Quick start::

    import nir
    from nir.akida import AkidaTargetProfile, AkidaValidator
    from nir.akida.export_keras import export_keras

    graph = nir.read("my_model.nir")
    profile = AkidaTargetProfile.v1()
    validator = AkidaValidator(profile)
    report = validator.validate(graph)
    if not report.is_valid:
        print(report)
    else:
        model = export_keras(graph, profile)
        model.save("my_model.h5")
"""

from .diagnostics import AkidaDiagnostic, ValidationReport, ValidationStage
from .export_keras import AkidaExportError, export_keras
from .metadata import QuantizationMetadata
from .profiles import AkidaGeneration, AkidaTargetProfile
from .validator import AkidaValidator

__all__ = [
    "AkidaDiagnostic",
    "AkidaExportError",
    "AkidaGeneration",
    "AkidaTargetProfile",
    "AkidaValidator",
    "QuantizationMetadata",
    "ValidationReport",
    "ValidationStage",
    "export_keras",
]
