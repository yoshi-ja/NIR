"""nir.akida — BrainChip Akida backend for NIR.

This package provides:

- :class:`~nir.akida.profiles.AkidaTargetProfile` — hardware target definitions
- :class:`~nir.akida.metadata.QuantizationMetadata` — quantization schema
- :class:`~nir.akida.validator.AkidaValidator` — graph validation
- :class:`~nir.akida.diagnostics.ValidationReport` — structured diagnostics

Keras export (Phase 3) and toolchain orchestration (Phase 4) are provided in
:mod:`nir.akida.export_keras` and :mod:`nir.akida.toolchain` respectively.

Quick start::

    import nir
    from nir.akida import AkidaTargetProfile, AkidaValidator

    graph = nir.read("my_model.nir")
    profile = AkidaTargetProfile.v1()
    validator = AkidaValidator(profile)
    report = validator.validate(graph)
    if not report.is_valid:
        print(report)
"""

from .diagnostics import AkidaDiagnostic, ValidationReport, ValidationStage
from .metadata import QuantizationMetadata
from .profiles import AkidaGeneration, AkidaTargetProfile
from .validator import AkidaValidator

__all__ = [
    "AkidaDiagnostic",
    "AkidaGeneration",
    "AkidaTargetProfile",
    "AkidaValidator",
    "QuantizationMetadata",
    "ValidationReport",
    "ValidationStage",
]
