"""BrainChip toolchain orchestration for the Akida deployment pipeline.

This module wraps the two mandatory BrainChip tools:

1. **QuantizeML** — post-training quantization of a Keras model
2. **CNN2SNN** — conversion of a quantized Keras model to an Akida model

Both tools are **optional dependencies**: the module can be imported without
them installed.  Only the individual stage functions raise ``ImportError`` at
call time when the corresponding tool is absent.

The high-level entry point :func:`run_akida_flow` orchestrates the full
pipeline and writes per-stage artefacts and logs to a user-supplied output
directory.

Usage::

    from nir.akida.toolchain import run_akida_flow, QuantizeConfig
    from nir.akida import AkidaTargetProfile

    result = run_akida_flow(
        graph=nir_graph,
        profile=AkidaTargetProfile.v1(),
        output_dir="./akida_output",
        quantize_config=QuantizeConfig(num_samples=1024),
    )
    print(result)

All failures are surfaced as :class:`AkidaToolchainError` tagged with the
failing stage, so callers can route them to the correct documentation.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import nir

from .export_keras import AkidaExportError, export_keras
from .profiles import AkidaTargetProfile
from .validator import AkidaValidator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public exceptions and enums
# ---------------------------------------------------------------------------


class ToolchainStage(Enum):
    """Stage of the Akida toolchain where a failure occurred."""

    VALIDATION = "validation"
    EXPORT = "export"
    QUANTIZATION = "quantization"
    CONVERSION = "conversion"
    MAPPING = "mapping"


class AkidaToolchainError(Exception):
    """Raised when any stage of the Akida toolchain fails.

    Attributes:
        stage: The :class:`ToolchainStage` where the failure occurred.
        cause: The original exception (if any) from the BrainChip tool.
    """

    def __init__(
        self,
        message: str,
        stage: ToolchainStage,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.cause = cause

    def __str__(self) -> str:
        base = f"[{self.stage.value.upper()}] {super().__str__()}"
        if self.cause is not None:
            return f"{base}\n  Caused by: {self.cause}"
        return base


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QuantizeConfig:
    """Configuration passed to QuantizeML for post-training quantization.

    Attributes:
        num_samples: Number of calibration samples to use.
        batch_size: Calibration batch size.
        extra_kwargs: Additional keyword arguments forwarded verbatim to
            ``quantizeml.models.quantize``.  Consult the QuantizeML
            documentation for available options.
    """

    num_samples: int = 1024
    batch_size: int = 32
    extra_kwargs: dict = field(default_factory=dict)


@dataclass
class ConvertConfig:
    """Configuration passed to CNN2SNN for model conversion.

    Attributes:
        extra_kwargs: Additional keyword arguments forwarded verbatim to
            ``cnn2snn.convert``.
    """

    extra_kwargs: dict = field(default_factory=dict)


@dataclass
class ToolchainResult:
    """Result of a completed :func:`run_akida_flow` execution.

    Attributes:
        output_dir: Directory where artefacts were written.
        keras_model_path: Path to the exported Keras model.
        quantized_model_path: Path to the quantized Keras model, or None if
            the quantization stage was not reached.
        akida_model_path: Path to the Akida model, or None if the conversion
            stage was not reached.
        validation_report_path: Path to the JSON validation report.
        stages_completed: List of stages that completed successfully.
    """

    output_dir: pathlib.Path
    keras_model_path: Optional[pathlib.Path] = None
    quantized_model_path: Optional[pathlib.Path] = None
    akida_model_path: Optional[pathlib.Path] = None
    validation_report_path: Optional[pathlib.Path] = None
    stages_completed: list = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"ToolchainResult(output_dir={self.output_dir})"]
        for stage in self.stages_completed:
            lines.append(f"  ✓ {stage}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lazy imports for BrainChip tools
# ---------------------------------------------------------------------------


def _import_quantizeml() -> Any:
    try:
        import quantizeml  # type: ignore
        return quantizeml
    except ImportError:
        raise ImportError(
            "quantizeml is required for the quantization step.\n"
            "Install BrainChip's QuantizeML package:\n"
            "  pip install quantizeml\n"
            "See https://doc.brainchipinc.com/ for installation instructions."
        )


def _import_cnn2snn() -> Any:
    try:
        import cnn2snn  # type: ignore
        return cnn2snn
    except ImportError:
        raise ImportError(
            "cnn2snn is required for the conversion step.\n"
            "Install BrainChip's CNN2SNN package:\n"
            "  pip install cnn2snn\n"
            "See https://doc.brainchipinc.com/ for installation instructions."
        )


def _import_keras_for_save() -> Any:
    try:
        import tf_keras as keras  # type: ignore
        return keras
    except ImportError:
        try:
            import tensorflow.keras as keras  # type: ignore
            return keras
        except ImportError:
            try:
                import keras
                return keras
            except ImportError:
                raise ImportError(
                    "keras is required to save/load models.\n"
                    "  pip install keras  OR  pip install tensorflow"
                )


# ---------------------------------------------------------------------------
# Individual stage functions
# ---------------------------------------------------------------------------


def validate_graph(
    graph: nir.NIRGraph,
    profile: AkidaTargetProfile,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    """Run the Akida validator and write a JSON report to *output_dir*.

    Args:
        graph: NIR graph to validate.
        profile: Akida target profile.
        output_dir: Directory for artefact output.

    Returns:
        Path to the written JSON validation report.

    Raises:
        AkidaToolchainError: If validation finds blocking issues.
    """
    logger.info("Stage: validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = AkidaValidator(profile).validate(graph)

    report_dict = {
        "is_valid": report.is_valid,
        "diagnostics": [
            {
                "stage": d.stage.value,
                "node_name": d.node_name,
                "node_type": d.node_type,
                "message": d.message,
                "target_profile": d.target_profile,
            }
            for d in report.diagnostics
        ],
    }

    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report_dict, indent=2))
    logger.info("Validation report written to %s", report_path)

    if not report.is_valid:
        blocking = [d for d in report.diagnostics if d.stage.value in ("quantization", "conversion")]
        if blocking:
            raise AkidaToolchainError(
                f"Validation failed with {len(blocking)} blocking issue(s).  "
                f"See {report_path} for details.",
                stage=ToolchainStage.VALIDATION,
            )

    return report_path


def export_to_keras(
    graph: nir.NIRGraph,
    profile: AkidaTargetProfile,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    """Export *graph* to Keras and save to *output_dir*.

    Args:
        graph: NIR graph (must have passed validation).
        profile: Akida target profile.
        output_dir: Directory for artefact output.

    Returns:
        Path to the saved Keras model.

    Raises:
        AkidaToolchainError: If the export fails.
    """
    logger.info("Stage: export")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        keras_model = export_keras(
            graph,
            profile,
            skip_validation=True,
            prefer_tf_keras=True,
        )
    except AkidaExportError as exc:
        raise AkidaToolchainError(
            f"Keras export failed: {exc}",
            stage=ToolchainStage.EXPORT,
            cause=exc,
        ) from exc

    model_path = output_dir / "model.keras"
    try:
        keras_model.save(str(model_path))
    except Exception as exc:
        raise AkidaToolchainError(
            f"Failed to save Keras model: {exc}",
            stage=ToolchainStage.EXPORT,
            cause=exc,
        ) from exc

    logger.info("Keras model saved to %s", model_path)
    return model_path


def quantize_model(
    keras_model_path: pathlib.Path,
    profile: AkidaTargetProfile,
    output_dir: pathlib.Path,
    config: Optional[QuantizeConfig] = None,
    calibration_data: Optional[Any] = None,
) -> pathlib.Path:
    """Quantize a Keras model with QuantizeML.

    Args:
        keras_model_path: Path to the Keras model to quantize.
        profile: Akida target profile (used to select quantization bitwidths).
        output_dir: Directory for artefact output.
        config: QuantizeML configuration.  Defaults to ``QuantizeConfig()``.
        calibration_data: Optional numpy array of calibration samples.
            If None, QuantizeML uses its default calibration strategy.

    Returns:
        Path to the quantized Keras model.

    Raises:
        AkidaToolchainError: If QuantizeML fails.
        ImportError: If ``quantizeml`` is not installed.
    """
    if config is None:
        config = QuantizeConfig()

    qml = _import_quantizeml()
    keras = _import_keras_for_save()

    logger.info("Stage: quantization (QuantizeML)")
    log_path = output_dir / "quantization.log"

    try:
        model = keras.models.load_model(str(keras_model_path))
        kwargs = dict(config.extra_kwargs)
        if calibration_data is not None:
            kwargs["samples"] = calibration_data
        quantized = qml.models.quantize(model, **kwargs)
    except Exception as exc:
        log_path.write_text(traceback.format_exc())
        raise AkidaToolchainError(
            f"QuantizeML quantization failed.  Log written to {log_path}.  "
            f"Error: {exc}",
            stage=ToolchainStage.QUANTIZATION,
            cause=exc,
        ) from exc

    q_model_path = output_dir / "model_quantized.keras"
    quantized.save(str(q_model_path))
    log_path.write_text("QuantizeML completed successfully.")
    logger.info("Quantized model saved to %s", q_model_path)
    return q_model_path


def convert_model(
    quantized_model_path: pathlib.Path,
    profile: AkidaTargetProfile,
    output_dir: pathlib.Path,
    config: Optional[ConvertConfig] = None,
) -> pathlib.Path:
    """Convert a quantized Keras model to an Akida model with CNN2SNN.

    Args:
        quantized_model_path: Path to the quantized Keras model.
        profile: Akida target profile.
        output_dir: Directory for artefact output.
        config: CNN2SNN configuration.

    Returns:
        Path to the Akida model file.

    Raises:
        AkidaToolchainError: If CNN2SNN fails.
        ImportError: If ``cnn2snn`` is not installed.
    """
    if config is None:
        config = ConvertConfig()

    cnn2snn = _import_cnn2snn()
    keras = _import_keras_for_save()

    logger.info("Stage: conversion (CNN2SNN)")
    log_path = output_dir / "conversion.log"

    try:
        q_model = keras.models.load_model(str(quantized_model_path))
        kwargs = dict(config.extra_kwargs)
        akida_model = cnn2snn.convert(q_model, **kwargs)
    except Exception as exc:
        log_path.write_text(traceback.format_exc())
        raise AkidaToolchainError(
            f"CNN2SNN conversion failed.  Log written to {log_path}.  "
            f"Error: {exc}",
            stage=ToolchainStage.CONVERSION,
            cause=exc,
        ) from exc

    akida_path = output_dir / "model.fbz"
    try:
        akida_model.save(str(akida_path))
    except Exception as exc:
        log_path.write_text(traceback.format_exc())
        raise AkidaToolchainError(
            f"Failed to save Akida model: {exc}",
            stage=ToolchainStage.CONVERSION,
            cause=exc,
        ) from exc

    log_path.write_text("CNN2SNN conversion completed successfully.")
    logger.info("Akida model saved to %s", akida_path)
    return akida_path


# ---------------------------------------------------------------------------
# High-level orchestration entry point
# ---------------------------------------------------------------------------


def run_akida_flow(
    graph: nir.NIRGraph,
    profile: AkidaTargetProfile,
    output_dir: str | os.PathLike,
    quantize_config: Optional[QuantizeConfig] = None,
    convert_config: Optional[ConvertConfig] = None,
    calibration_data: Optional[Any] = None,
    stop_after: Optional[ToolchainStage] = None,
) -> ToolchainResult:
    """Run the full NIR → Akida pipeline.

    Executes stages in order:
    1. **Validation** — AkidaValidator checks the graph
    2. **Export** — NIR → Keras via :func:`export_to_keras`
    3. **Quantization** — Keras → quantized Keras via QuantizeML
    4. **Conversion** — quantized Keras → Akida via CNN2SNN

    Set *stop_after* to halt the pipeline after a specific stage.  This is
    useful for inspecting intermediate artefacts.

    All artefacts and logs are written to *output_dir*.

    Args:
        graph: NIR graph to deploy.
        profile: Akida hardware target profile.
        output_dir: Directory to write artefacts and logs.
        quantize_config: QuantizeML configuration.
        convert_config: CNN2SNN configuration.
        calibration_data: Calibration data array for QuantizeML.
        stop_after: Stop the pipeline after this stage.  Options:
            ``ToolchainStage.VALIDATION``, ``ToolchainStage.EXPORT``,
            ``ToolchainStage.QUANTIZATION``.  Defaults to running all stages.

    Returns:
        :class:`ToolchainResult` describing completed stages and artefact paths.

    Raises:
        AkidaToolchainError: On failure at any stage, tagged with the stage name.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = ToolchainResult(output_dir=out)

    # Stage 1: Validation
    result.validation_report_path = validate_graph(graph, profile, out)
    result.stages_completed.append(ToolchainStage.VALIDATION.value)
    if stop_after == ToolchainStage.VALIDATION:
        return result

    # Stage 2: Export
    result.keras_model_path = export_to_keras(graph, profile, out)
    result.stages_completed.append(ToolchainStage.EXPORT.value)
    if stop_after == ToolchainStage.EXPORT:
        return result

    # Stage 3: Quantization (requires QuantizeML)
    result.quantized_model_path = quantize_model(
        result.keras_model_path,
        profile,
        out,
        config=quantize_config,
        calibration_data=calibration_data,
    )
    result.stages_completed.append(ToolchainStage.QUANTIZATION.value)
    if stop_after == ToolchainStage.QUANTIZATION:
        return result

    # Stage 4: Conversion (requires CNN2SNN)
    result.akida_model_path = convert_model(
        result.quantized_model_path,
        profile,
        out,
        config=convert_config,
    )
    result.stages_completed.append(ToolchainStage.CONVERSION.value)

    return result
