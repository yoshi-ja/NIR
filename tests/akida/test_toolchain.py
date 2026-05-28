"""Tests for the Akida toolchain orchestration module.

BrainChip tools (quantizeml, cnn2snn) are optional; all tests that require
them are skipped automatically when the packages are not installed.

Tests that do NOT require BrainChip tools:
- Stage routing and error classification
- validate_graph stage produces a JSON report
- run_akida_flow stops correctly after each stop_after option

Tests that require QuantizeML (skipped if absent):
- quantize_model invokes QuantizeML correctly

Tests that require CNN2SNN (skipped if absent):
- convert_model invokes CNN2SNN correctly
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

import nir
from nir.akida import AkidaTargetProfile, QuantizationMetadata
from nir.akida.toolchain import (
    AkidaToolchainError,
    ConvertConfig,
    QuantizeConfig,
    ToolchainStage,
    run_akida_flow,
    validate_graph,
)


def _has_keras() -> bool:
    try:
        import keras  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import tensorflow.keras  # noqa: F401
        return True
    except ImportError:
        return False


def _has_quantizeml() -> bool:
    try:
        import quantizeml  # noqa: F401
        return True
    except ImportError:
        return False


def _has_cnn2snn() -> bool:
    try:
        import cnn2snn  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_keras = pytest.mark.skipif(
    not _has_keras(),
    reason="keras/tensorflow not installed",
)
requires_quantizeml = pytest.mark.skipif(
    not _has_quantizeml(),
    reason="quantizeml not installed (BrainChip optional dependency)",
)
requires_cnn2snn = pytest.mark.skipif(
    not _has_cnn2snn(),
    reason="cnn2snn not installed (BrainChip optional dependency)",
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _quant_meta() -> dict:
    return QuantizationMetadata(weight_bits=8, activation_bits=4).to_dict()


def _valid_graph(n: int = 4) -> nir.NIRGraph:
    linear = nir.Linear(weight=np.eye(n))
    linear.metadata.update(_quant_meta())
    lif = nir.LIF(
        tau=np.ones(n),
        r=np.ones(n),
        v_leak=np.zeros(n),
        v_threshold=np.ones(n),
        v_reset=np.zeros(n),
    )
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type=np.array([n])),
            "linear": linear,
            "lif": lif,
            "output": nir.Output(output_type=np.array([n])),
        },
        edges=[
            ("input", "linear"),
            ("linear", "lif"),
            ("lif", "output"),
        ],
        type_check=False,
    )


def _invalid_graph() -> nir.NIRGraph:
    """Graph containing an unsupported node (Delay) for negative tests."""
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type=np.array([1])),
            "delay": nir.Delay(delay=np.array([0.001])),
            "output": nir.Output(output_type=np.array([1])),
        },
        edges=[("input", "delay"), ("delay", "output")],
        type_check=False,
    )


@pytest.fixture
def v1():
    return AkidaTargetProfile.v1()


@pytest.fixture
def tmp_out(tmp_path):
    return tmp_path / "akida_output"


# ---------------------------------------------------------------------------
# validate_graph tests (no BrainChip tools required)
# ---------------------------------------------------------------------------


class TestValidateGraph:
    def test_valid_graph_writes_report(self, v1, tmp_out):
        report_path = validate_graph(_valid_graph(), v1, tmp_out)
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["is_valid"] is True
        assert data["diagnostics"] == []

    def test_invalid_graph_raises_toolchain_error(self, v1, tmp_out):
        with pytest.raises(AkidaToolchainError) as exc_info:
            validate_graph(_invalid_graph(), v1, tmp_out)
        assert exc_info.value.stage == ToolchainStage.VALIDATION

    def test_invalid_graph_still_writes_report(self, v1, tmp_out):
        try:
            validate_graph(_invalid_graph(), v1, tmp_out)
        except AkidaToolchainError:
            pass
        report_path = tmp_out / "validation_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["is_valid"] is False
        assert len(data["diagnostics"]) >= 1

    def test_report_contains_stage_field(self, v1, tmp_out):
        try:
            validate_graph(_invalid_graph(), v1, tmp_out)
        except AkidaToolchainError:
            pass
        data = json.loads((tmp_out / "validation_report.json").read_text())
        for d in data["diagnostics"]:
            assert "stage" in d
            assert d["stage"] in ("quantization", "conversion", "mapping_risk")

    def test_output_dir_created_automatically(self, v1, tmp_path):
        nested = tmp_path / "deep" / "nested" / "output"
        assert not nested.exists()
        validate_graph(_valid_graph(), v1, nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# run_akida_flow: stop_after tests (no BrainChip tools required)
# ---------------------------------------------------------------------------


class TestRunAkidaFlowStopAfter:
    def test_stop_after_validation_no_keras_needed(self, v1, tmp_out):
        """stop_after=VALIDATION completes without Keras or BrainChip tools."""
        result = run_akida_flow(
            graph=_valid_graph(),
            profile=v1,
            output_dir=tmp_out,
            stop_after=ToolchainStage.VALIDATION,
        )
        assert ToolchainStage.VALIDATION.value in result.stages_completed
        assert ToolchainStage.EXPORT.value not in result.stages_completed
        assert result.validation_report_path is not None
        assert result.keras_model_path is None

    def test_stop_after_validation_invalid_graph_raises(self, v1, tmp_out):
        with pytest.raises(AkidaToolchainError) as exc_info:
            run_akida_flow(
                graph=_invalid_graph(),
                profile=v1,
                output_dir=tmp_out,
                stop_after=ToolchainStage.VALIDATION,
            )
        assert exc_info.value.stage == ToolchainStage.VALIDATION

    @requires_keras
    def test_stop_after_export(self, v1, tmp_out):
        result = run_akida_flow(
            graph=_valid_graph(),
            profile=v1,
            output_dir=tmp_out,
            stop_after=ToolchainStage.EXPORT,
        )
        assert ToolchainStage.EXPORT.value in result.stages_completed
        assert result.keras_model_path is not None
        assert result.keras_model_path.exists()
        assert result.quantized_model_path is None


# ---------------------------------------------------------------------------
# AkidaToolchainError tests
# ---------------------------------------------------------------------------


class TestAkidaToolchainError:
    def test_str_contains_stage(self):
        exc = AkidaToolchainError("test message", stage=ToolchainStage.QUANTIZATION)
        assert "QUANTIZATION" in str(exc)
        assert "test message" in str(exc)

    def test_cause_is_preserved(self):
        cause = ValueError("original error")
        exc = AkidaToolchainError("wrapper", stage=ToolchainStage.CONVERSION, cause=cause)
        assert exc.cause is cause
        assert "original error" in str(exc)

    def test_stage_attribute(self):
        exc = AkidaToolchainError("msg", stage=ToolchainStage.MAPPING)
        assert exc.stage == ToolchainStage.MAPPING


# ---------------------------------------------------------------------------
# Missing-dependency error messages
# ---------------------------------------------------------------------------


class TestMissingDependencyMessages:
    def test_quantize_without_quantizeml_raises_import_error(self, v1, tmp_out):
        if _has_quantizeml():
            pytest.skip("quantizeml is installed; skipping missing-dep test")
        from nir.akida.toolchain import quantize_model
        # Need a dummy keras model path; use a non-existent path to trigger
        # the import error before file-loading
        with pytest.raises(ImportError, match="quantizeml"):
            quantize_model(
                pathlib.Path("dummy_model.keras"), v1, tmp_out
            )

    def test_convert_without_cnn2snn_raises_import_error(self, v1, tmp_out):
        if _has_cnn2snn():
            pytest.skip("cnn2snn is installed; skipping missing-dep test")
        from nir.akida.toolchain import convert_model
        with pytest.raises(ImportError, match="cnn2snn"):
            convert_model(pathlib.Path("dummy.keras"), v1, tmp_out)


# ---------------------------------------------------------------------------
# QuantizeML integration tests (skipped without quantizeml)
# ---------------------------------------------------------------------------


@requires_quantizeml
class TestQuantizeModel:
    def test_quantize_model_produces_file(self, v1, tmp_out):
        from nir.akida.toolchain import export_to_keras, quantize_model

        graph = _valid_graph()
        keras_path = export_to_keras(graph, v1, tmp_out)
        q_path = quantize_model(keras_path, v1, tmp_out)
        assert q_path.exists()

    def test_quantize_log_written_on_success(self, v1, tmp_out):
        from nir.akida.toolchain import export_to_keras, quantize_model

        graph = _valid_graph()
        keras_path = export_to_keras(graph, v1, tmp_out)
        quantize_model(keras_path, v1, tmp_out)
        log_path = tmp_out / "quantization.log"
        assert log_path.exists()


# ---------------------------------------------------------------------------
# CNN2SNN integration tests (skipped without cnn2snn)
# ---------------------------------------------------------------------------


@requires_cnn2snn
class TestConvertModel:
    def test_convert_produces_fbz(self, v1, tmp_out):
        from nir.akida.toolchain import convert_model, export_to_keras, quantize_model

        graph = _valid_graph()
        keras_path = export_to_keras(graph, v1, tmp_out)
        q_path = quantize_model(keras_path, v1, tmp_out)
        akida_path = convert_model(q_path, v1, tmp_out)
        assert akida_path.exists()
        assert akida_path.suffix == ".fbz"
