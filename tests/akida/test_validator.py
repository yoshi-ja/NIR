"""Unit tests for the Akida validation module.

Covers:
- Valid graphs (zero diagnostics)
- CONVERSION diagnostics (unsupported op)
- QUANTIZATION diagnostics (missing / invalid metadata)
- MAPPING_RISK diagnostics (non-uniform threshold, non-zero reset)
- Profile-specific checks (Conv1d on v1, depthwise conv)

No BrainChip packages are required.
"""

import numpy as np
import pytest

import nir
from nir.akida import (
    AkidaDiagnostic,
    AkidaTargetProfile,
    AkidaValidator,
    QuantizationMetadata,
    ValidationReport,
    ValidationStage,
)

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _quant_meta(weight_bits: int = 8, activation_bits: int = 4) -> dict:
    return QuantizationMetadata(
        weight_bits=weight_bits, activation_bits=activation_bits
    ).to_dict()


def _make_linear(n: int = 4, with_quant: bool = True) -> nir.Linear:
    node = nir.Linear(weight=np.eye(n))
    if with_quant:
        node.metadata.update(_quant_meta())
    return node


def _make_affine(n: int = 4, with_quant: bool = True) -> nir.Affine:
    node = nir.Affine(weight=np.eye(n), bias=np.zeros(n))
    if with_quant:
        node.metadata.update(_quant_meta())
    return node


def _make_lif(n: int = 4, uniform_threshold: bool = True, zero_reset: bool = True):
    threshold = np.ones(n) if uniform_threshold else np.arange(1, n + 1, dtype=float)
    v_reset = np.zeros(n) if zero_reset else np.ones(n) * 0.1
    return nir.LIF(
        tau=np.ones(n),
        r=np.ones(n),
        v_leak=np.zeros(n),
        v_threshold=threshold,
        v_reset=v_reset,
    )


def _make_conv2d(
    h: int = 4,
    w: int = 4,
    dilation: int = 1,
    groups: int = 1,
    with_quant: bool = True,
) -> nir.Conv2d:
    c_out, c_in = 8, 4
    groups = min(groups, c_in)
    node = nir.Conv2d(
        input_shape=(h, w),
        weight=np.zeros((c_out, c_in // groups, 3, 3)),
        bias=np.zeros(c_out),
        stride=1,
        padding=0,
        dilation=dilation,
        groups=groups,
    )
    if with_quant:
        node.metadata.update(_quant_meta())
    return node


def _simple_graph(*node_pairs) -> nir.NIRGraph:
    """Build a linear NIRGraph from (name, node) pairs plus Input/Output wrappers.

    Uses type_check=False so that tests for individual node problems are not
    blocked by shape mismatches between nodes.
    """
    names = [p[0] for p in node_pairs]
    nodes = {p[0]: p[1] for p in node_pairs}

    nodes["input"] = nir.Input(input_type=np.array([4]))
    nodes["output"] = nir.Output(output_type=np.array([4]))

    edges = [("input", names[0])]
    for a, b in zip(names[:-1], names[1:]):
        edges.append((a, b))
    edges.append((names[-1], "output"))

    return nir.NIRGraph(nodes=nodes, edges=edges, type_check=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def v1():
    return AkidaTargetProfile.v1()


@pytest.fixture
def v2():
    return AkidaTargetProfile.v2()


# ---------------------------------------------------------------------------
# Valid graph tests
# ---------------------------------------------------------------------------


class TestValidGraphs:
    def test_linear_lif_v1(self, v1):
        """Linear + LIF with quant metadata → no diagnostics."""
        graph = _simple_graph(("linear", _make_linear()), ("lif", _make_lif()))
        report = AkidaValidator(v1).validate(graph)
        assert report.is_valid, str(report)

    def test_affine_lif_v1(self, v1):
        """Affine + LIF → no diagnostics."""
        graph = _simple_graph(("affine", _make_affine()), ("lif", _make_lif()))
        report = AkidaValidator(v1).validate(graph)
        assert report.is_valid, str(report)

    def test_conv2d_lif_v1(self, v1):
        """Conv2d + LIF → no diagnostics."""
        graph = _simple_graph(
            ("conv", _make_conv2d()), ("lif", _make_lif(n=8))
        )
        report = AkidaValidator(v1).validate(graph)
        assert report.is_valid, str(report)

    def test_flatten_linear_lif_v1(self, v1):
        """Flatten + Linear + LIF → no diagnostics."""
        flatten = nir.Flatten(
            start_dim=0,
            end_dim=-1,
            input_type={"input": np.array([4])},
        )
        graph = _simple_graph(
            ("flatten", flatten),
            ("linear", _make_linear()),
            ("lif", _make_lif()),
        )
        report = AkidaValidator(v1).validate(graph)
        assert report.is_valid, str(report)

    def test_conv1d_v2_valid(self, v2):
        """Conv1d is valid on v2."""
        conv1d = nir.Conv1d(
            input_shape=16,
            weight=np.zeros((8, 4, 3)),
            bias=np.zeros(8),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        conv1d.metadata.update(_quant_meta())
        graph = _simple_graph(("conv1d", conv1d))
        report = AkidaValidator(v2).validate(graph)
        assert report.is_valid, str(report)


# ---------------------------------------------------------------------------
# CONVERSION diagnostic tests
# ---------------------------------------------------------------------------


class TestConversionDiagnostics:
    def test_delay_node_raises_conversion(self, v1):
        delay = nir.Delay(delay=np.array([0.001]))
        graph = _simple_graph(("delay", delay))
        report = AkidaValidator(v1).validate(graph)
        assert report.has_stage(ValidationStage.CONVERSION)
        diag = report.by_stage(ValidationStage.CONVERSION)[0]
        assert diag.node_name == "delay"
        assert diag.node_type == "Delay"

    def test_cuba_lif_raises_conversion(self, v1):
        node = nir.CubaLIF(
            tau_syn=np.ones(4),
            tau_mem=np.ones(4),
            r=np.ones(4),
            v_leak=np.zeros(4),
            v_threshold=np.ones(4),
        )
        graph = _simple_graph(("cuba", node))
        report = AkidaValidator(v1).validate(graph)
        assert report.has_stage(ValidationStage.CONVERSION)

    def test_li_raises_conversion(self, v1):
        node = nir.LI(tau=np.ones(4), r=np.ones(4), v_leak=np.zeros(4))
        graph = _simple_graph(("li", node))
        report = AkidaValidator(v1).validate(graph)
        assert report.has_stage(ValidationStage.CONVERSION)

    def test_standalone_threshold_raises_conversion(self, v1):
        node = nir.Threshold(threshold=np.array(1.0))
        graph = _simple_graph(("thresh", node))
        report = AkidaValidator(v1).validate(graph)
        assert report.has_stage(ValidationStage.CONVERSION)
        assert "fused" in report.by_stage(ValidationStage.CONVERSION)[0].message

    def test_conv2d_with_dilation_raises_conversion(self, v1):
        conv = _make_conv2d(dilation=2)
        graph = _simple_graph(("conv", conv))
        report = AkidaValidator(v1).validate(graph)
        assert report.has_stage(ValidationStage.CONVERSION)

    def test_conv1d_on_v1_raises_conversion(self, v1):
        conv1d = nir.Conv1d(
            input_shape=16,
            weight=np.zeros((8, 4, 3)),
            bias=np.zeros(8),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        conv1d.metadata.update(_quant_meta())
        graph = _simple_graph(("conv1d", conv1d))
        report = AkidaValidator(v1).validate(graph)
        assert report.has_stage(ValidationStage.CONVERSION)
        diag = report.by_stage(ValidationStage.CONVERSION)[0]
        assert diag.target_profile == "v1"

    def test_scale_raises_conversion(self, v1):
        node = nir.Scale(scale=np.array(2.0))
        graph = _simple_graph(("scale", node))
        report = AkidaValidator(v1).validate(graph)
        assert report.has_stage(ValidationStage.CONVERSION)


# ---------------------------------------------------------------------------
# QUANTIZATION diagnostic tests
# ---------------------------------------------------------------------------


class TestQuantizationDiagnostics:
    def test_linear_without_quant_raises_quantization(self, v1):
        graph = _simple_graph(("linear", _make_linear(with_quant=False)))
        report = AkidaValidator(v1).validate(graph)
        assert report.has_stage(ValidationStage.QUANTIZATION)
        diag = report.by_stage(ValidationStage.QUANTIZATION)[0]
        assert "weight_bits" in diag.message

    def test_affine_without_quant_raises_quantization(self, v1):
        graph = _simple_graph(("affine", _make_affine(with_quant=False)))
        report = AkidaValidator(v1).validate(graph)
        assert report.has_stage(ValidationStage.QUANTIZATION)

    def test_conv2d_without_quant_raises_quantization(self, v1):
        graph = _simple_graph(("conv", _make_conv2d(with_quant=False)))
        report = AkidaValidator(v1).validate(graph)
        assert report.has_stage(ValidationStage.QUANTIZATION)

    def test_invalid_weight_bits_raises_quantization(self, v1):
        """weight_bits=3 is not a valid Akida value."""
        with pytest.raises(ValueError, match="weight_bits"):
            QuantizationMetadata(weight_bits=3, activation_bits=4)

    def test_unsupported_weight_bits_for_profile(self, v1):
        """weight_bits=16 is not in AkidaV1's supported set."""
        linear = nir.Linear(weight=np.eye(4))
        # Bypass QuantizationMetadata validation and inject directly
        linear.metadata["weight_bits"] = 16
        linear.metadata["activation_bits"] = 4
        graph = _simple_graph(("linear", linear))
        report = AkidaValidator(v1).validate(graph)
        assert report.has_stage(ValidationStage.QUANTIZATION)
        diag = report.by_stage(ValidationStage.QUANTIZATION)[0]
        assert "16" in diag.message


# ---------------------------------------------------------------------------
# MAPPING_RISK diagnostic tests
# ---------------------------------------------------------------------------


class TestMappingRiskDiagnostics:
    def test_non_uniform_threshold_on_v1_raises_mapping_risk(self, v1):
        graph = _simple_graph(("lif", _make_lif(uniform_threshold=False)))
        report = AkidaValidator(v1).validate(graph)
        assert report.has_stage(ValidationStage.MAPPING_RISK)
        diag = report.by_stage(ValidationStage.MAPPING_RISK)[0]
        assert diag.target_profile == "v1"
        assert "non-uniform" in diag.message

    def test_non_uniform_threshold_on_v2_no_risk(self, v2):
        """v2 does not require uniform threshold."""
        graph = _simple_graph(("lif", _make_lif(uniform_threshold=False)))
        report = AkidaValidator(v2).validate(graph)
        mapping_risks = report.by_stage(ValidationStage.MAPPING_RISK)
        # May have non-zero reset diagnostic but NOT non-uniform threshold
        threshold_diags = [d for d in mapping_risks if "non-uniform" in d.message]
        assert len(threshold_diags) == 0

    def test_non_zero_reset_raises_mapping_risk(self, v1):
        graph = _simple_graph(("lif", _make_lif(zero_reset=False)))
        report = AkidaValidator(v1).validate(graph)
        assert report.has_stage(ValidationStage.MAPPING_RISK)
        diag = report.by_stage(ValidationStage.MAPPING_RISK)[0]
        assert "reset" in diag.message.lower()

    def test_sumpool2d_raises_mapping_risk(self, v1):
        pool = nir.SumPool2d(
            kernel_size=np.array([2, 2]),
            stride=np.array([2, 2]),
            padding=np.array([0, 0]),
        )
        graph = _simple_graph(("pool", pool))
        report = AkidaValidator(v1).validate(graph)
        assert report.has_stage(ValidationStage.MAPPING_RISK)


# ---------------------------------------------------------------------------
# Report helper tests
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_summary_ok(self):
        r = ValidationReport()
        assert "OK" in r.summary()
        assert r.is_valid

    def test_summary_fail(self, v1):
        graph = _simple_graph(("delay", nir.Delay(delay=np.array([0.001]))))
        report = AkidaValidator(v1).validate(graph)
        assert "FAIL" in report.summary()
        assert not report.is_valid

    def test_str_contains_diagnostics(self, v1):
        graph = _simple_graph(("delay", nir.Delay(delay=np.array([0.001]))))
        report = AkidaValidator(v1).validate(graph)
        s = str(report)
        assert "delay" in s
        assert "CONVERSION" in s

    def test_by_stage_filters_correctly(self, v1):
        # Graph with both QUANTIZATION and CONVERSION issues
        graph = _simple_graph(
            ("linear", _make_linear(with_quant=False)),
            ("delay", nir.Delay(delay=np.array([0.001]))),
        )
        # Manually wire: linear → delay → output
        report = AkidaValidator(v1).validate(graph)
        quant = report.by_stage(ValidationStage.QUANTIZATION)
        conv = report.by_stage(ValidationStage.CONVERSION)
        assert len(quant) >= 1
        assert len(conv) >= 1
