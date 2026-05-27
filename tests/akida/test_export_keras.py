"""Tests for the NIR → Keras export for Akida.

All tests are skipped automatically when Keras / TensorFlow is not installed.
The golden fixture tests verify that the exported layer sequence matches the
expected sequence, ensuring that changes to the exporter are detected.
"""

import numpy as np
import pytest

keras = pytest.importorskip(
    "keras",
    reason="keras not installed; skipping export tests (pip install keras)",
)
# Fallback: try tensorflow.keras
try:
    import keras  # noqa: F811
except ImportError:
    try:
        import tensorflow.keras as keras  # type: ignore  # noqa: F811
    except ImportError:
        pytest.skip("Neither keras nor tensorflow is installed", allow_module_level=True)


import nir
from nir.akida import AkidaExportError, AkidaTargetProfile, QuantizationMetadata
from nir.akida.export_keras import export_keras

# ---------------------------------------------------------------------------
# Test helpers (mirror those from test_validator.py)
# ---------------------------------------------------------------------------


def _quant_meta(weight_bits: int = 8, activation_bits: int = 4) -> dict:
    return QuantizationMetadata(
        weight_bits=weight_bits, activation_bits=activation_bits
    ).to_dict()


def _make_linear(n: int = 4) -> nir.Linear:
    node = nir.Linear(weight=np.eye(n))
    node.metadata.update(_quant_meta())
    return node


def _make_affine(n: int = 4) -> nir.Affine:
    node = nir.Affine(weight=np.eye(n), bias=np.zeros(n))
    node.metadata.update(_quant_meta())
    return node


def _make_lif(n: int = 4) -> nir.LIF:
    return nir.LIF(
        tau=np.ones(n),
        r=np.ones(n),
        v_leak=np.zeros(n),
        v_threshold=np.ones(n),
        v_reset=np.zeros(n),
    )


def _make_conv2d(with_quant: bool = True) -> nir.Conv2d:
    node = nir.Conv2d(
        input_shape=(8, 8),
        weight=np.zeros((8, 4, 3, 3)),
        bias=np.zeros(8),
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    if with_quant:
        node.metadata.update(_quant_meta())
    return node


def _sequential_graph(nodes: dict, edges: list) -> nir.NIRGraph:
    return nir.NIRGraph(nodes=nodes, edges=edges, type_check=False)


@pytest.fixture
def v1():
    return AkidaTargetProfile.v1()


@pytest.fixture
def v2():
    return AkidaTargetProfile.v2()


# ---------------------------------------------------------------------------
# Layer sequence golden fixture tests
# ---------------------------------------------------------------------------


class TestLayerSequenceGolden:
    """Verify that the Keras layer names/types match expected golden sequences.

    These tests act as regression guards: if the exporter changes how a node
    is mapped, the test fails and the change must be intentional.
    """

    def _layer_types(self, model) -> list:
        # Skip the InputLayer at index 0
        return [type(l).__name__ for l in model.layers if not isinstance(l, keras.layers.InputLayer)]

    def test_linear_lif_sequence(self, v1):
        nodes = {
            "input": nir.Input(input_type=np.array([4])),
            "linear": _make_linear(),
            "lif": _make_lif(),
            "output": nir.Output(output_type=np.array([4])),
        }
        graph = _sequential_graph(
            nodes, [("input", "linear"), ("linear", "lif"), ("lif", "output")]
        )
        model = export_keras(graph, v1)
        types = self._layer_types(model)
        assert types == ["Dense", "ReLU"]

    def test_affine_lif_sequence(self, v1):
        nodes = {
            "input": nir.Input(input_type=np.array([4])),
            "affine": _make_affine(),
            "lif": _make_lif(),
            "output": nir.Output(output_type=np.array([4])),
        }
        graph = _sequential_graph(
            nodes,
            [("input", "affine"), ("affine", "lif"), ("lif", "output")],
        )
        model = export_keras(graph, v1)
        types = self._layer_types(model)
        assert types == ["Dense", "ReLU"]

    def test_conv2d_lif_sequence(self, v1):
        nodes = {
            "input": nir.Input(input_type=np.array([4, 8, 8])),
            "conv": _make_conv2d(),
            "lif": _make_lif(n=8),
            "output": nir.Output(output_type=np.array([8])),
        }
        graph = _sequential_graph(
            nodes,
            [("input", "conv"), ("conv", "lif"), ("lif", "output")],
        )
        model = export_keras(graph, v1)
        types = self._layer_types(model)
        assert types == ["Conv2D", "ReLU"]

    def test_flatten_linear_sequence(self, v1):
        flatten = nir.Flatten(
            start_dim=0,
            end_dim=-1,
            input_type={"input": np.array([4])},
        )
        nodes = {
            "input": nir.Input(input_type=np.array([4])),
            "flatten": flatten,
            "linear": _make_linear(),
            "output": nir.Output(output_type=np.array([4])),
        }
        graph = _sequential_graph(
            nodes,
            [("input", "flatten"), ("flatten", "linear"), ("linear", "output")],
        )
        model = export_keras(graph, v1)
        types = self._layer_types(model)
        assert types == ["Flatten", "Dense"]

    def test_avgpool2d_sequence(self, v1):
        pool = nir.AvgPool2d(
            kernel_size=np.array([2, 2]),
            stride=np.array([2, 2]),
            padding=np.array([0, 0]),
        )
        nodes = {
            "input": nir.Input(input_type=np.array([4, 8, 8])),
            "pool": pool,
            "output": nir.Output(output_type=np.array([4, 4, 4])),
        }
        graph = _sequential_graph(
            nodes, [("input", "pool"), ("pool", "output")]
        )
        model = export_keras(graph, v1)
        types = self._layer_types(model)
        assert types == ["AveragePooling2D"]

    def test_sumpool2d_exported_as_avgpool2d(self, v1):
        """SumPool2d is approximated as AveragePooling2D (MAPPING_RISK emitted by validator)."""
        pool = nir.SumPool2d(
            kernel_size=np.array([2, 2]),
            stride=np.array([2, 2]),
            padding=np.array([0, 0]),
        )
        nodes = {
            "input": nir.Input(input_type=np.array([4, 8, 8])),
            "pool": pool,
            "output": nir.Output(output_type=np.array([4, 4, 4])),
        }
        graph = _sequential_graph(
            nodes, [("input", "pool"), ("pool", "output")]
        )
        model = export_keras(graph, v1, skip_validation=True)
        types = self._layer_types(model)
        assert types == ["AveragePooling2D"]

    def test_conv1d_sequence_v2(self, v2):
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
        nodes = {
            "input": nir.Input(input_type=np.array([4, 16])),
            "conv1d": conv1d,
            "output": nir.Output(output_type=np.array([8, 14])),
        }
        graph = _sequential_graph(
            nodes, [("input", "conv1d"), ("conv1d", "output")]
        )
        model = export_keras(graph, v2)
        types = self._layer_types(model)
        assert types == ["Conv1D"]


# ---------------------------------------------------------------------------
# Validation gating tests
# ---------------------------------------------------------------------------


class TestExportValidationGating:
    def test_unsupported_node_raises_export_error(self, v1):
        """Export must fail fast on unsupported nodes (validation gate)."""
        nodes = {
            "input": nir.Input(input_type=np.array([1])),
            "delay": nir.Delay(delay=np.array([0.001])),
            "output": nir.Output(output_type=np.array([1])),
        }
        graph = _sequential_graph(
            nodes, [("input", "delay"), ("delay", "output")]
        )
        with pytest.raises(AkidaExportError, match="Delay"):
            export_keras(graph, v1)

    def test_missing_quant_meta_raises_export_error(self, v1):
        """Linear without quantization metadata must fail before Keras export."""
        nodes = {
            "input": nir.Input(input_type=np.array([4])),
            "linear": nir.Linear(weight=np.eye(4)),  # no quant metadata
            "output": nir.Output(output_type=np.array([4])),
        }
        graph = _sequential_graph(
            nodes, [("input", "linear"), ("linear", "output")]
        )
        with pytest.raises(AkidaExportError):
            export_keras(graph, v1)

    def test_skip_validation_exports_anyway(self, v1):
        """skip_validation=True bypasses the validation gate."""
        nodes = {
            "input": nir.Input(input_type=np.array([4])),
            "linear": nir.Linear(weight=np.eye(4)),  # no quant metadata
            "output": nir.Output(output_type=np.array([4])),
        }
        graph = _sequential_graph(
            nodes, [("input", "linear"), ("linear", "output")]
        )
        # Should not raise even without quant metadata
        model = export_keras(graph, v1, skip_validation=True)
        assert model is not None


# ---------------------------------------------------------------------------
# Weight preservation tests
# ---------------------------------------------------------------------------


class TestWeightPreservation:
    def test_linear_weights_preserved(self, v1):
        w = np.random.randn(4, 4).astype(np.float32)
        node = nir.Linear(weight=w)
        node.metadata.update(_quant_meta())
        nodes = {
            "input": nir.Input(input_type=np.array([4])),
            "linear": node,
            "output": nir.Output(output_type=np.array([4])),
        }
        graph = _sequential_graph(
            nodes, [("input", "linear"), ("linear", "output")]
        )
        model = export_keras(graph, v1)
        dense_layers = [l for l in model.layers if isinstance(l, keras.layers.Dense)]
        assert len(dense_layers) == 1
        keras_w = dense_layers[0].get_weights()[0]  # shape (in, out)
        np.testing.assert_allclose(keras_w, w.T, atol=1e-6)

    def test_affine_weights_and_bias_preserved(self, v1):
        w = np.random.randn(4, 4).astype(np.float32)
        b = np.random.randn(4).astype(np.float32)
        node = nir.Affine(weight=w, bias=b)
        node.metadata.update(_quant_meta())
        nodes = {
            "input": nir.Input(input_type=np.array([4])),
            "affine": node,
            "output": nir.Output(output_type=np.array([4])),
        }
        graph = _sequential_graph(
            nodes, [("input", "affine"), ("affine", "output")]
        )
        model = export_keras(graph, v1)
        dense_layers = [l for l in model.layers if isinstance(l, keras.layers.Dense)]
        assert len(dense_layers) == 1
        keras_w, keras_b = dense_layers[0].get_weights()
        np.testing.assert_allclose(keras_w, w.T, atol=1e-6)
        np.testing.assert_allclose(keras_b, b, atol=1e-6)
