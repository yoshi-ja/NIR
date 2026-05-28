"""Tests for BindsNET <-> NIR integration.

Tests cover:
- NIR -> BindsNET conversion of feedforward networks
- BindsNET -> NIR conversion of simple networks
- Round-trip conversion (NIR -> BindsNET -> NIR)
- Preservation of weights, shapes, and neuron parameters
- Informative failures for unsupported cases
"""

import numpy as np
import pytest
import torch

import nir

# nir_bindsnet applies a compatibility shim for newer PyTorch versions
# before importing BindsNET, so import it first.
import nir_bindsnet  # noqa: F401 - triggers _compat patch

from bindsnet.network import Network
from bindsnet.network.nodes import Input as BNInput
from bindsnet.network.nodes import IFNodes, LIFNodes
from bindsnet.network.topology import Connection

from nir_bindsnet.from_nir import from_nir, _topological_order, _validate_feedforward
from nir_bindsnet.to_nir import to_nir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_nir_graph(in_size=4, out_size=3):
    """Create a simple NIR graph: Input -> Linear -> LIF -> Output."""
    weight = np.random.randn(out_size, in_size)
    tau = np.ones(out_size) * 10.0
    r = np.ones(out_size)
    v_leak = np.zeros(out_size)
    v_threshold = np.ones(out_size)
    v_reset = np.zeros(out_size)

    nodes = {
        "input": nir.Input(input_type=np.array([in_size])),
        "linear": nir.Linear(weight=weight),
        "lif": nir.LIF(
            tau=tau, r=r, v_leak=v_leak,
            v_threshold=v_threshold, v_reset=v_reset,
        ),
        "output": nir.Output(output_type=np.array([out_size])),
    }
    edges = [
        ("input", "linear"),
        ("linear", "lif"),
        ("lif", "output"),
    ]
    return nir.NIRGraph(nodes=nodes, edges=edges)


def _make_affine_nir_graph(in_size=4, out_size=3):
    """Create a NIR graph: Input -> Affine -> LIF -> Output."""
    weight = np.random.randn(out_size, in_size)
    bias = np.random.randn(out_size)
    tau = np.ones(out_size) * 20.0
    r = np.ones(out_size)
    v_leak = np.zeros(out_size)
    v_threshold = np.ones(out_size) * 0.5
    v_reset = np.zeros(out_size)

    nodes = {
        "input": nir.Input(input_type=np.array([in_size])),
        "affine": nir.Affine(weight=weight, bias=bias),
        "lif": nir.LIF(
            tau=tau, r=r, v_leak=v_leak,
            v_threshold=v_threshold, v_reset=v_reset,
        ),
        "output": nir.Output(output_type=np.array([out_size])),
    }
    edges = [
        ("input", "affine"),
        ("affine", "lif"),
        ("lif", "output"),
    ]
    return nir.NIRGraph(nodes=nodes, edges=edges)


def _make_simple_bindsnet_network(in_size=4, out_size=3):
    """Create a simple BindsNET network: Input -> Connection -> LIFNodes."""
    net = Network(dt=1.0, batch_size=1, learning=False)

    input_layer = BNInput(n=in_size)
    lif_layer = LIFNodes(
        n=out_size, thresh=1.0, rest=0.0, reset=0.0, tc_decay=10.0, refrac=0,
    )
    net.add_layer(input_layer, name="input")
    net.add_layer(lif_layer, name="lif")

    w = torch.randn(in_size, out_size)
    conn = Connection(source=input_layer, target=lif_layer, w=w)
    net.add_connection(conn, source="input", target="lif")

    return net, w


# ---------------------------------------------------------------------------
# Tests: NIR -> BindsNET
# ---------------------------------------------------------------------------

class TestFromNIR:
    def test_simple_linear_lif(self):
        """Test converting a simple Linear -> LIF NIR graph."""
        in_size, out_size = 4, 3
        graph = _make_simple_nir_graph(in_size, out_size)
        net = from_nir(graph)

        assert isinstance(net, Network)
        assert "input" in net.layers
        assert "lif" in net.layers
        assert isinstance(net.layers["input"], BNInput)
        assert isinstance(net.layers["lif"], LIFNodes)
        assert net.layers["input"].n == in_size
        assert net.layers["lif"].n == out_size

    def test_affine_lif(self):
        """Test converting an Affine -> LIF NIR graph."""
        in_size, out_size = 4, 3
        graph = _make_affine_nir_graph(in_size, out_size)
        net = from_nir(graph)

        assert isinstance(net, Network)
        assert "input" in net.layers
        assert "lif" in net.layers
        assert net.layers["lif"].n == out_size

    def test_weight_preservation_linear(self):
        """Test that Linear weights are preserved during conversion."""
        in_size, out_size = 4, 3
        weight = np.array([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0, 12.0]])
        nodes = {
            "input": nir.Input(input_type=np.array([in_size])),
            "linear": nir.Linear(weight=weight),
            "lif": nir.LIF(
                tau=np.ones(out_size) * 10.0,
                r=np.ones(out_size),
                v_leak=np.zeros(out_size),
                v_threshold=np.ones(out_size),
            ),
            "output": nir.Output(output_type=np.array([out_size])),
        }
        edges = [("input", "linear"), ("linear", "lif"), ("lif", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph)

        # Find the connection. BindsNET stores weight as (in, out) = weight.T
        conn = list(net.connections.values())[0]
        np.testing.assert_allclose(
            conn.w.detach().cpu().numpy(), weight.T, rtol=1e-5,
        )

    def test_weight_preservation_affine(self):
        """Test that Affine weights and bias are preserved."""
        in_size, out_size = 2, 2
        weight = np.array([[1.0, 2.0], [3.0, 4.0]])
        bias = np.array([0.5, -0.5])
        nodes = {
            "input": nir.Input(input_type=np.array([in_size])),
            "affine": nir.Affine(weight=weight, bias=bias),
            "lif": nir.LIF(
                tau=np.ones(out_size) * 10.0,
                r=np.ones(out_size),
                v_leak=np.zeros(out_size),
                v_threshold=np.ones(out_size),
            ),
            "output": nir.Output(output_type=np.array([out_size])),
        }
        edges = [("input", "affine"), ("affine", "lif"), ("lif", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph)

        conn = list(net.connections.values())[0]
        # BindsNET stores weight as (in, out) = NIR weight.T
        np.testing.assert_allclose(
            conn.w.detach().cpu().numpy(), weight.T, rtol=1e-5,
        )
        np.testing.assert_allclose(
            conn.b.detach().cpu().numpy(), bias, rtol=1e-5,
        )

    def test_lif_parameters(self):
        """Test that LIF neuron parameters are preserved."""
        n = 3
        tau = np.ones(n) * 15.0
        v_leak = np.ones(n) * -0.5
        v_threshold = np.ones(n) * 1.5
        v_reset = np.ones(n) * -1.0

        nodes = {
            "input": nir.Input(input_type=np.array([n])),
            "lif": nir.LIF(
                tau=tau, r=np.ones(n), v_leak=v_leak,
                v_threshold=v_threshold, v_reset=v_reset,
            ),
            "output": nir.Output(output_type=np.array([n])),
        }
        edges = [("input", "lif"), ("lif", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph)

        lif = net.layers["lif"]
        assert float(lif.tc_decay) == pytest.approx(15.0)
        assert float(lif.rest) == pytest.approx(-0.5)
        assert float(lif.thresh) == pytest.approx(1.5)
        assert float(lif.reset) == pytest.approx(-1.0)

    def test_if_conversion(self):
        """Test converting an IF neuron."""
        n = 3
        nodes = {
            "input": nir.Input(input_type=np.array([n])),
            "if_node": nir.IF(
                r=np.ones(n),
                v_threshold=np.ones(n) * 2.0,
                v_reset=np.zeros(n),
            ),
            "output": nir.Output(output_type=np.array([n])),
        }
        edges = [("input", "if_node"), ("if_node", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)
        net = from_nir(graph)

        assert isinstance(net.layers["if_node"], IFNodes)
        assert float(net.layers["if_node"].thresh) == pytest.approx(2.0)
        assert float(net.layers["if_node"].reset) == pytest.approx(0.0)

    def test_from_list_sequential(self):
        """Test converting a graph created with NIRGraph.from_list."""
        graph = nir.NIRGraph.from_list(
            nir.Affine(
                weight=np.eye(3),
                bias=np.zeros(3),
            ),
            nir.LIF(
                tau=np.ones(3) * 10.0,
                r=np.ones(3),
                v_leak=np.zeros(3),
                v_threshold=np.ones(3),
            ),
        )
        net = from_nir(graph)
        assert isinstance(net, Network)
        assert len(net.layers) >= 2  # Input + LIF (at minimum)


# ---------------------------------------------------------------------------
# Tests: BindsNET -> NIR
# ---------------------------------------------------------------------------

class TestToNIR:
    def test_simple_network(self):
        """Test converting a simple BindsNET network to NIR."""
        net, w = _make_simple_bindsnet_network(4, 3)
        graph = to_nir(net)

        assert isinstance(graph, nir.NIRGraph)
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

    def test_input_layer_conversion(self):
        """Test that Input layers become nir.Input nodes."""
        net, _ = _make_simple_bindsnet_network(4, 3)
        graph = to_nir(net)

        # Should have an Input node
        input_nodes = [
            n for n in graph.nodes.values() if isinstance(n, nir.Input)
        ]
        assert len(input_nodes) >= 1

    def test_lif_layer_conversion(self):
        """Test that LIFNodes become nir.LIF nodes."""
        net, _ = _make_simple_bindsnet_network(4, 3)
        graph = to_nir(net)

        lif_nodes = [n for n in graph.nodes.values() if isinstance(n, nir.LIF)]
        assert len(lif_nodes) == 1

        lif = lif_nodes[0]
        assert lif.tau.shape == (3,)
        np.testing.assert_allclose(lif.tau, 10.0)
        np.testing.assert_allclose(lif.v_leak, 0.0)
        np.testing.assert_allclose(lif.v_threshold, 1.0)

    def test_weight_preservation(self):
        """Test that connection weights are preserved."""
        net, original_w = _make_simple_bindsnet_network(4, 3)
        graph = to_nir(net)

        # Find the Linear or Affine node
        linear_nodes = [
            n for n in graph.nodes.values()
            if isinstance(n, (nir.Linear, nir.Affine))
        ]
        assert len(linear_nodes) == 1

        # BindsNET weight (4, 3) -> NIR weight (3, 4) via transpose
        np.testing.assert_allclose(
            linear_nodes[0].weight, original_w.numpy().T, rtol=1e-5,
        )

    def test_if_layer_conversion(self):
        """Test that IFNodes become nir.IF nodes."""
        net = Network(dt=1.0, batch_size=1, learning=False)
        inp = BNInput(n=3)
        if_layer = IFNodes(n=3, thresh=2.0, reset=-1.0, refrac=0)
        net.add_layer(inp, name="input")
        net.add_layer(if_layer, name="if_layer")

        w = torch.eye(3)
        conn = Connection(source=inp, target=if_layer, w=w)
        net.add_connection(conn, source="input", target="if_layer")

        graph = to_nir(net)

        if_nodes = [n for n in graph.nodes.values() if isinstance(n, nir.IF)]
        assert len(if_nodes) == 1
        np.testing.assert_allclose(if_nodes[0].v_threshold, 2.0)
        np.testing.assert_allclose(if_nodes[0].v_reset, -1.0)

    def test_affine_with_bias(self):
        """Test that connections with non-zero bias become Affine nodes."""
        net = Network(dt=1.0, batch_size=1, learning=False)
        inp = BNInput(n=2)
        lif = LIFNodes(n=2, thresh=1.0, rest=0.0, reset=0.0, tc_decay=10.0, refrac=0)
        net.add_layer(inp, name="input")
        net.add_layer(lif, name="lif")

        w = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = torch.tensor([0.5, -0.5])
        conn = Connection(source=inp, target=lif, w=w, b=b)
        net.add_connection(conn, source="input", target="lif")

        graph = to_nir(net)

        affine_nodes = [n for n in graph.nodes.values() if isinstance(n, nir.Affine)]
        assert len(affine_nodes) == 1
        np.testing.assert_allclose(affine_nodes[0].bias, [0.5, -0.5], rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Round-trip (NIR -> BindsNET -> NIR)
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_linear_lif_roundtrip(self):
        """Test round-trip: NIR -> BindsNET -> NIR preserves structure."""
        in_size, out_size = 4, 3
        weight = np.random.randn(out_size, in_size)
        tau_val = 10.0
        thresh_val = 1.0

        original = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=np.array([in_size])),
                "linear": nir.Linear(weight=weight),
                "lif": nir.LIF(
                    tau=np.ones(out_size) * tau_val,
                    r=np.ones(out_size),
                    v_leak=np.zeros(out_size),
                    v_threshold=np.ones(out_size) * thresh_val,
                ),
                "output": nir.Output(output_type=np.array([out_size])),
            },
            edges=[
                ("input", "linear"),
                ("linear", "lif"),
                ("lif", "output"),
            ],
        )

        # NIR -> BindsNET
        bn_net = from_nir(original)

        # BindsNET -> NIR
        result = to_nir(bn_net)

        # Check that we have the right node types
        lif_nodes = [n for n in result.nodes.values() if isinstance(n, nir.LIF)]
        linear_nodes = [
            n for n in result.nodes.values()
            if isinstance(n, (nir.Linear, nir.Affine))
        ]

        assert len(lif_nodes) == 1
        assert len(linear_nodes) == 1

        # Check parameter preservation
        np.testing.assert_allclose(lif_nodes[0].tau, tau_val, rtol=1e-5)
        np.testing.assert_allclose(lif_nodes[0].v_threshold, thresh_val, rtol=1e-5)
        np.testing.assert_allclose(linear_nodes[0].weight, weight, rtol=1e-5)

    def test_affine_lif_roundtrip(self):
        """Test round-trip with Affine (weights + bias)."""
        in_size, out_size = 3, 2
        weight = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        bias = np.array([0.1, -0.1])

        original = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=np.array([in_size])),
                "affine": nir.Affine(weight=weight, bias=bias),
                "lif": nir.LIF(
                    tau=np.ones(out_size) * 20.0,
                    r=np.ones(out_size),
                    v_leak=np.zeros(out_size),
                    v_threshold=np.ones(out_size),
                ),
                "output": nir.Output(output_type=np.array([out_size])),
            },
            edges=[
                ("input", "affine"),
                ("affine", "lif"),
                ("lif", "output"),
            ],
        )

        bn_net = from_nir(original)
        result = to_nir(bn_net)

        affine_nodes = [n for n in result.nodes.values() if isinstance(n, nir.Affine)]
        assert len(affine_nodes) == 1
        np.testing.assert_allclose(affine_nodes[0].weight, weight, rtol=1e-5)
        np.testing.assert_allclose(affine_nodes[0].bias, bias, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unsupported_node_type(self):
        """Test that unsupported NIR node types raise NotImplementedError."""
        # Use a Delay node which doesn't cause type-inference issues
        nodes = {
            "input": nir.Input(input_type=np.array([3])),
            "delay": nir.Delay(delay=np.ones(3)),
            "output": nir.Output(output_type=np.array([3])),
        }
        edges = [("input", "delay"), ("delay", "output")]
        graph = nir.NIRGraph(nodes=nodes, edges=edges)

        with pytest.raises(NotImplementedError, match="Delay.*not supported"):
            from_nir(graph)

    def test_cyclic_graph(self):
        """Test that cyclic graphs raise NotImplementedError."""
        nodes = {
            "a": nir.Input(input_type=np.array([3])),
            "b": nir.LIF(
                tau=np.ones(3), r=np.ones(3), v_leak=np.zeros(3),
                v_threshold=np.ones(3),
            ),
            "c": nir.LIF(
                tau=np.ones(3), r=np.ones(3), v_leak=np.zeros(3),
                v_threshold=np.ones(3),
            ),
        }
        edges = [("a", "b"), ("b", "c"), ("c", "b")]  # cycle: b -> c -> b

        with pytest.raises(NotImplementedError, match="[Rr]ecurrent|[Cc]yclic"):
            _topological_order(nodes, edges)

    def test_branching_graph(self):
        """Test that branching patterns raise NotImplementedError."""
        nodes = {
            "input": nir.Input(input_type=np.array([3])),
            "lif_a": nir.LIF(
                tau=np.ones(3), r=np.ones(3), v_leak=np.zeros(3),
                v_threshold=np.ones(3),
            ),
            "lif_b": nir.LIF(
                tau=np.ones(3), r=np.ones(3), v_leak=np.zeros(3),
                v_threshold=np.ones(3),
            ),
            "output": nir.Output(output_type=np.array([3])),
        }
        edges = [
            ("input", "lif_a"),
            ("input", "lif_b"),  # branching
            ("lif_a", "output"),
        ]

        with pytest.raises(NotImplementedError, match="[Bb]ranching"):
            _validate_feedforward(nodes, edges)

    def test_merging_graph(self):
        """Test that merging patterns raise NotImplementedError."""
        nodes = {
            "input_a": nir.Input(input_type=np.array([3])),
            "input_b": nir.Input(input_type=np.array([3])),
            "lif": nir.LIF(
                tau=np.ones(3), r=np.ones(3), v_leak=np.zeros(3),
                v_threshold=np.ones(3),
            ),
            "output": nir.Output(output_type=np.array([3])),
        }
        edges = [
            ("input_a", "lif"),
            ("input_b", "lif"),  # merging
            ("lif", "output"),
        ]

        with pytest.raises(NotImplementedError, match="[Mm]erging"):
            _validate_feedforward(nodes, edges)


# ---------------------------------------------------------------------------
# Tests: Topological ordering
# ---------------------------------------------------------------------------

class TestTopologicalOrder:
    def test_simple_chain(self):
        """Test topological ordering of a simple chain."""
        nodes = {"a": None, "b": None, "c": None}
        edges = [("a", "b"), ("b", "c")]
        order = _topological_order(nodes, edges)
        assert order.index("a") < order.index("b") < order.index("c")

    def test_single_node(self):
        """Test topological ordering of a single node."""
        nodes = {"a": None}
        edges = []
        order = _topological_order(nodes, edges)
        assert order == ["a"]
