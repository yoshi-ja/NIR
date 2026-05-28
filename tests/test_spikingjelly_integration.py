"""Integration tests for the SpikingJelly <-> NIR round-trip.

All tests are guarded by ``pytest.importorskip`` so they are silently skipped
when SpikingJelly (or its dependencies) are not installed.

Milestones covered
------------------
M2 – Basic round-trip parity (Linear, Affine, IF, LIF, Conv2d).
M3 – Extended coverage (ParametricLIFNode export, multi-step forward pass).
"""

import tempfile

import numpy as np
import pytest

# Skip the entire module if spikingjelly is not installed.
sj = pytest.importorskip("spikingjelly")

import torch  # noqa: E402 – import after importorskip guard
import torch.nn as nn  # noqa: E402

import nir  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DT = 1e-4  # time step used throughout; must be consistent between export/import


def _export_import(net: nn.Module, example_input: torch.Tensor, step_mode: str = "s"):
    """Export *net* to NIR (via HDF5 round-trip) and re-import to SpikingJelly."""
    from spikingjelly.activation_based.nir_exchange import (
        export_to_nir,
        import_from_nir,
    )

    nir_graph = export_to_nir(net, example_input, dt=DT)

    with tempfile.NamedTemporaryFile(suffix=".nir", delete=False) as fp:
        path = fp.name

    nir.write(path, nir_graph)
    loaded = nir.read(path)
    imported = import_from_nir(loaded, dt=DT, step_mode=step_mode)
    return nir_graph, imported


def _forward_single_step(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Reset neuron states, run one forward pass, return output."""
    from spikingjelly.activation_based import functional

    net.eval()
    functional.reset_net(net)
    with torch.no_grad():
        return net(x)


def _get_linear_weight(net: nn.Module) -> np.ndarray:
    """Return the weight of the first nn.Linear-like layer as a numpy array."""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            return m.weight.detach().cpu().numpy()
    raise ValueError("No nn.Linear found in model")


def _get_linear_bias(net: nn.Module) -> np.ndarray:
    """Return the bias of the first nn.Linear-like layer as a numpy array (or None)."""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            b = m.bias
            return b.detach().cpu().numpy() if b is not None else None
    raise ValueError("No nn.Linear found in model")


# ---------------------------------------------------------------------------
# M2 – Basic round-trip tests
# ---------------------------------------------------------------------------


class TestLinearRoundtrip:
    """nn.Linear without bias → nir.Linear → SpikingJelly Linear."""

    def test_weight_preserved(self):
        from spikingjelly.activation_based import layer

        net = nn.Sequential(layer.Linear(8, 16, bias=False))
        example_input = torch.zeros(1, 8)
        _, imported = _export_import(net, example_input)

        w_orig = _get_linear_weight(net)
        w_imported = _get_linear_weight(imported)
        assert np.allclose(w_orig, w_imported, rtol=1e-5, atol=1e-6), (
            "Linear weight changed after NIR round-trip"
        )

    def test_no_bias(self):
        from spikingjelly.activation_based import layer

        net = nn.Sequential(layer.Linear(4, 8, bias=False))
        example_input = torch.zeros(1, 4)
        _, imported = _export_import(net, example_input)

        bias = _get_linear_bias(imported)
        assert bias is None or np.allclose(bias, 0), (
            "Expected no bias (or zero bias) after round-trip of bias-free Linear"
        )


class TestAffineRoundtrip:
    """nn.Linear with bias → nir.Affine → SpikingJelly Linear."""

    def test_weight_and_bias_preserved(self):
        from spikingjelly.activation_based import layer

        net = nn.Sequential(layer.Linear(8, 16, bias=True))
        example_input = torch.zeros(1, 8)
        _, imported = _export_import(net, example_input)

        w_orig = _get_linear_weight(net)
        w_imported = _get_linear_weight(imported)
        assert np.allclose(w_orig, w_imported, rtol=1e-5, atol=1e-6), (
            "Affine weight changed after NIR round-trip"
        )

        b_orig = _get_linear_bias(net)
        b_imported = _get_linear_bias(imported)
        assert b_orig is not None and b_imported is not None, (
            "Bias lost during NIR round-trip"
        )
        assert np.allclose(b_orig, b_imported, rtol=1e-5, atol=1e-6), (
            "Affine bias changed after NIR round-trip"
        )


class TestIFRoundtrip:
    """neuron.IFNode (hard reset) → nir.IF → SpikingJelly IFNode."""

    def test_nir_node_type(self):
        from spikingjelly.activation_based import layer, neuron

        net = nn.Sequential(
            layer.Linear(4, 4, bias=False),
            neuron.IFNode(v_threshold=1.0, v_reset=0.0),
        )
        example_input = torch.zeros(1, 4)
        nir_graph, _ = _export_import(net, example_input)

        # Expect at least one nir.IF node in the exported graph
        if_nodes = [n for n in nir_graph.nodes.values() if isinstance(n, nir.IF)]
        assert len(if_nodes) >= 1, "No nir.IF node found after export"

    def test_threshold_reset_preserved(self):
        from spikingjelly.activation_based import layer, neuron

        v_thr = 1.0
        v_rst = 0.0
        net = nn.Sequential(
            layer.Linear(4, 4, bias=False),
            neuron.IFNode(v_threshold=v_thr, v_reset=v_rst),
        )
        example_input = torch.zeros(1, 4)
        nir_graph, _ = _export_import(net, example_input)

        for node in nir_graph.nodes.values():
            if isinstance(node, nir.IF):
                assert np.allclose(node.v_threshold, v_thr), (
                    "IF v_threshold changed after export"
                )
                assert np.allclose(node.v_reset, v_rst), (
                    "IF v_reset changed after export"
                )

    def test_output_parity(self):
        from spikingjelly.activation_based import layer, neuron

        torch.manual_seed(42)
        net = nn.Sequential(
            layer.Linear(8, 8, bias=False),
            neuron.IFNode(v_threshold=1.0, v_reset=0.0),
        )
        example_input = torch.zeros(1, 8)
        _, imported = _export_import(net, example_input)

        x = torch.randn(1, 8)
        out_orig = _forward_single_step(net, x)
        out_imp = _forward_single_step(imported, x)
        assert torch.allclose(out_orig, out_imp, atol=1e-5), (
            f"IF output mismatch after round-trip: orig={out_orig}, imp={out_imp}"
        )


class TestLIFRoundtrip:
    """neuron.LIFNode → nir.LIF → SpikingJelly LIFNode."""

    def test_nir_node_type(self):
        from spikingjelly.activation_based import layer, neuron

        net = nn.Sequential(
            layer.Linear(4, 4, bias=False),
            neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0),
        )
        example_input = torch.zeros(1, 4)
        nir_graph, _ = _export_import(net, example_input)

        lif_nodes = [n for n in nir_graph.nodes.values() if isinstance(n, nir.LIF)]
        assert len(lif_nodes) >= 1, "No nir.LIF node found after export"

    def test_tau_preserved(self):
        """tau_nir = tau_sj * dt."""
        from spikingjelly.activation_based import layer, neuron

        tau_sj = 2.0
        net = nn.Sequential(
            layer.Linear(4, 4, bias=False),
            neuron.LIFNode(tau=tau_sj, v_threshold=1.0, v_reset=0.0),
        )
        example_input = torch.zeros(1, 4)
        nir_graph, _ = _export_import(net, example_input)

        for node in nir_graph.nodes.values():
            if isinstance(node, nir.LIF):
                expected_tau = tau_sj * DT
                assert np.allclose(node.tau, expected_tau, rtol=1e-3), (
                    f"LIF tau mismatch: expected {expected_tau}, got {node.tau}"
                )

    def test_threshold_reset_preserved(self):
        from spikingjelly.activation_based import layer, neuron

        v_thr = 1.5
        v_rst = 0.0
        net = nn.Sequential(
            layer.Linear(4, 4, bias=False),
            neuron.LIFNode(tau=2.0, v_threshold=v_thr, v_reset=v_rst),
        )
        example_input = torch.zeros(1, 4)
        nir_graph, _ = _export_import(net, example_input)

        for node in nir_graph.nodes.values():
            if isinstance(node, nir.LIF):
                assert np.allclose(node.v_threshold, v_thr), (
                    "LIF v_threshold changed after export"
                )
                assert np.allclose(node.v_reset, v_rst), (
                    "LIF v_reset changed after export"
                )

    def test_output_parity(self):
        from spikingjelly.activation_based import layer, neuron

        torch.manual_seed(0)
        net = nn.Sequential(
            layer.Linear(8, 8, bias=False),
            neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0),
        )
        example_input = torch.zeros(1, 8)
        _, imported = _export_import(net, example_input)

        x = torch.randn(1, 8)
        out_orig = _forward_single_step(net, x)
        out_imp = _forward_single_step(imported, x)
        assert torch.allclose(out_orig, out_imp, atol=1e-5), (
            f"LIF output mismatch after round-trip: orig={out_orig}, imp={out_imp}"
        )


class TestConv2dRoundtrip:
    """layer.Conv2d → nir.Conv2d → SpikingJelly Conv2d."""

    def test_nir_node_type(self):
        from spikingjelly.activation_based import layer

        net = nn.Sequential(layer.Conv2d(1, 4, kernel_size=3, padding=1))
        example_input = torch.zeros(1, 1, 8, 8)
        nir_graph, _ = _export_import(net, example_input)

        conv_nodes = [n for n in nir_graph.nodes.values() if isinstance(n, nir.Conv2d)]
        assert len(conv_nodes) >= 1, "No nir.Conv2d node found after export"

    def test_weight_preserved(self):
        from spikingjelly.activation_based import layer

        torch.manual_seed(7)
        net = nn.Sequential(layer.Conv2d(1, 4, kernel_size=3, padding=1))
        example_input = torch.zeros(1, 1, 8, 8)
        nir_graph, _ = _export_import(net, example_input)

        # Retrieve Conv2d module weight directly via isinstance check
        orig_conv = None
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                orig_conv = m
                break
        assert orig_conv is not None, "No nn.Conv2d found in source model"

        # Retrieve weight from NIR graph directly
        for node in nir_graph.nodes.values():
            if isinstance(node, nir.Conv2d):
                w_nir = node.weight
                w_orig = orig_conv.weight.detach().cpu().numpy()
                assert np.allclose(w_orig, w_nir, rtol=1e-5, atol=1e-6), (
                    "Conv2d weight changed after NIR export"
                )

    def test_meta_preserved(self):
        from spikingjelly.activation_based import layer

        stride, padding = 2, 1
        net = nn.Sequential(
            layer.Conv2d(1, 4, kernel_size=3, stride=stride, padding=padding)
        )
        example_input = torch.zeros(1, 1, 8, 8)
        nir_graph, _ = _export_import(net, example_input)

        for node in nir_graph.nodes.values():
            if isinstance(node, nir.Conv2d):
                node_stride = np.atleast_1d(np.asarray(node.stride).flatten())
                node_padding = np.atleast_1d(np.asarray(node.padding).flatten())
                assert np.array_equal(node_stride, np.array([stride, stride])), (
                    "Conv2d stride changed after NIR export"
                )
                assert np.array_equal(node_padding, np.array([padding, padding])), (
                    "Conv2d padding changed after NIR export"
                )


# ---------------------------------------------------------------------------
# M3 – Extended coverage
# ---------------------------------------------------------------------------


class TestParametricLIFExport:
    """neuron.ParametricLIFNode → nir.LIF with τ frozen at export time."""

    def test_exports_as_lif(self):
        from spikingjelly.activation_based import layer, neuron

        net = nn.Sequential(
            layer.Linear(4, 4, bias=False),
            neuron.ParametricLIFNode(init_tau=2.0, v_threshold=1.0, v_reset=0.0),
        )
        example_input = torch.zeros(1, 4)
        nir_graph, _ = _export_import(net, example_input)

        lif_nodes = [n for n in nir_graph.nodes.values() if isinstance(n, nir.LIF)]
        assert len(lif_nodes) >= 1, (
            "ParametricLIFNode should be exported as nir.LIF but no LIF node found"
        )

    def test_tau_is_frozen_at_export(self):
        """The exported τ should match init_tau * dt (τ is frozen, not learnable)."""
        from spikingjelly.activation_based import layer, neuron

        init_tau = 3.0
        net = nn.Sequential(
            layer.Linear(4, 4, bias=False),
            neuron.ParametricLIFNode(init_tau=init_tau, v_threshold=1.0, v_reset=0.0),
        )
        example_input = torch.zeros(1, 4)
        nir_graph, _ = _export_import(net, example_input)

        for node in nir_graph.nodes.values():
            if isinstance(node, nir.LIF):
                expected_tau = init_tau * DT
                assert np.allclose(node.tau, expected_tau, rtol=1e-2), (
                    f"ParametricLIF frozen tau mismatch: expected ~{expected_tau}, got {node.tau}"
                )


class TestMultiStepRoundtrip:
    """Multi-step mode (step_mode='m') round-trip and output parity."""

    def test_multistep_import(self):
        """Model imported with step_mode='m' should accept (T, B, N) inputs."""
        from spikingjelly.activation_based import functional, layer, neuron

        torch.manual_seed(1)
        net = nn.Sequential(
            layer.Linear(8, 8, bias=False),
            neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0),
        )
        functional.set_step_mode(net, "m")

        example_input = torch.zeros(1, 8)  # export uses a single-step example
        nir_graph, imported_m = _export_import(net, example_input, step_mode="m")

        # Run a 4-step multi-step forward pass
        T, B, N = 4, 2, 8
        x = torch.randn(T, B, N)

        net.eval()
        functional.reset_net(net)
        with torch.no_grad():
            out_orig = net(x)

        imported_m.eval()
        functional.reset_net(imported_m)
        with torch.no_grad():
            out_imp = imported_m(x)

        assert out_orig.shape == out_imp.shape, (
            f"Multi-step output shape mismatch: {out_orig.shape} vs {out_imp.shape}"
        )
        assert torch.allclose(out_orig, out_imp, atol=1e-5), (
            "Multi-step LIF output parity failed after round-trip"
        )

    def test_multistep_output_matches_repeated_single_step(self):
        """Multi-step output should equal concatenating T single-step forward passes."""
        from spikingjelly.activation_based import functional, layer, neuron

        torch.manual_seed(2)
        net_s = nn.Sequential(
            layer.Linear(4, 4, bias=False),
            neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0),
        )
        net_m = nn.Sequential(
            layer.Linear(4, 4, bias=False),
            neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0),
        )
        # Copy weights from net_s to net_m
        net_m.load_state_dict(net_s.state_dict())

        functional.set_step_mode(net_m, "m")

        T, B, N = 3, 1, 4
        xs = [torch.randn(B, N) for _ in range(T)]
        x_multi = torch.stack(xs, dim=0)  # (T, B, N)

        # Single-step: run T times, collect outputs
        net_s.eval()
        functional.reset_net(net_s)
        outs_single = []
        with torch.no_grad():
            for x in xs:
                outs_single.append(net_s(x))
        out_single = torch.stack(outs_single, dim=0)  # (T, B, N)

        # Multi-step: run once
        net_m.eval()
        functional.reset_net(net_m)
        with torch.no_grad():
            out_multi = net_m(x_multi)

        assert torch.allclose(out_single, out_multi, atol=1e-5), (
            "Multi-step output does not match repeated single-step outputs"
        )
