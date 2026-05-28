"""Convert a BindsNET Network into a NIR graph.

Supports BindsNET networks built from the following components:

- ``bindsnet.network.nodes.Input``
- ``bindsnet.network.nodes.LIFNodes``
- ``bindsnet.network.nodes.IFNodes``
- ``bindsnet.network.topology.Connection``

Other node or connection types raise ``NotImplementedError`` with
descriptive messages.
"""

from typing import Optional

import numpy as np

import nir

try:
    from bindsnet.network import Network
    from bindsnet.network.nodes import Input as BNInput
    from bindsnet.network.nodes import IFNodes, LIFNodes, Nodes
    from bindsnet.network.topology import Connection
except ImportError as e:
    raise ImportError(
        "BindsNET is required for nir_bindsnet. "
        "Install it with: pip install bindsnet"
    ) from e


def _bindsnet_lif_to_nir(layer: LIFNodes) -> nir.LIF:
    """Convert a BindsNET ``LIFNodes`` layer to a ``nir.LIF`` node.

    BindsNET LIF dynamics (discrete):
        v = decay * (v - rest) + rest + I
        where decay = exp(-dt / tc_decay)

    NIR LIF dynamics (continuous):
        tau * dv/dt = (v_leak - v) + R * I

    Mapping:
        tau         = tc_decay
        v_leak      = rest
        v_threshold = thresh
        v_reset     = reset
        r           = 1.0  (resistance is folded into connection weights)
    """
    n = layer.n
    tau = np.full(n, float(layer.tc_decay))
    r = np.ones(n)
    v_leak = np.full(n, float(layer.rest))
    v_threshold = np.full(n, float(layer.thresh))
    v_reset = np.full(n, float(layer.reset))

    return nir.LIF(
        tau=tau,
        r=r,
        v_leak=v_leak,
        v_threshold=v_threshold,
        v_reset=v_reset,
    )


def _bindsnet_if_to_nir(layer: IFNodes) -> nir.IF:
    """Convert a BindsNET ``IFNodes`` layer to a ``nir.IF`` node.

    BindsNET IF dynamics:
        v += I  (when not refractory)
        spike when v >= thresh, reset to reset

    NIR IF dynamics:
        dv/dt = R * I
        spike when v > v_threshold, reset to v_reset

    Mapping:
        r           = 1.0
        v_threshold = thresh
        v_reset     = reset
    """
    n = layer.n
    r = np.ones(n)
    v_threshold = np.full(n, float(layer.thresh))
    v_reset = np.full(n, float(layer.reset))

    return nir.IF(
        r=r,
        v_threshold=v_threshold,
        v_reset=v_reset,
    )


def to_nir(network: "Network") -> nir.NIRGraph:
    """Convert a BindsNET ``Network`` into a ``nir.NIRGraph``.

    Parameters
    ----------
    network : bindsnet.network.Network
        The BindsNET network to convert. Must contain only supported layer
        and connection types.

    Returns
    -------
    nir.NIRGraph
        A NIR graph representation of the BindsNET network.

    Raises
    ------
    NotImplementedError
        If the network contains unsupported layer or connection types.
    """
    nodes = {}
    edges = []

    # Convert layers to NIR nodes
    for name, layer in network.layers.items():
        if isinstance(layer, BNInput):
            nodes[name] = nir.Input(input_type=np.array([layer.n]))

        elif isinstance(layer, LIFNodes):
            nodes[name] = _bindsnet_lif_to_nir(layer)

        elif isinstance(layer, IFNodes):
            nodes[name] = _bindsnet_if_to_nir(layer)

        else:
            raise NotImplementedError(
                f"BindsNET layer type '{type(layer).__name__}' is not supported. "
                f"Supported types: Input, LIFNodes, IFNodes."
            )

    # Convert connections to NIR nodes and edges
    for (src_name, tgt_name), conn in network.connections.items():
        if not isinstance(conn, Connection):
            raise NotImplementedError(
                f"BindsNET connection type '{type(conn).__name__}' is not supported. "
                f"Only Connection (dense) is currently supported."
            )

        # Extract weight matrix
        # BindsNET Connection.w has shape (source.n, target.n)
        # NIR Linear/Affine weight has shape (out_features, in_features)
        # So we need to transpose.
        w = conn.w.detach().cpu().numpy().T
        b = conn.b.detach().cpu().numpy()

        # Determine whether to use Affine or Linear
        has_bias = np.any(b != 0)

        # Create a unique name for the connection node
        conn_name = f"{src_name}_to_{tgt_name}"

        if has_bias:
            nodes[conn_name] = nir.Affine(weight=w, bias=b)
        else:
            nodes[conn_name] = nir.Linear(weight=w)

        # Add edges: source -> connection -> target
        edges.append((src_name, conn_name))
        edges.append((conn_name, tgt_name))

    # Identify input and output layers (layers with no incoming or outgoing
    # connections in the NIR graph)
    sources = {e[0] for e in edges}
    targets = {e[1] for e in edges}

    # Find NIR-level inputs (nodes that are never a target)
    nir_inputs = []
    for name in nodes:
        if name not in targets:
            nir_inputs.append(name)

    # Find NIR-level outputs (nodes that are never a source)
    nir_outputs = []
    for name in nodes:
        if name not in sources:
            nir_outputs.append(name)

    # Add Input/Output wrapper nodes if the graph doesn't already have them
    for inp_name in nir_inputs:
        if not isinstance(nodes[inp_name], nir.Input):
            node = nodes[inp_name]
            input_node_name = f"input_{inp_name}"
            n = _get_node_input_size(node)
            nodes[input_node_name] = nir.Input(input_type=np.array([n]))
            edges.append((input_node_name, inp_name))

    for out_name in nir_outputs:
        if not isinstance(nodes[out_name], nir.Output):
            node = nodes[out_name]
            output_node_name = f"output_{out_name}"
            n = _get_node_output_size(node)
            nodes[output_node_name] = nir.Output(output_type=np.array([n]))
            edges.append((out_name, output_node_name))

    return nir.NIRGraph(nodes=nodes, edges=edges)


def _get_node_input_size(node):
    """Get the input size of a NIR node."""
    if hasattr(node, "input_type") and node.input_type is not None:
        return int(np.prod(node.input_type["input"]))
    return 1


def _get_node_output_size(node):
    """Get the output size of a NIR node."""
    if hasattr(node, "output_type") and node.output_type is not None:
        return int(np.prod(node.output_type["output"]))
    return 1
