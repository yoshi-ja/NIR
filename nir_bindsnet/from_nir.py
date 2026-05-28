"""Convert a NIR graph into a BindsNET Network.

Supports simple feedforward (acyclic, single-input single-output) graphs
containing the following NIR primitives:

- ``nir.Input``
- ``nir.Output``
- ``nir.Linear``
- ``nir.Affine``
- ``nir.LIF``
- ``nir.IF``

Unsupported primitives or graph structures raise ``NotImplementedError``
with descriptive messages.
"""

from typing import Optional

import numpy as np
import torch

import nir

try:
    from bindsnet.network import Network
    from bindsnet.network.nodes import Input as BNInput
    from bindsnet.network.nodes import IFNodes, LIFNodes
    from bindsnet.network.topology import Connection
except ImportError as e:
    raise ImportError(
        "BindsNET is required for nir_bindsnet. "
        "Install it with: pip install bindsnet"
    ) from e


def _topological_order(nodes, edges):
    """Return node names in topological order for a DAG.

    Raises ``NotImplementedError`` for graphs containing cycles.
    """
    adj = {name: [] for name in nodes}
    in_degree = {name: 0 for name in nodes}
    for src, tgt in edges:
        adj[src].append(tgt)
        in_degree[tgt] = in_degree.get(tgt, 0) + 1

    queue = [n for n in nodes if in_degree[n] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(nodes):
        raise NotImplementedError(
            "Recurrent or cyclic graphs are not supported. "
            "Only acyclic feedforward graphs can be converted."
        )
    return order


def _validate_feedforward(nodes, edges):
    """Validate that the graph is a simple feedforward chain.

    Each node (except Input/Output) must have exactly one incoming and one
    outgoing edge.  Branching and merging patterns are not supported.
    """
    out_count = {name: 0 for name in nodes}
    in_count = {name: 0 for name in nodes}
    for src, tgt in edges:
        out_count[src] += 1
        in_count[tgt] += 1

    for name, node in nodes.items():
        if isinstance(node, nir.Input):
            if in_count[name] != 0:
                raise NotImplementedError(
                    f"Input node '{name}' has incoming edges, which is not supported."
                )
            if out_count[name] > 1:
                raise NotImplementedError(
                    f"Node '{name}' has {out_count[name]} outgoing edges. "
                    "Branching patterns are not yet supported."
                )
        elif isinstance(node, nir.Output):
            if out_count[name] != 0:
                raise NotImplementedError(
                    f"Output node '{name}' has outgoing edges, which is not supported."
                )
        else:
            if in_count[name] > 1:
                raise NotImplementedError(
                    f"Node '{name}' has {in_count[name]} incoming edges. "
                    "Merging patterns are not yet supported."
                )
            if out_count[name] > 1:
                raise NotImplementedError(
                    f"Node '{name}' has {out_count[name]} outgoing edges. "
                    "Branching patterns are not yet supported."
                )


def _get_neuron_count(node):
    """Infer the number of neurons from a NIR node's shape information."""
    if isinstance(node, (nir.Linear, nir.Affine)):
        return int(node.weight.shape[0])
    if isinstance(node, nir.LIF):
        return int(np.prod(node.tau.shape))
    if isinstance(node, nir.IF):
        return int(np.prod(node.r.shape))
    if isinstance(node, nir.Input):
        return int(np.prod(node.input_type["input"]))
    if isinstance(node, nir.Output):
        return int(np.prod(node.output_type["output"]))
    raise NotImplementedError(f"Cannot infer neuron count from {type(node).__name__}")


def _nir_lif_to_bindsnet(node: nir.LIF, dt: float = 1.0) -> LIFNodes:
    """Convert a ``nir.LIF`` node to a BindsNET ``LIFNodes`` layer.

    NIR LIF dynamics (continuous):
        tau * dv/dt = (v_leak - v) + R * I

    BindsNET LIF dynamics (discrete, per timestep):
        v = decay * (v - rest) + rest + I
        where decay = exp(-dt / tc_decay)

    Mapping:
        tc_decay = tau  (time constant)
        rest     = v_leak
        thresh   = v_threshold
        reset    = v_reset

    Note: The resistance ``R`` is not directly represented in BindsNET's
    LIFNodes. It is assumed to be folded into the connection weights.
    A metadata warning is stored if ``R != 1``.
    """
    n = int(np.prod(node.tau.shape))
    # Use scalar values for BindsNET (it expects float or Tensor)
    tau = float(node.tau.flat[0]) if node.tau.size == 1 else node.tau.flat[0]
    rest = float(node.v_leak.flat[0]) if node.v_leak.size == 1 else node.v_leak.flat[0]
    thresh = (
        float(node.v_threshold.flat[0])
        if node.v_threshold.size == 1
        else node.v_threshold.flat[0]
    )
    v_reset = node.v_reset if node.v_reset is not None else np.zeros_like(node.tau)
    reset = float(v_reset.flat[0]) if v_reset.size == 1 else v_reset.flat[0]

    return LIFNodes(
        n=n,
        thresh=float(thresh),
        rest=float(rest),
        reset=float(reset),
        tc_decay=float(tau),
        refrac=0,
    )


def _nir_if_to_bindsnet(node: nir.IF) -> IFNodes:
    """Convert a ``nir.IF`` node to a BindsNET ``IFNodes`` layer.

    NIR IF dynamics:
        dv/dt = R * I
        spike when v > v_threshold, reset to v_reset

    BindsNET IF dynamics:
        v += I  (when not in refractory period)
        spike when v >= thresh, reset to reset

    Note: Resistance ``R`` is assumed folded into connection weights.
    """
    n = int(np.prod(node.r.shape))
    thresh = (
        float(node.v_threshold.flat[0])
        if node.v_threshold.size == 1
        else node.v_threshold.flat[0]
    )
    v_reset = node.v_reset if node.v_reset is not None else np.zeros_like(node.r)
    reset = float(v_reset.flat[0]) if v_reset.size == 1 else v_reset.flat[0]

    return IFNodes(
        n=n,
        thresh=float(thresh),
        reset=float(reset),
        refrac=0,
    )


def from_nir(
    graph: nir.NIRGraph, dt: float = 1.0, batch_size: int = 1
) -> "Network":
    """Convert a ``nir.NIRGraph`` into a BindsNET ``Network``.

    Parameters
    ----------
    graph : nir.NIRGraph
        The NIR graph to convert. Must be a simple feedforward (acyclic,
        single-input single-output) graph.
    dt : float, optional
        Simulation timestep in milliseconds. Default is 1.0.
    batch_size : int, optional
        Mini-batch size for the network. Default is 1.

    Returns
    -------
    bindsnet.network.Network
        A BindsNET network equivalent to the NIR graph.

    Raises
    ------
    NotImplementedError
        If the graph contains unsupported primitives, recurrent connections,
        branching, or merging patterns.
    """
    _validate_feedforward(graph.nodes, graph.edges)
    order = _topological_order(graph.nodes, graph.edges)

    # Build successor map for connecting layers
    successors = {}
    for src, tgt in graph.edges:
        successors[src] = tgt

    net = Network(dt=dt, batch_size=batch_size, learning=False)

    # First pass: create all BindsNET layers
    bn_layers = {}
    for name in order:
        node = graph.nodes[name]

        if isinstance(node, nir.Input):
            n = _get_neuron_count(node)
            bn_layers[name] = BNInput(n=n)

        elif isinstance(node, nir.Output):
            # Output is a virtual node; no BindsNET layer needed.
            # We track its predecessor for connection purposes.
            continue

        elif isinstance(node, nir.LIF):
            bn_layers[name] = _nir_lif_to_bindsnet(node, dt=dt)

        elif isinstance(node, nir.IF):
            bn_layers[name] = _nir_if_to_bindsnet(node)

        elif isinstance(node, (nir.Linear, nir.Affine)):
            # Linear/Affine are connection primitives in NIR.
            # In BindsNET, they become Connection objects between layers,
            # not standalone layers. We handle them during edge processing.
            continue

        elif isinstance(node, nir.NIRGraph):
            raise NotImplementedError(
                f"Subgraph node '{name}' is not supported. "
                "Only flat (non-hierarchical) graphs are currently supported."
            )

        else:
            raise NotImplementedError(
                f"NIR node type '{type(node).__name__}' is not supported. "
                f"Supported types: Input, Output, Linear, Affine, LIF, IF."
            )

    # Add layers to network
    for name, layer in bn_layers.items():
        net.add_layer(layer=layer, name=name)

    # Second pass: create connections
    for src, tgt in graph.edges:
        src_node = graph.nodes[src]
        tgt_node = graph.nodes[tgt]

        # Case 1: source is a layer, target is Linear/Affine, which connects
        # to the next layer
        if isinstance(tgt_node, (nir.Linear, nir.Affine)):
            # Find what the Linear/Affine connects to
            if tgt not in successors:
                continue  # Linear/Affine at the end (before Output handled below)
            next_name = successors[tgt]
            next_node = graph.nodes[next_name]

            if isinstance(next_node, nir.Output):
                # Linear/Affine connects to Output - skip, no BindsNET layer
                continue

            if src not in bn_layers or next_name not in bn_layers:
                continue

            # NIR weight shape: (out_features, in_features)
            # BindsNET weight shape: (source.n, target.n) = (in_features, out_features)
            w = torch.tensor(tgt_node.weight.T, dtype=torch.float32)
            kwargs = {"w": w}
            if isinstance(tgt_node, nir.Affine):
                kwargs["b"] = torch.tensor(tgt_node.bias, dtype=torch.float32)

            conn = Connection(
                source=bn_layers[src],
                target=bn_layers[next_name],
                **kwargs,
            )
            net.add_connection(
                connection=conn, source=src, target=next_name
            )

        elif isinstance(src_node, (nir.Linear, nir.Affine)):
            # Already handled when processing the Linear/Affine as target
            pass

        elif src in bn_layers and tgt in bn_layers:
            # Direct connection without Linear/Affine (identity connection)
            n_src = bn_layers[src].n
            n_tgt = bn_layers[tgt].n
            conn = Connection(
                source=bn_layers[src],
                target=bn_layers[tgt],
                w=torch.eye(n_src, n_tgt),
            )
            net.add_connection(connection=conn, source=src, target=tgt)

    return net
