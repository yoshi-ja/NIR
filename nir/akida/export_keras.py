"""NIR → Keras export for the Akida MVP op subset.

This module converts a validated NIR graph into a Keras ``Sequential`` or
``Functional`` model that can be passed directly to BrainChip's QuantizeML
for post-training quantization, and subsequently to CNN2SNN for Akida
deployment.

Only nodes listed in the capability matrix with status **MVP** are exported.
Any unsupported node causes :class:`AkidaExportError` with an actionable
message that names the problematic node and explains the restriction.

Keras is an optional dependency.  The module can be imported without Keras
installed; only :func:`export_keras` will raise ``ImportError`` at call time
if Keras is missing.

Usage::

    from nir.akida import AkidaTargetProfile, AkidaValidator
    from nir.akida.export_keras import export_keras

    profile = AkidaTargetProfile.v1()
    report = AkidaValidator(profile).validate(graph)
    if not report.is_valid:
        raise RuntimeError(str(report))
    keras_model = export_keras(graph, profile)
    keras_model.save("my_model.h5")
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

import nir

from .diagnostics import ValidationStage
from .metadata import QuantizationMetadata
from .profiles import AkidaTargetProfile
from .validator import AkidaValidator

# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class AkidaExportError(Exception):
    """Raised when a NIR graph cannot be exported to Keras for Akida.

    The message always names the problematic node and the reason, so that
    users can fix the graph before retrying.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Lazy Keras import so that the module can be loaded without Keras installed.
def _keras():
    try:
        import keras  # standalone Keras 3 / tf.keras
        return keras
    except ImportError:
        pass
    try:
        import tensorflow.keras as keras  # tf.keras fallback
        return keras
    except ImportError:
        raise ImportError(
            "Keras is required for export_keras.  Install it with:\n"
            "  pip install keras\n"
            "or\n"
            "  pip install tensorflow"
        )


def _topological_order(graph: nir.NIRGraph) -> List[str]:
    """Return node names in topological order (excluding Input/Output nodes)."""
    # Build adjacency from edges (NIRGraph auto-adds input_* / output_* wrappers)
    in_degree: dict = {name: 0 for name in graph.nodes}
    successors: dict = {name: [] for name in graph.nodes}
    for src, dst in graph.edges:
        if src in graph.nodes and dst in graph.nodes:
            successors[src].append(dst)
            in_degree[dst] += 1

    queue = [n for n, deg in in_degree.items() if deg == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    return order


def _input_shape_from_graph(graph: nir.NIRGraph) -> Tuple[int, ...]:
    """Extract the spatial input shape from the graph's Input node."""
    for node in graph.nodes.values():
        if isinstance(node, nir.Input):
            shape = node.input_type.get("input")
            if shape is not None:
                return tuple(int(x) for x in shape)
    raise AkidaExportError(
        "NIR graph must contain exactly one Input node to infer the Keras input shape."
    )


# ---------------------------------------------------------------------------
# Node-level export functions
# ---------------------------------------------------------------------------


def _export_linear(name: str, node: nir.Linear, keras: Any) -> Any:
    w = node.weight  # shape (out, in) in NIR convention
    return keras.layers.Dense(
        units=w.shape[0],
        use_bias=False,
        weights=[w.T],  # Keras Dense expects (in, out)
        name=name,
    )


def _export_affine(name: str, node: nir.Affine, keras: Any) -> Any:
    w = node.weight  # shape (out, in)
    b = node.bias
    return keras.layers.Dense(
        units=w.shape[0],
        use_bias=True,
        weights=[w.T, b],
        name=name,
    )


def _export_conv2d(name: str, node: nir.Conv2d, keras: Any) -> Any:
    w = node.weight  # (C_out, C_in, H, W) — NIR convention
    # Keras Conv2D expects kernel shape (H, W, C_in, C_out)
    kernel = np.transpose(w, (2, 3, 1, 0))
    use_bias = node.bias is not None and np.any(node.bias != 0)
    weights = [kernel, node.bias] if use_bias else [kernel]
    stride = node.stride if isinstance(node.stride, (list, tuple)) else (node.stride, node.stride)
    padding = node.padding
    if isinstance(padding, str):
        keras_padding = padding.lower()
    elif isinstance(padding, (int, float)):
        keras_padding = "valid" if int(padding) == 0 else "same"
    else:
        # Tuple padding: Keras doesn't support asymmetric, use 'same' as fallback
        keras_padding = "same"
    return keras.layers.Conv2D(
        filters=w.shape[0],
        kernel_size=(w.shape[2], w.shape[3]),
        strides=stride,
        padding=keras_padding,
        use_bias=use_bias,
        weights=weights,
        name=name,
    )


def _export_conv1d(name: str, node: nir.Conv1d, keras: Any) -> Any:
    w = node.weight  # (C_out, C_in, L) — NIR convention
    # Keras Conv1D expects kernel shape (L, C_in, C_out)
    kernel = np.transpose(w, (2, 1, 0))
    use_bias = node.bias is not None and np.any(node.bias != 0)
    weights = [kernel, node.bias] if use_bias else [kernel]
    stride = node.stride if isinstance(node.stride, (int,)) else node.stride[0]
    padding_val = node.padding
    if isinstance(padding_val, str):
        keras_padding = padding_val.lower()
    else:
        keras_padding = "valid" if int(padding_val) == 0 else "same"
    return keras.layers.Conv1D(
        filters=w.shape[0],
        kernel_size=w.shape[2],
        strides=stride,
        padding=keras_padding,
        use_bias=use_bias,
        weights=weights,
        name=name,
    )


def _export_avgpool2d(name: str, node: nir.AvgPool2d, keras: Any) -> Any:
    kernel_size = node.kernel_size
    if isinstance(kernel_size, np.ndarray):
        pool_size = tuple(int(x) for x in kernel_size)
    else:
        pool_size = (int(kernel_size), int(kernel_size))
    stride = node.stride
    if isinstance(stride, np.ndarray):
        strides = tuple(int(x) for x in stride)
    else:
        strides = (int(stride), int(stride))
    padding_val = node.padding
    if isinstance(padding_val, np.ndarray):
        keras_padding = "valid" if np.all(padding_val == 0) else "same"
    else:
        keras_padding = "valid" if padding_val == 0 else "same"
    return keras.layers.AveragePooling2D(
        pool_size=pool_size,
        strides=strides,
        padding=keras_padding,
        name=name,
    )


def _export_flatten(name: str, node: nir.Flatten, keras: Any) -> Any:
    return keras.layers.Flatten(name=name)


# LIF / IF → ReLU placeholder
# In a real CNN2SNN flow, QuantizeML/CNN2SNN maps ReLU activations to spiking
# neurons.  We export LIF and IF as ReLU here so that the Keras graph structure
# is preserved and the downstream tools can perform the substitution.
def _export_lif(name: str, node: nir.LIF, keras: Any) -> Any:
    threshold = float(np.asarray(node.v_threshold).flat[0])
    return keras.layers.ReLU(
        max_value=None,
        threshold=0.0,
        name=name,
    )


def _export_if(name: str, node: nir.IF, keras: Any) -> Any:
    return keras.layers.ReLU(
        max_value=None,
        threshold=0.0,
        name=name,
    )


# SumPool2d is approximated as AveragePooling2D (scale correction deferred)
def _export_sumpool2d(name: str, node: nir.SumPool2d, keras: Any) -> Any:
    kernel_size = node.kernel_size
    if isinstance(kernel_size, np.ndarray):
        pool_size = tuple(int(x) for x in kernel_size)
    else:
        pool_size = (int(kernel_size), int(kernel_size))
    stride = node.stride
    if isinstance(stride, np.ndarray):
        strides = tuple(int(x) for x in stride)
    else:
        strides = (int(stride), int(stride))
    padding_val = node.padding
    if isinstance(padding_val, np.ndarray):
        keras_padding = "valid" if np.all(padding_val == 0) else "same"
    else:
        keras_padding = "valid" if padding_val == 0 else "same"
    # SumPool → AvgPool approximation (MAPPING_RISK diagnostic emitted by validator)
    return keras.layers.AveragePooling2D(
        pool_size=pool_size,
        strides=strides,
        padding=keras_padding,
        name=name,
    )


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

#: Node types that are skipped during layer construction (not emitted as
#: Keras layers but used structurally).
_SKIP_TYPES = (nir.Input, nir.Output)


def export_keras(
    graph: nir.NIRGraph,
    profile: AkidaTargetProfile,
    *,
    skip_validation: bool = False,
) -> Any:
    """Convert a NIR graph to a Keras Sequential model for Akida deployment.

    The graph is first validated against *profile* (unless *skip_validation*
    is True).  Any validation failure raises :class:`AkidaExportError`.

    Args:
        graph: The NIR graph to export.  Must have been validated and must
            contain only MVP-supported nodes.
        profile: Akida target profile (:meth:`AkidaTargetProfile.v1` or
            :meth:`AkidaTargetProfile.v2`).
        skip_validation: If True, skip pre-export validation.  Use with care;
            an invalid graph will produce an unusable Keras model.

    Returns:
        A ``keras.Sequential`` model with layers corresponding to each
        non-structural NIR node, in topological order.

    Raises:
        AkidaExportError: If the graph contains unsupported nodes, or if
            validation fails (when *skip_validation* is False).
        ImportError: If Keras is not installed.
    """
    keras = _keras()

    if not skip_validation:
        report = AkidaValidator(profile).validate(graph)
        blocking = [
            d for d in report.diagnostics
            if d.stage in (ValidationStage.QUANTIZATION, ValidationStage.CONVERSION)
        ]
        if blocking:
            lines = "\n".join(f"  {d}" for d in blocking)
            raise AkidaExportError(
                f"Graph failed validation with {len(blocking)} blocking issue(s) "
                f"before Keras export.  Fix these and retry:\n{lines}"
            )

    input_shape = _input_shape_from_graph(graph)
    layers = [keras.layers.Input(shape=input_shape)]

    order = _topological_order(graph)

    for node_name in order:
        node = graph.nodes[node_name]

        if isinstance(node, _SKIP_TYPES):
            continue

        layer = _node_to_layer(node_name, node, keras)
        layers.append(layer)

    return keras.Sequential(layers, name="nir_akida_export")


def _node_to_layer(name: str, node: nir.NIRNode, keras: Any) -> Any:
    """Convert a single NIR node to a Keras layer.

    Raises:
        AkidaExportError: For nodes that passed validation but have no
            implemented Keras mapping (should not happen in normal use).
    """
    if isinstance(node, nir.Linear):
        return _export_linear(name, node, keras)
    if isinstance(node, nir.Affine):
        return _export_affine(name, node, keras)
    if isinstance(node, nir.Conv2d):
        return _export_conv2d(name, node, keras)
    if isinstance(node, nir.Conv1d):
        return _export_conv1d(name, node, keras)
    if isinstance(node, nir.AvgPool2d):
        return _export_avgpool2d(name, node, keras)
    if isinstance(node, nir.SumPool2d):
        return _export_sumpool2d(name, node, keras)
    if isinstance(node, nir.Flatten):
        return _export_flatten(name, node, keras)
    if isinstance(node, nir.LIF):
        return _export_lif(name, node, keras)
    if isinstance(node, nir.IF):
        return _export_if(name, node, keras)

    node_type = type(node).__name__
    raise AkidaExportError(
        f"Node '{name}' ({node_type}) has no Keras export implementation.  "
        "This node should have been rejected by the validator.  "
        "Please file a bug report."
    )
