"""Quantization metadata schema for Akida-targeted NIR nodes.

NIR nodes carry an unstructured ``metadata: Dict[str, Any]`` field.
``QuantizationMetadata`` provides a typed schema on top of that dict so that
the validator and exporter can read and write quantization parameters without
relying on ad-hoc key names.

Usage::

    # Attach metadata to a NIR node
    node = nir.Linear(weight=w)
    node.metadata.update(QuantizationMetadata(weight_bits=8, activation_bits=4).to_dict())

    # Read it back
    qm = QuantizationMetadata.from_node(node)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Sentinel used to detect missing required fields
_MISSING = object()

# Valid bit-width choices accepted by QuantizeML / Akida
_VALID_BITS = frozenset({1, 2, 4, 8})

_SCHEMA_VERSION = "1"
_SCHEMA_KEY = "_akida_qm_version"


@dataclass
class QuantizationMetadata:
    """Typed quantization parameters for a single NIR layer.

    Required for all quantizable layers (Linear, Affine, Conv1d, Conv2d).
    Not required for spiking neuron layers (LIF, IF) because spike output is
    inherently binary.

    Attributes:
        weight_bits: Bit-width for weight quantization (1, 2, 4, or 8).
        activation_bits: Bit-width for post-activation values (1, 2, 4, or 8).
        input_bits: Bit-width for input encoding (optional; for first layer only).
        per_channel: Whether quantization is per-channel rather than per-tensor.
    """

    weight_bits: int
    activation_bits: int
    input_bits: Optional[int] = None
    per_channel: bool = False

    # Validation --------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.weight_bits not in _VALID_BITS:
            raise ValueError(
                f"weight_bits must be one of {sorted(_VALID_BITS)}, got {self.weight_bits}"
            )
        if self.activation_bits not in _VALID_BITS:
            raise ValueError(
                f"activation_bits must be one of {sorted(_VALID_BITS)}, "
                f"got {self.activation_bits}"
            )
        if self.input_bits is not None and self.input_bits not in _VALID_BITS:
            raise ValueError(
                f"input_bits must be one of {sorted(_VALID_BITS)}, got {self.input_bits}"
            )

    # Serialisation -----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a flat dict suitable for merging into a NIR node's metadata."""
        d: Dict[str, Any] = {
            _SCHEMA_KEY: _SCHEMA_VERSION,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "per_channel": self.per_channel,
        }
        if self.input_bits is not None:
            d["input_bits"] = self.input_bits
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QuantizationMetadata":
        """Reconstruct from a flat metadata dict (as stored in a NIR node).

        Raises:
            KeyError: if required keys are absent.
            ValueError: if values are out of range.
        """
        return cls(
            weight_bits=d["weight_bits"],
            activation_bits=d["activation_bits"],
            input_bits=d.get("input_bits"),
            per_channel=bool(d.get("per_channel", False)),
        )

    @classmethod
    def from_node(cls, node: Any) -> "QuantizationMetadata":
        """Read metadata from a NIR node's ``metadata`` dict.

        Raises:
            KeyError: if required quantization keys are absent.
        """
        return cls.from_dict(node.metadata)

    @classmethod
    def is_present(cls, node: Any) -> bool:
        """Return True if *node* carries the required quantization fields."""
        m = getattr(node, "metadata", {})
        return "weight_bits" in m and "activation_bits" in m
