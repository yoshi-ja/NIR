"""Akida target profile definitions.

Two hardware generations are modelled:

- **AkidaV1** – the original BrainChip Akida (AKD1000 / AKD1500) event-based SNN
  accelerator.
- **AkidaV2** – Akida 2.0 (AKD2000), which adds temporal encoding (TENNs) and
  expanded layer support.

Profiles are passed to the validator and exporter to enable hardware-specific
checks and transformations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Tuple


class AkidaGeneration(Enum):
    """Hardware generation identifier."""

    V1 = "v1"
    V2 = "v2"


@dataclass(frozen=True)
class AkidaTargetProfile:
    """Immutable description of an Akida hardware target.

    Attributes:
        generation: Hardware generation (V1 or V2).
        supported_weight_bits: Allowed weight bit-widths for QuantizeML.
        supported_activation_bits: Allowed activation bit-widths.
        max_input_dims: Maximum number of spatial input dimensions accepted
            by this target (e.g. 2 for V1 Conv2D, 3 for V2 with temporal dim).
        supports_conv1d: Whether Conv1D layers are natively supported.
        supports_depthwise_conv: Whether depthwise (groups == in_channels) Conv2D
            is supported.
        requires_uniform_threshold: Whether neuron threshold must be a scalar
            (single value) rather than a per-neuron array.
        name: Human-readable profile name for diagnostics.
    """

    generation: AkidaGeneration
    supported_weight_bits: FrozenSet[int]
    supported_activation_bits: FrozenSet[int]
    max_input_dims: int
    supports_conv1d: bool
    supports_depthwise_conv: bool
    requires_uniform_threshold: bool
    name: str

    # Convenience class-methods -----------------------------------------------

    @classmethod
    def v1(cls) -> "AkidaTargetProfile":
        """Return the default Akida v1 (AKD1000/AKD1500) target profile."""
        return cls(
            generation=AkidaGeneration.V1,
            supported_weight_bits=frozenset({1, 2, 4, 8}),
            supported_activation_bits=frozenset({1, 2, 4, 8}),
            max_input_dims=2,
            supports_conv1d=False,
            supports_depthwise_conv=False,
            requires_uniform_threshold=True,
            name="AkidaV1",
        )

    @classmethod
    def v2(cls) -> "AkidaTargetProfile":
        """Return the default Akida v2 (AKD2000 / Akida 2.0) target profile."""
        return cls(
            generation=AkidaGeneration.V2,
            supported_weight_bits=frozenset({1, 2, 4, 8}),
            supported_activation_bits=frozenset({1, 2, 4, 8}),
            max_input_dims=3,
            supports_conv1d=True,
            supports_depthwise_conv=True,
            requires_uniform_threshold=False,
            name="AkidaV2",
        )

    def __str__(self) -> str:
        return self.name
