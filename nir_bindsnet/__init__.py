"""BindsNET integration for NIR (Neuromorphic Intermediate Representation).

Provides bidirectional conversion between NIR graphs and BindsNET networks:
- ``from_nir``: Convert a ``nir.NIRGraph`` into a ``bindsnet.network.Network``
- ``to_nir``: Convert a ``bindsnet.network.Network`` into a ``nir.NIRGraph``

Supported NIR primitives:
- ``nir.Input`` / ``nir.Output``
- ``nir.Linear`` / ``nir.Affine``
- ``nir.LIF`` / ``nir.IF``

Supported BindsNET components:
- ``bindsnet.network.nodes.Input``
- ``bindsnet.network.nodes.LIFNodes``
- ``bindsnet.network.nodes.IFNodes``
- ``bindsnet.network.topology.Connection``
"""

# Apply compatibility shim before importing anything from BindsNET
from . import _compat  # noqa: F401

from .from_nir import from_nir
from .to_nir import to_nir

__all__ = ["from_nir", "to_nir"]
