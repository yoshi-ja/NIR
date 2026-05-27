"""Akida graph validator.

``AkidaValidator`` inspects a NIR graph and classifies every potential problem
into one of three stages (see :mod:`nir.akida.diagnostics`):

- ``QUANTIZATION`` — missing or invalid quantization metadata
- ``CONVERSION``   — op or pattern that CNN2SNN cannot convert
- ``MAPPING_RISK`` — structurally convertible but may fail device mapping

Call :meth:`AkidaValidator.validate` before attempting export or toolchain
invocation.  The returned :class:`~nir.akida.diagnostics.ValidationReport`
lists every issue with its associated node so that users can fix problems
before calling into BrainChip tools.

No BrainChip packages are imported or required here.
"""

from __future__ import annotations

import numpy as np

import nir

from .diagnostics import AkidaDiagnostic, ValidationReport, ValidationStage
from .metadata import QuantizationMetadata
from .profiles import AkidaTargetProfile

# ---------------------------------------------------------------------------
# Node type sets
# ---------------------------------------------------------------------------

# NIR types that require quantization metadata before Akida export
_QUANTIZABLE_TYPES = (nir.Linear, nir.Affine, nir.Conv1d, nir.Conv2d)

# NIR types that are never convertible to Akida (emit CONVERSION diagnostic)
_UNSUPPORTED_TYPES = (nir.Delay, nir.CubaLI, nir.CubaLIF, nir.LI, nir.I)

# NIR types that are unsupported as *standalone* nodes (must be fused)
_STANDALONE_UNSUPPORTED = (nir.Threshold,)

# NIR types that are supported as passthrough/structural nodes
_STRUCTURAL_TYPES = (nir.Input, nir.Output, nir.Flatten)

# NIR types that have potential mapping risks requiring inspection
_SPIKING_TYPES = (nir.LIF, nir.IF)

# NIR pooling types
_POOL_TYPES = (nir.AvgPool2d, nir.SumPool2d)


class AkidaValidator:
    """Validates a NIR graph for Akida compatibility.

    Args:
        profile: The Akida hardware target profile to validate against.
            Use :meth:`~nir.akida.profiles.AkidaTargetProfile.v1` or
            :meth:`~nir.akida.profiles.AkidaTargetProfile.v2`.
    """

    def __init__(self, profile: AkidaTargetProfile) -> None:
        self._profile = profile

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, graph: nir.NIRGraph) -> ValidationReport:
        """Validate *graph* against the configured profile.

        Args:
            graph: The top-level NIR graph to check.

        Returns:
            A :class:`~nir.akida.diagnostics.ValidationReport` containing all
            found issues.  Check :attr:`~ValidationReport.is_valid` to decide
            whether to proceed.
        """
        report = ValidationReport()
        for node_name, node in graph.nodes.items():
            self._validate_node(node_name, node, report)
        return report

    # ------------------------------------------------------------------
    # Per-node checks (each method appends to *report*)
    # ------------------------------------------------------------------

    def _validate_node(
        self, name: str, node: nir.NIRNode, report: ValidationReport
    ) -> None:
        node_type = type(node).__name__

        # Structural nodes: no checks needed
        if isinstance(node, _STRUCTURAL_TYPES):
            return

        # Completely unsupported nodes
        if isinstance(node, _UNSUPPORTED_TYPES):
            report.diagnostics.append(
                AkidaDiagnostic(
                    stage=ValidationStage.CONVERSION,
                    node_name=name,
                    node_type=node_type,
                    message=(
                        f"{node_type} has no Akida hardware equivalent and cannot be "
                        "converted by CNN2SNN.  Remove this node or replace it with a "
                        "supported primitive."
                    ),
                )
            )
            return

        if isinstance(node, _STANDALONE_UNSUPPORTED):
            report.diagnostics.append(
                AkidaDiagnostic(
                    stage=ValidationStage.CONVERSION,
                    node_name=name,
                    node_type=node_type,
                    message=(
                        f"Standalone {node_type} is not supported.  Threshold must be "
                        "fused into a spiking neuron layer (LIF or IF)."
                    ),
                )
            )
            return

        # Conv checks
        if isinstance(node, nir.Conv2d):
            self._check_conv2d(name, node, report)
            return

        if isinstance(node, nir.Conv1d):
            self._check_conv1d(name, node, report)
            return

        # Linear / Affine: quantization metadata required
        if isinstance(node, _QUANTIZABLE_TYPES):
            self._check_quantization_metadata(name, node, report)
            return

        # Spiking neurons: threshold uniformity check
        if isinstance(node, _SPIKING_TYPES):
            self._check_spiking_neuron(name, node, report)
            return

        # SumPool2d: always a mapping risk (approximation)
        if isinstance(node, nir.SumPool2d):
            report.diagnostics.append(
                AkidaDiagnostic(
                    stage=ValidationStage.MAPPING_RISK,
                    node_name=name,
                    node_type=node_type,
                    message=(
                        "SumPool2d is translated to AveragePooling2D with a scale "
                        "correction factor.  Post-quantization rounding may degrade "
                        "accuracy.  Consider replacing with AvgPool2d if possible."
                    ),
                )
            )
            return

        # Scale: must be absorbed into an adjacent weight layer; not yet supported
        if isinstance(node, nir.Scale):
            report.diagnostics.append(
                AkidaDiagnostic(
                    stage=ValidationStage.CONVERSION,
                    node_name=name,
                    node_type=node_type,
                    message=(
                        "Scale nodes must be absorbed into an adjacent weight layer "
                        "before export.  Standalone Scale is not yet supported."
                    ),
                )
            )
            return

        # Nested NIRGraph: not yet supported
        if isinstance(node, nir.NIRGraph):
            report.diagnostics.append(
                AkidaDiagnostic(
                    stage=ValidationStage.CONVERSION,
                    node_name=name,
                    node_type=node_type,
                    message=(
                        "Nested NIRGraph nodes are not yet handled by the Akida "
                        "validator.  Flatten the graph before validation."
                    ),
                )
            )
            return

    # ------------------------------------------------------------------
    # Specific checks
    # ------------------------------------------------------------------

    def _check_quantization_metadata(
        self, name: str, node: nir.NIRNode, report: ValidationReport
    ) -> None:
        node_type = type(node).__name__
        if not QuantizationMetadata.is_present(node):
            report.diagnostics.append(
                AkidaDiagnostic(
                    stage=ValidationStage.QUANTIZATION,
                    node_name=name,
                    node_type=node_type,
                    message=(
                        f"{node_type} is missing required quantization metadata "
                        "('weight_bits' and 'activation_bits').  Add a "
                        "QuantizationMetadata entry to node.metadata before running "
                        "QuantizeML."
                    ),
                )
            )
            return

        # Validate bit-width values
        try:
            qm = QuantizationMetadata.from_node(node)
        except (KeyError, ValueError) as exc:
            report.diagnostics.append(
                AkidaDiagnostic(
                    stage=ValidationStage.QUANTIZATION,
                    node_name=name,
                    node_type=node_type,
                    message=f"Invalid quantization metadata: {exc}",
                )
            )
            return

        # Check against profile-supported bit-widths
        if qm.weight_bits not in self._profile.supported_weight_bits:
            report.diagnostics.append(
                AkidaDiagnostic(
                    stage=ValidationStage.QUANTIZATION,
                    node_name=name,
                    node_type=node_type,
                    message=(
                        f"weight_bits={qm.weight_bits} is not supported by "
                        f"{self._profile.name}.  "
                        f"Supported values: {sorted(self._profile.supported_weight_bits)}"
                    ),
                    target_profile=self._profile.generation.value,
                )
            )
        if qm.activation_bits not in self._profile.supported_activation_bits:
            report.diagnostics.append(
                AkidaDiagnostic(
                    stage=ValidationStage.QUANTIZATION,
                    node_name=name,
                    node_type=node_type,
                    message=(
                        f"activation_bits={qm.activation_bits} is not supported by "
                        f"{self._profile.name}.  "
                        f"Supported values: "
                        f"{sorted(self._profile.supported_activation_bits)}"
                    ),
                    target_profile=self._profile.generation.value,
                )
            )

    def _check_conv2d(
        self, name: str, node: nir.Conv2d, report: ValidationReport
    ) -> None:
        # Check quantization metadata first
        self._check_quantization_metadata(name, node, report)

        # Dilation check (not supported on either target)
        dilation = node.dilation
        if isinstance(dilation, (list, tuple)):
            has_dilation = any(d > 1 for d in dilation)
        else:
            has_dilation = dilation > 1
        if has_dilation:
            report.diagnostics.append(
                AkidaDiagnostic(
                    stage=ValidationStage.CONVERSION,
                    node_name=name,
                    node_type="Conv2d",
                    message=(
                        f"Conv2d with dilation={node.dilation} is not supported by "
                        "Akida.  Only dilation=1 is permitted."
                    ),
                )
            )

        # Groups check
        groups = node.groups
        in_channels = node.weight.shape[1]
        if groups > 1:
            if not self._profile.supports_depthwise_conv:
                report.diagnostics.append(
                    AkidaDiagnostic(
                        stage=ValidationStage.CONVERSION,
                        node_name=name,
                        node_type="Conv2d",
                        message=(
                            f"Conv2d with groups={groups} requires depthwise "
                            f"convolution support, which is not available on "
                            f"{self._profile.name}."
                        ),
                        target_profile=self._profile.generation.value,
                    )
                )
            elif groups != in_channels:
                # Non-depthwise grouped conv not supported even on v2
                report.diagnostics.append(
                    AkidaDiagnostic(
                        stage=ValidationStage.CONVERSION,
                        node_name=name,
                        node_type="Conv2d",
                        message=(
                            f"Conv2d with groups={groups} (not depthwise) is not "
                            "supported.  Only groups=1 or groups==in_channels "
                            "(depthwise) is permitted."
                        ),
                    )
                )

    def _check_conv1d(
        self, name: str, node: nir.Conv1d, report: ValidationReport
    ) -> None:
        if not self._profile.supports_conv1d:
            report.diagnostics.append(
                AkidaDiagnostic(
                    stage=ValidationStage.CONVERSION,
                    node_name=name,
                    node_type="Conv1d",
                    message=(
                        f"Conv1d is not supported on {self._profile.name}.  "
                        "Upgrade to Akida v2 or replace with Conv2d."
                    ),
                    target_profile=self._profile.generation.value,
                )
            )
            return
        # Conv1d on v2: still needs quantization metadata
        self._check_quantization_metadata(name, node, report)

    def _check_spiking_neuron(
        self, name: str, node: nir.NIRNode, report: ValidationReport
    ) -> None:
        node_type = type(node).__name__
        threshold = node.v_threshold

        if self._profile.requires_uniform_threshold:
            # Scalar is fine; array must be all-equal
            if isinstance(threshold, np.ndarray) and threshold.ndim > 0:
                if not np.all(threshold == threshold.flat[0]):
                    report.diagnostics.append(
                        AkidaDiagnostic(
                            stage=ValidationStage.MAPPING_RISK,
                            node_name=name,
                            node_type=node_type,
                            message=(
                                f"{node_type} has a non-uniform (per-neuron) threshold "
                                f"array.  {self._profile.name} requires a single scalar "
                                "threshold per layer.  Device mapping may fail."
                            ),
                            target_profile=self._profile.generation.value,
                        )
                    )

        # v_reset must be zero for Akida
        v_reset = getattr(node, "v_reset", None)
        if v_reset is not None:
            reset_array = np.asarray(v_reset)
            if not np.all(reset_array == 0):
                report.diagnostics.append(
                    AkidaDiagnostic(
                        stage=ValidationStage.MAPPING_RISK,
                        node_name=name,
                        node_type=node_type,
                        message=(
                            f"{node_type} has a non-zero v_reset.  Akida always resets "
                            "membrane potential to 0 after a spike.  "
                            "Non-zero reset may not be honoured after conversion."
                        ),
                    )
                )
