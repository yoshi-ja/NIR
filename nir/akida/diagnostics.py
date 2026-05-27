"""Structured diagnostics for the Akida validation pipeline.

Failures are classified by stage so that users know exactly *where* in the
BrainChip toolchain (QuantizeML → CNN2SNN → akida.Model.map) a problem will
surface, even before the tools are invoked.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ValidationStage(Enum):
    """The stage of the Akida toolchain at which a problem will manifest."""

    QUANTIZATION = "quantization"
    """Missing or invalid quantization metadata; QuantizeML will fail."""

    CONVERSION = "conversion"
    """Op or pattern unsupported by CNN2SNN; conversion will fail."""

    MAPPING_RISK = "mapping_risk"
    """Structurally valid but may fail during akida.Model.map (e.g. SRAM limits,
    non-uniform threshold)."""


@dataclass(frozen=True)
class AkidaDiagnostic:
    """A single validation finding for one node in a NIR graph."""

    stage: ValidationStage
    node_name: str
    node_type: str
    message: str
    target_profile: Optional[str] = None
    """Non-None when the finding is specific to a particular Akida hardware
    generation (e.g. 'v1' or 'v2')."""

    def __str__(self) -> str:
        profile_tag = f" [profile={self.target_profile}]" if self.target_profile else ""
        return (
            f"[{self.stage.value.upper()}] {self.node_name} ({self.node_type})"
            f"{profile_tag}: {self.message}"
        )


@dataclass
class ValidationReport:
    """Aggregated result of running AkidaValidator on a NIR graph."""

    diagnostics: List[AkidaDiagnostic] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no diagnostics were raised (graph is safe to export)."""
        return len(self.diagnostics) == 0

    def by_stage(self, stage: ValidationStage) -> List[AkidaDiagnostic]:
        """Return only diagnostics for a given stage."""
        return [d for d in self.diagnostics if d.stage == stage]

    def has_stage(self, stage: ValidationStage) -> bool:
        """Return True if any diagnostic belongs to *stage*."""
        return any(d.stage == stage for d in self.diagnostics)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        if self.is_valid:
            return "ValidationReport: OK (no issues found)"
        counts = {s: 0 for s in ValidationStage}
        for d in self.diagnostics:
            counts[d.stage] += 1
        parts = [f"{v} {k.value}" for k, v in counts.items() if v > 0]
        return f"ValidationReport: FAIL ({', '.join(parts)})"

    def __str__(self) -> str:
        lines = [self.summary()]
        for d in self.diagnostics:
            lines.append(f"  {d}")
        return "\n".join(lines)
