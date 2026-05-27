# RFC: BrainChip Akida Backend Integration for NIR

**Status**: Draft  
**Authors**: NIR Contributors  
**Date**: 2026-05

---

## 1. Problem Statement

NIR is a hardware-agnostic neuromorphic intermediate representation.
BrainChip Akida is a commercial event-based AI inference chip that requires
a specific multi-step toolchain (QuantizeML → CNN2SNN) before execution.

NIR currently has no mechanism for:
- Expressing per-node quantization metadata
- Validating whether a NIR graph is compatible with Akida's constrained op set
- Exporting to Keras, which is the required entry point for the BrainChip toolchain
- Invoking QuantizeML and CNN2SNN in a reproducible way

This RFC describes a layered integration: **validation → Keras export →
toolchain orchestration**, treating the BrainChip tools as the authoritative
source for quantization and conversion behaviour.

---

## 2. Non-Goals

The following are explicitly **out of scope** for this integration:

- Direct generation of Akida `.fbz` binary artefacts inside NIR
- Replacing or wrapping BrainChip's closed-source tooling behaviour
- Floating-point inference on Akida (the chip is inference-only at low precision)
- Support for spiking dynamics beyond the Akida v1/v2 hardware-supported subset
- Graph partitioning and host/device split execution (Phase 5 / deferred)
- ONNX export (Phase 5 / deferred, unless trivially mirrored from Keras path)
- Automatic quantization calibration inside NIR
- Training or fine-tuning workflows

---

## 3. Target Architecture

The integration follows BrainChip's documented four-step philosophy:

```
NIR graph
    │
    ▼  [nir.akida.validator]
Validation (quantization / conversion / mapping-risk checks)
    │
    ▼  [nir.akida.export_keras]
Keras model (.h5 / SavedModel)
    │
    ▼  [QuantizeML — BrainChip tool, external]
Quantized Keras model
    │
    ▼  [CNN2SNN — BrainChip tool, external]
Akida model
    │
    ▼  [optional: akida.Model.map — BrainChip tool, external]
Device mapping report
```

**Module layout** (within the `nir` package):

```
nir/akida/
    __init__.py          – public API surface
    profiles.py          – AkidaTargetProfile (v1, v2) definitions
    metadata.py          – QuantizationMetadata dataclass / schema
    validator.py         – AkidaValidator: stages, diagnostics, report
    diagnostics.py       – ValidationStage enum, AkidaDiagnostic, ValidationReport
    export_keras.py      – NIRGraph → Keras Sequential / Functional model
    toolchain.py         – QuantizeML / CNN2SNN orchestration wrappers
    cli.py               – CLI entry points

tests/akida/
    __init__.py
    fixtures/            – golden NIR graphs stored as .nir files
    test_validator.py
    test_export_keras.py
    test_toolchain.py

docs/
    akida_integration_rfc.md        (this file)
    akida_capability_matrix.md
    akida_delivery_plan.md
```

NIR core is **not modified**; the Akida integration lives entirely inside
`nir/akida/` and depends only on public NIR APIs.

---

## 4. MVP Supported Op Subset

The following NIR primitives are targeted for Akida v1/v2 compatibility in
the MVP.  Justification is based on BrainChip's published model zoo and
CNN2SNN documentation (as of 2025).

| NIR Primitive | Akida representation | Notes |
|---------------|----------------------|-------|
| `Input`       | Input layer          | shape must be 1-D, 2-D or 3-D (no batch dim) |
| `Linear`      | Dense                | integer weights required after quantization |
| `Affine`      | Dense + bias         | bias folded into Dense |
| `Conv2d`      | Conv2D               | stride ≤ 2, no dilation > 1, groups == 1 for v1 |
| `Conv1d`      | Conv1D (v2 only)     | v1 does not support 1-D convolution |
| `LIF`         | SpikingConv2D / SpikingDense | v_reset must be 0, uniform threshold |
| `IF`          | SpikingConv2D / SpikingDense | uniform threshold across neurons |
| `AvgPool2d`   | AveragePooling2D     | integer kernel size only |
| `SumPool2d`   | (scaled to AvgPool2d) | divided by kernel area at export time |
| `Flatten`     | Flatten              | no reshape beyond flattening spatial dims |
| `Output`      | Output marker        | consumed at export, not emitted as layer |

---

## 5. Unsupported Ops / Deferred Items

| NIR Primitive | Status | Reason |
|---------------|--------|--------|
| `CubaLIF`     | Deferred | Two-compartment dynamics not directly mapped to Akida |
| `CubaLI`      | Deferred | Same as CubaLIF |
| `LI`          | Deferred | Non-spiking leaky integrator — no Akida analogue |
| `I`           | Deferred | Pure integrator — no Akida analogue |
| `Delay`       | Unsupported | No temporal delay primitive on Akida |
| `Threshold`   | Unsupported | Standalone threshold — must be fused with neuron |
| `Scale`       | Partial | Can be absorbed into adjacent weight layer during export |
| `Conv2d` (dilation > 1) | Unsupported | Akida v1 does not support dilated convolutions |
| `Conv2d` (groups > 1)   | Deferred  | Depthwise supported on v2 only |

The validator must emit a `CONVERSION` or `UNSUPPORTED` diagnostic for every
encountered op from this list.

---

## 6. Akida Target Profile Strategy

Two target profiles are defined at integration time:

**AkidaV1**
- Event-based convolutional SNN
- Supported layers: Dense, Conv2D, SeparableConv2D, AvgPool2D
- Weight precision: 1, 2, 4, or 8 bits
- No recurrent connections in MVP
- Input encoding: rate or delta

**AkidaV2 (Akida 2.0 / TENNs)**
- Extends v1 with temporal encoding
- Adds Conv1D, LSTM-like temporal layers (TENNs)
- Expanded bitwidth support

The profile is passed to the validator and exporter, allowing them to raise
profile-specific diagnostics.  All validation failures that are
profile-specific are tagged with `target_profile` in the diagnostic.

---

## 7. Quantization Metadata Requirements

NIR nodes carry an unstructured `metadata: Dict[str, Any]` field.
The Akida integration defines a typed `QuantizationMetadata` schema layered
on top of this field.

Required fields for quantizable layers (Linear, Affine, Conv1d, Conv2d):

| Field | Type | Description |
|-------|------|-------------|
| `weight_bits` | `int` | Bit-width for weight quantization (1, 2, 4, 8) |
| `activation_bits` | `int` | Bit-width for post-activation values (1, 2, 4, 8) |

Optional fields:

| Field | Type | Description |
|-------|------|-------------|
| `input_bits` | `int` | Bit-width for input quantization (first layer only) |
| `per_channel` | `bool` | Whether quantization is per-channel (default: False) |

For spiking neuron layers (LIF, IF), no explicit quantization fields are
required because the spike encoding is already binary.  The threshold must
be **uniform** (scalar or all-equal array) for Akida v1 compatibility.

The validator checks for presence and validity of quantization metadata and
raises a `QUANTIZATION` stage diagnostic on failure.

---

## 8. Validation Stages

Validation failures are classified into three stages to aid debugging:

| Stage | Description | Example |
|-------|-------------|---------|
| `QUANTIZATION` | Missing or invalid quantization metadata | `weight_bits` absent on Conv2d |
| `CONVERSION` | Op or pattern not convertible by CNN2SNN | Delay node present |
| `MAPPING_RISK` | Structurally convertible but may fail device mapping | Tensor size exceeds on-chip SRAM |

Each diagnostic carries: `stage`, `node_name`, `node_type`, `message`, and
optional `target_profile`.

---

## 9. Testing Strategy

- **Unit tests** for validator and export independently (no BrainChip tools required)
- **Golden fixture tests** for Keras export: reference `.nir` files stored in
  `tests/akida/fixtures/`, expected Keras layer sequences stored alongside
- **Integration tests** (optional, skipped if BrainChip tools absent) for
  QuantizeML / CNN2SNN round-trip
- **Negative tests** for each diagnostic category: unsupported op, missing
  quantization metadata, profile mismatch
- Tests use `pytest.importorskip` to skip Keras / BrainChip tool tests when
  the relevant packages are not installed

---

## 10. Risks and Open Questions

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | BrainChip CNN2SNN API changes between versions | Pin version in optional extras; document tested version |
| 2 | Keras version compatibility (tf.keras vs standalone keras) | Abstract Keras import; test on both; document |
| 3 | NIR `metadata` dict is untyped; silent metadata loss on round-trip | Use `QuantizationMetadata.to_dict()` / `from_dict()` with explicit schema version |
| 4 | Akida v2 TENN layers have no direct NIR analogue | Document as unsupported; track in capability matrix |
| 5 | Uniform threshold assumption for IF/LIF may be wrong for some models | Validator raises `MAPPING_RISK` if threshold is non-uniform |
| 6 | SumPool2d → AvgPool2d scaling may affect quantized accuracy | Document the transformation; emit `MAPPING_RISK` diagnostic |

**Open questions:**
- Should NIR define a first-class `QuantizationMetadata` extension point in core, or keep it in `nir.akida`?
- How should subgraph NIRGraph nodes (nested graphs) be handled in validation?
- Is per-channel quantization within BrainChip's documented support?
