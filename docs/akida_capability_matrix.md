# Akida Backend Capability Matrix

This matrix maps every NIR primitive to its Akida compatibility status,
required attributes, quantization requirements, and export support.

`Status` legend:
- **MVP** — targeted in the initial Akida integration PR set
- **Later** — structurally possible, deferred to a follow-up PR
- **Unsupported** — not mappable to Akida hardware; validator emits `CONVERSION` diagnostic

`Keras export` and `ONNX export` columns reflect support within the
`nir.akida` integration; upstream NIR has no exporter.

---

## Primitive compatibility table

| NIR op / pattern | Required attributes (NIR fields) | Quantization requirements | Keras export support | ONNX export support | Akida v1 compatibility | Akida v2 compatibility | Status |
|---|---|---|---|---|---|---|---|
| `Input` | `input_type` shape (≤ 3-D spatial) | None | ✅ `keras.Input` | ⬜ Deferred | ✅ | ✅ | MVP |
| `Output` | `output_type` shape | None | ✅ (consumed as endpoint) | ⬜ Deferred | ✅ | ✅ | MVP |
| `Linear` | `weight` (2-D) | `weight_bits`, `activation_bits` | ✅ `Dense` | ⬜ Deferred | ✅ | ✅ | MVP |
| `Affine` | `weight` (2-D), `bias` (1-D) | `weight_bits`, `activation_bits` | ✅ `Dense` (bias folded) | ⬜ Deferred | ✅ | ✅ | MVP |
| `Conv2d` | `weight`, `stride`, `padding`, `dilation=1`, `groups=1` | `weight_bits`, `activation_bits` | ✅ `Conv2D` | ⬜ Deferred | ✅ (stride ≤ 2, no dilation) | ✅ | MVP |
| `Conv1d` | `weight`, `stride`, `padding`, `dilation=1` | `weight_bits`, `activation_bits` | ✅ `Conv1D` | ⬜ Deferred | ❌ not supported | ✅ | MVP (v2 only) |
| `LIF` | `tau`, `r`, `v_leak`, `v_threshold`, `v_reset=0` | None (spike output is binary) | ✅ `SpikingDense` / `SpikingConv2D` | ⬜ Deferred | ✅ (uniform threshold) | ✅ | MVP |
| `IF` | `r`, `v_threshold`, `v_reset=0` | None (spike output is binary) | ✅ `SpikingDense` / `SpikingConv2D` | ⬜ Deferred | ✅ (uniform threshold) | ✅ | MVP |
| `AvgPool2d` | `kernel_size`, `stride`, `padding` | None | ✅ `AveragePooling2D` | ⬜ Deferred | ✅ | ✅ | MVP |
| `SumPool2d` | `kernel_size`, `stride`, `padding` | None | ✅ via scale factor → `AveragePooling2D` | ⬜ Deferred | ⚠️ scaled approximation | ⚠️ scaled approximation | MVP (with MAPPING_RISK diagnostic) |
| `Flatten` | `start_dim`, `end_dim` | None | ✅ `Flatten` | ⬜ Deferred | ✅ | ✅ | MVP |
| `Scale` | `scale` | None | ⚠️ absorbed into adjacent Dense/Conv weight | ⬜ Deferred | ⚠️ must be absorbed | ⚠️ must be absorbed | Later |
| `Conv2d` (dilation > 1) | — | — | ❌ validator rejects | ❌ | ❌ | ❌ | Unsupported |
| `Conv2d` (groups > 1, not depthwise) | — | — | ❌ validator rejects | ❌ | ❌ | ⚠️ depthwise only | Later (depthwise v2) |
| `CubaLIF` | — | — | ❌ validator rejects | ❌ | ❌ | ❌ | Deferred |
| `CubaLI` | — | — | ❌ validator rejects | ❌ | ❌ | ❌ | Deferred |
| `LI` | — | — | ❌ validator rejects | ❌ | ❌ | ❌ | Deferred |
| `I` | — | — | ❌ validator rejects | ❌ | ❌ | ❌ | Deferred |
| `Delay` | — | — | ❌ validator rejects | ❌ | ❌ | ❌ | Unsupported |
| `Threshold` (standalone) | — | — | ❌ validator rejects (must be fused) | ❌ | ❌ | ❌ | Unsupported |
| `NIRGraph` (nested) | — | — | ⬜ Not yet handled | ⬜ | ⬜ | ⬜ | Later |

---

## Pattern-level notes

### Conv2d + LIF / IF (spiking convolutional block)
The most common Akida-targeted pattern.  Exported as `SpikingConv2D` (Akida
layer type) when the neuron node immediately follows a Conv2d with matching
spatial shape.  Both nodes must be present and adjacent in the edge list.

### Linear + LIF / IF (spiking dense block)
Exported as `SpikingDense`.  Same adjacency requirement as above.

### SumPool2d → AvgPool2d translation
SumPool2d is translated to `AveragePooling2D` with a scale correction factor
`kernel_h * kernel_w` applied to the next weight layer.  A `MAPPING_RISK`
diagnostic is always emitted because post-quantization the scale factor may
introduce rounding error.

### Uniform threshold requirement (v1)
Akida v1 requires a single threshold value per layer.  If a LIF or IF node
carries a non-uniform (per-neuron) threshold array, the validator emits a
`MAPPING_RISK` diagnostic.

---

## Quantization metadata schema (per quantizable layer)

| Field | Type | Required | Valid values | Notes |
|-------|------|----------|--------------|-------|
| `weight_bits` | `int` | Yes | 1, 2, 4, 8 | Weight bitwidth for QuantizeML |
| `activation_bits` | `int` | Yes | 1, 2, 4, 8 | Post-activation bitwidth |
| `input_bits` | `int` | No | 8 | Input encoding bitwidth (first layer only) |
| `per_channel` | `bool` | No | True / False | Per-channel vs per-tensor quantization |

Missing required fields on a quantizable layer raise a `QUANTIZATION` stage
diagnostic.  Invalid values (e.g. `weight_bits=3`) also raise `QUANTIZATION`.

---

## Diagnostic classification reference

| Diagnostic stage | Validator raises | Meaning |
|-----------------|-----------------|---------|
| `QUANTIZATION` | Missing / invalid quant metadata | Layer cannot be quantized by QuantizeML |
| `CONVERSION` | Unsupported op / pattern | CNN2SNN cannot convert this node |
| `MAPPING_RISK` | Valid but risky pattern | May fail during akida.Model.map |
