# Exporting NIR to Keras for Akida

This page describes how to export a NIR graph to a Keras model that can be
passed to BrainChip's QuantizeML and CNN2SNN toolchain.

## Prerequisites

The NIR → Keras exporter lives in `nir.akida` and requires one of:

```bash
pip install keras             # standalone Keras 3
# or
pip install tensorflow        # includes tf.keras as fallback
```

BrainChip tools (QuantizeML, CNN2SNN) are **not** required for the export
step itself; they are needed only for the quantization and conversion steps
described in the [toolchain orchestration docs](akida_toolchain.md).

## Quick start

```python
import nir
from nir.akida import AkidaTargetProfile, AkidaValidator, QuantizationMetadata
from nir.akida.export_keras import export_keras

# 1. Load or construct a NIR graph
graph = nir.read("my_model.nir")

# 2. Attach quantization metadata to quantizable layers
for node_name, node in graph.nodes.items():
    if isinstance(node, (nir.Linear, nir.Affine, nir.Conv1d, nir.Conv2d)):
        node.metadata.update(
            QuantizationMetadata(weight_bits=8, activation_bits=4).to_dict()
        )

# 3. Choose a target profile
profile = AkidaTargetProfile.v1()  # or AkidaTargetProfile.v2()

# 4. Validate before export
report = AkidaValidator(profile).validate(graph)
if not report.is_valid:
    print(report)   # prints stage-classified diagnostics
    raise SystemExit(1)

# 5. Export
keras_model = export_keras(graph, profile)
keras_model.summary()

# 6. Save for downstream use with QuantizeML
keras_model.save("my_model.h5")
```

## Validation gate

`export_keras` always runs the validator internally before building the Keras
model.  If the graph has any `QUANTIZATION` or `CONVERSION` stage diagnostics,
it raises `AkidaExportError` with a list of blocking issues.

`MAPPING_RISK` diagnostics do **not** block export; they are logged and the
export proceeds.  Inspect the `ValidationReport` to review any risks before
passing the model to CNN2SNN.

To bypass validation (e.g. for debugging):

```python
keras_model = export_keras(graph, profile, skip_validation=True)
```

## Supported NIR primitives

See the [capability matrix](akida_capability_matrix.md) for the full table.
The MVP subset exported in this phase:

| NIR node | Keras layer |
|----------|-------------|
| `Input` | `keras.Input` (shape only) |
| `Output` | _(consumed as endpoint)_ |
| `Linear` | `Dense` (no bias) |
| `Affine` | `Dense` (with bias) |
| `Conv2d` | `Conv2D` |
| `Conv1d` | `Conv1D` (v2 profile only) |
| `LIF` | `ReLU` (CNN2SNN maps to spiking) |
| `IF` | `ReLU` (CNN2SNN maps to spiking) |
| `AvgPool2d` | `AveragePooling2D` |
| `SumPool2d` | `AveragePooling2D` + `MAPPING_RISK` diagnostic |
| `Flatten` | `Flatten` |

## LIF / IF mapping note

NIR `LIF` and `IF` nodes are exported as Keras `ReLU` layers.  This is
intentional: BrainChip's CNN2SNN tool recognises ReLU activations and converts
them to the corresponding spiking neuron type during the conversion step.
The NIR threshold value is preserved in the `ValidationReport` for reference
but is not encoded in the Keras model.

## SumPool2d note

`SumPool2d` is approximated as `AveragePooling2D` during export.  A
`MAPPING_RISK` diagnostic is always emitted.  The scale correction factor
(`kernel_h * kernel_w`) should be absorbed into the next weight layer if
possible; this is currently a manual step.

## Error handling

All export failures raise `AkidaExportError` with an actionable message:

```python
from nir.akida.export_keras import AkidaExportError

try:
    model = export_keras(graph, profile)
except AkidaExportError as e:
    print(f"Export failed: {e}")
```
