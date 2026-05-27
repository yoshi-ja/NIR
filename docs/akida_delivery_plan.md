# Akida Integration Delivery Plan

This document breaks the Akida backend integration into small, independently
reviewable PRs.  Each milestone has a clear scope, acceptance criteria, and
list of files touched.

---

## Milestone 0 — Repository Reconnaissance (complete)

**Scope**: Understand NIR architecture; identify gaps; produce Phase 1 planning artefacts.

**Acceptance criteria**:
- Architecture summary written
- Proposed module layout agreed
- No code changes

**Files created**: `docs/akida_integration_rfc.md`, `docs/akida_capability_matrix.md`, `docs/akida_delivery_plan.md`

---

## Milestone 1 — Planning Artefacts (this PR)

**Scope**: RFC, capability matrix, delivery plan documents only.

**Acceptance criteria**:
- RFC covers all required sections (see RFC doc)
- Capability matrix covers all 17 NIR primitives
- Delivery plan lists all milestones with acceptance criteria
- No code changes in `nir/`

**Files**: `docs/akida_integration_rfc.md`, `docs/akida_capability_matrix.md`, `docs/akida_delivery_plan.md`

---

## Milestone 2 — Validation Module

**Scope**: Akida target profiles, quantization metadata schema, validator, diagnostics.  **No exporter or toolchain code.**

**Acceptance criteria**:
- `AkidaTargetProfile` enum/class with v1 and v2 variants
- `QuantizationMetadata` dataclass with `to_dict` / `from_dict`
- `AkidaValidator.validate(graph, profile)` returns `ValidationReport`
- `ValidationReport` lists zero or more `AkidaDiagnostic` items
- Each diagnostic carries `stage`, `node_name`, `node_type`, `message`
- Test coverage:
  - Valid linear+LIF graph → no diagnostics
  - `Delay` node → `CONVERSION` diagnostic
  - `Linear` without `weight_bits` → `QUANTIZATION` diagnostic
  - LIF with non-uniform threshold on v1 profile → `MAPPING_RISK` diagnostic
- All tests pass with zero BrainChip dependencies installed

**Files**:
```
nir/akida/__init__.py
nir/akida/profiles.py
nir/akida/metadata.py
nir/akida/validator.py
nir/akida/diagnostics.py
tests/akida/__init__.py
tests/akida/fixtures/  (golden .nir files)
tests/akida/test_validator.py
```

---

## Milestone 3 — Keras Export

**Scope**: NIR → Keras model export for the validated MVP subset.  **No toolchain invocation.**

**Acceptance criteria**:
- `export_keras(graph, profile)` returns a `keras.Model`
- Export raises `AkidaExportError` with actionable message on any unsupported pattern
- Export is only attempted after validator passes (or caller bypasses with `skip_validation=True`)
- Golden fixture round-trip: exported Keras model layer sequence matches expected
- Tests use `pytest.importorskip("keras")` / `pytest.importorskip("tensorflow")`
- Docs section added: "Exporting to Keras for Akida"

**Files**:
```
nir/akida/export_keras.py
tests/akida/test_export_keras.py
docs/akida_keras_export.md  (or section in RFC update)
```

---

## Milestone 4 — Toolchain Orchestration

**Scope**: Wrappers for QuantizeML and CNN2SNN invocation; CLI entry points; artifact/log capture.

**Acceptance criteria**:
- `quantize(keras_model, config)` invokes QuantizeML; returns quantized model path
- `convert(quantized_model_path, profile)` invokes CNN2SNN; returns Akida model path
- `run_akida_flow(graph, profile, output_dir)` orchestrates full pipeline
- CLI command `python -m nir.akida.cli` exposes `validate`, `export`, `quantize`, `convert` subcommands
- Each stage writes a log file to `output_dir/`
- Failures from BrainChip tools are surfaced as `AkidaToolchainError` with stage tag
- Tests use `pytest.importorskip("quantizeml")` / `pytest.importorskip("cnn2snn")`; all tests skipped cleanly without BrainChip tools
- README / docs updated with installation note for optional BrainChip dependencies

**Files**:
```
nir/akida/toolchain.py
nir/akida/cli.py
tests/akida/test_toolchain.py
docs/akida_toolchain.md
```

---

## Milestone 5 — Later Work (planned, not yet scheduled)

The following items are planned but not yet scheduled.  They should each be
their own PR when the time comes.

### 5a — ONNX Export
- Export NIR MVP subset to ONNX as an alternative to Keras
- Useful if BrainChip adds ONNX→CNN2SNN support
- **Blocked on**: confirmation that CNN2SNN accepts ONNX input

### 5b — Subgraph Partitioning
- Identify the maximal Akida-compatible subgraph of an arbitrary NIR graph
- Emit a `PartitionedGraph` with `akida_subgraph` and `host_subgraph` components
- **Blocked on**: Milestone 2 and 3 being stable

### 5c — Host/Device Split Execution Plan
- Given a partitioned graph, produce an execution plan that routes tensors between host and Akida
- **Blocked on**: Milestone 5b

### 5d — Device Mapping Reports
- Invoke `akida.Model.map` and parse its output into a structured report
- Surface per-layer SRAM usage, throughput estimates
- **Blocked on**: Milestone 4

### 5e — Embedded Deployment Guidance
- Document `.fbz` generation workflow (fully via BrainChip tools)
- Add example scripts for AKD1000 and AKD1500 targets
- **Blocked on**: Milestone 4 being stable

---

## Dependency graph

```
M1 (docs)
  └─► M2 (validation)
        └─► M3 (Keras export)
              └─► M4 (toolchain)
                    └─► M5a–e (later)
```

Each milestone is independently reviewable and mergeable.
No milestone introduces changes that break a preceding milestone's tests.
