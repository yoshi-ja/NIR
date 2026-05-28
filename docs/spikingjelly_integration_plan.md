# NIR ↔ SpikingJelly Integration Plan

**Status:** Implementation-ready  
**Date:** 2026-05-27  
**Author:** Copilot  
**Version:** 1.0

---

## Executive Summary

SpikingJelly (`spikingjelly`) already ships a production-quality `nir_exchange` subpackage
(`spikingjelly.activation_based.nir_exchange`) that provides bidirectional NIR exchange via
`export_to_nir` and `import_from_nir`. The NIR core repository (`nir` package) defines the
common IR primitives that SpikingJelly translates to and from.

The integration work required in this repository is therefore **documentation-first**:

1. Add SpikingJelly to the official support table in `README.md`.
2. Add a canonical conversion example notebook under `docs/source/examples/spikingjelly/`.
3. Maintain a clear record of semantic constraints and known limitations.
4. Optionally validate round-trip numerical parity in CI (future milestone).

The first implementation slice (M1) consists of the two items above plus this planning
document itself. No new Python source code is needed in the `nir` package for M1.

---

## 1. Repository Audit

### 1.1 Layout of `yoshi-ja/NIR`

```
nir/
  __init__.py            – package entry point, exposes ir.*, data_ir.*, read/write
  ir/
    __init__.py          – exports all IR primitives + dict2NIRNode / str2NIRNode
    node.py              – NIRNode base dataclass
    neuron.py            – CubaLI, CubaLIF, I, IF, LI, LIF
    linear.py            – Affine, Linear, Scale
    conv.py              – Conv1d, Conv2d
    flatten.py           – Flatten
    graph.py             – NIRGraph, Input, Output
    delay.py             – Delay
    pooling.py           – AvgPool2d, SumPool2d
    threshold.py         – Threshold
    typing.py            – NIRNode type alias, Edges, Nodes, Types
    utils.py             – calc_flatten_output, calculate_conv_output, …
  data_ir/
    __init__.py
    graph.py             – NIRData graph (data exchange IR)
  serialization.py       – HDF5 read/write via h5py
docs/
  source/
    examples/            – per-framework notebook examples
      lava/, nengo/, norse/, rockpool/, sinabs/, snntorch/, spinnaker2/, spyx/ …
tests/
  test_ir.py, test_readwrite.py, test_architectures.py, test_data_ir.py, test_utils.py
pyproject.toml           – build, ruff, black, pytest config
```

### 1.2 Key Python Modules

| File | Purpose |
|------|---------|
| `nir/ir/neuron.py` | Defines IF, LIF, CubaLI, CubaLIF, LI, I |
| `nir/ir/graph.py` | NIRGraph (directed graph with Input/Output sentinels) |
| `nir/serialization.py` | HDF5 read / write |
| `nir/ir/linear.py` | Linear, Affine, Scale |
| `nir/ir/conv.py` | Conv1d, Conv2d |

### 1.3 Existing Serialization / Exchange Logic

- All primitives expose `to_dict()` / `from_dict()` (defined on `NIRNode`).
- `nir.write(path, graph)` and `nir.read(path)` provide HDF5 persistence.
- `NIRGraph.from_list(...)` auto-generates Input/Output sentinels and edges.
- Companion package `nirtorch` (not in this repo) provides PyTorch FX graph extraction
  and `torch_to_nir` / `nir_to_torch` utilities used by SpikingJelly and others.

### 1.4 Current Test Strategy

Tests live in `tests/` and run with `pytest`. They cover:
- IR construction and equality semantics.
- Serialization round-trips (HDF5 write + read).
- Architecture patterns (sequential, nested graphs, recurrent).
- Data IR.

CI uses GitHub Actions (`build.yml`) and tests Python 3.10–3.14 on ubuntu-latest.

### 1.5 Existing Integration Points

Each framework provides its own converter package (e.g., `norse.to_nir`,
`snntorch.export_nir`). The NIR repo itself only hosts documentation/example notebooks —
it does **not** contain framework-specific code. The same pattern applies to SpikingJelly.

---

## 2. External Dependency Audit

### 2.1 SpikingJelly Overview

- **Repo:** https://github.com/fangwei123456/spikingjelly
- **Docs:** https://spikingjelly.readthedocs.io
- **Primary module for NIR:** `spikingjelly.activation_based.nir_exchange`
  - `to_nir.py` → `export_to_nir(net, example_input, save_path=None, dt=1e-4)`
  - `from_nir.py` → `import_from_nir(graph, dt=1e-4, device="cpu", dtype=torch.float32, step_mode="s")`

### 2.2 Relevant SpikingJelly Modules

| Module | Description |
|--------|-------------|
| `spikingjelly.activation_based.neuron.IFNode` | Integrate-and-Fire neuron |
| `spikingjelly.activation_based.neuron.LIFNode` | Leaky Integrate-and-Fire neuron |
| `spikingjelly.activation_based.neuron.ParametricLIFNode` | LIF with learnable τ |
| `spikingjelly.activation_based.layer.Linear` | Linear with step-mode awareness |
| `spikingjelly.activation_based.layer.Conv2d` | Conv2d with step-mode awareness |
| `spikingjelly.activation_based.layer.AvgPool2d` | AvgPool2d with step-mode awareness |
| `spikingjelly.activation_based.layer.Flatten` | Flatten with step-mode awareness |
| `spikingjelly.activation_based.functional` | `set_step_mode`, `reset_net`, etc. |

### 2.3 Time-Step Conventions

SpikingJelly uses **activation-based** (clock-driven) simulation with discrete time steps.
Each forward call processes one or multiple time steps (`step_mode='s'` or `'m'`).

SpikingJelly's LIF dynamics (discrete):
```
H[t] = V[t-1] + (1/τ) * (-V[t-1] + X[t])     # if decay_input=True
H[t] = V[t-1] * (1 - 1/τ) + X[t]              # if decay_input=False
```

NIR's LIF dynamics (continuous ODE, Euler-discretised):
```
τ·dv/dt = (v_leak - v) + R·I
v[t+1] = v[t] + dt/τ·(v_leak - v[t]) + dt·R/τ·I[t]
```

**Mapping:**
- `tau_nir = tau_sj * dt`
- `r_nir = 1.0` when `decay_input=True`; `r_nir = tau_sj` when `decay_input=False`
- `v_leak_nir = v_reset_sj`  (SpikingJelly sets v_leak = v_reset in the NIR representation)

### 2.4 `nir_exchange` Status

As of master branch (2026-05), SpikingJelly's `nir_exchange` is fully functional and
covers the modules listed in section 2.2. It uses `nirtorch` for FX-graph-level graph
extraction and assembly.

### 2.5 Version Concerns

- SpikingJelly's `nir_exchange` uses `nirtorch` which in turn depends on `nir`.
  The latest `nirtorch` (0.3.x) is compatible with `nir` ≥ 1.1.
- SpikingJelly's even-numbered versions are stable; the GitHub master is the odd
  ("developing") version. NIR exchange is available on GitHub master and will appear
  in the next stable release.
- Pin `spikingjelly >= 0.0.0.0.15` (or install from GitHub master) in the example
  notebook's requirements cell to ensure `nir_exchange` is present.

---

## 3. Proposed Integration Scope

### 3.1 MVP Objective

Add SpikingJelly to the set of NIR-supported frameworks by:

1. Creating a canonical example notebook (`docs/source/examples/spikingjelly/nir-conversion.ipynb`)
   showing bidirectional conversion.
2. Updating `README.md` to list SpikingJelly in the support table.
3. Creating this planning document.

### 3.2 Non-Goals for v1

- No new Python source code in the `nir` package.
- No automated round-trip CI test (future milestone; blocked on SpikingJelly stable release).
- No support for `neuron.ParametricLIFNode` in the example (reduce scope; it maps to
  `nir.LIF` anyway via `export_to_nir`).
- No support for `nir.CubaLIF` ↔ SpikingJelly (no direct SJ equivalent).
- No support for recurrent or custom SpikingJelly neuron types.
- No support for SpiNNaker2 or hardware deployment via SpikingJelly.

### 3.3 Supported Module Subset (v1)

| SpikingJelly | NIR |
|---|---|
| `nn.Linear` / `layer.Linear` (no bias) | `nir.Linear` |
| `nn.Linear` / `layer.Linear` (with bias) | `nir.Affine` |
| `nn.Conv2d` / `layer.Conv2d` | `nir.Conv2d` |
| `nn.AvgPool2d` / `layer.AvgPool2d` | `nir.AvgPool2d` |
| `nn.Flatten` / `layer.Flatten` | `nir.Flatten` |
| `neuron.IFNode` | `nir.IF` |
| `neuron.LIFNode` | `nir.LIF` |

### 3.4 v1 Direction

**Bidirectional** (limited round-trip). SpikingJelly's `nir_exchange` already provides
both export and import, so the example notebook demonstrates both directions.

---

## 4. Semantic Compatibility Analysis

| Aspect | NIR | SpikingJelly | Compatibility |
|--------|-----|--------------|---------------|
| **Graph representation** | `NIRGraph` with named nodes + explicit edge list | `torch.nn.Sequential` or `fx.GraphModule` (FX trace) | ✅ `nirtorch` bridges FX ↔ NIRGraph |
| **Time axis** | Absent in IR (framework defines dt) | `step_mode='s'` (1-step) or `'m'` (T-step) | ✅ handled via `dt` parameter |
| **Neuron state** | Not serialized in NIR (transient) | `v` stored as `nn.Module` attribute, reset via `functional.reset_net` | ✅ state excluded from NIR graph; reset handled by caller |
| **Threshold/reset** | `v_threshold`, `v_reset` as numpy arrays | `v_threshold`, `v_reset` as floats or tensors | ✅ exact mapping |
| **Soft vs hard reset** | NIR IF/LIF always hard-reset (v_reset field) | `v_reset=None` = soft reset; float = hard reset | ⚠️ soft-reset maps to hard-reset with `v_reset=0` in NIR; information is lost |
| **Parameter representation** | numpy arrays (element-wise per neuron) | scalar or tensor | ✅ `np.full(shape, scalar)` in exporter |
| **τ convention** | Continuous-time `τ` (seconds) | Dimensionless decay factor | ✅ mapping: `tau_nir = tau_sj * dt` |
| **R convention** | Resistance term in ODE | Not explicit; folded into τ | ✅ `r=1.0` (decay_input=True) or `r=tau_sj` |
| **v_leak** | Explicit leak voltage | Implicit (= v_reset in most cases) | ⚠️ `v_leak_nir = v_reset_sj`; round-trip OK only when `v_leak==v_reset` |
| **Training semantics** | Out of scope (inference IR) | Uses surrogate gradients | ✅ NIR scoped to inference; gradients ignored |
| **Surrogate gradients** | `nir.Threshold` node (experimental) | `surrogate.ATan`, `PiecewiseLinear`, etc. | ❌ not mapped; out of scope for v1 |
| **ParametricLIF** | No direct primitive | `neuron.ParametricLIFNode` (learnable τ) | ✅ exported as `nir.LIF` with fixed τ snapshot |
| **CubaLIF** | `nir.CubaLIF` | No direct equivalent | ❌ not mapped |

### What can be preserved exactly

- Linear / Affine weights and biases.
- Conv2d weights, stride, padding, dilation, groups.
- AvgPool2d kernel/stride/padding.
- Flatten start/end dims.
- IF: v_threshold, v_reset (hard reset only).
- LIF: τ (via dt), v_threshold, v_reset, decay_input flag.

### What is approximate

- LIF v_leak (forced to equal v_reset; loses non-zero leak offset).
- IF soft-reset → hard-reset with v_reset=0.

### What is not preserved

- Surrogate gradient type.
- Training parameters beyond weight values (optimiser state, etc.).

---

## 5. Compatibility Matrix

| SpikingJelly Construct | NIR Equivalent | Support Status | Notes / Caveats | Test Needed |
|---|---|---|---|---|
| `layer.Linear` (no bias) | `nir.Linear` | ✅ Supported | weight only | Round-trip weight check |
| `layer.Linear` (with bias) | `nir.Affine` | ✅ Supported | weight + bias | Round-trip weight+bias check |
| `nn.Linear` | `nir.Linear` / `nir.Affine` | ✅ Supported | same as layer.Linear | idem |
| `layer.Conv2d` | `nir.Conv2d` | ✅ Supported | bias forced to zeros if None | Round-trip weight+meta check |
| `layer.AvgPool2d` | `nir.AvgPool2d` | ✅ Supported | kernel/stride/padding preserved | Spot-check |
| `layer.Flatten` | `nir.Flatten` | ✅ Supported | dim offset adjusted for T/B dims | Spot-check |
| `neuron.IFNode` (hard reset) | `nir.IF` | ✅ Supported | v_threshold, v_reset preserved | Numerical parity test |
| `neuron.IFNode` (soft reset) | `nir.IF` | ⚠️ Lossy | v_reset mapped to 0; not round-trippable | Note in docs |
| `neuron.LIFNode` | `nir.LIF` | ✅ Supported | tau, v_threshold, v_reset, decay_input | Numerical parity test |
| `neuron.ParametricLIFNode` | `nir.LIF` | ✅ Export-only | τ snapshot at export time; not re-trainable | Export check |
| `neuron.IFNode` heterogeneous params | `nir.IF` | ❌ Not supported | SJ importer requires uniform v_thr/v_reset | Raise AssertionError |
| `nir.CubaLIF` | (none) | ❌ Not mapped | No CUBA LIF in SJ activation_based | N/A |
| `nir.I` (integrator) | (none) | ❌ Not mapped | No direct SJ equivalent | N/A |
| `nir.LI` | (none) | ❌ Not mapped | No direct SJ equivalent | N/A |
| `nir.Delay` | (none) | ❌ Not mapped | No direct SJ equivalent | N/A |
| `nir.Conv1d` | (none) | ❌ Not mapped | SJ lacks Conv1d in nir_exchange | N/A |
| Recurrent / feedback edges | (none) | ❌ Not supported | nirtorch may not trace recurrence | Future milestone |

---

## 6. Architecture Proposal

### 6.1 Module / Package Layout

No changes to the `nir` Python package source code are required for M1–M3.

The integration lives in two places:

```
# SpikingJelly side (already implemented)
spikingjelly/
  activation_based/
    nir_exchange/
      __init__.py          # from .from_nir import *; from .to_nir import *
      to_nir.py            # export_to_nir()
      from_nir.py          # import_from_nir()

# NIR side (to add in M1)
docs/
  spikingjelly_integration_plan.md   ← this file
  source/
    examples/
      spikingjelly/
        nir-conversion.ipynb         ← M1
```

### 6.2 Key APIs

**Export (SpikingJelly → NIR):**
```python
from spikingjelly.activation_based.nir_exchange import export_to_nir
import torch
import nir

net = torch.nn.Sequential(...)  # SpikingJelly network
example_input = torch.zeros(1, C)  # single time-step, batch=1
nir_graph = export_to_nir(net, example_input, dt=1e-4)
nir.write("model.nir", nir_graph)
```

**Import (NIR → SpikingJelly):**
```python
from spikingjelly.activation_based.nir_exchange import import_from_nir
import nir

nir_graph = nir.read("model.nir")
sj_model = import_from_nir(nir_graph, dt=1e-4, step_mode='s')
```

### 6.3 Data Flow — Export

```
SpikingJelly nn.Module
    │
    ▼  nirtorch.torch_tracer.NIRTorchTracer (FX trace)
FX GraphModule
    │
    ▼  _ModuleMapper.map_dict  (to_nir.py)
    │     nn.Linear → nir.Linear / nir.Affine
    │     nn.Conv2d → nir.Conv2d
    │     nn.AvgPool2d → nir.AvgPool2d
    │     nn.Flatten → nir.Flatten
    │     neuron.IFNode → nir.IF
    │     neuron.LIFNode → nir.LIF
    │     neuron.ParametricLIFNode → nir.LIF (τ frozen)
    ▼  nirtorch.torch_to_nir
nir.NIRGraph
    │
    ▼  nir.write (HDF5)
model.nir
```

### 6.4 Data Flow — Import

```
model.nir (HDF5)
    │
    ▼  nir.read
nir.NIRGraph
    │
    ▼  _NodeMapper.map_dict  (from_nir.py)
    │     nir.Linear  → layer.Linear (no bias)
    │     nir.Affine  → layer.Linear (with bias)
    │     nir.Conv2d  → layer.Conv2d
    │     nir.AvgPool2d → layer.AvgPool2d
    │     nir.Flatten → layer.Flatten
    │     nir.IF      → neuron.IFNode
    │     nir.LIF     → neuron.LIFNode
    ▼  nirtorch.nir_to_torch
fx.GraphModule
    │
    ▼  functional.set_step_mode(gm, step_mode)
SpikingJelly model
```

### 6.5 Validation Points

- `_NodeMapper.map_lif`: asserts uniform τ, r, v_reset, v_leak across neurons.
- `_NodeMapper.map_if`: asserts uniform v_threshold, v_reset, r.
- `export_to_nir` raises `KeyError` on unsupported module types via `type_check=True`.

### 6.6 Error Handling Strategy

- Unsupported modules in export: `nirtorch` raises `KeyError` / `ValueError`.
- Heterogeneous neuron parameters in import: `AssertionError` with message.
- Soft-reset: silently maps v_reset to 0 with a docstring warning (no runtime error).
- Missing `v_reset` in HDF5: handled by `LIF.from_dict` / `IF.from_dict` defaults.

### 6.7 Version Pinning Strategy

The example notebook should pin:
- `spikingjelly` installed from GitHub master until a stable release with `nir_exchange`
  is published to PyPI (`>= 0.0.0.0.15`).
- `nirtorch >= 0.3.0`
- `nir >= 1.1.0`

---

## 7. Milestone Plan

### M0 — Discovery / Spec (complete)

**Goal:** Audit both repos and document the integration landscape.  
**Files:** `docs/spikingjelly_integration_plan.md` (this file)  
**Dependencies:** None  
**Risks:** SpikingJelly API may shift on master  
**Acceptance criteria:** Plan reviewed and approved

---

### M1 — Example Notebook + README Update

**Goal:** Add SpikingJelly to the official NIR support table and provide a canonical
bidirectional conversion example notebook.

**Files to create/modify:**
- `docs/source/examples/spikingjelly/nir-conversion.ipynb` ← **create**
- `README.md` ← add row to support table

**Dependencies:** M0  
**Risks:** spikingjelly not installable without heavy CUDA; example must run in CPU-only
environment  
**Acceptance criteria:**
- Notebook imports `spikingjelly.activation_based.nir_exchange`
- Notebook exports a Sequential(Linear + LIFNode) to NIR and reloads it
- Notebook imports a `nir.NIRGraph` into SpikingJelly
- README lists SpikingJelly with ✓ for both Write and Read

---

### M2 — Integration Tests

**Goal:** Add pytest tests in this repo that verify round-trip parity for the supported
module subset.

**Files to create:**
- `tests/test_spikingjelly_integration.py`

**Dependencies:** M1, stable PyPI release of SpikingJelly with `nir_exchange`  
**Risks:** Heavy dependency on torch + spikingjelly; must be skipped if not installed  
**Acceptance criteria:**
- `test_linear_roundtrip`: weights preserved to float32 precision
- `test_lif_roundtrip`: τ, v_threshold, v_reset preserved within `dt` tolerance
- `test_if_roundtrip`: v_threshold, v_reset preserved exactly
- `test_conv2d_roundtrip`: weight/bias/meta preserved exactly
- All tests decorated with `pytest.importorskip("spikingjelly")`

---

### M3 — Extended Features

**Goal:** Extend coverage to multi-step mode, nested graphs, and `ParametricLIFNode`.

**Files to modify:**
- `docs/source/examples/spikingjelly/nir-conversion.ipynb` (extend)
- `tests/test_spikingjelly_integration.py` (extend)

**Dependencies:** M2  
**Risks:** Multi-step FX tracing may differ from single-step  
**Acceptance criteria:** Multi-step forward pass output matches single-step numerically

---

### M4 — Hardware Deployment Path

**Goal:** Add a section showing how to deploy a SpikingJelly-exported NIR graph to
a hardware platform (e.g., SpiNNaker2 or Speck).

**Files:** Documentation only  
**Dependencies:** M3, hardware platform SDK access  
**Out of scope:** Actual hardware verification

---

## 8. Verification Strategy

### Unit Tests (M2)

```python
# tests/test_spikingjelly_integration.py
import pytest
sj = pytest.importorskip("spikingjelly")

def test_linear_roundtrip():
    ...  # export nn.Linear → nir → import; compare weights

def test_lif_roundtrip():
    ...  # export LIFNode → nir → import; compare τ, v_threshold, v_reset

def test_if_roundtrip():
    ...  # export IFNode → nir → import; compare v_threshold, v_reset

def test_conv2d_roundtrip():
    ...  # export Conv2d → nir → import; compare weight, bias, stride
```

### Notebook Demo (M1)

`docs/source/examples/spikingjelly/nir-conversion.ipynb` — executable notebook that:
1. Builds a `Sequential(layer.Linear + neuron.LIFNode)` in SpikingJelly.
2. Exports to `nir.NIRGraph` via `export_to_nir`.
3. Saves to an HDF5 file via `nir.write`.
4. Reloads via `nir.read`.
5. Imports back to SpikingJelly via `import_from_nir`.
6. Runs both models on the same input and prints output (visual parity check).

Also demonstrates the reverse: constructing a `nir.NIRGraph` manually and importing to
SpikingJelly.

### Numerical Parity

For M2, parity is defined as:
- Weight matrices: `np.allclose(w_original, w_roundtrip, rtol=1e-5, atol=1e-6)`
- Time constants: match within `dt=1e-4` relative tolerance
- Forward pass outputs: `torch.allclose(out_original, out_roundtrip, atol=1e-5)` on
  random input, single time step

### "Success" per Layer Type

| Layer | Success Criterion |
|---|---|
| Linear/Affine | Weight (and bias) bitwise identical after round-trip |
| Conv2d | Weight, bias, stride, padding, dilation, groups identical |
| AvgPool2d | kernel_size, stride, padding identical |
| Flatten | start_dim, end_dim equivalent (adjusted for T/B dims) |
| IFNode | v_threshold, v_reset identical; single-step output matches |
| LIFNode | τ (via dt), v_threshold, v_reset, decay_input flag restored; output matches |

---

## 9. Risk Register

| Risk | Severity | Likelihood | Mitigation | Fallback |
|---|---|---|---|---|
| SpikingJelly API drift (master branch) | High | Medium | Pin to tagged commit in notebook; update on release | Lock to last-known-good commit |
| `nirtorch` version incompatibility | High | Low | Pin `nirtorch >= 0.3.0` | Test with multiple nirtorch versions |
| Soft-reset semantic loss | Medium | High | Document clearly in notebook and plan | Mark as known limitation; error if downstream platform requires exact reset |
| Heterogeneous neuron params not supported | Medium | Medium | `AssertionError` from SJ importer; document limitation | Pre-validate params before export |
| FX tracing fails on custom neuron subclasses | High | Medium | Only support listed primitives; raise `KeyError` otherwise | Advise wrapping in `torch.fx.wrap` |
| Notebook-heavy repo: notebooks not tested in CI | Medium | High | Include import/export as code cells that produce no errors; add test fixture in M2 | Keep notebook as documentation-only |
| torch installation size in CI | Medium | High | Use `pytest.importorskip`; skip SJ tests if not installed | Mark tests as optional |
| v_leak ≠ v_reset in imported LIF | Medium | Low | Import raises `AssertionError`; document assumption | Extend SJ importer to handle v_leak offset in future milestone |
| CubaLIF / LI / I / Delay not mapped | Low | High | Document in compatibility matrix | Raise informative error; suggest alternative mappings |

---

## 10. Recommended First Implementation Slice

The safest and highest-leverage first slice is **M1: Example Notebook + README update**.

**Rationale:**
- SpikingJelly's `nir_exchange` is already complete. There is no new Python code to write
  in the `nir` package.
- Adding the example notebook and README entry is low-risk, immediately useful to users,
  and follows the established pattern of all other NIR-supported frameworks.
- The notebook acts as living documentation and a smoke test.
- Updating the README support table raises SpikingJelly's visibility within the NIR
  ecosystem.

**Slice definition:**

1. `docs/source/examples/spikingjelly/nir-conversion.ipynb`  
   — Shows export from SpikingJelly to NIR and import back.  
   — Demonstrates manual NIR graph construction + SpikingJelly import.  
   — Plain Python cells; no GPU required; minimal dependencies.

2. `README.md` — Add SpikingJelly row with ✓ for both Write and Read.

---

## Open Questions / Assumptions

1. **Which SpikingJelly version to target?** Assumed: GitHub master (odd, developing
   version) which includes `nir_exchange`. Stable PyPI release with this feature is
   pending. The example notebook should note this.

2. **Should the example be runnable in CI?** For M1, no — too heavy. For M2, via
   `pytest.importorskip`. Document this decision.

3. **Does `nirtorch.torch_to_nir` support multi-step (`step_mode='m'`) networks?**
   Likely yes (FX trace is step-mode agnostic), but not verified. Assumed single-step
   for M1.

4. **Is there a `supported_primitives.md` auto-generation workflow for SpikingJelly?**
   The repo has a `supported_primitives.py` and workflow
   (`update_supported_primitives.yml`). SpikingJelly should be added if/when the
   framework registers itself.

5. **Recurrent networks:** SpikingJelly supports recurrent topologies. NIRGraph supports
   feedback edges. However, FX tracing of truly recurrent modules may require additional
   handling. Deferred to M3/M4.
