# ICP Execution Harness — Design Specification

This document captures the architecture and design decisions for the configuration and execution harness around the ICP system-level model. The harness exists to run the ICP algorithm over point cloud pairs with a fully specified, reproducible set of parameters, and to support parameter sweeps — the primary objective being the evaluation of different quantization levels of point cloud coordinate values.

The core library (`ICP`, `PointCloud`, `KDTree`, `TargetCloud`) is unchanged by this design except for one small addition noted at the end. What is being designed is the layer that sits on top: how runs are configured, validated, and executed.

---

## 1. Motivation and scope

The original implementation used a marimo notebook that combined two distinct concerns: exploratory data analysis (point count and coordinate distribution plots) and execution (an options form feeding a `process()` loop). Mixing these created coupling — marimo's constraint that a `UIElement`'s `.value` cannot be read in the cell that created it forced awkward multi-cell state gymnastics for what was essentially a parameter form.

The resolution is to separate the two concerns by role:

- **Exploratory analysis** stays in a notebook (marimo or Jupyter), where reactivity is a genuine benefit and there is no parameter-form friction.
- **Execution** moves to a small command-line harness driven by configuration files, because the parameters are known before each run and the real goal — comparing across quantization levels — is inherently a sweep, which a config-driven script expresses as data rather than as repeated manual UI interaction.

A purely interactive terminal-prompt approach was considered and rejected: it cannot be scripted, cannot re-run a configuration without re-typing, and is slower per run, all without buying anything when the inputs are known in advance.

---

## 2. System architecture

![System architecture](figures/01_architecture.svg)

The harness (`run_icp.py`) reads configuration files, resolves and validates them, computes the quantization range where needed, and drives the core library. Outputs are written per run, with a manifest tying a sweep together. The analysis notebook is a separate artifact that consumes the same trace outputs but is not part of the execution path.

The core library classes are deliberately left untouched. Their clean separation from orchestration is what makes swapping the harness on top inexpensive — `process()` already has essentially the right signature.

---

## 3. Configuration file design

### 3.1 Plain config — parameters at the top level

A single-run configuration carries its parameters at the top level, with no wrapper key. Grouping into logical sections keeps it readable.

```yaml
quantization:
  n_coord_bits: 16
  norm_range: [-2.5, 2.5]   # explicit; or null to auto-compute
  norm_margin_frac: 0.05    # fractional padding per side; used only when norm_range is null

memory:
  n_addrs: 16384            # addr_width derived as int(log2(n_addrs))

point_counts:
  n_P: 3824
  n_Q: 14000

icp:
  n_icp_iters: 10
  n_jacobi_sweeps: 8

run:
  n_pairs: 30               # null = all pairs

paths:
  validation: ../KTH_dataset_2/Validation_Data
  trace: ./ICP_Trace
```

This same flat shape is exactly what `config_used.yaml` dumps for each run, so a resolved configuration is itself a valid input to the `run` command. "Re-run exactly this point" is literally `run config_used.yaml`.

### 3.2 Sweep file — reference plus axes

A sweep is a *separate* file that references a config rather than duplicating it. This preserves modularity while eliminating the drift hazard that copying a base block would create (edit one copy, forget the other, and single runs silently diverge from sweeps).

```yaml
extends: config.yaml
sweep:
  quantization.n_coord_bits: [8, 10, 12, 16]
  icp.n_icp_iters: [5, 10, 20]
```

A sweep file is **strict**: it may contain *only* `extends:` and `sweep:`. Any other key is an error. This keeps the file unambiguously "a reference plus the axes to vary" and avoids any question of precedence between the sweep file and the config it extends.

Design rules:

- `extends:` resolves **relative to the sweep file's own location**, not the current working directory, so a sweep file is portable regardless of where it is invoked from.
- A sweep file with a `sweep:` block but no `extends:` is an error — there are no parameters to vary.

### 3.3 Derived and special values

- `addr_width` is **derived** as `int(log2(n_addrs))` and never appears in the config.
- `n_pairs: null` means "all pairs."
- `norm_range: null` triggers auto-computation (Section 5).

---

## 4. Command dispatch and loading

![Config loading and command dispatch](figures/02_config_resolution.svg)

There are two commands, `run` and `sweep`, with distinct expectations about the file handed to them. Validation happens **before any I/O** — structural problems are reported immediately rather than after a potentially expensive dataset scan.

### 4.1 `run`

`run` executes a single configuration. It accepts either a plain config or a sweep file:

- Given a **plain config**, it uses the top-level parameters directly.
- Given a **sweep file**, it resolves `extends:` to obtain the parameters, runs that single configuration, and **warns** that the `sweep:` section was ignored (use `sweep` for the full grid).

### 4.2 `sweep`

`sweep` requires a sweep file (`extends:` + a non-empty `sweep:`). It validates that every axis is permitted, enumerates the grid, and runs each point.

### 4.3 Sweep point selection

Selecting a single point from a sweep is done by **index**, which is terse and composes cleanly with external job schedulers (e.g. a cluster array job passing its task ID straight to `--index`):

- `--index N` — run grid point *N*.
- `--list` — print the enumerated grid with indices and exit, without running anything.

An earlier `--point key=value` selector was removed: once selection is restricted to points already on the grid, it duplicates `--index` with more typing and more room for error. `--index` plus `--list` (so the index is never a guessing game) covers the need with fewer moving parts.

---

## 5. Quantization range (`norm_range`)

![norm_range resolution](figures/03_norm_range.svg)

`norm_range` defines the coordinate interval mapped onto the quantization grid. It is either given explicitly or computed automatically.

### 5.1 Explicit

If `norm_range: [lo, hi]` is given, it is used verbatim and `norm_margin_frac` is ignored.

### 5.2 Automatic (global, over P ∪ Q)

If `norm_range: null`, the range is computed from the data:

- Over the **union of source (P) and target (Q)** coordinates. Covering both clouds guarantees neither clips. This is a deliberate change from the original implementation, which derived the range from targets only — the union is more correct.
- **Globally** across all pairs that will actually be processed (honoring `n_pairs`), producing **one** range for the whole run. A global range keeps the quantization grid constant, which is what makes comparisons across a sweep meaningful. It is computed over exactly the pairs processed — not all pairs on disk — so the range never depends on data the run did not use.
- Padded on each side by a **fractional** margin, `norm_margin_frac × (max − min)`. A fractional (relative) margin is scale-independent: 5% means the same thing regardless of the physical scale of a given cloud, unlike an absolute margin.

The resolved range is recorded in each `config_used.yaml`, so an auto run is as reproducible as an explicit one.

### 5.3 Reuse across a sweep

The auto-range depends only on the coordinate data and which clouds are read (`n_pairs`, `validation`) — *not* on bit width or iteration count. It is therefore computed **once per run and reused across every sweep point**. This reuse is valid precisely because the sweep allowlist (Section 6) excludes range-affecting parameters; the two decisions reinforce each other.

### 5.4 A note on asymmetry and resolution

The quantizer is symmetric in *code* space (it maps `norm_range` onto `[-1, 1]` and then onto the signed integer code range), but `norm_range` itself is generally **asymmetric** because real coordinate data is not centered on zero. This is the correct choice for this application:

- Fitting `[lo, hi]` tightly to the data **maximizes resolution** (minimizes the quantization step). It does not provide "more dynamic range" — every scheme uses all `2ⁿ` codes — it spends those codes more finely. Symmetrizing the range would widen it to cover a region the data never occupies on the shorter side, coarsening the step for no benefit.
- The cost of asymmetry is that coordinate `0.0` does not map to integer code `0`. For ICP this is harmless: the algorithm operates on *relative* geometry (centroids, centered coordinates, covariances) and is invariant to a global translation of the coordinate system. The centering step subtracts the centroid, so any constant encoding offset cancels before it reaches the cross-covariance, eigendecomposition, or rotation. No part of the pipeline depends on zero-preservation.
- The global-range choice (Section 5.2) trades a little resolution — the union span is at least as wide as any single pair's span, so the step is slightly coarser — for grid consistency across the sweep. This is an accepted, understood tradeoff, not a correctness issue. The fractional margin keeps every individual cloud strictly inside the range, well clear of the clip boundaries.

---

## 6. Sweepable parameters

Only certain parameters may appear as sweep axes, enforced by a flat allowlist:

```python
SWEEPABLE_AXES = {
    "quantization.n_coord_bits",
    "icp.n_icp_iters",
}
```

The `sweep` command fails fast if any axis is not in this set, naming the offending axis and listing what is permitted. Expanding the set later is a one-line edit.

The restriction serves two purposes. First, **safety**: `run.n_pairs` and `paths.validation` change which clouds are read and would break the "compute the auto-range once and reuse" assumption, so they must never be swept. Second, **domain sensibility**: changing the dataset is a different experiment, better expressed as a new config than as a grid axis. Parameters such as `n_P`, `n_Q`, `n_addrs`, and `n_jacobi_sweeps` are safe to add later (they do not affect the auto-range), but the starting set is kept minimal.

---

## 7. Sweep grid enumeration

![Sweep grid — Cartesian product](figures/04_sweep_grid.svg)

The grid is the full **Cartesian product** of the axes, produced by `itertools.product`. For axes `n_coord_bits = [8, 12, 16]` and `n_icp_iters = [5, 10, 20]`, this is 3 × 3 = 9 points — every value of one axis paired with every value of the other — not an element-wise pairing.

Properties relevant to implementation:

- The **last-listed axis varies fastest** (it is the inner loop of the product). Index 0 is the first combination; the index advances with the inner axis cycling first.
- Axes are passed to `product` in the order written in the `sweep:` section, and `--list` displays the grid in that same order, so `--index N` and the displayed grid always agree.
- A single-axis sweep is the degenerate case `product(*[values])`, yielding one-tuples — the same code path handles one axis or several with no special-casing.

The enumeration index maps directly to the output directory: index `N` → `run_NNN`.

---

## 8. Execution flow and outputs

  <p align="center">
    <img src="figures/05_execution_flow.svg" width="350">
  </p>

A sweep runs in two passes. Range discovery (pass 1) runs only when `norm_range` is `null`, scanning P ∪ Q once over the pairs that will be processed. Pass 2 is the per-point loop: for each grid point, the configuration is resolved, ICP is run, and traces are written.

### 8.1 Output layout

  - Each grid point writes to a numbered subdirectory **`run_NNN`**, zero-padded to the width of the largest index (so lexical and numeric ordering agree in a file browser). Numbered directories were chosen over key-value names (e.g. `n_coord_bits=8__n_P=1000`) because the latter become unwieldy and filesystem-unfriendly as grids grow; the numbering also aligns with `--index`.
  - Each `run_NNN` contains a **`config_used.yaml`** — the fully resolved, flat configuration including the resolved `norm_range` and the specific axis values for that point. This is the per-run source of truth and is itself a valid `run` input. It records the index and axis values so the historical output remains self-describing even if the sweep file is later edited.
  - The sweep root contains a **`sweep_manifest.yaml`** mapping every `run_NNN` to its axis values, giving the whole grid at a glance — the natural artifact for the analysis notebook to read when comparing quantization levels.

---

## 9. Core library change

  The one change to the core library is to thread the Jacobi sweep count through, replacing the currently hardcoded value:

  ```
  run(..., n_jacobi_sweeps)
    → _estimate_transform(..., n_jacobi_sweeps)
      → _jacobi_eigen_4x4(N, n_sweeps=n_jacobi_sweeps)
  ```

  This makes `icp.n_jacobi_sweeps` from the config effective rather than a placeholder. The previous `n_power_iters` parameter is vestigial after the switch from power iteration to the 4×4 Jacobi eigendecomposition and is removed.

---

## 10. Summary of decisions

| Area | Decision |
|------|----------|
| Roles | Exploratory plots in a notebook; execution in a CLI harness |
| Config shape | Parameters at top level, no wrapper; grouped sections |
| Sweep file | Separate file, `extends:` a config; strict (only `extends:` + `sweep:`) |
| `extends:` resolution | Relative to the sweep file's location |
| `norm_range` | Explicit, or `null` → auto over **P ∪ Q**, **global**, fractional margin |
| Range reuse | Computed once per run, reused across sweep points |
| Range asymmetry | Kept — maximizes resolution; harmless for translation-invariant ICP |
| `addr_width` | Derived as `int(log2(n_addrs))`; not in config |
| `n_pairs: null` | Means all pairs |
| Sweepable axes | Flat allowlist: `n_coord_bits`, `n_icp_iters` |
| Grid | Full Cartesian product via `itertools.product`; last axis fastest |
| `run` command | Plain config, or sweep file (resolve `extends:`, warn on ignored `sweep:`) |
| Sweep selection | `--index N` and `--list` only (`--point` removed) |
| Output | `run_NNN/` + `config_used.yaml` each; `sweep_manifest.yaml` at root |
| Core change | Thread `n_jacobi_sweeps` through; remove vestigial `n_power_iters` |

---

## 11. Implementation readiness
 
All design decisions in this document are settled. Implementation will produce:
 
- `run_icp.py` — the `run` and `sweep` commands, `--index`/`--list` selection, allowlist validation, two-pass execution with global auto-range, and output writing (`run_NNN/`, `config_used.yaml`, `sweep_manifest.yaml`).
- Example `config.yaml` and `sweep.yaml`.
- The `n_jacobi_sweeps` threading through `icp.py` (Section 9), and removal of the vestigial `n_power_iters`.
The `SWEEPABLE_AXES` starting set is `{quantization.n_coord_bits, icp.n_icp_iters}`, easily expanded later to any parameter that does not affect the auto-range.