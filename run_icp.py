"""ICP execution harness.

A single command that auto-detects the file it is given. A plain config (top-
level parameters, no wrapper) is run directly. A sweep file — containing only
`extends:` (a path to a config, resolved relative to the sweep file) and
`sweep:` (dotted-path axes) — requires one of --all, --index, or --list to say
what to do with the grid.

Usage:
  python run_icp.py config.yaml            # run a plain config
  python run_icp.py sweep.yaml --all       # run the whole grid
  python run_icp.py sweep.yaml --index 3   # run one grid point
  python run_icp.py sweep.yaml --list      # print the grid and exit
"""

from __future__ import annotations

import copy
import itertools
import math
from pathlib import Path
from typing import Any, Optional

import typer
import yaml

import pypcd4
import numpy as np
from natsort import natsorted
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from pointcloud import PointCloud
from targetcloud import TargetCloud
from icp import ICP, MemTrace


# ── Policy ────────────────────────────────────────────────────────────────────

# Axes permitted in a sweep. Safe to expand later with any parameter that does
# NOT change which point clouds are read (and therefore does not affect the
# global auto-range) — e.g. point_counts.n_P, point_counts.n_Q, memory.n_addrs,
# icp.n_jacobi_sweeps. Never add run.n_pairs or paths.validation: those change
# which clouds are read and would break the "compute auto-range once, reuse"
# assumption.
SWEEPABLE_AXES = {
    "quantization.n_coord_bits",
    "icp.n_icp_iters",
}


# ── Small helpers for dotted-path access on nested dicts ──────────────────────

def get_dotted(cfg: dict, path: str) -> Any:
    """Return cfg["a"]["b"] for path "a.b". Raises KeyError if absent."""
    node = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def set_dotted(cfg: dict, path: str, value: Any) -> None:
    """Set cfg["a"]["b"] = value for path "a.b". Intermediate keys must exist."""
    node = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def path_exists(cfg: dict, path: str) -> bool:
    try:
        get_dotted(cfg, path)
        return True
    except KeyError:
        return False


# ── Loading & validation ──────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise typer.BadParameter(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise typer.BadParameter(f"Config file is not a YAML mapping: {path}")
    return data


def is_sweep_file(data: dict) -> bool:
    """A sweep file is identified by the presence of `extends:` and/or `sweep:`."""
    return "extends" in data or "sweep" in data


def load_sweep_file(path: Path) -> tuple[dict, dict]:
    """Load a sweep file, returning (base_config, sweep_axes).

    Enforces the strict rule: a sweep file may contain ONLY `extends:` and
    `sweep:`. `extends:` is resolved relative to the sweep file's own location.
    """
    data = _load_yaml(path)

    extra = set(data) - {"extends", "sweep"}
    if extra:
        raise typer.BadParameter(
            f"Sweep file {path} may contain only 'extends:' and 'sweep:'. "
            f"Unexpected key(s): {', '.join(sorted(extra))}. "
            f"To override base parameters, edit the referenced config instead."
        )
    if "extends" not in data:
        raise typer.BadParameter(
            f"Sweep file {path} must reference a config via 'extends:'."
        )

    base_path = (path.parent / data["extends"]).resolve()
    base = _load_yaml(base_path)
    if is_sweep_file(base):
        raise typer.BadParameter(
            f"'extends:' must point to a plain config, but {base_path} looks "
            f"like a sweep file (contains 'extends:'/'sweep:')."
        )

    sweep = data.get("sweep") or {}
    return base, sweep


def validate_sweep_axes(sweep: dict, base: dict) -> None:
    """Fail fast if the sweep is empty, references unknown keys, or uses
    a non-allowlisted axis."""
    if not sweep:
        raise typer.BadParameter(
            "Sweep section is empty; add at least one axis, e.g. "
            "`quantization.n_coord_bits: [8, 12, 16]`."
        )
    for axis, values in sweep.items():
        if axis not in SWEEPABLE_AXES:
            raise typer.BadParameter(
                f"Sweep axis '{axis}' is not permitted. "
                f"Allowed axes: {', '.join(sorted(SWEEPABLE_AXES))}."
            )
        if not path_exists(base, axis):
            raise typer.BadParameter(
                f"Sweep axis '{axis}' does not match any key in the base config."
            )
        if not isinstance(values, list) or len(values) == 0:
            raise typer.BadParameter(
                f"Sweep axis '{axis}' must be a non-empty list of values."
            )


# ── Grid enumeration ──────────────────────────────────────────────────────────

def enumerate_grid(sweep: dict) -> tuple[list[str], list[tuple]]:
    """Return (axis_names, points) where points is the Cartesian product.

    Axes are taken in the order written in the sweep file; the last axis varies
    fastest (itertools.product semantics). A single-axis sweep yields 1-tuples.
    """
    axes = list(sweep.keys())
    value_lists = [sweep[a] for a in axes]
    points = list(itertools.product(*value_lists))
    return axes, points


def resolve_point(base: dict, axes: list[str], point: tuple) -> dict:
    """Produce a fully resolved config for one grid point."""
    cfg = copy.deepcopy(base)
    for axis, value in zip(axes, point):
        set_dotted(cfg, axis, value)
    return cfg


# ── Memory-trace config ───────────────────────────────────────────────────────

def resolve_mem_iters(cfg: dict) -> frozenset[int]:
    """Resolve and validate trace.mem_iters against the run mode.

    Memory tracing is fixed-iteration only. Returns the set of iteration indices
    to trace, or an empty set when tracing is disabled. Rules:
      - missing/empty trace.mem_iters -> no tracing.
      - convergence mode (icp.n_icp_iters is None) with a non-empty mem_iters ->
        warn and disable (the iteration count is unknown; the hardware target
        runs a fixed count anyway).
      - fixed-iteration mode -> every index must be in [0, n_icp_iters); fail
        fast otherwise rather than silently producing no iter_NN/ directory.
    """
    trace = cfg.get("trace") or {}
    raw = trace.get("mem_iters") or []
    if not raw:
        return frozenset()

    n_icp_iters = cfg["icp"]["n_icp_iters"]
    if n_icp_iters is None:
        typer.secho(
            "Warning: trace.mem_iters is set but icp.n_icp_iters is null "
            "(convergence mode); memory images will not be written.",
            fg=typer.colors.YELLOW,
        )
        return frozenset()

    iters = sorted({int(v) for v in raw})
    out_of_range = [v for v in iters if not (0 <= v < n_icp_iters)]
    if out_of_range:
        raise typer.BadParameter(
            f"trace.mem_iters contains out-of-range value(s) {out_of_range}; "
            f"valid range is [0, {n_icp_iters}) for icp.n_icp_iters="
            f"{n_icp_iters}."
        )
    return frozenset(iters)


def _count_dataset_pairs(validation: Path) -> int:
    """Total (source frame, target) pairs in the dataset.

    The sum of each frame's target count. Metadata-only (no point cloud loads)
    and order-independent: we need the cardinality for a range check, not the
    traversal order, so plain glob/iterdir suffices.
    """
    total = 0
    for scan_dir in validation.glob("source*"):
        for frame_dir in scan_dir.iterdir():
            meta = yaml.safe_load((frame_dir / "metadata.yaml").read_text())
            total += len(meta["targets"])
    return total


def resolve_mem_pairs(cfg: dict) -> frozenset[int]:
    """Resolve and validate trace.mem_pairs against the available pair count.

    Selects which processed pairs are eligible for ICP-output logging. This is
    independent of run.n_pairs: n_pairs still controls how many pairs are
    processed (and, in auto-range mode, how many define norm_range); mem_pairs
    only narrows which of those processed pairs get traced. Returns the set of
    requested pair ids, or an empty set meaning "all processed pairs eligible".

    Rules:
      - missing/empty trace.mem_pairs -> empty set ("all eligible"). Whether any
        tracing happens at all is governed by trace.mem_iters (the master
        switch); mem_pairs only ever narrows.
      - the range-check cap is run.n_pairs when set, otherwise the dataset's
        total pair count (a cheap metadata-only count). Every requested id must
        be in [0, cap); fail fast otherwise. The end-of-run warning remains a
        backstop for on-disk drift between counting and processing.
    """
    trace = cfg.get("trace") or {}
    raw = trace.get("mem_pairs") or []
    if not raw:
        return frozenset()

    pairs = sorted({int(v) for v in raw})
    n_pairs = cfg["run"]["n_pairs"]
    cap = (
        n_pairs if n_pairs is not None
        else _count_dataset_pairs(Path(cfg["paths"]["validation"]))
    )

    out_of_range = [v for v in pairs if not (0 <= v < cap)]
    if out_of_range:
        raise typer.BadParameter(
            f"trace.mem_pairs contains out-of-range value(s) {out_of_range}; "
            f"valid range is [0, {cap})."
        )
    return frozenset(pairs)



# ── norm_range ────────────────────────────────────────────────────────────────

def needs_auto_range(cfg: dict) -> bool:
    return cfg["quantization"].get("norm_range") is None


def compute_global_range(cfg: dict) -> tuple[float, float]:
    """Compute the global min/max over P ∪ Q across the pairs that will be
    processed (honoring n_pairs), padded by a fractional margin.

    This is a pure-I/O pass: it reads coordinates only, mirroring the traversal
    that `_process` performs so the range covers exactly the processed pairs.
    """

    validation = Path(cfg["paths"]["validation"])
    n_pairs = cfg["run"]["n_pairs"]
    margin_frac = cfg["quantization"]["norm_margin_frac"]

    lo = math.inf
    hi = -math.inf
    pair_count = 0

    def _coord_extent(pcd_path: Path) -> tuple[float, float]:
        pc = pypcd4.PointCloud.from_path(pcd_path)
        xyz = pc.numpy(("x", "y", "z"))
        return float(np.min(xyz)), float(np.max(xyz))

    for scan_dir in natsorted(validation.glob("source*")):
        for frame_dir in natsorted(scan_dir.iterdir()):
            source_path = next(frame_dir.glob("*.pcd"))
            metadata = yaml.safe_load((frame_dir / "metadata.yaml").read_text())

            for target_meta in metadata["targets"]:
                if n_pairs is not None and pair_count >= n_pairs:
                    _finalize_check(lo, hi)
                    return _pad(lo, hi, margin_frac)

                target_path = validation / target_meta["path"]

                s_lo, s_hi = _coord_extent(source_path)
                t_lo, t_hi = _coord_extent(target_path)
                lo = min(lo, s_lo, t_lo)
                hi = max(hi, s_hi, t_hi)
                pair_count += 1

    _finalize_check(lo, hi)
    return _pad(lo, hi, margin_frac)


def _finalize_check(lo: float, hi: float) -> None:
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise typer.BadParameter(
            "Could not compute an auto norm_range: no point cloud pairs were "
            "found to process. Check paths.validation and run.n_pairs."
        )


def _pad(lo: float, hi: float, margin_frac: float) -> tuple[float, float]:
    span = hi - lo
    pad = margin_frac * span
    return (lo - pad, hi + pad)


# ── Per-frame results metadata (ICP output) ──────────────────────────────────

# Width budget for the results metadata.yaml, matching validation_builder.
RESULTS_YAML_WIDTH = 100


def _homogeneous(R: np.ndarray, t: np.ndarray) -> list[list[float]]:
    """Assemble a 4x4 homogeneous transform from a (3,3) R and (3,1)/(3,) t.

    Returns plain Python floats (no numpy scalars) so the YAML is clean.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return [[float(v) for v in row] for row in T]


def _flow_matrix(T: list[list[float]]) -> CommentedSeq:
    """A 4x4 matrix as a block sequence of flow-style rows (one row per line),
    matching the transformation_matrix rendering in validation_builder."""
    rows = CommentedSeq(
        (lambda r: r.fa.set_flow_style() or r)(CommentedSeq(row)) for row in T
    )
    return rows


def _matrix_visual(T: list[list[float]], label: str) -> str:
    """Human-readable visual block for a 4x4 transform, used as a YAML comment."""
    out = f"{label}:\n"
    for row in T:
        out += f"[ {', '.join(f'{v:10.4f}' for v in row)} ]\n"
    return out


def _target_result_block(
    target_meta: dict,
    pair_id: int,
    source_csv_rel: Path,
    target_csv_rel: Path,
    total_transform: dict,
    p2p: dict,
    p2pl: dict,
) -> CommentedMap:
    """One target's results entry for a frame's results metadata.yaml.

    Carries both path kinds (dataset-relative `path` for joining to the
    validation dataset; run-relative `*_csv` for the float+quant CSV that the
    metrics were computed from), the pair id (join key into pairs_manifest), the
    4x4 float and quantized transforms, and the float/quant error metrics.
    """
    T_float = _homogeneous(*total_transform["float"])
    T_quant = _homogeneous(*total_transform["quantized"])

    transform = CommentedMap({
        "float": _flow_matrix(T_float),
        "quantized": _flow_matrix(T_quant),
    })

    metrics = CommentedMap({
        "p2p": CommentedMap({
            "float": float(p2p["float"]),
            "quantized": float(p2p["quantized"]),
        }),
        "p2pl": CommentedMap({
            "float": float(p2pl["float"]),
            "quantized": float(p2pl["quantized"]),
        }),
    })

    entry = CommentedMap({
        "path": str(target_meta["path"]),       # dataset-relative; joins to validation
        "target_csv": str(target_csv_rel),      # run-relative; carries float + quant
        "pair_id": pair_id,                      # join key into pairs_manifest
        "transform": transform,
        "metrics": metrics,
    })

    # Visual representation of the float transform, as a comment before
    # `transform`. The quantized matrix is fixed-point (Q1.(R_width-1)) and not
    # human-readable as raw integers, so only the float form is shown.
    visual = _matrix_visual(T_float, "Float transform")
    entry.yaml_set_comment_before_after_key("transform", before=visual, indent=2)

    return entry


def _write_frame_metadata(
    frame_dir: Path,
    source_prov: Path,
    source_csv_rel: Path,
    target_results: list[CommentedMap],
) -> None:
    """Write a frame's ICP-results metadata.yaml next to its written source CSV.

    Lives at the same per-frame location as the validation dataset's input
    metadata.yaml (source_*/XX/), mirroring its shape: one `source` plus a
    `targets:` sequence. The list reflects the targets actually PROCESSED this
    run, which may be fewer than the validation dataset lists if run.n_pairs
    capped the run mid-frame.
    """
    yaml_rt = YAML()
    yaml_rt.default_flow_style = False
    yaml_rt.width = RESULTS_YAML_WIDTH

    targets_seq = CommentedSeq(target_results)
    for i in range(len(targets_seq)):
        label = f" TARGET {i + 1} ".center(RESULTS_YAML_WIDTH - 4, "#")
        targets_seq.yaml_set_comment_before_after_key(i, before=f"\n{label}", indent=2)

    meta = CommentedMap()
    meta["source"] = str(source_prov)            # dataset-relative provenance
    meta["source_csv"] = str(source_csv_rel)     # run-relative; carries float + quant
    meta["targets"] = targets_seq

    meta.yaml_set_comment_before_after_key(
        "source_csv",
        before="Run-relative source CSV (float + quantized coordinates)",
    )
    meta.yaml_set_comment_before_after_key(
        "targets",
        before="\nICP results for the targets processed this run "
               "(may be fewer than the validation dataset lists if n_pairs "
               "capped the run mid-frame)",
    )

    frame_dir.mkdir(parents=True, exist_ok=True)
    with open(frame_dir / "metadata.yaml", "w") as fh:
        yaml_rt.dump(meta, fh)


# ── Core processing for a single resolved config ──────────────────────────────

def _process(cfg: dict, trace_path: Path, norm_range: tuple[float, float]) -> None:
    """Run the ICP loop for one fully resolved config.

    This is the standalone replacement for the notebook's `process()`. It takes
    an already-resolved norm_range so a sweep can reuse one computed range.
    """

    n_coord_bits = cfg["quantization"]["n_coord_bits"]
    n_P = cfg["point_counts"]["n_P"]
    n_Q = cfg["point_counts"]["n_Q"]
    n_pairs = cfg["run"]["n_pairs"]
    n_icp_iters = cfg["icp"]["n_icp_iters"]
    n_jacobi_sweeps = cfg["icp"]["n_jacobi_sweeps"]
    R_width = cfg["icp"]["R_width"]
    addr_width = int(math.log2(cfg["memory"]["n_addrs"]))
    noise_coeffs = cfg["error"]["p2pl_noise_coeffs"]
    validation = Path(cfg["paths"]["validation"])

    # Iterations whose per-unit memory images are written (fixed-iter only;
    # already validated against n_icp_iters).
    mem_iters = resolve_mem_iters(cfg)

    # Pairs eligible for logging. Empty set = all processed pairs eligible;
    # mem_iters is the master on/off switch, mem_pairs only narrows. Composes
    # with mem_iters as an AND: a pair is traced iff it passes this gate AND
    # mem_iters is non-empty.
    mem_pairs = resolve_mem_pairs(cfg)

    # Track which requested pair ids were actually processed, so we can warn
    # about any that never occurred (the case the up-front range check can't
    # catch: n_pairs is null, or fewer pairs existed than n_pairs allowed).
    requested_pairs_seen: set[int] = set()

    # Per-pair manifest, always written. Records dataset-relative provenance
    # (the stable identity of each pair) plus run-relative paths to the written
    # csv trace files. A natural home for future per-pair data (e.g. the
    # transform matrix or error metrics).
    pairs_manifest: dict[str, dict] = {}

    pair_count = 0

    done = False  # set when the n_pairs cap is reached; breaks all loops

    for scan_idx, scan_dir in enumerate(natsorted(validation.glob("source*"))):
        if done:
            break
        for frame_idx, frame_dir in enumerate(natsorted(scan_dir.iterdir())):
            if n_pairs is not None and pair_count >= n_pairs:
                done = True
                break

            source_path = next(frame_dir.glob("*filtered_full*.pcd"))
            print(f"SOURCE (scan {scan_idx}, frame {frame_idx}): {source_path}")
            source = PointCloud(source_path, norm_range, n_coord_bits)
            source.downsample(n_P)

            trace_source_subpath = Path(
                trace_path,
                *source_path.parts[
                    next(i for i, p in enumerate(source_path.parts)
                         if p.startswith("source")):
                ],
            ).with_suffix("")
            source.write_csv(trace_source_subpath)
            source.write_mem(trace_source_subpath)

            # Written source csv filename (matches PointCloud.write_csv), made
            # relative to the run trace root for the manifest.
            source_csv = (
                trace_source_subpath.parent
                / f"{trace_source_subpath.stem}-{source.n_points}_{source.n_bits}.csv"
            )
            source_csv_rel = source_csv.relative_to(trace_path)

            # Dataset-relative source provenance (stable pair identity).
            source_prov = Path(
                *source_path.parts[
                    next(i for i, p in enumerate(source_path.parts)
                         if p.startswith("source")):
                ]
            )

            metadata = yaml.safe_load((frame_dir / "metadata.yaml").read_text())

            # Per-frame accumulator for ICP results, flushed to a results
            # metadata.yaml after this frame's target loop.
            frame_target_results: list[CommentedMap] = []

            for target_idx, target_meta in enumerate(metadata["targets"]):
                if n_pairs is not None and pair_count >= n_pairs:
                    done = True
                    break
                
                target_path = validation / target_meta["path"]
                print(f"  TARGET {target_idx}: {target_path}")

                pair_id = pair_count  # contiguous 0..n-1; stable within this run

                target = TargetCloud(target_path, norm_range, n_coord_bits, n_Q)

                # Trace path for this target, with an extra per-id directory
                # level so each target's csv/mem/tree files live in their own
                # folder (targets/<bolt>/<id>/<id>...) rather than scattered
                # among sibling targets in the bolt directory.
                target_rel = Path(
                    *target_path.parts[target_path.parts.index("targets"):]
                )  # targets/<bolt>/<id>.pcd
                tree_subpath = (
                    trace_path / target_rel.parent / target_rel.stem / target_rel.stem
                )  # targets/<bolt>/<id>/<id>
                target.write(tree_subpath, addr_width)

                # Written target csv filename (matches TargetCloud/PointCloud
                # write_csv), made relative to the run trace root.
                target_csv = (
                    tree_subpath.parent
                    / f"{tree_subpath.stem}-{target.point_cloud.n_points}_{target.point_cloud.n_bits}.csv"
                )
                target_csv_rel = target_csv.relative_to(trace_path)

                # Whether this pair is traced: the master switch (mem_iters
                # non-empty) AND the pair gate (mem_pairs empty = all eligible,
                # else pair_id must be listed).
                pair_eligible = (not mem_pairs) or (pair_id in mem_pairs)
                if mem_pairs and pair_id in mem_pairs:
                    requested_pairs_seen.add(pair_id)
                traced = bool(mem_iters) and pair_eligible

                # Per-pair memory trace under pair_NNN/ at the run trace root,
                # a sibling of source.../ and targets/. Disabled (None) unless
                # this pair is traced.
                pair_dir = trace_path / f"pair_{pair_id:03d}"
                mem_trace = (
                    MemTrace(iters=mem_iters, out_dir=pair_dir) if traced else None
                )

                icp = ICP(source, target, R_width=R_width)
                total_transform = icp.run(
                    n_icp_iters,
                    n_jacobi_sweeps=n_jacobi_sweeps,
                    mem_trace=mem_trace,
                )

                # print(icp.aligned_source.points_f)
                # print(icp.aligned_source.dequantize())

                # Capture each metric dict once (each re-runs NN search).
                p2p = icp.p2p_error()
                p2pl = icp.p2pl_error(noise_coeffs=noise_coeffs)
                print(p2p)
                print(p2pl)

                # Accumulate this target's results entry for the frame's
                # metadata.yaml (flushed after the target loop).
                frame_target_results.append(
                    _target_result_block(
                        target_meta,
                        pair_id,
                        source_csv_rel,
                        target_csv_rel,
                        total_transform,
                        p2p,
                        p2pl,
                    )
                )

                # Record the pair: dataset-relative provenance + run-relative
                # written csv paths + whether ICP output was traced.
                pairs_manifest[f"pair_{pair_id:03d}"] = {
                    "source": str(source_prov),
                    "target": str(target_meta["path"]),
                    "source_csv": str(source_csv_rel),
                    "target_csv": str(target_csv_rel),
                    "traced": traced,
                }

                pair_count += 1

            # Flush this frame's results metadata.yaml next to the written
            # source CSV (source_*/XX/), mirroring the validation dataset's
            # per-frame metadata location and shape. Runs on both normal target-
            # loop exhaustion and the n_pairs `break`, so a capped run still
            # writes complete metadata for every frame it finished targets in.
            if frame_target_results:
                _write_frame_metadata(
                    trace_source_subpath.parent,
                    source_prov,
                    source_csv_rel,
                    frame_target_results,
                )

            print()

    # Warn about requested pair ids that never occurred during traversal. This
    # is the safety net the up-front range check can't provide: it catches ids
    # too high for the pairs that actually existed (fewer than n_pairs, or
    # n_pairs null so no check was possible).
    if mem_pairs and mem_iters:
        missing = sorted(mem_pairs - requested_pairs_seen)
        if missing:
            typer.secho(
                f"Warning: trace.mem_pairs requested pair id(s) {missing} that "
                f"were never processed ({pair_count} pair(s) total); no trace "
                f"was written for them.",
                fg=typer.colors.YELLOW,
            )

    # Dump the per-pair manifest at the run trace root (always written).
    trace_path.mkdir(parents=True, exist_ok=True)
    (trace_path / "pairs_manifest.yaml").write_text(
        yaml.safe_dump({"pairs": pairs_manifest}, sort_keys=False)
    )


def _run_resolved(cfg: dict, trace_path: Path,
                  shared_range: Optional[tuple[float, float]] = None) -> dict:
    """Resolve norm_range (using shared_range if provided), run, and dump
    config_used.yaml. Returns the resolved config (with norm_range filled in)."""
    resolved = copy.deepcopy(cfg)

    if needs_auto_range(resolved):
        rng = shared_range if shared_range is not None else compute_global_range(resolved)
        resolved["quantization"]["norm_range"] = [float(rng[0]), float(rng[1])]

    norm_range = tuple(resolved["quantization"]["norm_range"])

    trace_path.mkdir(parents=True, exist_ok=True)
    (trace_path / "config_used.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False)
    )

    _echo_config(resolved, trace_path, norm_range)
    _process(resolved, trace_path, norm_range)
    return resolved


def _echo_config(cfg: dict, trace_path: Path, norm_range) -> None:
    print(f"Validation path : {cfg['paths']['validation']}")
    print(f"Trace path      : {trace_path}")
    print(f"Norm range      : {tuple(norm_range)}")
    print(f"Coord bits      : {cfg['quantization']['n_coord_bits']}")
    print(f"Tree addr width : {int(math.log2(cfg['memory']['n_addrs']))}")
    print(f"P points        : {cfg['point_counts']['n_P']}")
    print(f"Q points        : {cfg['point_counts']['n_Q']}")
    print(f"Pairs           : {cfg['run']['n_pairs']}")
    print(f"ICP iterations  : {cfg['icp']['n_icp_iters']}")
    print(f"Jacobi sweeps   : {cfg['icp']['n_jacobi_sweeps']}")
    print(f"R width         : {cfg['icp']['R_width']}")
    print("\n----- PROCESSING -----\n")


# ── Sweep actions ─────────────────────────────────────────────────────────────

def _list_grid(base: dict, sweep: dict) -> None:
    """Print the sweep grid with indices and exit (no runs)."""
    validate_sweep_axes(sweep, base)
    axes, points = enumerate_grid(sweep)
    width = max(3, len(str(len(points) - 1)))
    typer.echo(f"Sweep grid: {len(points)} point(s) over axes {axes}")
    for i, pt in enumerate(points):
        mapping = ", ".join(f"{a}={v}" for a, v in zip(axes, pt))
        typer.echo(f"  run_{i:0{width}d}: {mapping}")


def _run_one_point(base: dict, sweep: dict, index: int) -> None:
    """Run a single grid point selected by index."""
    validate_sweep_axes(sweep, base)
    axes, points = enumerate_grid(sweep)
    if not (0 <= index < len(points)):
        raise typer.BadParameter(
            f"--index {index} out of range; sweep has {len(points)} "
            f"point(s) (0–{len(points) - 1})."
        )
    width = max(3, len(str(len(points) - 1)))
    cfg = resolve_point(base, axes, points[index])
    trace_path = Path(base["paths"]["trace"]) / f"run_{index:0{width}d}"
    _run_resolved(cfg, trace_path)


def _run_grid(base: dict, sweep: dict) -> None:
    """Execute the full Cartesian-product grid."""
    validate_sweep_axes(sweep, base)
    axes, points = enumerate_grid(sweep)
    width = max(3, len(str(len(points) - 1)))
    trace_root = Path(base["paths"]["trace"])

    # Compute the auto-range once and reuse it across all points (valid because
    # the allowlist excludes range-affecting parameters).
    shared_range: Optional[tuple[float, float]] = None
    if needs_auto_range(base):
        print("Computing global norm_range over P ∪ Q (once for the sweep)...\n")
        shared_range = compute_global_range(base)
        print(f"Resolved norm_range: {shared_range}\n")

    manifest = {
        "axes": axes,
        "n_points": len(points),
        "norm_range": list(shared_range) if shared_range is not None
                      else base["quantization"]["norm_range"],
        "runs": {},
    }

    for i, pt in enumerate(points):
        run_name = f"run_{i:0{width}d}"
        print(f"\n===== SWEEP {run_name}  "
              f"({', '.join(f'{a}={v}' for a, v in zip(axes, pt))}) =====\n")
        cfg = resolve_point(base, axes, pt)
        trace_path = trace_root / run_name
        _run_resolved(cfg, trace_path, shared_range=shared_range)
        manifest["runs"][run_name] = {a: v for a, v in zip(axes, pt)}

    trace_root.mkdir(parents=True, exist_ok=True)
    (trace_root / "sweep_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )
    print(f"\nWrote manifest: {trace_root / 'sweep_manifest.yaml'}")


# ── Command ───────────────────────────────────────────────────────────────────

def main(
    config: Path = typer.Argument(..., help="Path to a config or sweep file."),
    all_points: bool = typer.Option(
        False, "--all", help="Run the whole grid (sweep files only)."),
    index: Optional[int] = typer.Option(
        None, "--index", help="Run grid point N (sweep files only)."),
    list_grid: bool = typer.Option(
        False, "--list", help="Print the sweep grid with indices and exit "
                              "(sweep files only)."),
):
    """Execute an ICP configuration.

    The file kind is detected from its contents. A plain config is run directly.
    A sweep file requires exactly one of --all, --index, or --list to say what
    to do with the grid.
    """
    data = _load_yaml(config)
    sweep_flags = [bool(all_points), index is not None, bool(list_grid)]

    # ── Plain config ────────────────────────────────────────────────────────
    if not is_sweep_file(data):
        if any(sweep_flags):
            raise typer.BadParameter(
                "--all/--index/--list require a sweep file (with 'extends:' "
                "and 'sweep:'). This is a plain config; pass it with no flags."
            )
        # A plain config writes to base/ under the trace root. Sweeps write
        # run_NNN/ dirs (plus the manifest) under the same root, so the two
        # never collide and the root is always a container — base/ for the
        # one-off config, run_NNN/ for sweep points.
        _run_resolved(data, Path(data["paths"]["trace"]) / "base")
        return

    # ── Sweep file: require exactly one action ────────────────────────────────
    if sum(sweep_flags) == 0:
        raise typer.BadParameter(
            f"{config} is a sweep file; choose what to do with the grid: "
            "--all (run every point), --index N (run one), or --list (preview)."
        )
    if sum(sweep_flags) > 1:
        raise typer.BadParameter(
            "--all, --index, and --list are mutually exclusive; pass exactly one."
        )

    base, sweep = load_sweep_file(config)

    if list_grid:
        _list_grid(base, sweep)
    elif index is not None:
        _run_one_point(base, sweep, index)
    else:  # all_points
        _run_grid(base, sweep)


if __name__ == "__main__":
    typer.run(main)