"""Validation dataset builder.

Restructures the collaborators' raw dataset into a validation dataset for the
ICP harness (run_icp.py). For each source frame it:

  - copies the filtered full source point cloud (xyz + intensity preserved),
  - copies the keypoints cloud (the PCL VoxelGrid-downsampled source) when
    present — carried as its own cloud, since voxel centroids are not a subset
    of the full cloud,
  - copies every referenced target point cloud into <out>/targets/<bolt>/<id>.pcd
    (deduplicated across frames),
  - scores each collaborator-supplied transformation_matrix against our own
    error metrics, in float space, for BOTH the full and keypoints clouds, and
  - writes a metadata.yaml describing the source, keypoints, targets, and the
    best-target selection criterion.

The error metrics mirror ICP.p2p_error / ICP.p2pl_error (float space, no
quantization), all as mean-squared error (mean of squares, no square root). The
transform under test is the collaborators' ground-truth matrix, not the output
of our ICP. Under metrics.computed, the `full` and `keypoints` blocks each
record:

  transformation_error_p2p           point-to-point MSE, mean(sum((s-t)^2))
  transformation_error_p2pl          raw point-to-plane MSE, mean(|d|^2)
  transformation_error_p2pl_denoised denoised point-to-plane MSE,
                                     mean((|d| - noise)^2) over correspondences
                                     whose |d| meets the per-point noise floor
                                     (others dropped); null when the cloud has
                                     no intensity field.

The keypoints denoised value is the apples-to-apples comparison against
metrics.original.transformation_error: the collaborators computed their error on
the voxel-downsampled keypoints. The `keypoints` block is omitted for frames
with no keypoints cloud.

Usage:
  python validation_dataset_builder.py <dataset_path>
  python validation_dataset_builder.py <dataset_path> --out ../Validation_Data
  python validation_dataset_builder.py <dataset_path> --noise-coeffs 96.72 -0.762
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from pypcd4 import PointCloud as PCD4, Encoding
from rich.progress import track
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from kdtree import KDTree

app = typer.Typer(add_completion=False, help=__doc__)

YAML_WIDTH = 100


# ── Error metrics (float space, scoring a known transform) ────────────────────

def _apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an (N, 3) array of points."""
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points.T).T + t


def _nearest(target_xyz: np.ndarray, query_xyz: np.ndarray) -> np.ndarray:
    """Indices into target_xyz of the nearest neighbor for each query point.

    Uses the project KDTree so correspondences match the harness's notion of
    nearest neighbor exactly.
    """
    tree = KDTree(target_xyz)
    return tree.nn_search(query_xyz)


def _p2pl_abs_d(
    source_xyz: np.ndarray,
    target_nearest_xyz: np.ndarray,
    target_nearest_normals: np.ndarray,
) -> np.ndarray:
    """Absolute per-point point-to-plane distance |(s - t) · n|."""
    return np.abs(
        np.sum((source_xyz - target_nearest_xyz) * target_nearest_normals, axis=1)
    )


def p2p_error(source_xyz: np.ndarray, target_nearest_xyz: np.ndarray) -> float:
    """Point-to-point MSE: mean of squared Euclidean distances (no sqrt)."""
    return float(
        np.mean(np.sum((source_xyz - target_nearest_xyz) ** 2, axis=1))
    )


def p2pl_error(
    source_xyz: np.ndarray,
    target_nearest_xyz: np.ndarray,
    target_nearest_normals: np.ndarray,
) -> float:
    """Raw point-to-plane MSE: mean(|d|^2) over all correspondences (no sqrt)."""
    abs_d = _p2pl_abs_d(source_xyz, target_nearest_xyz, target_nearest_normals)
    return float(np.mean(abs_d ** 2))


def _expected_noise(
    intensity: np.ndarray, noise_coeffs: tuple[float, float]
) -> np.ndarray:
    """Per-point expected noise floor: A * amplitude**B * NOISE_FACTOR.

    NOISE_FACTOR mirrors the fixed constant in ICP.p2pl_error.
    """
    NOISE_FACTOR = 2
    A, B = noise_coeffs
    amplitude = np.asarray(intensity, dtype=np.float64).reshape(-1)
    return A * (amplitude ** B) * NOISE_FACTOR


def p2pl_denoised_error(
    source_xyz: np.ndarray,
    target_nearest_xyz: np.ndarray,
    target_nearest_normals: np.ndarray,
    intensity: np.ndarray,
    noise_coeffs: tuple[float, float],
) -> float:
    """Denoised point-to-plane MSE, subtract-first form: mean((|d| - noise)^2).

    The per-point expected noise floor is subtracted from each correspondence's
    point-to-plane distance; correspondences with |d| below the floor are
    dropped (not used in the mean), matching the collaborators' description.
    Returns NaN when no correspondence survives the floor.
    """
    abs_d = _p2pl_abs_d(source_xyz, target_nearest_xyz, target_nearest_normals)
    expected_noise = _expected_noise(intensity, noise_coeffs)

    keep = abs_d >= expected_noise
    survivors = abs_d[keep] - expected_noise[keep]
    if survivors.size == 0:
        return float("nan")
    return float(np.mean(survivors ** 2))


def score_transform(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    target_normals: np.ndarray,
    T: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    noise_coeffs: Optional[tuple[float, float]] = None,
) -> tuple[float, float, Optional[float]]:
    """Score a candidate transform T (4x4) that maps source onto target.

    Returns (p2p, p2pl, p2pl_denoised) MSE values in float space (mean of
    squares, no sqrt), where:
      p2p          point-to-point MSE
      p2pl         raw point-to-plane MSE over |d|
      p2pl_denoised denoised point-to-plane MSE, subtract-first
                   mean((|d| - noise)^2); None when intensity or noise_coeffs
                   is unavailable.
    """
    moved = _apply_transform(source_xyz, T)
    nn_idx = _nearest(target_xyz, moved)
    nearest_xyz = target_xyz[nn_idx]
    nearest_normals = target_normals[nn_idx]

    p2p = p2p_error(moved, nearest_xyz)
    p2pl = p2pl_error(moved, nearest_xyz, nearest_normals)

    if intensity is None or noise_coeffs is None:
        p2pl_denoised: Optional[float] = None
    else:
        p2pl_denoised = p2pl_denoised_error(
            moved, nearest_xyz, nearest_normals, intensity, noise_coeffs
        )

    return p2p, p2pl, p2pl_denoised


# ── Build ─────────────────────────────────────────────────────────────────────

def _copy_cloud(
    src_path: Path, frame_out: Path
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a point cloud, copy it into the frame output dir (preserving
    intensity when present), and return (xyz, intensity).

    intensity is None when the cloud has no intensity field.
    """
    pc = PCD4.from_path(src_path)
    dst = str(frame_out / src_path.name)
    if "intensity" in pc.fields:
        arr = pc.numpy(("x", "y", "z", "intensity"))
        PCD4.from_xyzi_points(arr).save(dst, encoding=Encoding.ASCII)
        return arr[:, :3], arr[:, 3]
    arr = pc.numpy(("x", "y", "z"))
    PCD4.from_xyz_points(arr).save(dst, encoding=Encoding.ASCII)
    return arr, None


def build(
    dataset_path: Path,
    out_path: Path,
    noise_coeffs: Optional[tuple[float, float]],
) -> None:
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = YAML_WIDTH

    scan_dirs = [d for d in dataset_path.glob("source_*") if d.is_dir()]
    if not scan_dirs:
        raise typer.BadParameter(
            f"No scan directories (source_*) found under {dataset_path}."
        )

    # Track targets already copied so we copy each referenced one once.
    copied_targets: set[Path] = set()

    for scan_dir in track(sorted(scan_dirs), description="Processing scans"):

        # Group files in the scan directory by their leading numeric prefix.
        frame_map: dict[str, list[Path]] = {}
        for f in scan_dir.glob("*_*"):
            frame_map.setdefault(f.name.split("_")[0], []).append(f)

        for prefix in track(
            sorted(frame_map.keys()),
            description=f"  {scan_dir.name} frames",
        ):
            files = frame_map[prefix]
            source_path = next(
                f for f in files if f.name.endswith("_filtered_full_pointcloud.pcd")
            )
            metadata_path = next(f for f in files if f.name.endswith("_metadata.json"))
            keypoints_path = next(
                (f for f in files if f.name.endswith("_keypoints_pointcloud.pcd")),
                None,
            )

            # Output directory for this frame.
            frame_out = out_path / scan_dir.name / prefix
            frame_out.mkdir(parents=True, exist_ok=True)

            # Copy the full source cloud, preserving intensity when present.
            full_xyz, full_intensity = _copy_cloud(source_path, frame_out)

            # Copy the keypoints cloud (voxel-downsampled source) when present.
            # Keypoints are PCL VoxelGrid centroids, NOT a subset of the full
            # cloud, so they are carried as their own cloud rather than a mask.
            keypoints_rel: Optional[str] = None
            kp_xyz = kp_intensity = None
            if keypoints_path is not None:
                kp_xyz, kp_intensity = _copy_cloud(keypoints_path, frame_out)
                keypoints_rel = str(
                    (frame_out / keypoints_path.name).relative_to(out_path)
                )

            with open(metadata_path, "r") as fh:
                metadata = json.load(fh)
            filtered_targets = metadata.get("filtered_targets") or []

            filtered_targets_validation = []

            for target in track(
                filtered_targets, description="    targets", transient=True
            ):
                bolt = str(target.get("bolt"))
                tid = str(target.get("id"))

                # Collaborators' ground-truth transform (note the transpose,
                # preserved from the original builder).
                T_matrix = np.array(
                    target.get("transformation_matrix"), dtype=float
                ).T
                T_error = float(np.array(target.get("transformation_error")))

                # Source target path in the raw dataset, and its destination in
                # the validation dataset.
                target_src = dataset_path / "targets" / bolt / f"{tid}.pcd"
                target_rel = Path("targets", bolt, f"{tid}.pcd")
                target_dst = out_path / target_rel

                # Load target geometry + normals once for scoring.
                pc = PCD4.from_path(target_src)
                Q = pc.numpy(("x", "y", "z"))
                N = pc.numpy(("normal_x", "normal_y", "normal_z"))

                # Score the collaborators' transform with our metrics, on the
                # full cloud and (when present) the keypoints cloud. The
                # keypoints denoised point-to-plane error is the apples-to-apples
                # comparison against the collaborators' own transformation_error,
                # which was computed on the voxel-downsampled keypoints as a
                # point-to-plane mean-squared error with the per-point sensor
                # noise ignored.
                def _block(p2p: float, p2pl: float,
                           den: Optional[float]) -> dict:
                    """One cloud's computed-metrics block, all MSE (mean of
                    squares, no sqrt). The denoised entry is the subtract-first
                    MSE mean((|d| - noise)^2), or null when intensity/noise are
                    unavailable."""
                    return {
                        "transformation_error_p2p": p2p,
                        "transformation_error_p2pl": p2pl,
                        "transformation_error_p2pl_denoised": den,
                    }

                full_p2p, full_p2pl, full_p2pl_den = score_transform(
                    full_xyz, Q, N, T_matrix,
                    intensity=full_intensity, noise_coeffs=noise_coeffs,
                )

                computed = CommentedMap({"full": _block(full_p2p, full_p2pl, full_p2pl_den)})
                if kp_xyz is not None:
                    kp_p2p, kp_p2pl, kp_p2pl_den = score_transform(
                        kp_xyz, Q, N, T_matrix,
                        intensity=kp_intensity, noise_coeffs=noise_coeffs,
                    )
                    computed["keypoints"] = _block(kp_p2p, kp_p2pl, kp_p2pl_den)

                # Copy the referenced target into the validation set (once).
                if target_dst not in copied_targets:
                    target_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target_src, target_dst)
                    copied_targets.add(target_dst)

                T_matrix_native = [[float(v) for v in row] for row in T_matrix.tolist()]

                target_validation = CommentedMap({
                    "path": str(target_rel),
                    "transformation_matrix": [
                        (lambda r: r.fa.set_flow_style() or r)(CommentedSeq(row))
                        for row in T_matrix_native
                    ],
                    "metrics": {
                        "original": {
                            "transformation_error": T_error
                        },
                        "computed": computed,
                    },
                })

                T_matrix_visual = "Visual Representation:\n"
                for row in T_matrix:
                    T_matrix_visual += f"[ {', '.join(f'{v:8.3f}' for v in row)} ]\n"
                target_validation.yaml_set_comment_before_after_key(
                    "metrics", before=T_matrix_visual, indent=2
                )

                filtered_targets_validation.append(target_validation)

            # Assemble metadata.yaml for this frame.
            targets_seq = CommentedSeq(filtered_targets_validation)
            for i in range(len(targets_seq)):
                label = f" TARGET {i + 1} ".center(YAML_WIDTH - 4, "#")
                targets_seq.yaml_set_comment_before_after_key(
                    i, before=f"\n{label}", indent=2
                )

            metadata_validation = CommentedMap()
            metadata_validation["source"] = str(frame_out / source_path.name)
            metadata_validation["keypoints"] = (
                str(frame_out / keypoints_path.name)
                if keypoints_path is not None else None
            )
            metadata_validation["targets"] = targets_seq
            metadata_validation["criterion"] = {
                "metric": "metrics.original.transformation_error",
                "objective": "minimize",
            }

            for key in list(metadata_validation.keys())[1:]:
                metadata_validation.yaml_set_comment_before_after_key(key, before="\n")
            metadata_validation.yaml_set_comment_before_after_key(
                "keypoints",
                before="Voxel-downsampled source cloud (PCL VoxelGrid centroids); "
                       "null if absent for this frame",
            )
            metadata_validation.yaml_set_comment_before_after_key(
                "targets", before="Target point clouds identified as potential matches"
            )
            metadata_validation.yaml_set_comment_before_after_key(
                "criterion", before="Best target selection criterion"
            )

            with open(frame_out / "metadata.yaml", "w") as fh:
                yaml.dump(metadata_validation, fh)


# ── CLI ────────────────────────────────────────────────────────────────────────

@app.command()
def main(
    dataset_path: Path = typer.Argument(
        ..., help="Path to the collaborators' raw dataset (contains source_* dirs)."
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Output path for the validation dataset. "
             "Defaults to '<dataset_path>/../Validation_Data'.",
    ),
    noise_coeffs: tuple[float, float] = typer.Option(
        (96.72, -0.762),
        "--noise-coeffs",
        help="Sensor-noise floor coefficients (A, B) for the denoised p2pl "
             "metric: expected_noise = A * amplitude**B * 2. Matches "
             "error.p2pl_noise_coeffs in config.yaml.",
    ),
):
    """Build the validation dataset from the raw collaborator dataset."""
    if not dataset_path.exists():
        raise typer.BadParameter(f"Dataset path not found: {dataset_path}")

    out_path = out if out is not None else dataset_path.parent / "Validation_Data"
    out_path.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Dataset path    : {dataset_path.resolve()}")
    typer.echo(f"Validation path : {out_path.resolve()}")
    typer.echo(f"Noise coeffs    : {tuple(noise_coeffs)}\n")

    build(dataset_path, out_path, noise_coeffs=tuple(noise_coeffs))

    typer.secho(
        f"\nDone. Validation dataset at: {out_path.resolve()}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()