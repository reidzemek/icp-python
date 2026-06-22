from pointcloud import PointCloud
from targetcloud import TargetCloud
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from drra_memory import DrraMemory


# Width of a DRRA memory row, in bits (see memory_map.md).
DRRA_ROW_WIDTH = 256

# Fixed field width for the cross-covariance accumulators. Held at 64 bits
# regardless of n_coord_bits / point count: this is both the accumulation
# headroom and a deliberate interface choice, so the RISC-V int64->float32
# promotion is a trivial fixed-width load (see memory_map.md, section 4).
H_ACCUM_WIDTH = 64


@dataclass(frozen=True)
class MemTrace:
    """Per-pair memory-trace request handed to ICP.run().

    Attributes:
        iters (frozenset[int]): ICP iteration indices (0-based) whose per-unit
            memory images should be written. Fixed-iteration mode only.
        out_dir (Path): Output directory for this (source, target) pair. Each
            selected iteration writes its images under out_dir / "iter_NN".
    """
    iters: frozenset[int]
    out_dir: Path


# Declarative layout table: the executable twin of memory_map.md. One entry per
# memory image (.mem file) emitted per traced iteration, named "<unit>_<side>".
# Every functional unit records BOTH its input and output, even where that
# duplicates an array another unit wrote: each hardware unit is validated in
# isolation, so each must carry its full input/output independently.
#
# `width` is a token resolved at write time against the only free widths, so no
# concrete bit count is baked in here:
#   "coord" -> n_coord_bits   "rot" -> R_width   "accum" -> H_ACCUM_WIDTH (64)
#
# `packing` is one of:
#   "interleaved"     -> whole (x, y, z) rows, point_size = 3
#   "per_axis"        -> three passes (all x, then all y, then all z)
#   "replicated"      -> a single (x, y, z) centroid written to `repeat` rows
#   "center_input"    -> composite: centroid row 0, then per-axis points
#                        (source_key is the points array; `centroid_key` names
#                        the centroid array). Dedicated writer.
#   "covariance_input"-> P/Q rows interleaved per axis: P_x, Q_x, P_y, Q_y, ...
#                        (source_key = P array, `second_key` = Q array).
#                        Dedicated writer.
#   "transform_input" -> R|t header rows, then interleaved points
#                        (source_key = points; R/t from rotation/translation
#                        log keys). Dedicated writer.
#   "transform_output"-> transformed P twice: interleaved group, then per-axis
#                        group. Dedicated writer.
#
# Each entry is a dict so the irregular ones can carry extra keys (centroid_key,
# second_key) without bloating the common case. The writer owns the layout; the
# log holds only raw integer arrays.
LAYOUT_SPEC = {
    # ── Nearest Neighbor Search ──
    "nn_search_input":   {"key": "source",          "packing": "interleaved", "width": "coord"},
    "nn_search_output":  {"key": "nearest_target",  "packing": "per_axis",    "width": "coord"},

    # ── Compute Centroid (Average) ──
    # Inputs are the iteration's P (source) and nearest-Q, per-axis (= NN output
    # form). Outputs are the centroids, two identical rows each.
    "mean_source_input":  {"key": "source",         "packing": "per_axis",   "width": "coord"},
    "mean_source_output": {"key": "source_mean",    "packing": "replicated", "width": "coord", "repeat": 2},
    "mean_target_input":  {"key": "nearest_target", "packing": "per_axis",   "width": "coord"},
    "mean_target_output": {"key": "target_mean",    "packing": "replicated", "width": "coord", "repeat": 2},

    # ── Center (Vector Processor) ──
    # Inputs are composite: centroid row 0, then per-axis points. Outputs are the
    # centered clouds, per-axis (= NN output form).
    "center_source_input":  {"key": "source",          "packing": "center_input", "width": "coord", "centroid_key": "source_mean"},
    "center_source_output": {"key": "source_centered", "packing": "per_axis",     "width": "coord"},
    "center_target_input":  {"key": "nearest_target",  "packing": "center_input", "width": "coord", "centroid_key": "target_mean"},
    "center_target_output": {"key": "target_centered", "packing": "per_axis",     "width": "coord"},

    # ── Compute Covariance (Matrix-Matrix Multiplication) ──
    # Input interleaves the two centered clouds P/Q per axis; output is H at the
    # fixed 64-bit accumulator width, row-major.
    "covariance_input":  {"key": "source_centered", "packing": "covariance_input", "width": "coord", "second_key": "target_centered"},
    "covariance_output": {"key": "cross_covariance_matrix", "packing": "interleaved", "width": "accum"},

    # ── Apply Transformation (Vector Processor) ──
    # Input is the R|t header then interleaved P; output carries transformed P in
    # both interleaved and per-axis forms for its two consumers.
    "transform_input":  {"key": "source",             "packing": "transform_input",  "width": "coord", "rot_key": "rotation_matrix", "trans_key": "translation_vector"},
    "transform_output": {"key": "source_transformed", "packing": "transform_output", "width": "coord"},
}


class ICP:
    """
    Iterative closest point algorithm.

    Args:
        P (PointCloud): Source point cloud (P).
        Q (TargetCloud): k-dimensional tree data structure for the target point cloud (Q).
        n_iter (int): ICP iteration count.
        R_width (int): Signed fixed-point bit-width for the rotation matrix `R`. Defaults to 64.

    ## Attributes
        **source** : *PointCloud*
        Source point cloud.<br>

        **target** : *TargetCloud*
        k-d dimensional tree data structure built on target point cloud.<br>

        **n_iter** : *int*
        ICP iterations.<br>

        **R_width** : *int*
        Signed fixed-point bit-width for the rotation matrix `R`.

        **total_rotation** : *np.ndarray*
        Cumulative rotation (R) for all ICP iterations.<br>

        **total_translation** : *np.ndarray*
        Cumulative translation (t) for all ICP iterations.
    """

    def __init__(self, P: PointCloud, Q: TargetCloud, R_width: int=32, seed: int=42):
        """Load source and target point cloud data and initialize ICP algorithm parameters."""

        # Assign attributes
        self.source = P
        self._aligned_source = None
        self.target = Q
        self.R_width = R_width
        self.rng = np.random.default_rng(seed)

        # Total transformation
        self._total_transform = None

    @property
    def aligned_source(self):
        if self._aligned_source is None:
            raise RuntimeError("Call run() before accessing aligned_source.")
        return self._aligned_source
    
    @property
    def total_transform(self):
        if self._total_transform is None:
            raise RuntimeError("Call run() before accessing total_transform.")
        return self._total_transform

    def run(
        self,
        n_icp_iters: int=None,
        tol: float=1e-3,
        max_iters: int=10_000,
        n_jacobi_sweeps: int=8,
        mem_trace: Optional[MemTrace]=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the ICP algorithm for `n_icp_iters` iterations.

        Args:
            n_icp_iters (int): Number of ICP iterations. If None, the algorithm runs until
                convergence. Defaults to None.
            n_jacobi_sweeps (int): Number of sweeps used by the 4x4 Jacobi eigen decomposition
                when estimating the optimal rotation (Horn's method). Defaults to 8.
            mem_trace (MemTrace | None): When provided, the per-unit DRRA memory images for the
                selected iterations are written under `mem_trace.out_dir / "iter_NN"`. Honored only
                in fixed-iteration mode (`n_icp_iters` is not None); ignored with a warning in
                convergence mode, since the hardware target runs a fixed iteration count.
                Defaults to None.

        Returns:
            tuple: `self.total_rotation` and `self.total_translation`.
        """

        # Memory tracing is a fixed-iteration-only feature: in convergence mode
        # the iteration count is not known ahead of time, and the hardware
        # target runs a fixed number of iterations regardless. Warn and disable.
        if mem_trace is not None and n_icp_iters is None:
            import warnings
            warnings.warn(
                "mem_trace is ignored in convergence mode (n_icp_iters is None); "
                "no memory images will be written.",
                stacklevel=2,
            )
            mem_trace = None

        trace_iters = mem_trace.iters if mem_trace is not None else frozenset()

        # Total transformation (rotation and translation) parameters
        R_f_total = np.eye(3,)
        t_f_total = np.zeros((3, 1))
        R_q_total = np.eye(3, dtype=np.int64) * (2 ** (self.R_width - 1)) # Q1.(self.R_width - 1)
        t_q_total = np.zeros((3, 1), dtype=np.int64)

        # Assign aligned source attribute
        self._aligned_source = self.source.copy()

        # Source point clouds
        P_f = self._aligned_source.points_f     # floating point space
        P_q = self._aligned_source.points_q     # quantized space

        # ICP algorithm iterations
        i = 0   # ICP iteration count
        while True:
            # Pre-transform source snapshot: the quantized cloud as fed INTO this
            # iteration (the NN-search input). Held cheaply for the whole loop
            # body; only committed to a log if this iteration is traced.
            P_q_pre = P_q

            # Nearest points in target point clouds (floating point and quantized space)
            Q_nearest_f, Q_nearest_q = self.target.nearest(self._aligned_source)

            # ----- FLOATING POINT SPACE -----
            # Compute centroids (source and nearest points in target)
            P_mean_f = np.mean(P_f, axis=0).reshape(1, 3)
            Q_mean_f = np.mean(Q_nearest_f.points_f, axis=0).reshape(1, 3)

            # Center P and Q_nearest
            P_centered_f = P_f - P_mean_f
            Q_centered_f = Q_nearest_f.points_f - Q_mean_f

            # Compute the cross covariance matrix (H)
            H_f = P_centered_f.T @ Q_centered_f

            # print(f"\nIteration {i}: float")
            # Compute the transformation parameters (R and t)
            R_f, t_f = self._estimate_transform(H_f, P_mean_f, Q_mean_f, n_jacobi_sweeps)

            # Apply the transformation to P
            P_f = ((R_f @ P_f.T) + t_f).T

            # Update the total transformation
            R_f_total = R_f @ R_f_total         # rotation
            t_f_total = R_f @ t_f_total + t_f   # translation

            # ----- QUANTIZED SPACE -----
            # Compute centroids (source and nearest points in target)
            P_mean_q = self._mean(P_q)
            Q_mean_q = self._mean(Q_nearest_q.points_q)

            # Center P and Q_nearest
            P_centered_q = self._center(P_q, P_mean_q)
            Q_centered_q = self._center(Q_nearest_q.points_q, Q_mean_q)

            # Compute the cross covariance matrix (H)
            H_q = self._xcovariance(P_centered_q, Q_centered_q)

            if self._det3_int(H_q) == 0:
                raise RuntimeError(
                    "Rank-deficient cross-covariance matrix H_q in quantized space "
                    "(det(H_q) == 0): degenerate point geometry, rotation is "
                    "ill-defined. This arises when quantization or too few points "
                    "(n_coord_bits, n_P/n_Q) collapse the cloud onto a plane, line, "
                    f"or point.\nH_q = {np.array2string(H_q, prefix="H_q = ")}"
                )

            # print(f"\nIteration {i}: quantized")
            # Compute and quantize the transformation parameters (R and t)
            R_q, t_q = self._estimate_transform(H_q, P_mean_q, Q_mean_q, n_jacobi_sweeps)
            R_q, t_q = self._quantize_transform(R_q, t_q)

            # Apply transformation to P
            P_q = self._apply_transform(P_q, R_q, t_q)

            # ----- MEMORY TRACE (selected iterations only) -----
            # `i` is still the 0-based index of the iteration just computed
            # (incremented below). Assemble the per-unit log only when traced.
            if i in trace_iters:
                iter_log = {
                    "source": P_q_pre,                          # NN-search input (pre-transform)
                    "nearest_target": Q_nearest_q.points_q,
                    "source_mean": P_mean_q,
                    "target_mean": Q_mean_q,
                    "source_centered": P_centered_q,
                    "target_centered": Q_centered_q,
                    "cross_covariance_matrix": H_q,
                    "rotation_matrix": R_q,
                    "translation_vector": t_q,
                    "source_transformed": P_q,                  # apply-transform output
                }
                self._write_mem(iter_log, mem_trace.out_dir / f"iter_{i:02d}")

            # Update the total transformation (rotation and translation)
            R_q_total = ((R_q.astype(object) @ R_q_total.astype(object)) \
                >> (self.R_width - 1)).astype(np.int64)                     # Q1.(self.R_width - 1)
            t_q_total = ((R_q.astype(object) @ t_q_total.astype(object)) \
                >> (self.R_width - 1)).astype(np.int64) + t_q               # coordinate data type

            # ----- INCREMENT ITERATION COUNT -----
            i += 1

            # ----- CONVERGENCE MODE -----
            if n_icp_iters is None:
                if np.mean(np.sum((P_q - Q_nearest_q.points_q)**2, axis=1)) < tol:
                    break
                if i >= max_iters:
                    raise RuntimeError("ICP did not converge.")

            # ----- FIXED ITERATION MODE -----
            if n_icp_iters is not None and i >= n_icp_iters:
                break

        # Update aligned source attribute with transformed source from ICP algorithm
        self._aligned_source.points_f = P_f
        self._aligned_source.points_q = P_q

        # Apply total transformation to surface normal vectors
        self._aligned_source.normals = (R_f_total @ self._aligned_source.normals.T).T if self._aligned_source.normals is not None else None

        # Assign total transformation attribute
        self._total_transform = {
            "float": (R_f_total, t_f_total),
            "quantized": (R_q_total, t_q_total)
        }

        return self.total_transform

    # ── Memory tracing ──────────────────────────────────────────────────────

    def _resolve_width(self, token: str) -> int:
        """Resolve a LAYOUT_SPEC width token to a concrete bit width.

        The only free widths are the coordinate width (from the source point
        cloud's quantization) and the rotation fixed-point width; the covariance
        accumulator is the fixed H_ACCUM_WIDTH.
        """
        if token == "coord":
            return self.source.n_bits
        if token == "rot":
            return self.R_width
        if token == "accum":
            return H_ACCUM_WIDTH
        raise ValueError(f"Unknown width token: {token!r}")

    def _write_mem(self, iter_log: dict, iter_dir: Path) -> None:
        """Write every memory image for one traced iteration.

        Walks LAYOUT_SPEC and dispatches each entry to the writer for its
        packing, producing one "<unit>_<side>.mem" file under `iter_dir`. Each
        functional unit records both its input and output independently (the
        same array may be written several ways), so every unit can be validated
        in isolation against its RTL.

        Args:
            iter_log (dict): Raw int64 arrays keyed by data name (see run()).
            iter_dir (Path): Output directory for this iteration (iter_NN).
        """
        iter_dir.mkdir(parents=True, exist_ok=True)

        for name, spec in LAYOUT_SPEC.items():
            packing = spec["packing"]
            width = self._resolve_width(spec["width"])
            mem = DrraMemory(row_width=DRRA_ROW_WIDTH)

            if packing == "interleaved":
                arr = self._as_points(iter_log[spec["key"]])
                self._pack_interleaved(mem, arr, width)

            elif packing == "per_axis":
                arr = self._as_points(iter_log[spec["key"]])
                self._pack_per_axis(mem, arr, width)

            elif packing == "replicated":
                point = self._as_points(iter_log[spec["key"]]).reshape(-1)
                if point.size != 3:
                    raise ValueError(
                        f"{name}: replicated layout expects a single (x, y, z) "
                        f"point, got {point.size} values."
                    )
                for r in range(spec["repeat"]):
                    mem.write(point, width=width, point_size=3, row=r, bit=0)

            elif packing == "center_input":
                # Composite: centroid on row 0, then per-axis points below.
                centroid = self._as_points(iter_log[spec["centroid_key"]]).reshape(-1)
                points = self._as_points(iter_log[spec["key"]])
                mem.write(centroid, width=width, point_size=3, row=0, bit=0)
                # per-axis groups start on the next fresh row
                self._pack_per_axis(mem, points, width, start_row=1)

            elif packing == "covariance_input":
                # P/Q rows interleaved per axis: P_x, Q_x, P_y, Q_y, P_z, Q_z.
                # Each axis group starts on a fresh row; a group spanning more
                # than one row pushes the next group down accordingly.
                P = self._as_points(iter_log[spec["key"]])
                Q = self._as_points(iter_log[spec["second_key"]])

                def _next_fresh_row() -> int:
                    r, b = mem.tell()
                    return r + 1 if b != 0 else r

                row = 0
                for axis in range(3):
                    mem.write(P[:, axis], width=width, point_size=1, row=row, bit=0)
                    row = _next_fresh_row()
                    mem.write(Q[:, axis], width=width, point_size=1, row=row, bit=0)
                    row = _next_fresh_row()

            elif packing == "transform_input":
                # R|t header (rows 0-1), then interleaved points.
                self._pack_transform_header(
                    mem,
                    iter_log[spec["rot_key"]],
                    iter_log[spec["trans_key"]],
                )
                points = self._as_points(iter_log[spec["key"]])
                self._pack_interleaved(mem, points, width, start_row=2)

            elif packing == "transform_output":
                # Transformed P carried twice: interleaved group, then per-axis.
                arr = self._as_points(iter_log[spec["key"]])
                self._pack_interleaved(mem, arr, width)
                next_row, bit = mem.tell()
                if bit != 0:
                    next_row += 1
                self._pack_per_axis(mem, arr, width, start_row=next_row)

            else:
                raise ValueError(f"{name}: unknown packing {packing!r}")

            mem.to_file(str(iter_dir / f"{name}.mem"))

    # ── Packing primitives ──────────────────────────────────────────────────

    @staticmethod
    def _as_points(arr) -> np.ndarray:
        """Coerce a logged array to int64 (N, 3)."""
        return np.asarray(arr, dtype=np.int64).reshape(-1, 3)

    @staticmethod
    def _pack_interleaved(mem: DrraMemory, arr: np.ndarray, width: int,
                          start_row: int = 0) -> None:
        """Whole (x, y, z) points in sequence, point_size = 3, atomic per row."""
        mem.write(arr.reshape(-1), width=width, point_size=3, row=start_row, bit=0)

    @staticmethod
    def _pack_per_axis(mem: DrraMemory, arr: np.ndarray, width: int,
                       start_row: int = 0) -> None:
        """Three consecutive groups (all x, then all y, then all z). Each group
        starts on a fresh row; the previous group zero-fills its last row."""
        first = True
        for axis in range(3):
            if first:
                row = start_row
                first = False
            else:
                row, bit = mem.tell()
                if bit != 0:
                    row += 1
            mem.write(arr[:, axis], width=width, point_size=1, row=row, bit=0)

    def _pack_transform_header(self, mem: DrraMemory, R, t) -> None:
        """Write the apply-transform R|t header into rows 0-1 of `mem`.

        Layout (memory_map.md, section 6):
            row 0 : r00, r01, r02, t0, r10, r11, r12, t1
            row 1 : r20, r21, r22, t2
        R uses the rotation fixed-point width (R_width); t uses the coordinate
        width (n_coord_bits). The two widths differ, so each scalar is written
        with its own width rather than packed as a uniform point.
        """
        R = np.asarray(R, dtype=np.int64).reshape(3, 3)
        t = np.asarray(t, dtype=np.int64).reshape(3)

        r_width = self.R_width
        t_width = self.source.n_bits

        row0 = [
            (R[0, 0], r_width), (R[0, 1], r_width), (R[0, 2], r_width), (t[0], t_width),
            (R[1, 0], r_width), (R[1, 1], r_width), (R[1, 2], r_width), (t[1], t_width),
        ]
        row1 = [
            (R[2, 0], r_width), (R[2, 1], r_width), (R[2, 2], r_width), (t[2], t_width),
        ]

        for row_idx, scalars in enumerate((row0, row1)):
            mem.seek(row_idx, 0)
            for value, width in scalars:
                mem.write([value], width=width, point_size=1)

    def _mean(self, pc: np.ndarray) -> np.ndarray:
        """
        Compute the centroid of a 3D point cloud using hierarchical reduction.

        Points are hierarchically reduces using an 8-input adder tree (with zero padding where
        required) until a single sum remains. A final fixed-point correction factor is then applied
        to account for non-power-of-two input sizes.

        Args:
            pc (np.ndarray): A signed integer array of shape (N, 3).

        Returns:
            np.ndarray: A signed integer array of shape (1, 3).
        """

        def _reduce(points: np.ndarray) -> np.ndarray:
            """
            Reduce one hierarchy level by summing blocks of up to 8 points.
            
            Each output point represents the sum of one block of at most 8 input points.
            
            Args:
                points (np.ndarray): An array of shape (N, 3).
                
            Returns:
                np.ndarray: Reduced array of shape (ceil(N/8), 3).
            """
            reduced = []

            for i in range(0, points.shape[0], 8):
                block = points[i:i + 8]
                reduced.append(np.sum(block, axis=0))

            return np.asarray(reduced, dtype=points.dtype)
        
        # Hierarchical reduction until a single vector remains
        points = pc
        while points.shape[0] > 1:
            points = _reduce(points)

        # Fixed-point scaling factor to approximate division by N using power-of-two scaling
        # A hardware implementation would use a pre-computed scaling factor
        frac_bits = 16                  # fractional bits for fixed-point scaling factor
        N = pc.shape[0]                 # number of points in input point cloud
        tree_depth = N.bit_length()     # number of 8-input reduction levels
        scale = np.rint((2**tree_depth / N) * (2**frac_bits)).astype(np.int64)

        # Compute the average using the fixed-point scaling factor and bit-shift division
        return (points * scale) >> (tree_depth + frac_bits)
    
    def _center(self, pc: np.ndarray, mean: np.ndarray) -> np.ndarray:
        """
        Center point cloud by subtracting centroid (mean).
        
        Args:
            pc (np.ndarray): A signed integer array of shape (N, 3).
            mean (np.ndarray): A signed integer array of shape (1, 3) representing the centroid.

        Returns:
            np.ndarray: A signed integer array of shape (N, 3).        
        """

        return pc - mean

    def _xcovariance(self, P, Q) -> np.ndarray:
        """
        Compute the 3x3 cross-covariance matrix.

        The point clouds must have the same number of points (N), they must be centered
        (ie. zero mean), and the points in Q should be the those nearest to the corresponding
        points in P.

        Args:
            P (np.ndarray): Centered source point cloud array of shape (N, 3).
            Q (np.ndarray): Centered array of the nearest target points of shape (N, 3).

        Returns:
            np.ndarray: 3x3 cross-covariance matrix `H` = `P.T` x `Q`.
        """

        return P.T @ Q
    
    def _det3_int(self, H):
        """Exact 3x3 determinant in Python ints (no overflow, no float cast)."""
        a, b, c, d, e, f, g, h, i = (int(v) for v in H.ravel())
        return (a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g))
    
    def _estimate_transform(
        self,
        H: np.ndarray,
        P_mean: np.ndarray,
        Q_mean: np.ndarray,
        n_jacobi_sweeps: int=8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Using Horn's quaternion method, extimate the rigid transformation aligning `P` with `Q`.

        Args:
            H (np.ndarray): Cross covariance matrix of shape (3, 3).
            P_mean (np.ndarray): P centroid of shape (1, 3).
            Q_mean (np.ndarray): Q centroid of shape (1, 3).
            n_jacobi_sweeps (int): Number of sweeps used by the 4x4 Jacobi eigen decomposition
                when computing the unit quaternion. Defaults to 8.

        Returns:
            tuple: `R` and `t` outlined below.

            **R** : *np.ndarray of shape (3, 3)*<br>
            Rotation matrix in double precision floating point representation.

            **t** : *np.ndarray of shape (3, 1)*
            Translation vector in double precision floating point representation.
        """

        # Extract the vector components of the skew-symmetric parts of H
        delta = np.array([
            H[1, 2] - H[2, 1],
            H[2, 0] - H[0, 2],
            H[0, 1] - H[1, 0]
        ])

        # use Horn's method to construc the 4x4 quaternion characteristic matrix N.
        tr = np.trace(H)
        N = np.zeros((4, 4))
        N[0, 0] = tr
        N[0, 1:4] = delta
        N[1:4, 0] = delta
        N[1:4, 1:4] = np.array([
            [2*H[0, 0] - tr,      H[0, 1] + H[1, 0],  H[0, 2] + H[2, 0]],
            [H[1, 0] + H[0, 1],   2*H[1, 1] - tr,     H[1, 2] + H[2, 1]],
            [H[2, 0] + H[0, 2],   H[2, 1] + H[1, 2],  2*H[2, 2] - tr]
        ])

        # 4x4 Jacobi eigen decomposition
        q = self._jacobi_eigen_4x4(N, n_sweeps=n_jacobi_sweeps)
        q_ref = np.linalg.eigh(N)[1][:, -1]
        q_ref = q_ref / np.linalg.norm(q_ref)
        if np.dot(q, q_ref) < 0:
            q_ref = -q_ref
        error = np.linalg.norm(q - q_ref)
        # print(f"q = {q}")
        # print(f"q_ref = {q_ref}")
        # print(f"error = {error}")
        
        # Extract the quaternion components
        q0, q1, q2, q3 = q

        # Use the unit quaternion to construct the rotation matrix
        R = np.array([
            [1 - 2*(q2*q2 + q3*q3),   2*(q1*q2 - q3*q0),     2*(q1*q3 + q2*q0)],
            [2*(q1*q2 + q3*q0),       1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q1*q0)],
            [2*(q1*q3 - q2*q0),       2*(q2*q3 + q1*q0),     1 - 2*(q1*q1 + q2*q2)]
        ])

        # Translation vector
        t = Q_mean.T - R @ P_mean.T

        return R, t

    def _jacobi_eigen_4x4(self, A: np.ndarray, n_sweeps: int = 8):
        A = A.copy()
        V = np.eye(4)

        def rotate(i, j):
            a = A[i, i]
            b = A[j, j]
            c = A[i, j]

            if abs(c) < 1e-12:
                return

            # rotation parameter (stable form)
            tau = (b - a) / (2.0 * c)
            t = np.sign(tau) / (abs(tau) + np.sqrt(1.0 + tau * tau))
            cs = 1.0 / np.sqrt(1.0 + t * t)
            sn = t * cs

            # --- update matrix A (rows/cols i,j) ---
            for k in range(4):
                aki = A[k, i]
                akj = A[k, j]
                A[k, i] = cs * aki - sn * akj
                A[k, j] = sn * aki + cs * akj

            for k in range(4):
                aki = A[i, k]
                akj = A[j, k]
                A[i, k] = cs * aki - sn * akj
                A[j, k] = sn * aki + cs * akj

            # force symmetry (important for numerical stability)
            A[i, j] = 0.0
            A[j, i] = 0.0

            # --- update eigenvector matrix V ---
            for k in range(4):
                vki = V[k, i]
                vkj = V[k, j]
                V[k, i] = cs * vki - sn * vkj
                V[k, j] = sn * vki + cs * vkj

        # sweep schedule (fixed, deterministic)
        pairs = [(0,1), (0,2), (0,3),
                (1,2), (1,3),
                (2,3)]

        for _ in range(n_sweeps):
            for (i, j) in pairs:
                rotate(i, j)

        # dominant eigenvector
        eigenvalues = np.diag(A)
        idx = np.argmax(eigenvalues)
        q = V[:, idx]

        # normalize quaternion
        q = q / np.linalg.norm(q)
        return q

    def _power_iteration(
        self,
        N: np.ndarray,
        n_iters: int=None,
        q: np.ndarray=None,
        tol: float=1e-8, #np.finfo(float).eps,
        max_iters: int=10_000
    ):
        """
        Power iteration method for computing the dominant eigen vector.

        Supports two modes of operation:
        - Fixed-iteration mode: if `n_iters` is specified, the method runs for exactly `n_iters`
          iterations
        - Convergence mode: if `n_iters` is None, the  method runs until the change between
          successive estimates falls below `tol`, or until `max_iters` is reached.

        Args:
            N (np.ndarray): Square matrix whose dominant eigen vector is to be computed.
            n_iters (int): Number of iterations to perform. If None, the method runs until
                convergence. Defaults to None.
            q (np.ndarray): Initial estimate for the dominant eigen vector. If None, a random
                vector is initialized using a standard normal distribution.
            tol (float): Convergence tolerance. Iteration stops when the L2 norm of the difference
                between successive vectors falls below this value. Only used when `n_iters` is
                None. Defaults to machine epsilon.
            max_iters (int): Maximum number of iterations in convergence mode. Prevents infinite
                loops if convergence is slow or fails. Defaults to 10_000.
        
        Returns:
            np.ndarray: Dominant eigen vector array of shape [N.shape[0],], normalized to unit
                length.

        Raises:
            RuntimeError:
                If convergence mode is used (`n_iters is None`) and the algorithm fails to converge
                within `max_iters` iterations.
        """
        if q is None:
            q = self.rng.standard_normal(N.shape[0])

        print(N)

        # Initial normalization (ensures validity of stopping criterion for consistent convergence)
        q = q / np.linalg.norm(q)

        # Power iteration loop
        print("\nITERATING")
        i = 0
        while True:
            q_new = N @ q
            q_new /= np.linalg.norm(q_new)
            i += 1

            temp = min(np.linalg.norm(q - q_new), np.linalg.norm(q + q_new)) 

            # Convergence mode
            if n_iters is None:
                if temp < tol:
                    return q_new
                if i >= max_iters:
                    raise RuntimeError("Power iteration did not converge.")

            print(f"{temp} < {tol}")
            
            q = q_new

            # Fixed-iteration mode
            if n_iters is not None and i >= n_iters:
                return q
    
    def _quantize_transform(self, R, t):
        """
        Quantize the transformation parameters.

        Args:
            **R** : *np.ndarray of shape (3, 3)*<br>
            Rotation matrix in double precision floating point representation.

            **t** : *np.ndarray of shape (3, 1)*
            Translation vector in double precision floating point representation.

        Returns:
            tuple: `R` and `t` outlined below.

            **R** : *np.ndarray of shape (3, 3)*<br>
            Rotation matrix in Q1.(`self.R_width - 1`) fixed-point representation.

            **t** : *np.ndarray of shape (3, 1)*
            Translation vector using the same signed integer representation as `P`.
        """

        # Quantize the rotation matrix (R) and the translation vector (t)
        R = np.rint(np.clip(R, -1.0, 1.0 - (2**-(self.R_width - 1))) * 2**(self.R_width - 1))
        t = np.rint(t)

        return R.astype(np.int64), t.astype(np.int64)

    
    def _apply_transform(self, P: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Tasnform the source point cloud `P`.

        Args:
            P (np.ndarray): Source point cloud array of shape (N, 3).
            R (np.ndarray): 3x3 rotation matrix.
            t (np.ndarray): 3x1 rotation vector.

        Returns:
            np.ndarray: Transformed point cloud of shape (N, 3).
        """

        return (((R @ P.T) >> (self.R_width - 1)) + t).T

    def p2p_error(self) -> float:
        """
        Compute the point-to-point alignment error.

        Returns:
            float: Root mean square (RMS) point-to-point alignment error.
        """

        # Nearest points in target point cloud (floating point space)
        Q_nearest_f, _ = self.target.nearest(self.aligned_source)

        source_f = self.aligned_source.points_f
        source_q = self.aligned_source.dequantize()
        target = Q_nearest_f.points_f

        # Point-to-point transformation error for floating point and quantized space ICP
        p2p_error_f = np.mean(np.sum((source_f - target)**2, axis=1))
        p2p_error_q = np.mean(np.sum((source_q - target)**2, axis=1))

        p2p_error = {
            "float": p2p_error_f,
            "quantized": p2p_error_q
        }

        return p2p_error

    def p2pl_error(self, noise_coeffs: tuple[float, float]=None) -> float:
        """
        Compute the point-to-plane alignment error.
 
        Args:
            noise_coeffs (tuple[float, float] | None): Coefficients (A, B) of the
                per-point expected sensor noise floor, derived from the source
                intensity (amplitude):
 
                    expected_noise = A * (amplitude ** B) * NOISE_FACTOR
 
                When provided, correspondences with |d| < expected_noise are
                excluded, and survivors contribute their noise-subtracted distance
                (|d| - expected_noise) to the RMS. When None, noise masking is
                disabled and all correspondences contribute their raw distance |d|.
                Defaults to None.
 
        Returns:
            float: Root mean square (RMS) point-to-plane alignment error.
        """
 
        # Constant multiplier on the noise floor, kept separate from the (A, B)
        # curve-fit coefficients to stay faithful to the collaborators' formula.
        # Its meaning is unconfirmed. Open questions for the collaborators:
        #   - Is `A * amplitude**B` a 1-sigma noise estimate? If so, this factor
        #     of 2 is likely a coverage factor (~2-sigma / ~95% confidence
        #     threshold), in which case it is a tunable choice and should
        #     graduate to a `C` parameter (and a candidate sweep axis).
        #   - Alternatively, does it represent a round-trip / two-way path
        #     factor, which would make it a fixed physical constant, not a knob?
        #   - What units are the noise output and the amplitude in? That likely
        #     also clarifies this factor (e.g. a radius/diameter convention).
        NOISE_FACTOR = 2
 
        # Nearest points in target point cloud (floating point space)
        Q_nearest_f, _ = self.target.nearest(self.aligned_source)
 
        source_f = self.aligned_source.points_f
        source_q = self.aligned_source.dequantize()
        target = Q_nearest_f.points_f
        normals = Q_nearest_f.normals
 
        # Per-point expected noise floor from source amplitude (intensity)
        expected_noise = None
        if noise_coeffs is not None:
            if self.aligned_source.intensities is None:
                raise RuntimeError(
                    "p2pl_error with noise_coeffs requires source intensities "
                    "(amplitude) to compute the expected noise floor, but "
                    "aligned_source.intensities is None."
                )
            A, B = noise_coeffs
            amplitude = np.asarray(self.aligned_source.intensities, dtype=np.float64).reshape(-1)
            expected_noise = A * (amplitude ** B) * NOISE_FACTOR
 
        def _mse(source):
            # Absolute per-point point-to-plane distance
            abs_d = np.abs(np.sum((source - target) * normals, axis=1))
 
            if noise_coeffs is None:
                return np.mean(abs_d**2)
 
            # Keep only correspondences above the expected noise floor,
            # contributing their noise-subtracted distance to the RMS
            keep = abs_d >= expected_noise
            survivors = abs_d[keep] - expected_noise[keep]
 
            if survivors.size == 0:
                return float("nan")
            return np.mean(survivors**2)
 
        p2pl_error = {
            "float": _mse(source_f),
            "quantized": _mse(source_q),
        }

        return p2pl_error