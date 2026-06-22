from __future__ import annotations
import pypcd4
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import re
import warnings
import copy

class PointCloud:
    """
    3-dimensional point cloud, with quantization for hardware modeling.

    Supports reading from CSV files with pre-quantized points or PCD files that are normalized and 
    quantized on load.

    Args:
        path (str): Path to the point cloud file. Can accept either PCD or CSV files.<br>
            **PCD** files contain raw point cloud data and require `norm_range` and `n_bits` for
            min-max quantization.<br>
            **CSV** files should be quantized, with the following filename format
            `<name>-<n_points>_<n_bits>.csv`. The bit width (`n_bits`) is parsed from the filename
            and validated. `norm_range` should be included to support `dequantize()`.
        norm_range (Tuple[float, float], optional): Minimum and maximum values of the coordinate 
            range used for min-max quantization. Required for both CSV and PCD files.
        n_bits (int, optional): Bit width for quantized coordinates. Required for PCD files.

    ## Attributes
        **points_f** : *np.ndarray, optional*
        Unmodified double precision floating point coordinate values of shape (`n_points`, 3).<br>

        **points_q** : *np.ndarray*
        Quantized `n_bits`-bit signed integer coordinate values of shape (`n_points`, 3).<br>

        **intensities** : *float, optional*
        Unmodified double precision floating point intensity values for each point.<br>

        **normals** : *np.ndarray, optional*
        Unmodified double precision floating point surface normal vectors of for each point.<br>
        
        **n_points** : *int*
        Number of points in the point cloud.<br>

        **n_bits** : *int*
        Bit width used for quantization.<br>

        **norm_range** : *Tuple[float, float]*
        Minimum and maximum values of the coordinate range range used for min-max quantizaiton.
    """

    def __init__(
        self,
        path: Path,
        norm_range: Tuple[float, float],
        n_bits: Optional[int]=None
    ):
        """Load point cloud and quantize if required."""

        # Load PCD file
        if path.suffix.lower() == ".pcd":
            if n_bits is None:
                raise ValueError("n_bits required for PCD files.")

            pc = pypcd4.PointCloud.from_path(path)
            self.points_f = pc.numpy(("x", "y", "z"))

            # Quantize
            points_norm = 2 * (self.points_f - norm_range[0]) / (norm_range[1] - norm_range[0]) - 1
            self.points_q = np.clip(
                np.rint(points_norm * (2**(n_bits - 1) - 1)),
                -2**(n_bits - 1),
                2**(n_bits - 1) - 1
            ).astype(np.int64)

            # Assign attributes
            self.intensities = pc.numpy(("intensity",)) if "intensity" in pc.fields else None
            normal_fields = ("normal_x", "normal_y", "normal_z")
            self.normals = pc.numpy(normal_fields) if all(f in pc.fields for f in normal_fields) else None
            self.n_points = self.points_q.shape[0]
            self.n_bits = n_bits
        
        # Load CSV file
        elif path.suffix.lower() == ".csv":
            parsed_n_points, parsed_n_bits = self._parse_csv_filename(path)
            pc = pd.read_csv(path).to_numpy()
            points_q = pc[:, 0:3]
            
            # Content checking
            if pc.shape[1] not in (3, 4, 7):
                raise ValueError("CSV must have 3, 4 or 7 columns.")
            if pc.shape[0] != parsed_n_points:
                warnings.warn("CSV point count not consistent with filename.")
            if not np.all(
                (points_q >= -2**(parsed_n_bits - 1)) & (points_q <= (2**(parsed_n_bits - 1) - 1))
            ):
                raise ValueError("Point cloud contains out or range values.")
            
            # Assign attributes
            self.points_f = None
            self.points_q = points_q.astype(np.int64)
            self.intensities = pc[:, 3] if not np.all(np.isnan(pc[:, 3])) else None
            self.normals = pc[:, 4:7] if not np.all(np.isnan(pc[:, 4:7])) else None
            self.n_points = pc.shape[0]
            self.n_bits = parsed_n_bits
        
        # Assign norm_range attribute (required to support dequantization)
        self.norm_range = norm_range

    def copy(self) -> PointCloud:
        return copy.deepcopy(self)

    def subset(self, indices: np.ndarray):
        pc = PointCloud.__new__(PointCloud)

        pc.points_f = self.points_f[indices, :]
        pc.points_q = self.points_q[indices, :]

        pc.normals = self.normals[indices, :]
        pc.intensities = self.intensities[indices]

        pc.n_points = len(indices)
        pc.n_bits = self.n_bits
        pc.norm_range = self.norm_range

        return pc
    
    def dequantize(self) -> np.ndarray:
        """
        Convert quantized point cloud coordinate values back to the original floating-point range.

        Reverses symmetric uniform quantization applied in the constructor.
        - Maps integer values back to [-1, 1]
        - Scales the original normalization range

        Returns:
            np.ndarray: Double precision floating point coordinate values of shape (`n_points`, 3).
        """
        return (
            ((self.points_q.astype(np.float64) / (2**(self.n_bits - 1) - 1)) + 1) / 2
        ) * (self.norm_range[1] - self.norm_range[0]) + self.norm_range[0]

    def downsample(self, n_points: int):
        """
        Downsample point cloud using farthest point sampling (FPS).
        
        Args:
            n_points (int): Desired number of point cloud point after downsampling.
        """
        centroids = np.zeros(n_points, dtype=int)
        distances = np.full(self.n_points, np.inf)

        # Deterministic first point: farthest from the center
        center = self.points_q.mean(axis=0)
        farthest = np.argmax(np.linalg.norm(self.points_q - center, axis=1))

        # Perform FPS for n_points
        for i in range(n_points):
            centroids[i] = farthest
            centroid = self.points_q[farthest]
            dist = np.sum((self.points_q - centroid) ** 2, axis=1)
            distances = np.minimum(distances, dist)
            farthest = np.argmax(distances)

        # Assign attributes
        self.points_f = self.points_f[centroids] if self.points_f is not None else None
        self.points_q = self.points_q[centroids]
        self.intensities = self.intensities[centroids] if self.intensities is not None else None
        self.normals = self.normals[centroids] if self.normals is not None else None
        self.n_points = n_points

    def write_csv(self, subpath: Path):
        """
        Write point cloud data to CSV. Written data includes both the original unmodified and the 
        quantized point cloud coordinate values along with unmodified double precision floating 
        point intensity and surface normal vector values.
        
        Args:
            subpath (Path): Path including filename (without extension) where the point cloud CSV
            file will be stored.
        """
        subpath.parent.mkdir(parents=True, exist_ok=True)
        path = Path(subpath.parent, f"{subpath.stem}-{self.n_points}_{self.n_bits}.csv")
        columns = [
            "x_f", "y_f", "z_f",
            "x_q", "y_q", "z_q",
            "intensity",
            "normal_x", "normal_y", "normal_z"
        ]

        intensities = (
            self.intensities
            if self.intensities is not None
            else np.full((self.n_points, 1), np.nan)
        )
        normals = (
            self.normals
            if self.normals is not None
            else np.full((self.n_points, 3), np.nan)
        )
        data = np.hstack([self.points_f, self.points_q, intensities, normals])

        # Create data frame and write to CSV
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(path, index=False)

    def write_mem(self, subpath: Path):
        """
        Write point cloud data to binary memory image file.

        Args:
            subpath (Path): Path including filename (without extension) where the point cloud
            memory image file will be stored.
        """
        subpath.parent.mkdir(parents=True, exist_ok=True)
        path = Path(subpath.parent, f"{subpath.stem}-{self.n_points}_{self.n_bits}.mem")

        with path.open("w") as f:
            for i, point in enumerate(self.points_q):
                # Convert each coordinate to n_bits binary
                line = "".join(np.binary_repr(val, width=self.n_bits) for val in point)
                f.write(line + "\n")

    def _parse_csv_filename(self, path: Path) -> Tuple[int, int]:
        """
        Parse the CSV point cloud filename to extract number of points and bit width.

        Expected format: `<name>-<n_points>_<n_bits>.csv`

        Args:
            path (str): Path to the CSV file.

        Returns:
            tuple: `n_points` and `n_bits` outlined below.

            **n_points** : *int*<br>
            Number of points parsed from the filename.

            **n_bits** : *int* <br>
            Bit width parsed from the filename.

        Raises:
            ValueError: If the filename does not match the expected pattern.
        """
        filename = path.stem  # remove the .csv suffix

        # Match pattern: anything-<n_points>_<n_bits>
        match = re.match(r".*-(\d+)_(\d+)$", filename)
        if not match:
            raise ValueError(f"Filename does not match expected pattern.")

        n_points = int(match.group(1))
        n_bits   = int(match.group(2))
        
        return n_points, n_bits
