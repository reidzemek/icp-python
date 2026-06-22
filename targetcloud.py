from kdtree import KDTree
from pointcloud import PointCloud
from pathlib import Path
from typing import Tuple, Optional

class TargetCloud:
    """
    Target point cloud and k-d tree container.

    Args:
        path (str): Path to the target point cloud file. Accepts PCD of CSV.
            - **PCD files:** Contain raw point cloud data. The k-d tree will be built from scratch.
            Requires `norm_range` and `n_bits` for min-max quantization.
            - **CSV files:** Must be pre-quantized, with the filename format
            `<name>-<n_points>_<n_bits>.csv`. The bit width (`n_bits`) is parsed from the filename
            and validated. `norm_range` should be included to support `dequantize()`. This expects
            a companion BFS-ordered tree file named `<name>-<n_points>_<n_bits>-tree.csv`.
        norm_range (Tuple[float, float], optional): Minimum and maximum values of the coordinate 
            range used for min-max quantization. Required for both CSV and PCD formats.
        n_bits (int, optional): Bit width for quantized coordinates. Required for PCD files.

    ## Attributes:
        **point_cloud** : *PointCloud*
        Point cloud data.<br>

        **tree** : *KDTree*
        k-d tree datastructure.
    """

    def __init__(
        self,
        path: Path,
        norm_range: Tuple[float, float],
        n_bits: Optional[int],
        n_points: Optional[int]=None
    ):
        """
        Initialize target cloud by loading points and resolving the k-d tree.
        """

        # Load target (PCD or CSV file)
        self.point_cloud = PointCloud(path, norm_range, n_bits)

        # KDTree csv filenames
        tree_f_csv = (
            f"{path.stem}_tree_f-{self.point_cloud.n_points}_{self.point_cloud.n_bits}.csv"
        )
        tree_q_csv = (
            f"{path.stem}_tree_q-{self.point_cloud.n_points}_{self.point_cloud.n_bits}.csv"
        )

        # Load tree (from CSV for CSV point clouds)
        if path.suffix.lower() == ".csv":
            self.tree_f = KDTree(target=path.parent / tree_f_csv, pc=self.point_cloud.points_f)
            self.tree_q = KDTree(target=path.parent / tree_q_csv, pc=self.point_cloud.points_q)
        else:
            self.point_cloud.downsample(n_points) if n_points is not None else None
            self.tree_f = KDTree(self.point_cloud.points_f)
            self.tree_q = KDTree(self.point_cloud.points_q)

    def nearest(self, P: PointCloud):
        """
        Returns a list of the nearest points in `self` for each point in `P`.

        Args:
            P (PointCloud): Source point cloud.
            mode (str): Representation used for nearest neighbor search, either "quantized" or
                "float". Default is "quantized".

        Returns:
            Q_nearest (PointCloud): Point cloud of the nearest points in target point cloud (Q).
        """

        # Find the indices for the nearest points in point cloud Q
        nearest_indices_f = self.tree_f.nn_search(P.points_f)
        nearest_indices_q = self.tree_q.nn_search(P.points_q)

        # nearest point clouds
        nearest_f = self.point_cloud.subset(nearest_indices_f)
        nearest_q = self.point_cloud.subset(nearest_indices_q)

        return nearest_f, nearest_q

    def write(self, subpath: Path, addr_width: int):
        """
        Write target point cloud data along with its and tree data to csv and binary text format
        memory image files.

        Args:
            subpath (Path): Path including filename (without extension) where the data files will
            be stored.
        """
        subpath.parent.mkdir(parents=True, exist_ok=True)
        path_pc = Path(subpath.parent, subpath.stem)
        path_tree_f = Path(
            subpath.parent,
            f"{subpath.stem}_tree_f-{self.point_cloud.n_points}_{self.point_cloud.n_bits}"
        )
        path_tree_q = Path(
            subpath.parent,
            f"{subpath.stem}_tree_q-{self.point_cloud.n_points}_{self.point_cloud.n_bits}"
        )

        # Write point cloud data files
        self.point_cloud.write_csv(Path(f"{path_pc}.csv"))
        self.point_cloud.write_mem(Path(f"{path_pc}.mem"))

        # Write k-d tree data files
        self.tree_f.write_csv(Path(f"{path_tree_f}.csv"))
        self.tree_q.write_csv(Path(f"{path_tree_q}.csv"))
        self.tree_q.write_mem(Path(f"{path_tree_q}.mem"), self.point_cloud.n_bits, addr_width)
