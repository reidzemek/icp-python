from __future__ import annotations
from typing import Optional, List, Tuple, Union
import numpy as np
from collections import deque, defaultdict
from pathlib import Path
import csv
from pypcd4 import PointCloud
import pandas as pd

class KDTree:
    """k-dimensional (k-d) tree for exact nearest neighbor (nn) search.

    Args:
        Q (np.ndarray): Target point cloud of shape (M, 3) containing M points.

    ## Attributes:
        **_nodes** : *tuple[KDTree._Node]*
        BFS-ordered nodes (immutable). Read-only after construction.<br>

        **log_q_nn** : *list[list[int]]*
        Nearest neighbors in Q for each point in the source point cloud (P).<br>

        **log_leaf** : *list[list[int]]*
        List of lists of node indices for the leaf nodes of each downward pass during the nearest\
        neighbor search for a single query point.<br>
        
        **log_best** : *list[list[tuple[int, float]]]*
        List of lists of tuples containing the best node index and the best node distance of each\
        downward pass during the nearest neighbor search for a single query point.<br>

        **log_branch** : *list[list[list[int]]]*
        Same as `leaf_log` but this time, instead of just the index of a single leaf, there is\
        another list of all leaves traversed for the corresponding downward pass.)
    """

    class _Node:
        """A single node (corresponding to a single point) used in the construction of the
        k-dimensional (k-d) tree data structure for the 3-dimensional (3D) target point cloud.

        Attributes:
            point (tuple[float, float, float]): The (x, y, z) coordinate values.
            normal (tuple[float, float, float] | None): The surface normal, or None if 
                normals were not provided when building the tree.
            axis (int): Splitting axis used at this node (0 = x, 1 = y, 2 = z).
            addr1 (int | None): Index of the first child node (from left to right) in the
                k-d tree array, or None if the node has no children.
            addrP (int): Index of the parent node in the k-d tree array.
            type (int): Node type (2 = binary, 1 = unary, 0 = leaf).
        """

        __slots__ = ("point", "normal", "axis", "addr1", "addrP", "type")

        def __init__(
                self,
                point: Tuple[float, float, float],
                normal: Optional[Tuple[float, float, float]],
                axis: int,
                left_child: int | None,
                parent: int | None,
                node_type: int
        ):
            """Initialize a k-d tree node."""

            self.point = point
            self.normal = normal
            self.axis = axis
            self.addr1 = left_child
            self.addrP = parent
            self.type = node_type
    
    def __init__(self, target: Union[Path, np.ndarray]):
        """Built the k-d tree data structure using BFS ordering."""

        # Initialize the k-d tree node container
        self._nodes: tuple[KDTree._Node] # immutable

        # Tree metadata
        self.max_depth = 0 # computed during construction

        # Build k-d tree data structure from point cloud numpy array
        if isinstance(target, np.ndarray):
            self._build(target)

        # Load point cloud from file
        else:

            # Load target point cloud (Q) from pcd file
            if target.suffix.lower() == ".pcd":
                Q = PointCloud.from_path(str(target)).numpy(("x", "y", "z"))
                self._build(Q)

            # Load serialized k-d tree from csv file
            elif target.suffix.lower() == ".csv":
                self._load_from_csv(target)

            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")

        # Tree metadata
        self.size = len(self._nodes)

        # Nearest neighbor search metadata initialization
        self._visited_count = 0
        self._down_count = 0

        # Initialize lists for nearest neighbor search logging
        self._log_q_nn = []
        self._log_leaf = []
        self._log_best = []
        self._log_branch = []

    def nn_search(self, P: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Find the nearest neighbors for each point in point cloud `P`.

        Args:
            P (np.ndarray): Source point cloud of shape (N, 3) containing N points to query.

        Returns:
            tuple: `Q_nn` and `Optional[N_nn]` outlined below.

            **Q_nn** : *np.ndarray of shape (N, 3)*<br>
            Nearest neighbor points in Q for each point in P.

            **N_nn** : *np.ndarray of shape (N, 3) | None*<br>
            Surface normals corresponding to each point in `Q_nn` or None for trees
            without surface normal vectors.
        """

        # Initialize arrays to store the nearest point clouds and corresponding normal vectors
        Q_nn = np.empty_like(P)
        N_nn = np.empty_like(P) if self._nodes[0].normal is not None else None

        # Reset logging
        self._log_leaf = []
        self._log_best = []
        self._log_branch = []

        # For each point in the source point cloud
        for idx, query in enumerate(P):

            # Append an empty list to leaf and best log lists
            self._log_leaf.append([])
            self._log_best.append([])
            self._log_branch.append([])

            self._down_count = 0

            # Append 
            branch_stack = []
            best = [0, float("inf")]
            self._descend(query, branch_stack, best, idx)
            Q_nn[idx] = self._nodes[best[0]].point
            if N_nn is not None:
                N_nn[idx] = self._nodes[best[0]].normal

        # Write point cloud of nearest target points to nearest targets log
        self._log_q_nn = Q_nn.tolist()

        return Q_nn, N_nn

    def _descend(
        self,
        query: np.ndarray,
        branch_stack: List[Tuple[int, float, int]],
        best: List[int, float],
        point_idx: int,
        node_idx: Optional[int]=0,
    ):
        """Descend the tree from a given node until a leaf node is reached, updating `best`
        along the way.
        
        This function iteratively traverses nodes starting from `node_idx`, updating `best` with
        the closest node found so far. All visited node indices are pushed onto `branch_stack`
        along with their split distance and far child, enabling efficient evaluation of per-node
        search flags after reaching the leaf.

        Args:
            query (np.ndarray): Query point of shape (3,).
            branch_stack (List[Tuple[int, float, int]]): Stack of visited nodes as tuples of
                (node index, split distance, far child index).
            best (List[int, float]): Current best node index and its associated squared distance.
            node_idx (int): Index of the node where the descent starts. Defaults to root (0).
        """

        self._log_branch[point_idx].append([])
        
        # Continue until leaf node is reached
        while True:

            self._log_branch[point_idx][self._down_count].append(node_idx)

            self._visited_count += 1

            # Current node
            node = self._nodes[node_idx]
            node_point = np.array(node.point)
            dist_sq = np.sum((query - node_point) ** 2)

            # Update best if current node is closer
            if dist_sq < best[1]:
                best[:] = [node_idx, dist_sq]

            # For binary nodes
            if node.type == 2: # binary node
                # Select the near and far children based on the query relative to the split axis
                if query[node.axis] < node_point[node.axis]:
                    near, far = node.addr1, node.addr1 + 1
                else:
                    near, far = node.addr1 + 1, node.addr1

            # For unary nodes
            elif node.type == 1: # unary node
                near, far = node.addr1, None

            # For leaf nodes
            else:
                near = far = None

            # Calculate the squared splitting distance for the current node
            split_dist_sq= (query[node.axis] - node_point[node.axis]) ** 2

            # Push current node to branch stack for potential far-branch search during backtracking
            branch_stack.append((node_idx, split_dist_sq, far))

            temp = node_idx

            # Update the current node index
            node_idx = near

            # Leaf node reached
            if node.type == 0:
                break

        self._down_count += 1

        # Add the leaf node and best node addresses for the current downward pass to the leaf log
        self._log_leaf[point_idx].append(temp)
        self._log_best[point_idx].append(best[0])

        # Begin backward pass towards the root
        self._backtrack(query, branch_stack, best, point_idx)

    def _backtrack(
        self,
        query: np.ndarray,
        branch_stack: List[Tuple[int, float, int]],
        best: List[int, float],
        point_idx: int
    ):
        """Backtrack through the branch stack and explore far branches where required, updating
        `best` along the way.

        For each node in `branch_stack`, this function evaluates a search flag and, if indicated,
        descends down the corresponding far branch. The process continues until all relevant far
        branches have been explored, ending back at the root node.

        Args:
            query (np.ndarray): Query point of shape (3,).
            branch_stack (List[Tuple[int, float, int]]): Stack of visited nodes as tuples of
                (node index, split distance, far child index).
            best (List[int, float]): Current best node index and its associated squared distance.
        """
        
        # Continue until back at root
        while len(branch_stack) > 0:

            # Pop current branch node
            branch_node = branch_stack.pop()

            # Continue if search of far branch is not indicated by search flag
            if best[1] < branch_node[1] or branch_node[2] is None:
                continue

            # If search is indicated, descend far branch down to leaf
            self._descend(query, branch_stack, best, point_idx, branch_node[2])

    def _build(self, Q: np.ndarray, Q_N: Optional[np.ndarray]=None) -> Tuple[_Node, ...]:
        """Construct a k-d tree from a point cloud using BFS ordering.

        Splits at the median along alternating axes (x=0, y=1, z=2) recursively.
        Returns the tree as a tuple of Node objects for immutability.

        Args:
            Q (np.ndarray): Array of shape (M, 3) containing M points.
            Q_N (np.ndarray | None): Array of shape (M, 3) containing M normal vectors or 
                None if building a tree without surface normal vectors.

        Returns:
            tuple: BFS ordered k-d tree as a tuple of Node objects.
        """

        if Q.ndim != 2 or Q.shape[1] != 3:
            raise ValueError("Q must have shape (M, 3).")

        if Q_N is not None and Q.shape != Q_N.shape:
            raise ValueError("Q and Q_N must have the same shape (M, 3).")
        
        # Initialize an empty list to hold the BFS ordered tree
        nodes_list: list[KDTree._Node] = []

        # Double-ended queue to aid in constructing the BFS ordered tree
        queue = deque()

        # Seed the BFS queue: (points_subset, normals_subset, depth, parent_idx, node_idx)
        queue.append((Q, Q_N, 0, None, 0))

        # Initialize root index in nodes list
        nodes_list.append(None)

        # Fill nodes in BFS order
        while queue:

            # Current sub-tree
            points, normals, depth, parent, node_idx = queue.popleft()

            # Empty child
            if points.shape[0] == 0:
                continue

            # Cycle through splitting axes for each layer starting with x (0)
            axis = depth % 3

            # Sort points along current axis and find median
            sorted_idx = np.argsort(points[:, axis])
            median_idx = len(points) // 2

            # Select median point coordinates and surface normal vector if using
            median_point = points[sorted_idx[median_idx]]
            median_normal = normals[sorted_idx[median_idx]] if normals is not None else None

            # Left and right sub-trees
            left_points = points[sorted_idx[:median_idx]]
            right_points = points[sorted_idx[median_idx + 1:]]
            left_normals = normals[sorted_idx[:median_idx]] if normals is not None else None
            right_normals = normals[sorted_idx[median_idx + 1:]] if normals is not None else None

            # Evaluate node type
            if left_points.shape[0] > 0 and right_points.shape[0] > 0:
                node_type = 2 # binary

                # Reserve index in nodes list for left and right child nodes
                left_idx = len(nodes_list)
                nodes_list.append(None)
                right_idx = len(nodes_list)
                nodes_list.append(None)

            elif left_points.shape[0] > 0 or right_points.shape[0] > 0:
                node_type = 1 # unary

                # Reserve index in nodes list for single child
                left_idx, right_idx = len(nodes_list), None
                nodes_list.append(None)
            else:
                node_type = 0 # leaf

                # No children
                left_idx = right_idx = None

            # Add top node of current sub-tree to BFS ordered tree list
            nodes_list[node_idx] = self._Node(
                point=tuple(median_point),
                normal=tuple(median_normal) if median_normal is not None else None,
                axis=axis,
                left_child=left_idx,
                parent=parent,
                node_type=node_type
            )

            # Enqueue (add to the end of the queue) children in BFS order
            if left_points.shape[0] > 0:
                queue.append((left_points, left_normals, depth + 1, node_idx, left_idx))
            if right_points.shape[0] > 0:
                queue.append((right_points, right_normals, depth + 1, node_idx, right_idx))

        # Store BFS-ordered nodes list as an immutable tuple
        self._nodes = tuple(nodes_list)

        # Store the max depth of the tree
        self.max_depth = depth + 1

    def write_search_trace(self, path: Path, n_P, n_Q, n_coord_bits):
        """Write the nearest neighbor search trace logs to csv.

        Args:
            path (Path): Directory path where to store the logging data.
        """

        # Write leaf nodes
        leaf_nodes_dir = Path(path, f"leaf_nodes-{n_P}_{n_Q}_{n_coord_bits}")
        leaf_nodes_dir.mkdir(parents=True, exist_ok=True)
        leaf_df = pd.DataFrame(self._log_leaf)
        for down_pass in leaf_df.columns:
            coords = []
            for node_idx in leaf_df[down_pass]:
                if pd.isna(node_idx):
                    coords.append([np.nan, np.nan, np.nan])
                else:
                    coords.append(self._nodes[int(node_idx)].point)
            df = pd.DataFrame(coords, columns=["x", "y", "z"])
            csv_path=Path(leaf_nodes_dir, f"leaf_nodes-{down_pass}.csv")
            df.to_csv(csv_path, index=False, na_rep=" ")


        # Write best nodes
        best_nodes_dir = Path(path, f"best_nodes-{n_P}_{n_Q}_{n_coord_bits}")
        best_nodes_dir.mkdir(parents=True, exist_ok=True)
        best_df = pd.DataFrame(self._log_best)
        for down_pass in best_df.columns:
            coords = []
            for node_idx in best_df[down_pass]:
                if pd.isna(node_idx):
                    coords.append([np.nan, np.nan, np.nan])
                else:
                    coords.append(self._nodes[int(node_idx)].point)
            df = pd.DataFrame(coords, columns=["x", "y", "z"])
            csv_path = Path(best_nodes_dir, f"best_nodes-{down_pass}.csv")
            df.to_csv(csv_path, index=False, na_rep=" ")

        # Write branch nodes
        branches_dir = Path(path, f"branches-{n_P}_{n_Q}_{n_coord_bits}")
        branch_nodes_dir = Path(branches_dir)
        # branch_nodes_dir.mkdir(parents=True, exist_ok=True)
        
        n_points = len(self._log_branch)
        grouped = defaultdict(lambda: [[] for _ in range(n_points)])

        for i, query_point in enumerate(self._log_branch):
            for j, down_pass in enumerate(query_point):
                grouped[j][i] = down_pass  # assign instead of append

        for down_idx, branch_nodes in grouped.items():
            branch_dir = Path(branch_nodes_dir, f"pass_{down_idx}")
            branch_dir.mkdir(parents=True, exist_ok=True)
            branch_nodes_df = pd.DataFrame(branch_nodes)
            for i, row in branch_nodes_df.iterrows():
                coords = []
                for node_idx in row:
                    if pd.isna(node_idx):
                        coords.append([np.nan, np.nan, np.nan])
                    else:
                        coords.append(self._nodes[int(node_idx)].point)
                df = pd.DataFrame(coords, columns=["x", "y", "z"])
                csv_path = Path(branch_dir, f"point_{i}.csv")
                df.to_csv(csv_path, index=False, na_rep=" ")


            # print(branch_nodes_df)

        # branches_df = pd.DataFrame(self._log_branch)
        # print(self._log_branch)

        # leaf_df.to_csv(leaf_nodes_path, index=False)

        # best_nodes_path = Path(path, f"best_nodes-{n_P}_{n_Q}_{n_coord_bits}.csv")
        # best_df = pd.DataFrame(self._log_best)
        # best_df.to_csv(best_nodes_path, index=False)

    def write_tree(self, path: Path):
        """Write the target point cloud k-d tree to csv.

        Args:
            path (Path): Path to the k-d tree data structure memory file.
        """

        print(f"Writing k-d tree to: {path}")
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(mode='w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "x", "y", "z",
                "nx", "ny", "nz",
                "axis",
                "addr1",
                "addrP",
                "type"
            ])
            
            for node in self._nodes:
                x, y, z = node.point
                if node.normal is None:
                    nx, ny, nz = "", "", ""
                else:
                    nx, ny, nz = node.normal
                
                writer.writerow([
                    x, y, z,
                    nx, ny, nz,
                    node.axis,
                    node.addr1 if node.addr1 is not None else "",
                    node.addrP if node.addrP is not None else "",
                    node.type
                ])
    
    def write_tree_bin(self, path: Path, n_coord_bits: int, addr_width: int):
        """Write the target k-d tree to a binary text format.

        Each line represents one node:
        <x><y><z><unary_flag><axis><addr1><addrP>

        - Coordinates are signed integers encoded in two's complement.
        - unary_flag: 1 if node.type == "unary", else 0
        - axis: 0=x, 1=y, 2=z, 3=leaf
        - addr1 NULL = 0
        - addrP NULL = max address
        - last line ends with ';', others with ','

        Args:
            path (Path): Output file path.
            n_coord_bits (int): Bit width for coordinates.
            addr_width (int): Bit width for addresses.
        """

        print(f"Writing k-d tree (binary) to: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        MAX_ADDR = (1 << addr_width) - 1

        def to_twos_complement(val: int, bits: int) -> str:
            """Convert signed integer to fixed-width two's complement binary."""
            if val < 0:
                val = (1 << bits) + val
            return format(val, f"0{bits}b")

        def to_bin(val: int, bits: int) -> str:
            """Convert unsigned integer to fixed-width binary."""
            return format(val, f"0{bits}b")

        with path.open("w") as f:
            n_nodes = len(self._nodes)

            for i, node in enumerate(self._nodes):
                x, y, z = node.point

                # coordinates
                x_bin = to_twos_complement(int(x), n_coord_bits)
                y_bin = to_twos_complement(int(y), n_coord_bits)
                z_bin = to_twos_complement(int(z), n_coord_bits)

                # unary flag
                unary_flag = "1" if node.type == 1 else "0"

                # axis encoding
                if node.type == 0:
                    axis_val = 3
                else:
                    axis_val = node.axis

                axis_bin = format(axis_val, "02b")

                # addresses
                addr1_val = node.addr1 if node.addr1 is not None else 0
                addrP_val = node.addrP if node.addrP is not None else MAX_ADDR

                addr1_bin = to_bin(addr1_val, addr_width)
                addrP_bin = to_bin(addrP_val, addr_width)

                # separator
                sep = ";" if i == n_nodes - 1 else ","

                f.write(
                    f"{x_bin}{y_bin}{z_bin}{unary_flag}{axis_bin}{addr1_bin}{addrP_bin}{sep}\n"
                )

    def _load_from_csv(self, path: Path):
        """Load the serialized target point cloud k-d tree.
        
        Args:
            path (Path): Path to the serialized k-d tree data structure in csv format.
        """
        nodes = []

        with path.open(mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert point
                point = (float(row["x"]), float(row["y"]), float(row["z"]))

                # Convert normal, handle empty strings
                nx, ny, nz = row["nx"], row["ny"], row["nz"]
                if nx == "" or ny == "" or nz == "":
                    normal: Optional[tuple[float, float, float]] = None
                else:
                    normal = (float(nx), float(ny), float(nz))

                # Convert optional integer addresses
                addr1 = int(row["addr1"]) if row["addr1"] != "" else None
                addrP = int(row["addrP"]) if row["addrP"] != "" else None

                # Axis and type
                axis = int(row["axis"])
                node_type = int(row["type"])

                # Create _Node
                node = KDTree._Node(point, normal, axis, addr1, addrP, node_type)
                nodes.append(node)

        self._nodes = tuple(nodes)
        print(f"Loaded {len(self._nodes)} nodes from {path}")