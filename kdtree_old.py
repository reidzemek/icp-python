from __future__ import annotations
from typing import Optional, List, Tuple, Union
import numpy as np
from collections import deque, defaultdict
from pathlib import Path
import csv
import pandas as pd
from pointcloud import PointCloud

class KDTree:
    """k-dimensional (k-d) tree for exact nearest neighbor (nn) search.

    Args:
        target (PointCloud): Target (Q) point cloud object.

    ## Attributes
        **_nodes** : *tuple[KDTree._Node]*
        BFS-ordered nodes (immutable). Read-only after construction.<br>

        **size** : int
        <br>

        **max_depth** : int
        <br>

        **_visited_count** : **
        <br>

        **_down_count** : **
        <br>

        **_log** : *list*
        <br>

        **_log_q_nn** : *list[list[int]]*
        Nearest neighbors in Q for each point in the source point cloud (P).<br>

        **_log_leaf** : *list[list[int]]*
        List of lists of node indices for the leaf nodes of each downward pass during the nearest\
        neighbor search for a single query point.<br>
        
        **_log_best** : *list[list[tuple[int, float]]]*
        List of lists of tuples containing the best node index and the best node distance of each\
        downward pass during the nearest neighbor search for a single query point.<br>

        **_log_branch** : *list[list[list[int]]]*
        Same as `leaf_log` but this time, instead of just the index of a single leaf, there is\
        another list of all leaves traversed for the corresponding downward pass.)
    """

    class _Node:
        """A single node (corresponding to a single point) used in the construction of the
        k-dimensional (k-d) tree data structure for the 3-dimensional (3D) target point cloud.

        Attributes:
            idx (int): The index into PointCloud.
            axis (int): Splitting axis used at this node (0 = x, 1 = y, 2 = z).
            addr1 (int | None): Index of the first child node (from left to right) in the
                k-d tree array, or None if the node has no children.
            type (int): Node type (2 = binary, 1 = unary, 0 = leaf).
        """

        __slots__ = ("idx", "axis", "addr1", "type")

        def __init__(
                self,
                idx: int,
                axis: int,
                left_child: Optional[int],
                node_type: int
        ):
            """Initialize a k-d tree node."""

            self.idx = idx
            self.axis = axis
            self.addr1 = left_child
            self.type = node_type
    
    def __init__(self, source: Union[Path, np.ndarray], pc: Optional[np.ndarray] = None):
        """Built the k-d tree data structure using BFS ordering."""

        # Initialize the k-d tree node container
        self._nodes: tuple[KDTree._Node] # immutable

        # Tree metadata (computed during construction)
        self.size = 0
        self.max_depth = 0

        # Build k-d tree data structure from point cloud numpy array
        if isinstance(source, np.ndarray):
            self._pc = source
            self._build(source)

        # Load serialized k-d tree from CSV file
        elif isinstance(source, Path):
            if source.suffix.lower() == ".csv":
                if pc is None:
                    raise ValueError("pc array must be provided when loading KDTree from CSV.")
                self._pc = pc
                self._load_csv(source)
            else:
                raise ValueError(f"Unsupported file type: {source.suffix}")
        else:
            raise ValueError("Invalid source.")

        # Nearest neighbor search metadata initialization
        self._visited_count = 0
        self._down_count = 0

        # Initialize list for nearest neighbor search log
        self._log = []
        
    def nn_search(self, P: np.ndarray) -> np.ndarray:
        """Find the nearest neighbor indices in point cloud `Q` for each point in point cloud `P`.

        Args:
            P (np.ndarray): Source point cloud of shape (N, 3) containing N points to query.

        Returns:
            np.ndarray: Nearest neighbor indices into `PointCloud.points_q` of shape (N,).
        """

        # Array to store the nearest neighbor indices
        nn_idx = np.empty(P.shape[0], dtype=np.int64)

        # Reset log
        self._log = []

        # States
        DESCEND = 0
        BACKTRACK = 1

        # For each point in the source point cloud
        for i, query in enumerate(P):

            self._down_count = 0
            self._visited_count = 0

            # Branch stack: list of visited nodes from the root to the current node
            # [node index, perpendicular split distance squared, far child address]
            branch_stack = []

            # Best node found so far
            # [node index, distance squared]
            best = [0, float("inf")]

            # Initialize the current node and state
            node_idx = 0
            state = DESCEND

            while True:
                if state == DESCEND:
                    node_idx = self._descend(query, branch_stack, best, node_idx)
                    state = BACKTRACK

                elif state == BACKTRACK:
                    next_node = self._backtrack(query, branch_stack, best)

                    if next_node is None:
                        break               # search complete

                    node_idx = next_node    # need to search a far branch
                    state = DESCEND

            # Add nearest neighbor index to array
            nn_idx[i] = self._nodes[best[0]].idx

        return nn_idx

    def _descend(
        self,
        query: np.ndarray,
        branch_stack: List[Tuple[int, float, int]],
        best: List[int, float],
        node_idx: Optional[int]=0,
    ) -> None:
        """Descend the tree from a given node until a leaf node is reached, updating `best`
        along the way.
        
        This function iteratively traverses nodes starting from `node_idx`, updating `best` with
        the closest node found so far. All visited node indices are pushed onto `branch_stack`
        along with their split distance and far child, enabling efficient evaluation of per-node
        search flags after reaching the leaf.

        Args:
            query (np.ndarray): Query point of shape (3,).
            branch_stack (List[Tuple[int, float, int]]): Stack of visited nodes as tuples of
                (node index, perpendicular split distance squared, far child index).
            best (List[int, float]): Current best node index and its associated squared distance.
            node_idx (int): Index of the node where the descent starts. Defaults to root (0).
        """

        # Continue until leaf node is reached
        while True:

            self._visited_count += 1

            # Current node
            node = self._nodes[node_idx]
            node_point = self._pc[node.idx]
            dist_sq = np.sum((query - node_point) ** 2)

            # Update best if current node is closer
            if dist_sq < best[1]:
                best[:] = [node_idx, dist_sq]

            # Determine children
            if node.type == 2:      # for binary nodes
                if query[node.axis] < node_point[node.axis]:
                    near, far = node.addr1, node.addr1 + 1
                else:
                    near, far = node.addr1 + 1, node.addr1

            elif node.type == 1:    # for unary nodes
                near, far = node.addr1, None

            else:                   # for leaf nodes
                near = far = None

            # Calculate the squared splitting distance for the current node
            split_dist_sq = (query[node.axis] - node_point[node.axis]) ** 2

            # Logging
            self._log.append([
                node_idx,
                query[0], query[1], query[2],
                self._down_count,
                node_idx,
                node_point[0], node_point[1], node_point[2],
                node.type,
                dist_sq,
                split_dist_sq
            ])

            # Push current node to branch stack for potentian far-branch search during backtracking
            branch_stack.append((node_idx, split_dist_sq, far))

            # Leaf node reached
            if node.type == 0:
                break

            # Update the current node index
            node_idx = near

        self._down_count += 1

        return None  # signal BACKTRACK

    def _backtrack(
        self,
        query: np.ndarray,
        branch_stack: List[Tuple[int, float, int]],
        best: List[int, float],
    ) -> Optional[int]:
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
        while branch_stack:

            # Pop current branch node
            node_idx, split_dist_sq, far = branch_stack.pop()

            # Continue if unary node or search of far branch is not indicated by search flag
            if far is None or best[1] < split_dist_sq:
                continue

            # go back to DESCEND and explore far branch
            return far

        return None  # finished

    def _build(self, pc: np.ndarray) -> None:
        """Construct a k-d tree using BFS ordering.

        Splits at the median along alternating axes (x = 0, y = 1, z = 2) recursively.
        Stores the tree as a tuple of Node objects for immutability.

        Args:
            pc (np.ndarray): Point cloud array.
        """

        # Initialize node container (empty list to hold BFS-ordered tree)
        node_list: list[Optional[KDTree._Node]] = []

        # BFS Queue holds: (point indices, depth, node index)
        queue = deque()

        # Seed the BFS queue
        indices = np.arange(pc.shape[0])    # target point cloud indices
        queue.append((indices, 0, 0))       # start with the entire point cloud

        # Reserve root in node list
        node_list.append(None)

        self.max_depth = 0

        # Fill nodes in BFS order
        while queue:

            # Current sub-tree
            indices, depth, node_idx = queue.popleft()

            # Empty child
            if len(indices) == 0:
                continue

            if depth > self.max_depth:
                self.max_depth = depth

            # Cycle through splitting axes for each layer starting with x (0)
            axis = depth % 3

            # Sort points along current axis and find median
            pts = pc[indices]
            sorted_idx = np.argsort(pts[:, axis])
            median_local = len(indices) // 2
            median_idx = indices[sorted_idx[median_local]] # map to global index for current node

            # Left and right sub-trees either side of median split
            left_indices = indices[sorted_idx[:median_local]]
            right_indices = indices[sorted_idx[median_local + 1:]]

            # Determine node type & reserve children
            if len(left_indices) > 0 and len(right_indices) > 0:
                node_type = 2  # binary

                # Reserve index in nodes list for left and right child nodes
                left_idx = len(node_list)
                node_list.append(None)
                right_idx = len(node_list)
                node_list.append(None)

            elif len(left_indices) > 0 or len(right_indices) > 0:
                node_type = 1  # unary

                # Reserve index in nodes list for a single child node
                left_idx = len(node_list)
                node_list.append(None)
                right_idx = None

            else:
                node_type = 0  # leaf

                # No children
                left_idx = None
                right_idx = None

            # Create node: add top node of current sub-tree to BFS ordered tree list
            node_list[node_idx] = self._Node(
                idx=median_idx,
                axis=axis,
                left_child=left_idx,
                node_type=node_type
            )

            # Enqueue children (BFS order)
            if len(left_indices) > 0:
                queue.append((left_indices, depth + 1, left_idx))
            if len(right_indices) > 0:
                queue.append((right_indices, depth + 1, right_idx))

        # Store BFS-ordered nodes list as an immutable tuple
        self._nodes = tuple(node_list)
        self.size = len(self._nodes)

    def get_search_trace(self):
        """
        Get the nearest neighbor search trace.

        Returns:
            pd.DataFrame: DataFrame with the following headings.

            *Query idx. (P) | p_x | p_y | p_z | Trav. ID | Node addr. (Q_tree) | q_x | q_y | q_z |
            Node Type | Euclid. dist. sq. | Split dist. sq | Search Flag | Trav. Best*.
        """

        # Column labels
        columns = [
            "Query idx. (P)",
            "p_x", "p_y", "p_z",
            "Trav. ID",
            "Node addr. (Q_tree)",
            "q_x", "q_y", "q_z",
            "Node Type",
            "Euclid. dist. sq.",
            "Split dist. sq."
        ]

        # Log data
        log_df = pd.DataFrame.from_records(self._log, columns=columns)

        # Minimum euclidean distance per query-traversal, broadcast to all visited nodes (rows)
        trav_min = log_df.groupby(
            ["Query idx. (P)", "Trav. ID"]
        )["Euclid. dist. sq."].transform("min")

        # Evaluate search flag using best node so far
        log_df["Search Flag"] = (trav_min > log_df["Split dist. sq."]).astype(int)

        # Evaluate best flag per query-traversal
        log_df["Trav. Best"] = (log_df["Euclid. dist. sq."] == log_df.groupby(
            ["Query idx. (P)", "Trav. ID"]
        )["Euclid. dist. sq."].transform("min")).astype(int)

        return log_df

    def write_csv(self, path: Path):
        """Write the k-d tree structure to CSV.

        Args:
            path (Path): Output file path.
        """

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(mode='w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "idx",
                "axis",
                "addr1",
                "type"
            ])

            for node in self._nodes:
                writer.writerow([
                    node.idx,
                    node.axis,
                    node.addr1 if node.addr1 is not None else "",
                    node.type
                ])
    
    def write_mem(self, path: Path, n_coord_bits: int, addr_width: int):
        """Write the k-d tree to a binary text format memory image file.

        Each line represents one node:
        <x><y><z><unary_flag><axis><addr1>

        - Coordinates are signed integers encoded in two's complement (from PointCloud).
        - unary_flag: 1 if node.type == 1, else 0
        - axis: 0=x, 1=y, 2=z, 3=leaf
        - addr1 NULL = 0

        Args:
            path (Path): Output file path.
            n_coord_bits (int): Bit width for coordinates.
            addr_width (int): Bit width for addresses.
        """

        path.parent.mkdir(parents=True, exist_ok=True)

        # TODO There is a problem with this writing the float tree when the bit width is 2
        def to_twos_complement(val: int, bits: int) -> str:
            if val < 0:
                val = (1 << bits) + val
            return format(val, f"0{bits}b")

        def to_bin(val: int, bits: int) -> str:
            return format(val, f"0{bits}b")

        with path.open("w") as f:
            for node in self._nodes:

                # fetch coordinates via index
                x, y, z = self._pc[node.idx]

                # coordinates
                x_bin = to_twos_complement(int(x), n_coord_bits)
                y_bin = to_twos_complement(int(y), n_coord_bits)
                z_bin = to_twos_complement(int(z), n_coord_bits)

                # unary flag
                unary_flag = "1" if node.type == 1 else "0"

                # axis encoding
                axis_val = 3 if node.type == 0 else node.axis
                axis_bin = format(axis_val, "02b")

                # addr1
                addr1_val = node.addr1 if node.addr1 is not None else 0
                addr1_bin = to_bin(addr1_val, addr_width)

                # write fixed-width binary line
                f.write(f"{x_bin}{y_bin}{z_bin}{unary_flag}{axis_bin}{addr1_bin}\n")

    def _load_csv(self, path: Path):
        """Load the serialized k-d tree structure from CSV.

        Expected columns:
            idx, axis, addr1, type

        Args:
            path (Path): Path to the serialized k-d tree CSV.
        """

        nodes = []

        with path.open(mode='r', newline='') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Required fields
                idx = int(row["idx"])
                axis = int(row["axis"])
                node_type = int(row["type"])

                # Optional child address
                addr1 = int(row["addr1"]) if row["addr1"] != "" else None

                # Create node (structure only)
                node = KDTree._Node(
                    idx=idx,
                    axis=axis,
                    left_child=addr1,
                    node_type=node_type
                )

                nodes.append(node)

        # Finalize
        self._nodes = tuple(nodes)
        self.size = len(self._nodes)

        print(f"Loaded {self.size} nodes from {path}")
