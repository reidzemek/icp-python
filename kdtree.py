from typing import Optional, Tuple
import numpy as np

class Node:
    """A single node (corresponding to a single point) used in the construction of a
    k-dimensional (k-d) tree data structure for a 3-dimensional (3D) point cloud.

    Attributes:
        point (tuple[float, float, float]): The (x, y, z) coordinates.
        normal (tuple[float, float, float] | None): The surface normal, or None if 
            normals were not provided when building the tree.
        axis (int): Splitting axis used at this node (0 = x, 1 = y, 2 = z).
        left (int | None): Index of the left child node in the k-d tree array, or None
            if this node has no left child.
        right (int | None): Index of the right child node in the k-d tree array, or
            None if this node has no right child.
    """

    __slots__ = ("point", "normal", "axis", "left", "right")

    def __init__(
            self,
            point: Tuple[float, float, float],
            normal: Optional[Tuple[float, float, float]],
            axis: int,
            left: int | None,
            right: int | None
    ):
        """Initialize a k-d tree node."""

        self.point = point
        self.normal = normal
        self.axis = axis
        self.left = left
        self.right = right

def build(Q: np.ndarray, Q_N: Optional[np.ndarray]=None) -> Tuple[Node, ...]:
    """Construct a pre-ordered k-d tree from a point cloud.

    Splits at the median along alternating axes (x=0, y=1, z=2) recursively.
    Returns the tree as a tuple of nodes for immutability.

    Args:
        Q (np.ndarray): Array of shape (M, 3) containing M points.
        Q_N (np.ndarray | None): Array of shape (M, 3) containing M normal vectors or 
            None if building a tree without surface normal vectors.

    Returns:
        tuple: Pre-ordered k-d tree as a tuple of Node objects.
    """    
    if Q.ndim != 2 or Q.shape[1] != 3:
        raise ValueError("Q must have shape (M, 3).")

    if Q_N is not None and Q.shape != Q_N.shape:
        raise ValueError("Q and Q_N must have the same shape (M, 3).")

    nodes: list[Node] = [None] * Q.shape[0]
    idx = 0

    def _build(
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
        depth: int=0
    ) -> Optional[int]:
        """Recursive helper to fill nodes list in pre-order sequence.

        Args:
            points (np.ndarray): Subset of points from which to build a tree.
            normals (np.ndarray | None): Subset of surface normals for each point in
                `points` or None if building a tree without surface normal vectors.
            depth (int): Current depth in tree (to determine axis).

        Returns:
            Optional[int]: Index of this node in the nodes list or None for leaf nodes.
        """
        nonlocal idx

        # Empty child
        if points.shape[0] == 0:
            return None
        
        # Cycle through splitting axes for each layer starting with x (0)
        axis = depth % 3

        # Sort points along current axis and find median
        sorted_idx = np.argsort(points[:, axis])
        median_idx = len(points) // 2

        # Select median point coordinates and surface normal vector if using
        median_point = points[sorted_idx[median_idx]]
        median_normal = normals[sorted_idx[median_idx]] if normals is not None else None

        # Reserve index in nodes list for this node
        node_idx = idx
        idx += 1

        # Recursively build the left and right subtrees
        left_child = _build(
            points[sorted_idx[:median_idx]],
            normals[sorted_idx[:median_idx]] if normals is not None else None,
            depth + 1
        )
        right_child = _build(
            points[sorted_idx[1 + median_idx:]],
            normals[sorted_idx[1 + median_idx:]] if normals is not None else None,
            depth + 1
        )

        # Fill node in pre-allocated list
        nodes[node_idx] = Node(
            point=tuple(median_point),
            normal=tuple(median_normal) if median_normal is not None else None,
            axis=axis,
            left=left_child,
            right=right_child
        )

        # Return the tree index of the current node
        return node_idx
    
    # Recursively construct the tree starting from the root
    _build(Q, Q_N, depth=0)

    # Return the tree as a tuple
    return tuple(nodes)

def nn_search(
    tree: Tuple[Node, ...],
    P: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Find nearest neighbors in `tree` (k-d tree) for each point in `P` (point cloud).

    Args:
        P (np.ndarray): Array of shape (N, 3) containing N points to query.
        tree (tuple[Node, ...]): k-d tree represented as a pre-ordered tuple of Node
            objects, as returned by `kdtree.build`.

    Returns:
        tuple: `Q_nearest` and `Optional[Q_nearest_N]` outlined below.

        **Q_nearest** : *np.ndarray of shape (N, 3)*<br>
        Nearest neighbor points in Q for each point in P.

        **Q_nearest_N** : *np.ndarray of shape (N, 3) | None*<br>
        Surface normals corresponding to each point in `Q_nearest` or None for trees
        without surface normal vectors.
    """
    Q_nearest = np.empty_like(P)
    Q_nearest_N = np.empty_like(P) if tree[0].normal is not None else None

    def _nn_search(
        point:np.ndarray,
        node_idx: int,
        best: Tuple[int, float]
    ) -> Tuple[int, float]:
        """Recursive helper to find the nearest neighbor of a single point.

        Args:
            point (np.ndarray): Query point of shape (3,).
            node_idx (int): Index of the current node in the tree.
            best (Tuple[int, float]): Tuple containing the index of the best node
                found so far and its squared distance.

        Returns:
            best (Tuple[int, float]): Updated best node index and squared distance.
        """

        # Leaf node
        if node_idx is None:
            return best
        
        # Current node
        node = tree[node_idx]
        node_point = np.array(node.point)
        dist_sq = np.sum((point - node_point) ** 2)

        # Update best if current node is closer
        if dist_sq < best[1]:
            best = (node_idx, dist_sq)

        # Check on which side of the current nodes splitting axis the point lies
        if point[node.axis] < node_point[node.axis]:
            near, far = node.left, node.right           # near (or same) side
        else:
            near, far = node.right, node.left           # far (or opposite) side

        # Explore the near sub-tree
        best = _nn_search(point, near, best)

        # Determine if we need to explore the far sub-tree
        if (point[node.axis] - node_point[node.axis]) ** 2 < best[1]:
            best = _nn_search(point, far, best)

        return best
    
    # Find the nearest point in Q for each point in P
    for i in range(P.shape[0]):
        best_idx, _ = _nn_search(P[i], 0, (None, float("inf")))
        Q_nearest[i] = np.array(tree[best_idx].point)           # nearest neighbor
        if tree[0].normal is not None:
            Q_nearest_N[i] = np.array(tree[best_idx].normal)    # surface normal

    return Q_nearest, Q_nearest_N
