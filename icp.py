from typing import Optional
import logging
import numpy as np
import math

import utils

logger = logging.getLogger(__name__)

# NOTE Overflow checking assumes signed 64-bit integer will not overflow

def mean(
    point_cloud: np.ndarray,
    ID: Optional[str]=None,
    pc_width: Optional[int]=None,
    acc_width: Optional[int]=None,
) -> np.ndarray:
    """Compute the centroid of a 3D point cloud using hierarchical reduction.

    Points are hierarchically reduced using an 8-input adder (with zero padding where 
    required) until a single sum remains. A final fixed-point correction factor is then
    applied to account for non-power-of-two input sizes.

    Args:
        point_cloud (np.ndarray): An integer array of shape (N, 3) containing
            N points in 3D.
        ID (str, optional): Function call identifier for logger.
        pc_width (int, optional): Signed integer bit-width of `point_cloud`.
            Must be provided if `ID` is provided.
        acc_width (int, optional): Signed integer bit-width used for the adder tree.
            Must be provided if `ID` is provided.

    Returns:
        np.ndarray: A (1, 3) integer array representing the centroid (mean x, y, z).

    Raises:
        ValueError:
            - If `point_cloud` is not a np.integer array with shape (N, 3).
            - If `ID` is provided and either or both `pc_width` and `acc_width` are
            not specified.
    """
    if (
        point_cloud.ndim != 2
        or point_cloud.shape[1] != 3
        or not np.issubdtype(point_cloud.dtype, np.integer)
    ):
        raise ValueError("point_cloud must be a np.integer array with shape (N, 3)")
    
    if ID is not None and (pc_width is None or acc_width is None):
        raise ValueError("pc_width and acc_width must be provided alongside ID.")
    
    # Compute the bit width of the point cloud coordinate values
    pc_min, pc_max = point_cloud.min(), point_cloud.max()
    _pc_width = math.ceil(math.log2(max(abs(pc_min), abs(pc_max)) + 1)) + 1

    if ID is not None and pc_width < _pc_width:
        logger.warning(
            f"[ID: {ID}] Coordinate overflow: configured width ({pc_width}) "
            f"< required width for input data ({_pc_width})."
        )

    # Compute the worst case bit width for the adder tree accumulator
    N = point_cloud.shape[0]        # point count
    tree_depth = N.bit_length()     # number of 8-input reduction levels
    wc_acc_width = _pc_width + tree_depth

    if ID is not None and acc_width < wc_acc_width:
        logger.info(
            f"[ID: {ID}] Accumulator overflow risk: configured width ({acc_width}) "
            f"< worst-case required ({wc_acc_width})."
        )
    
    # Promote to avoid overflow
    points = point_cloud.astype(np.int64)

    def _reduce(points: np.ndarray) -> np.ndarray:
        """Reduce one hierarchy level by summing blocks of up to 8 points.

        Each output point represents the sum of one block of at most 8 input points.

        Args:
            points (np.ndarray): An array of shape (N, 3) containing N points in 3D.

        Returns:
            np.ndarray: Reduced array of shape (ceil(N/8), 3).
        """
        reduced = []

        for i in range(0, points.shape[0], 8):
            block = points[i:i + 8]
            reduced.append(np.sum(block, axis=0))

        return np.asarray(reduced, dtype=points.dtype)

    # Hierarchical reduction until a single vector remains
    while points.shape[0] > 1:
        points = _reduce(points)

    # Compute the required bit width for the adder tree accumulator
    _acc_width = math.ceil(math.log2(max(abs(points.min()), abs(points.max())) + 1)) + 1

    if ID is not None and acc_width < _acc_width:
        logger.warning(
            f"[ID: {ID}] Accumulator overflow: configured width ({acc_width}) "
            f"< required derived from input data ({_acc_width})."
        )

    # Fixed-point scaling factor to approximate division by N using power-of-two scaling
    # A hardware implementation would use pre-computed scaling factor(s).
    frac_bits = 16
    scale = np.round((2**tree_depth / N) * (2**frac_bits)).astype(np.int64)

    # Compute the average using the fixed-point scaling factor and bit-shift division
    return (points * scale) >> (tree_depth + frac_bits)

def center(
    point_cloud: np.ndarray,
    mean: np.ndarray,
    ID: Optional[str]=None,
    pc_width: Optional[int]=None,
    m_width: Optional[int]=None,
    c_width: Optional[int]=None
) -> np.ndarray:
    """Center a 3D point_cloud by subtracting a centroid (mean).

    Args:
        point_cloud (np.ndarray): Array of shape (N, 3) with N 3D points.
        mean (np.ndarray): Array of shape (1, 3) representing the centroid.
        ID (str, optional): Function call identification for logger.
        pc_width (int, optional): Signed integer bit-width of `point_cloud`.
            Must be provided if `ID` is provided.
        m_width (int, optional): Signed integer bit-width of `mean`.
            Must be provided if `ID` is provided.
        c_width (int, optional): Signed integer bit-width used for the centered
            point cloud. Must be provided if `ID` is provided.

    Returns:
        point_cloud_centered (np.ndarray): Array of shape (N, 3) where each point is
            shifted by subtracting the centroid.

    Raises:
        ValueError:
            - If `point_cloud` is not a np.integer array with shape (N, 3).
            - If `mean` is not a np.integer array with shape (1, 3).
            - If `ID` is provided and one or more of `pc_width`, `m_width` and
            `c_width` are missing.
    """

    if (
        point_cloud.ndim != 2
        or point_cloud.shape[1] != 3
        or not np.issubdtype(point_cloud.dtype, np.integer)
    ):
        raise ValueError("point_cloud must be a np.integer array with shape (N, 3)")

    if (mean.shape != (1, 3) or not np.issubdtype(mean.dtype, np.integer)):
        raise ValueError("mean must be a np.integer array with shape (1, 3).")
    
    if (
        ID is not None
        and (pc_width is None or m_width is None or c_width is None)
    ):
        raise ValueError(
            "pc_width, m_width and c_width must be provided alongside ID."
        )
    
    # Compute the bit width of the point cloud coordinate values
    pc_min, pc_max = point_cloud.min(), point_cloud.max()
    _pc_width = math.ceil(math.log2(max(abs(pc_min), abs(pc_max)) + 1)) + 1

    if ID is not None and pc_width < _pc_width:
        logger.warning(
            f"[ID: {ID}] Coordinate overflow: configured width ({pc_width}) "
            f"< required width for input data ({_pc_width})."
        )

    # Compute the bit width of the centroid values
    m_min, m_max = mean.min(), mean.max()
    _m_width = math.ceil(math.log2(max(abs(m_min), abs(m_max)) + 1)) + 1

    if ID is not None and m_width < _m_width:
        logger.warning(
            f"[ID: {ID}] Centroid overflow: configured width ({m_width}) "
            f"< required width for input data ({_m_width})."
        )

    # Compute the worst case bit width for the centered point cloud
    wc_c_width = math.ceil(math.log2(2**(_pc_width-1) + 2**(_m_width-1))) + 1

    if ID is not None and c_width < wc_c_width:
        logger.info(
            f"[ID: {ID}] Centered point cloud overflow risk: configured with "
            f"({c_width}) < worst-case required ({wc_c_width})."
        )

    # Compute centered point cloud (promote to avoid overflow)
    point_cloud_centered = point_cloud.astype(np.int64) - mean.astype(np.int64)

    # Compute the required bit width for the centered point cloud
    pc_c_min, pc_c_max = point_cloud_centered.min(), point_cloud_centered.max()
    _c_width = math.ceil(math.log2(max(abs(pc_c_min), abs(pc_c_max)) + 1)) + 1

    if ID is not None and c_width < _c_width:
        logger.warning(
            f"[ID: {ID}] Centered point cloud overflow: configured width ({c_width}) "
            f"< required derived from input data ({_c_width})."
        )
    
    return point_cloud_centered

def xcovariance(
        P_c: np.ndarray,
        Q_n_c: np.ndarray,
        P_width: Optional[int]=None,
        Q_width: Optional[int]=None,
        H_width: Optional[int]=None
    ) -> np.ndarray:
    """Compute the 3x3 cross-covariance matrix.

    The point clouds must have the same number of points (N), they must be centered
    (i.e., zero mean), and Q_n_c should be the centered points in the target point 
    cloud (Q) nearest to the corresponding points in the source point cloud (P).

    `P_width`, `Q_width` and `H_width` can be used for explicit hardware validation.

    Args:
        P_c (np.ndarray): Centered source point cloud array of shape (N, 3).
        Q_n_c (np.ndarray): Centered array of nearest target point cloud points
            of shape (N, 3).
        P_width (int, optional): Signed integer bit-width of `P_c`.
            Must be provided alongside `Q_width` and `H_width`.
        Q_width (int, optional): Signed integer bit-width of `Q_n_c`.
            Must be provided alongside `P_width` and `H_width`.
        H_width (int, optional): Signed integer bit-width for the cross-covariance
            matrix `H`. Must be provided alongside `P_width` and `Q_width`.

    Returns:
        H (np.ndarray): 3x3 cross-covariance matrix `H` = `P_c.T` x `Q_n_c`.

    Raises:
        ValueError:
            - `P_c` and/or `Q_n_c` are not of shape (N, 3).
            - `P_c` and `Q_n_c` do not have the same number of points (N).
            - Incomplete set of explicit hardware validation parameters (`P_width`,
            `Q_width`, `H_width`).
    """

    if P_c.ndim != 2 or P_c.shape[1] != 3 or Q_n_c.ndim != 2 or Q_n_c.shape[1] != 3:
        raise ValueError("P_n and/or Q_n_c both must have shape (N, 3).")
    
    if P_c.shape[0] != Q_n_c.shape[0]:
        raise ValueError("P_c and Q_n_c must have the same number of points (same N).")

    if any(x is None for x in (P_width, Q_width, H_width)):
        raise ValueError("P_width, Q_width and H_width must all be specified.")
    
    # Compute the required bit-width of the coordinate values of P_c
    _min, _max = P_c.min(), P_c.max()
    _P_width = math.ceil(math.log2(max(abs(_min), abs(_max)) + 1)) + 1

    if all(x is not None for x in (P_width, Q_width, H_width)) and P_width < _P_width:
        logger.warning(
            f"P_c coordinate overflow: configured width ({P_width}) "
            f"< required width for input data ({_P_width})."
        )

    # Compute the required bit-width of the coordinate values of Q_n_c
    _min, _max = Q_n_c.min(), Q_n_c.max()
    _Q_width = math.ceil(math.log2(max(abs(_min), abs(_max)) + 1)) + 1

    if all(x is not None for x in (P_width, Q_width, H_width)) and Q_width < _Q_width:
        logger.warning(
            f"P_n_c coordinate overflow: configured width ({Q_width}) "
            f"< required width for input data ({_Q_width})."
        )

    # Compute the worse case bit width of the cross-covariance matrix
    wc_H_width = _P_width + _Q_width - 1 + math.ceil(math.log2(P_c.shape[0]))

    if all(x is not None for x in (P_width, Q_width, H_width)) and H_width < wc_H_width:
        logger.info(
            f"Cross-covariance matrix overflow risk: configured width ({H_width}) "
            f"< worst-case required ({wc_H_width})."
        )

    # Compute the cross covariance matrix (promote to avoid overflow)
    H = P_c.T.astype(np.int64) @ Q_n_c.astype(np.int64)

    # Compute the required bit width for the centered point cloud
    _min, _max = H.min(), H.max()
    _H_width = math.ceil(math.log2(max(abs(_min), abs(_max)) + 1)) + 1

    if all(x is not None for x in (P_width, Q_width, H_width)) and H_width < _H_width:
        logger.warning(
            f"Cross-covariance matrix overflow: configured width ({H_width}) "
            f"< required derived from input data ({_H_width})."
        )
    
    return H

def T_matrix(
    H: np.ndarray,
    P_centroid: np.ndarray,
    Q_centroid: np.ndarray
) -> np.ndarray:
    """Compute the 4x4 rigid transformation matrix using Horn's quaternion method.

    Args:
        H (np.ndarray): 3x3 cross-covariance matrix.
        P_centroid (np.ndarray): (1, 3) centroid of point cloud P.
        Q_centroid (np.ndarray): (1, 3) centroid of point cloud Q.

    Returns:
        np.ndarray: A 4x4 transformation matrix, with R | t in the top 3x4 block
                    and [0 0 0 1] as the bottom row.

    Raises:
        ValueError:
            - `H` is not a 3x3 matrix.
            - `P_centroid` is not of shape (1, 3).
            - `Q_centroid` is not of shape (1, 3).
    """

    if H.shape != (3, 3):
        raise ValueError("H must be a 3x3 matrix.")
    if P_centroid.shape != (1, 3):
        raise ValueError("P_centroid must be shape (1, 3).")
    if Q_centroid.shape != (1, 3):
        raise ValueError("Q_centroid must be shape (1, 3).")

    # Extract the vector components of the skew-symmetric parts of H
    delta = np.array([
        H[1, 2] - H[2, 1],
        H[2, 0] - H[0, 2],
        H[0, 1] - H[1, 0]
    ])

    # Use Horn's method to construct the 4x4 quaternion characteristic matrix N.
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

    # Use the power iteration method to compute dominant eigenvector (unit quaternion)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    K = 100
    for _ in range(K):
        q_new = N @ q
        q_new /= np.linalg.norm(q_new)
        if np.linalg.norm(q - q_new) < np.finfo(float).eps:
            break
        q = q_new

    # Extract quaternion components
    q0, q1, q2, q3 = q

    # Use the unit quaternion to construct the rotation matrix
    R = np.array([
        [1 - 2*(q2*q2 + q3*q3),   2*(q1*q2 - q3*q0),     2*(q1*q3 + q2*q0)],
        [2*(q1*q2 + q3*q0),       1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q1*q0)],
        [2*(q1*q3 - q2*q0),       2*(q2*q3 + q1*q0),     1 - 2*(q1*q1 + q2*q2)]
    ])

    # Translation vector
    t = (Q_centroid.T - R @ P_centroid.T).flatten()

    # 4x4 rigid transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T

def transformation(
    H: np.ndarray,
    P_mean: np.ndarray,
    Q_mean: np.ndarray,
    H_width: Optional[int]=None,
    P_m_width: Optional[int]=None,
    Q_m_width: Optional[int]=None,
    R_width: int=64,
    t_width: int=64
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the rigid transformation aligning point cloud `P` to `Q`.

    The rotation matrix `R` is returned using Q1.(`R_width - 1`) fixed-point
    representation, and the translation vector `t` is returned using `t_width`-bit
    signed integer representation.

    `H_width`, `P_m_width` and `Q_m_width` can be used for explicit hardware validation.
    
    Args:
        H (np.ndarray): (3, 3) cross-covariance matrix.
        P_mean (np.ndarray): (1, 3) centroid/mean of point cloud P.
        Q_mean (np.ndarray): (1, 3) centroid/mean of point cloud Q.
        H_width (int, optional): Signed integer bit-width of `H`. Must be provided
            alongside `P_m_width` and `Q_m_width`.
        P_m_width (int, optional): Signed integer bit-width of `P_mean`. Must be
            provided alongside `H_width` and `Q_m_width`.
        Q_m_width (int, optional): Signed integer bit-width of `Q_mean`. Must be
            provided alongside `H_width` and `Q_m_width`.
        R_width (int, optional): Signed fixed-point bit-width for the rotation
            matrix `R`. Defaults to 64.
        t_width (int, optional): Signed integer bit-width for the translation vector
            `t`. Defaults to 64.

    Returns:
        tuple: `R` and `t` outlined below.

        **R** : *np.ndarray of shape (3, 3)*<br>
        Rotation matrix in Q1.15 fixed-point representation.

        **t** : *np.ndarray of shape (3,1)*<br>
        Translation vector in 11-bit integer representation.
    
    Raises:
        ValueError:
            - `H` is not a 3x3 matrix.
            - `P_mean` is not of shape (1, 3).
            - `Q_mean` is not of shape (1, 3).
            - Incomplete set of explicit hardware validation parameters (`H_width`,
            `P_m_width`, `Q_m_width`).
    """
    val_params = all(x is not None for x in (H_width, P_m_width, Q_m_width))
    if not val_params:
        raise ValueError("H_width, P_m_width, and Q_m_width must all be specified.")
    
    # Compute the required bit width for the cross-covariance matrix
    _min, _max = H.min(), H.max()
    _H_width = math.ceil(math.log2(max(abs(_min), abs(_max)) + 1)) + 1

    if val_params and H_width < _H_width:
        logger.warning(
            f"Cross-covariance matrix overflow: configured width ({H_width}) "
            f"< required width for input ({_H_width})."
        )

    # Compute the required bit width for the centroids (P_mean and Q_mean)
    _min, _max = P_mean.min(), P_mean.max()
    _P_m_width = math.ceil(math.log2(max(abs(_min), abs(_max)) + 1)) + 1
    _min, _max = Q_mean.min(), Q_mean.max()
    _Q_m_width = math.ceil(math.log2(max(abs(_min), abs(_max)) + 1)) + 1


    if val_params and P_m_width < _P_m_width:
        logger.warning(
            f"P centroid overflow: configured width ({P_m_width}) "
            f"< required width for input ({_P_m_width})."
        )

    if val_params and Q_m_width < _Q_m_width:
        logger.warning(
            f"Q centroid overflow: configured width ({Q_m_width}) "
            f"< required width for input ({_Q_m_width})."
        )

    # Compute the transformation matrix
    T = T_matrix(H, P_mean, Q_mean)

    # Compute the rotation matrix (Q1.15) and translation vector (int11)
    R = np.rint(np.clip(T[:3, :3], -1.0, 1.0-(2**-(R_width - 1))) * 2**(R_width - 1))
    t = np.rint(T[:3, 3].reshape(3, 1))

    # Compute the required bit width for the transformation vector (t)
    _min, _max = t.min(), t.max()
    _t_width = math.ceil(math.log2(max(abs(_min), abs(_max)) + 1)) + 1

    if val_params and t_width < _t_width:
        logger.warning(
            f"transformation vector (t) overflow: configured width ({t_width}) "
            f"< required ({_t_width})."
        )

    return R.astype(np.int64), t.astype(np.int64)


def transform(
    P: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    P_width: Optional[int]=None,
    R_width: Optional[int]=None,
    t_width: Optional[int]=None    
) -> np.ndarray:
    """Transform the source point cloud `P`.

    `P_width`, `R_width` and `t_width` can be used for explicit hardware validation.

    Args:
        P (np.ndarray): Source point cloud array of shape (N, 3).
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3x1 translation vector.
        P_width (int, optional): Signed integer bit width of both `P` and the 
            transformed output point cloud. Must be provided alongside `R_width`
            and `t_width`.
        R_width (int, optional): Signed integer bit width of `R`. Must be provided
            alongside `P_width` and `t_width`.
        t_width (int, optional): Signed integer bit width of `t`. Must be provided
            alongside `P_width` and `R_width`.

    Returns:
        np.ndarray: Transformed point cloud of shape (N, 3).

    Raises:
        ValueError:
            - `P` is not of shape (N, 3).
            - `R` is not of shape (3, 3).
            - `t` is not of shape (3, 1).
            - Incomplete set of explicit hardware validation parameters (`P_width`,
            `R_width`, `t_width`)
    """

    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P must have shape (N, 3).")
    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 rotation matrix.")
    if t.shape != (3, 1):
        raise ValueError("t must be a 3x1 translation vector.")
    
    val_params = all(x is not None for x in (P_width, R_width, t_width))
    if not val_params:
        raise ValueError("P_width, R_width and t_width must all be specified.")
    
    # Compute the required bit width for the source point cloud P
    _min, _max = P.min(), P.max()
    _P_width = math.ceil(math.log2(max(abs(_min), abs(_max)) + 1)) + 1

    if val_params and P_width < _P_width:
        logger.warning(
            f"Source point cloud (P) overflow: configured width ({P_width}) "
            f"< required width for input data ({_P_width})."
        )

    # Compute the required bit width for the Rotation matrix and translation vector
    _min, _max = R.min(), R.max()
    _R_width = math.ceil(math.log2(max(abs(_min), abs(_max)) + 1)) + 1
    _min, _max = t.min(), t.max()
    _t_width = math.ceil(math.log2(max(abs(_min), abs(_max)) + 1)) + 1

    if val_params and R_width < _R_width:
        logger.warning(
            f"Rotation matrix (R) overflow: configured width ({R_width}) "
            f"< required width for input ({_R_width})."
        )

    if val_params and t_width < _t_width:
        logger.warning(
            f"Transformation vector (t) overflow: configured width ({t_width}) "
            f"< required width for input ({_t_width})."
        )

    # Apply the transformation
    P_transformed = (((R @ P.T) >> (R_width - 1)) + t).T

    # Compute the required bit width for the transformed point cloud
    _min, _max = P_transformed.min(), P_transformed.max()
    _P_transformed_width = math.ceil(math.log2(max(abs(_min), abs(_max)) + 1)) + 1

    if val_params and P_width < _P_transformed_width:
        logger.warning(
            f"Transformed point cloud overflow: configured width ({P_width}) "
            f"< required ({_P_transformed_width})."
        )

    return P_transformed

# TODO add a function to compute the required bit width

def p2p_error(P: np.ndarray, Q_nearest: np.ndarray) -> float:
    """Compute the point-to-point RMS alignment error between two point clouds.

    Args:
        P (np.ndarray): Source point cloud of shape (N, 3).
        Q_nearest (np.ndarray): Nearest neighbor points in the target
            point cloud, also of shape (N, 3).

    Returns:
        float: The RMS point-to-point alignment error.

    Raises:
        ValueError: If the inputs do not have matching shape (N, 3).
    """
    if P.shape != Q_nearest.shape:
        raise ValueError("P and Q_nearest must have the same shape (N, 3).")

    diff = P - Q_nearest                # (N, 3)
    sq_dist = np.sum(diff ** 2, axis=1) # (N,)
    rms_error = np.sqrt(np.mean(sq_dist))

    return rms_error

# TODO fix comment/function
def p2pl_error(
    P: np.ndarray,
    Q_nearest: np.ndarray,
    Q_nearest_N: np.ndarray,
) -> float:
    """Compute the point-to-plane RMS alignment error between two point clouds.

    Args:
        P (np.ndarray): Source point cloud of shape (N, 3).
        Q_nearest (np.ndarray): Nearest neighbor points in the target
            point cloud, also of shape (N, 3).
        Q_nearest_N (np.ndarray): Surface normals at Q_nearest, shape (N, 3).
            Normals are assumed to be unit-length.

    Returns:
        float: The RMS point-to-plane alignment error.

    Raises:
        ValueError: If inputs do not have matching shape (N, 3).
    """
    if P.shape != Q_nearest.shape or P.shape != Q_nearest_N.shape:
        raise ValueError("P, Q_nearest, and Q_nearest_N must all have shape (N, 3).")

    diff = P - Q_nearest                                    # (N, 3)
    temp = diff * Q_nearest_N
    signed_dist = np.sum(temp, axis=1)  # (N,)
    rms_error = np.sqrt(np.mean(signed_dist ** 2))

    return rms_error
