import os
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
from pypcd4 import PointCloud, Encoding
import matplotlib.pyplot as plt
import seaborn as sns

import icp
import kdtree
import utils

# 1. Set the aesthetic theme (removes the top/right box lines)
sns.set_theme(context="paper", style="dark", palette="muted")
colors = sns.color_palette() # Access the muted palette colors

def collect_validation_paths():
    """Prompt the user for validation-pair directories until they choose to stop.

    The user is repeatedly asked to enter a directory path. Each valid directory
    is added to a list. Input terminates when the user presses Enter without
    providing a path.

    Each provided directory must:
      - exist and be a directory
      - contain only sub-directories (each sub-directory is assumed to be
        a validation pair)

    Returns:
        list[pathlib.Path]: A list of valid validation directory paths.
    """
    paths = []

    while True:
        path_str = input(
            "Enter path to validation directory (or press Enter to finish): "
        ).strip()
        if not path_str:
            break

        root = Path(path_str)

        if not root.is_dir():
            print(f"Invalid path: '{root}' is not a valid directory.")
            continue

        if not all((root / name).is_dir() for name in os.listdir(root)):
            print(f"'{root}' must contain only directories.")
            continue

        paths.append(root)

    return paths

def evaluate_transformation(T, filepath):
    """Overwrite file contents after the first 4 lines with T and its relative error."""

    # ---- Read reference transform (first 4 lines) ----
    with open(filepath, "r") as f:
        lines = f.readlines()

    if len(lines) < 4:
        raise ValueError("File must contain at least 4 lines for reference transform")

    header = lines[:4]

    T_ref = np.array([
        [float(x) for x in line.strip("[]\n").split(",")]
        for line in header
    ])

    # ---- Relative transform ----
    T_rel = np.linalg.inv(T_ref) @ T
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]

    # ---- Rotation error (degrees) ----
    trace = np.trace(R_rel)
    cos_theta = np.clip((trace - 1) / 2, -1.0, 1.0)
    rot_err_deg = np.degrees(np.arccos(cos_theta))

    # ---- Translation error ----
    trans_err = np.linalg.norm(t_rel)

    # ---- Format matrix ----
    formatted_T = "\n".join(
        f"[{T[i,0]:11.6f}, {T[i,1]:11.6f}, {T[i,2]:11.6f}, {T[i,3]:11.6f}]"
        for i in range(4)
    )

    # ---- Overwrite file ----
    with open(filepath, "w") as f:
        f.writelines(header)
        f.write("\nTotal Transformation:\n")
        f.write(formatted_T + "\n")
        f.write("\n# Transformation error compared to the above reference:\n")
        f.write(f"# Relative Rotation error (deg): {rot_err_deg:.6f}\n")
        f.write(f"# Relative Translation error:    {trans_err:.6f}\n")

def transformation_error(src_pcd_path, tgt_pcd_path):
    """
    Compute point-to-point ICP error between two PCD files.

    Parameters
    ----------
    src_pcd_path : str
        Path to source point cloud (.pcd)
    tgt_pcd_path : str
        Path to target point cloud (.pcd)

    Returns
    -------
    float
        Point-to-point transformation error
    """

    # ---- Load PCD files into Nx3 numpy arrays ----
    P = PointCloud.from_path(src_pcd_path).numpy(("x", "y", "z"))
    Q = PointCloud.from_path(tgt_pcd_path).numpy(("x", "y", "z"))

    # ---- Build KD-tree on target ----
    tree = kdtree.build(Q)

    # ---- Nearest neighbor search ----
    Q_nn = kdtree.nn_search(tree, P)

    # ---- Compute point-to-point error ----
    error = icp.p2p_error(P, Q_nn)

    return error

# Data file suffixes for a single validation pair
suffixes = [
    "_source.pcd",
    "_target.pcd",
    "_source_transformed.pcd",
    "_transformation.txt"
]

# User input required to specify the path to the base directories of the validation data
validation_paths = collect_validation_paths()

# # User input required to specify the path to the base directory of the validation data
# root = input("Enter the base directory containing the grouped validation dataset: ")

# # Check validity and raise an error if invalid
# if not os.path.isdir(root):
#     raise ValueError(f"Invalid path provided: '{root}' is not a valid directory.")
# if not all(
#     os.path.isdir(os.path.join(root, name))
#     for name in os.listdir(root)
# ):
#     raise ValueError(f"'{root}' must contain only directories.")

reference_transformation_error = []
icp_transformation_error = []

hist = {
    "P": [],
    "P_nn_dx": [],
    "P_nn_dy": [],
    "P_nn_dz": [],
    "Q": [],
    "Q_nn_dx": [],
    "Q_nn_dy": [],
    "Q_nn_dz": [],
    "neighbor_gap": [],
    "P_centered": [],
    "Q_centered": [],
    "H": [],
    "T": [],
    "T_total": [],
    "P_diff": []
}

def collect(hist, key, x):
    hist[key].extend(np.asarray(x).ravel())

# Iterate through each validation path
for validation_path in validation_paths:

    # Iterate through each validation pair directory
    for dirname in os.listdir(validation_path):
        dirpath = os.path.join(validation_path, dirname)
        
        # Find the data files for a single validation pair
        data_files = []
        for suffix in suffixes:
            data = os.path.join(dirpath, dirname + suffix)
            if os.path.isfile(data):
                data_files.append(data)
        
        # Check data files exist
        if len(data_files) != 4:
            print(f"{dirpath} contains no valid data.")
            continue

        # Initialize source (P) and target (Q) point clouds for the ICP algorithm
        P = PointCloud.from_path(data_files[0]).numpy(("x", "y", "z"))
        Q = PointCloud.from_path(data_files[1]).numpy(("x", "y", "z"))

        collect(hist, "Q", Q)

        # Build the k-d tree to enable efficient nearest neighbor search
        Q_tree = kdtree.build(Q)

        # For Q, build a tree find each point's nearest neighbor
        Q_scipy_tree = KDTree(Q) # SciPy version
        _, indices = Q_scipy_tree.query(Q, k=2) # (k=2 to skip self)

        # For each point in Q, record the nearest  neighbor per-axis deltas
        delta = np.abs(Q - Q[indices[:, 1]])
        collect(hist, "Q_nn_dx", delta[:, 0])
        collect(hist, "Q_nn_dy", delta[:, 1])
        collect(hist, "Q_nn_dz", delta[:, 2])

        # Total transformation accumulator
        T_total = np.identity(4)

        # Iterative Closest Point (ICP) algorithm (10 iterations)
        for i in range(10):
            collect(hist, "P", P)

            # For P, build a tree and find each point's nearest neighbor
            P_scipy_tree = KDTree(P) # SciPy version
            _, indices = P_scipy_tree.query(P, k=2) # (k=2 to skip self)

            # For each point in P, record the nearest neighbor per-axis deltas
            delta = np.abs(P - P[indices[:, 1]])
            collect(hist, "P_nn_dx", delta[:, 0])
            collect(hist, "P_nn_dy", delta[:, 1])
            collect(hist, "P_nn_dz", delta[:, 2])

            # For each point in P find the 2 nearest points in Q
            distances, _ = Q_scipy_tree.query(P, k=2)

            # Compute the 1st-to-2nd neighbor distance gap
            neighbor_gap = distances[:, 1] - distances[:, 0]
            collect(hist, "neighbor_gap", neighbor_gap)

            # For each point in P, find the nearest point in Q
            Q_nearest = kdtree.nn_search(Q_tree, P)

            # Shift point clouds to be centered around the origin
            P_centroid = icp.centroid(P)
            P_centered = icp.center(P, P_centroid)
            Q_nearest_centroid = icp.centroid(Q_nearest)
            Q_nearest_centered = icp.center(Q_nearest, Q_nearest_centroid)

            collect(hist, "P_centered", P_centered)
            collect(hist, "Q_centered", Q_nearest_centered)

            # Compute the cross covariance matrix H
            H = icp.xcovariance(P_centered, Q_nearest_centered)
            collect(hist, "H", H)

            # Compute the 4x4 rigid transformation matrix for P
            T = icp.transformation(H, P_centroid, Q_nearest_centroid)
            collect(hist, "T", T[:3, :])

            P_prev = P

            # Apply the transformation to P
            P = icp.transform(P, T)

            P_diff = P - P_prev

            collect(hist, "P_diff", P_diff)

            # Update the total transformation (order is important)
            T_total = T @ T_total
            collect(hist, "T_total", T_total[:3, :])

        pc = PointCloud.from_xyz_points(P)
        data_file = os.path.join(dirpath, dirname + "_source_transformed_icp.pcd")
        pc.save(
            data_file,
            encoding=Encoding.ASCII
        )

        evaluate_transformation(T_total, data_files[3])

        reference_transformation_error.append(
            transformation_error(data_files[2], data_files[1]))
        icp_transformation_error.append(
            transformation_error(data_file, data_files[1]))

monokai_classic_rc = {
    "axes.facecolor": "#1D1D19",
    "figure.facecolor": "#272822",
    "text.color": "#F8F8F2",
    "axes.labelcolor": "#F8F8F2",
    "xtick.color": "#75715E",
    "ytick.color": "#75715E",

    # --- Font Styling: Helvetica ---
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    
    # --- Title Styling ---
    "axes.titlesize": "large",
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    
    # Grid Styling: Horizontal only
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#3E3D32",
    "grid.alpha": 0.7,
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "axes.edgecolor": "#3E3D32",

    # Legend settings to hide grid lines seamlessly
    "legend.frameon": True,
    "legend.facecolor": "#272822",
    "legend.edgecolor": "none",
    "legend.framealpha": 1.0,
    
    # Cleanup
    "axes.spines.top": False,
    "axes.spines.right": False,
}

monokai_pro_light_rc = {
    "axes.facecolor": "#fdf9f3",      # Core light background
    "figure.facecolor": "#f8efe7",    # Matching figure background
    "text.color": "#3d3d3d",          # High-contrast charcoal text
    "axes.labelcolor": "#3d3d3d",     # High-contrast label color
    "xtick.color": "#969893",         # Muted gray for ticks (Light comment color)
    "ytick.color": "#969893",         # Muted gray for ticks

    # --- Font Styling: Helvetica ---
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    
    # --- Title Styling ---
    "axes.titlesize": "large",
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    
    # Grid Styling: Horizontal only
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#d4d5d4",          # Very light guide color
    "grid.alpha": 1.0,
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "axes.edgecolor": "#adafac",      # Soft border for the axes

    # Legend settings
    "legend.frameon": True,
    "legend.facecolor": "#f8efe7",
    "legend.edgecolor": "none",
    "legend.framealpha": 1.0,
    
    # Cleanup
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Apply theme
sns.set_theme(context="notebook", rc=monokai_pro_light_rc, palette="muted")

plt.figure()
plt.hist([icp_transformation_error, reference_transformation_error], 
         bins=15, label=['Custom ICP', 'Reference Implementation'])
plt.legend()
plt.xlabel('Transformation Error')
plt.ylabel('Frequency (Count)')
plt.title('Transformation Error (Reference vs. Custom ICP)')
plt.show()

data = hist["neighbor_gap"]
p = 5
percentile = np.percentile(data, p)
rng = (0, percentile * 20)
plt.hist(data, bins=100, range=rng, log=True)
plt.axvline(percentile, color="red", label=f"{p}th percentile")
plt.legend()
plt.show()

for key, values in hist.items():
    plt.figure()
    plt.hist(values, bins=50)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title(key)
