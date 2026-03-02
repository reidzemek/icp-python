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
import rc_themes

# Apply theme
sns.set_theme(context="notebook", rc=rc_themes.monokai_pro_light_rc, palette="muted")

# Data file suffixes for a single validation pair
suffixes = [
    "_source.pcd",
    "_target.pcd",
    "_source_transformed.pcd",
    "_transformation.txt"
]

# User input required to specify the path to the base directories of the validation data
validation_paths = utils.collect_validation_paths()

reference_transformation_error = []
icp_transformation_error = []

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

        # Find global bounds across both clouds to preserve relative spatial frame
        all_pts = np.concatenate([P, Q])
        min_val, max_val = all_pts.min(), all_pts.max()

        # Normalize P and Q to [-1, 1] using shared global bounds
        P_norm = 2 * (P - min_val) / (max_val - min_val) - 1
        Q_norm = 2 * (Q - min_val) / (max_val - min_val) - 1

        # Convert P and Q to signed 10-bit integers
        P_int10 = np.clip(np.rint(P_norm * 511), -512, 511).astype(np.int16)
        Q_int10 = np.clip(np.rint(Q_norm * 511), -512, 511).astype(np.int16)

        # Build the k-d tree to enable efficient nearest neighbor search
        Q_tree = kdtree.build(Q_int10)

        # Total transformation accumulator
        T_total = np.identity(4)

        # Iterative Closest Point (ICP) algorithm (10 iterations)
        for i in range(10):
            # For each point in P, find the nearest point in Q
            Q_nearest = kdtree.nn_search(Q_tree, P_int10)

            # Shift point clouds to be centered around the origin
            P_centroid = icp.centroid(P_int10)
            P_centered = icp.center(P, P_centroid)
            Q_nearest_centroid = icp.centroid(Q_nearest)
            Q_nearest_centered = icp.center(Q_nearest, Q_nearest_centroid)

            # Compute the cross covariance matrix H
            H = icp.xcovariance(P_centered, Q_nearest_centered)

            # Compute the 4x4 rigid transformation matrix for P
            T = icp.transformation(H, P_centroid, Q_nearest_centroid)

            # Apply the transformation to P
            P = icp.transform(P, T)

            # Update the total transformation (order is important)
            T_total = T @ T_total

        pc = PointCloud.from_xyz_points(P)
        data_file = os.path.join(dirpath, dirname + "_source_transformed_icp.pcd")
        pc.save(
            data_file,
            encoding=Encoding.ASCII
        )

        utils.evaluate_transformation(T_total, data_files[3])

        reference_transformation_error.append(
            utils.transformation_error(data_files[2], data_files[1]))
        icp_transformation_error.append(
            utils.transformation_error(data_file, data_files[1]))

plt.figure()
plt.hist([icp_transformation_error, reference_transformation_error], 
         bins=15, label=['Custom ICP', 'Reference Implementation'])
plt.legend()
plt.xlabel('Transformation Error')
plt.ylabel('Frequency (Count)')
plt.title('Transformation Error (Reference vs. Custom ICP)')
plt.show()
