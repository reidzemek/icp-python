import platform
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pypcd4 import PointCloud
import kdtree_old
import icp

def system_dark_mode():
    os_name = platform.system()

    # ---------------- Windows ----------------
    if os_name == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            )
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return value == 0  # 0 = dark mode
        except Exception:
            return False

    # ---------------- macOS ----------------
    elif os_name == "Darwin":
        try:
            output = subprocess.run(
                "defaults read -g AppleInterfaceStyle",
                shell=True,
                capture_output=True,
                text=True
            )
            return output.stdout.strip().lower() == "dark"
        except Exception:
            return False

    # ---------------- Linux / WSL ----------------
    elif os_name == "Linux":
        # Detect WSL
        is_wsl = False
        try:
            with open("/proc/version", "r") as f:
                is_wsl = "microsoft" in f.read().lower()
        except Exception:
            pass

        if is_wsl:
            # Attempt to read Windows dark mode via reg.exe
            try:
                cmd = [
                    "reg.exe",
                    "query",
                    r"HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
                    "/v",
                    "AppsUseLightTheme"
                ]
                output = subprocess.run(cmd, capture_output=True, text=True)
                # Example output line: "AppsUseLightTheme    REG_DWORD    0x0"
                for line in output.stdout.splitlines():
                    if "AppsUseLightTheme" in line:
                        value = int(line.strip().split()[-1], 16)
                        return value == 0
                return False
            except Exception:
                return False
        else:
            # Regular Linux: use GNOME theme
            try:
                output = subprocess.run(
                    ["gsettings", "get", "org.gnome.desktop.interface", "gtk-theme"],
                    capture_output=True,
                    text=True
                )
                theme = output.stdout.strip().strip("'").lower()
                return "dark" in theme
            except Exception:
                return False

    return False  # fallback to light

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
    tree = kdtree_old.build(Q)

    # ---- Nearest neighbor search ----
    Q_nn = kdtree_old.nn_search(tree, P)

    # ---- Compute point-to-point error ----
    error = icp.p2p_error(P, Q_nn)

    return error

def is_intn(data, n):
    """Checks if the input data fits within the signed n-bit integer range
    [-2⁽ⁿ⁻¹⁾, 2⁽ⁿ⁻¹⁾-1].

    Args:
        data (int, float, list, or np.ndarray): The numerical data to check. 
            Can be a single scalar or a multi-dimensional array.

    Returns:
        bool: True if all values are within the range [-2⁽ⁿ⁻¹⁾, 2⁽ⁿ⁻¹⁾-1], 
            False otherwise.
    """
    data = np.asanyarray(data)
    return np.all((data >= -2**(n-1)) & (data <= 2**(n-1)-1))

def plot(pc: np.ndarray, title: str="Point Cloud", ax=None) -> None:
    """Visualize a 3D point cloud with equal axis scaling and a 3D color gradient.

    Args:
        pc (np.ndarray): Point cloud of shape (N, 3).
        title (str): Title for the plot.

    Raises:
        ValueError: If pc is not of shape (N, 3).
    """
    
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise ValueError("pc must have shape (N, 3).")
    
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        show = True
    else:
        fig = ax.figure
        show = False

    ax.clear()
    ax.set_box_aspect([1, 1, 1])
    ax.set_facecolor(ax.figure.get_facecolor())

    # --- 3D color gradient ---
    # Normalize each coordinate to [0, 1]
    min_val = pc.min(axis=0)
    max_val = pc.max(axis=0)
    norm_pc = (pc - min_val) / (max_val - min_val + 1e-12)

    # Use normalized xyz as RGB for a full 3D gradient
    colors = norm_pc  # shape (N, 3)

    # Scatter
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=5, c=colors)

    # View and grid
    ax.view_init(elev=30, azim=-45)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Equal scaling
    ranges = max_val - min_val
    max_range = ranges.max()
    center = (min_val + max_val) / 2

    ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
    ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
    ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)

    # Labels + Title
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)

    if show:
        plt.show()
