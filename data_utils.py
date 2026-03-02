import os
import re
import shutil

# Matching pattern for a single validation pair
PREFIX_RE = re.compile(
    r"^(\d{1,2}_\d{1,2})_(source|target|source_transformed|transformation)\.(pcd|txt)$"
)

def fix_pcd(input_path, output_path=None):
    """
    Fix corrupt ASCII PCD files by:
      - removing rows with incorrect field counts
      - removing rows containing non-numeric values
      - removing rows containing NUL characters ('\\x00')
      - updating POINTS, WIDTH, HEIGHT in the header

    Prints to terminal where changes are made or if file is clean.
    If output_path is None, modifies the file in-place.
    """
    if output_path is None:
        output_path = input_path

    with open(input_path, 'r', errors='replace') as f:
        lines = f.readlines()

    # ---- Split header and data ----
    header = []
    data_start_idx = None
    for i, line in enumerate(lines):
        header.append(line)
        if line.strip().startswith("DATA"):
            data_start_idx = i + 1
            break

    if data_start_idx is None:
        raise ValueError(f"Invalid PCD file: {input_path}")

    # ---- Parse expected number of fields ----
    fields_line = next((l for l in header if l.startswith("FIELDS")), None)
    if fields_line is None:
        raise ValueError(f"PCD missing FIELDS keyword: {input_path}")

    num_fields = len(fields_line.strip().split()[1:])

    # ---- Clean data rows ----
    cleaned_data = []
    removed_rows = 0
    nul_rows = 0
    non_numeric_rows = 0

    for idx, line in enumerate(lines[data_start_idx:], start=data_start_idx + 1):

        # ---- NUL corruption ----
        if '\x00' in line:
            nul_rows += 1
            removed_rows += 1
            print(f"Removing NUL-corrupted row {idx}")
            continue

        parts = line.strip().split()

        # ---- Field count check ----
        if len(parts) != num_fields:
            removed_rows += 1
            print(f"Removing malformed row {idx}: wrong field count")
            continue

        # ---- Numeric validation ----
        try:
            _ = [float(p) for p in parts]
        except ValueError:
            non_numeric_rows += 1
            removed_rows += 1
            print(f"Removing non-numeric row {idx}: {line.strip()}")
            continue

        cleaned_data.append(line)

    # ---- Header helpers ----
    def get_header_value(keyword):
        for line in header:
            if line.startswith(keyword):
                return int(line.strip().split()[1])
        return None

    old_points = get_header_value("POINTS")
    old_width  = get_header_value("WIDTH")
    old_height = get_header_value("HEIGHT")

    new_points = len(cleaned_data)
    new_width = new_points
    new_height = 1

    # ---- Report changes ----
    header_changed = False
    if old_points != new_points:
        print(f"Updating POINTS: {old_points} -> {new_points}")
        header_changed = True
    if old_width != new_width:
        print(f"Updating WIDTH: {old_width} -> {new_width}")
        header_changed = True
    if old_height != new_height:
        print(f"Updating HEIGHT: {old_height} -> {new_height}")
        header_changed = True

    if nul_rows > 0:
        print(f"Removed {nul_rows} NUL-corrupted rows")
    if non_numeric_rows > 0:
        print(f"Removed {non_numeric_rows} non-numeric rows")

    if removed_rows == 0 and not header_changed:
        print(f"{input_path} is clean, no changes needed.")

    # ---- Update header ----
    def replace_header_value(pattern, replacement):
        nonlocal header
        for i, line in enumerate(header):
            if re.match(pattern, line.strip()):
                header[i] = replacement + "\n"

    replace_header_value(r"POINTS\s+.*", f"POINTS {new_points}")
    replace_header_value(r"WIDTH\s+.*",  f"WIDTH {new_width}")
    replace_header_value(r"HEIGHT\s+.*", f"HEIGHT {new_height}")

    # ---- Write output ----
    with open(output_path, 'w') as f:
        f.writelines(header)
        f.writelines(cleaned_data)

def num_pts(path):
    """
    Extract the number of points from the header of a PCD file.

    This function scans the file line-by-line until it finds a line beginning
    with the keyword ``POINTS`` and returns the integer value that follows it.
    It assumes the PCD file uses a standard ASCII PCD header.

    Args:
        path (str): Path to the PCD file to read.

    Returns:
        int: The number of points specified in the file's ``POINTS`` header field.

    Raises:
        ValueError: If no ``POINTS`` line is found in the file.
    """
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("POINTS"):
                return int(line.strip().split()[1])
    raise ValueError(f"POINTS line not found in {path}")

def refactor(root: str):
    """Verify and, where required, repair corrupt PCD files before refactoring based 
    on a specific filename pattern.

    Recursively scans the root directory for files belonging to a single validation
    pair matching the pattern `D1_D2_LABEL.EXT`. Verifies and repairs corrupt PCD files
    where required.

    ## Pattern details
        **D1**, **D2** : *1 or 2 digit integers*
        Pair identifier.<br>
        **LABEL** : *"source" | "target" | "source_transformed" | "transformation"*
        File label.<br>
        **EXT** : *"pcd" | "txt"*
        File extension/type.<br>

    ## Behavior
    - Verifies and, where required, repairs corrupt point cloud data (PCD) files.
    - Copies all matching files for a single validation pair into a new subdirectory:
      `./root_grouped/D1_D2/`.
    """

    # Create output_dir on same level as root
    root = os.path.abspath(root)
    parent_dir = os.path.dirname(root)
    output_dir = os.path.join(parent_dir, os.path.basename(root) + "_grouped")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Dictionary of dictionaries for each validation pair, mapping each label to its
    # corresponding file path.
    pairs = {}

    # Recursively scan all files in root and populate pairs
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            m = PREFIX_RE.fullmatch(fname)
            if not m:
                continue
            ident, label, ext = m.groups()
            full_path = os.path.join(dirpath, fname)
            pairs.setdefault(ident, {})[label] = full_path

    # File categories (labels)
    labels = {"source", "target", "source_transformed", "transformation"}

    # For each validation pair
    for ident, src_paths in pairs.items():

        # Skip any validation pairs with incomplete data
        if set(src_paths.keys()) != labels:
            print(
                f"Skipping incomplete group: {ident} (found: {set(src_paths.keys())})"
            )
            continue

        # Create a new directory in output_dir
        out_dir = os.path.join(output_dir, ident)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nGrouping {ident} → {out_dir}")

        # Track destination paths for point count comparison
        dest_paths = {}

        # Copy each data file to the new directory
        for label, path in src_paths.items():
            dest = os.path.join(out_dir, os.path.basename(path))
            dest_paths[label] = dest

            if path.endswith(".pcd"):
                # Verify and, where required, repair corrupt PCD files
                print(f"Checking PCD: {path}")
                fix_pcd(path, dest)
            else:
                # Copy txt files
                shutil.copy2(path, dest)

        # If the number of points in the source and its transformation do not match
        if num_pts(dest_paths["source"]) != num_pts(dest_paths["source_transformed"]):
            # Remove the directory for the current validation pair
            shutil.rmtree(out_dir, ignore_errors=True)

            print("\033[1;31m"
                  f"Deleted group {ident}: point count mismatch between source and "
                  f"source_transformed point clouds."
                  "\033[0m")
