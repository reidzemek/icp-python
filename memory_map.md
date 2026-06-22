# ICP Memory Maps

Memory layout for each functional unit in the ICP pipeline.

## Conventions

- **Each row is 256 bits wide.** The number of rows grows with the data.
- Coordinate **bit width `b` is not yet fixed.** Layouts are therefore given as an *ordering* of values within each row; concrete positions follow once `b` is chosen.
- **Values per row** is `floor(256 / b)` for single-axis rows, or `floor(256 / (3·b))` points per row when a row holds full `(x, y, z)` triplets.
- **Leftover bits in every row are zero-padded.**
- The final row of a group may hold fewer values than the others.
- Two packing styles appear below:
  - **Interleaved** — `x0, y0, z0, x1, y1, z1, …` (whole points, in sequence).
  - **Per-axis** — a row (or group of rows) holds values for a single axis: all `x`, then all `y`, then all `z`.

## 1. Nearest Neighbor Search — *Vector Nearest Neighbor*

### Input (interleaved points)

Each row holds whole points in sequence; every point keeps all three coordinates together. `r` points per row, where `r = floor(256 / (3·b))`. As many rows as needed for `n` points.

```
row 0 : x0, y0, z0, x1, y1, z1, ... , x(r-1), y(r-1), z(r-1)   | 0-pad
row 1 : xr, yr, zr, ...                                        | 0-pad
  :
row k : ... up to n points (final row may be partial)          | 0-pad
```

### Output (per-axis)

Coordinates are split by axis into three consecutive groups: all `x`, then all `y`, then all `z`. Values per row `= floor(256 / b)`.

```
--- X group ---
row 0      : x0, x1, x2, ...           | 0-pad
  :
--- Y group ---
row a      : y0, y1, y2, ...           | 0-pad
  :
--- Z group ---
row b      : z0, z1, z2, ...           | 0-pad
  :
```

| | Input | Output |
|---|---|---|
| Packing | Interleaved `(x,y,z)` per point | Per-axis: X group, Y group, Z group |
| Values / row | `floor(256/(3·b))` points | `floor(256/b)` coordinates |
| Padding | trailing bits per row | trailing bits per row |

## 2. Compute Centroid — *Average*

### Input

Identical to the **NN search output** (per-axis: all `x`, then all `y`, then all `z`).

### Output

Two identical rows, each holding the centroid `(x_c, y_c, z_c)`, zero-padded to 256 bits.

```
row 0 : x_c, y_c, z_c   | 0-pad
row 1 : x_c, y_c, z_c   | 0-pad   (identical to row 0)
```

| | Input | Output |
|---|---|---|
| Packing | Per-axis (= NN output) | `(x_c, y_c, z_c)` |
| Rows | as many as the point data needs | 2 (identical) |

## 3. Center — *Vector Processor*

### Input

- **Row 0:** one row of the **compute-centroid output** (the `x_c, y_c, z_c` row).
- **Rows 1…:** identical to the **NN search output** (per-axis groups).

```
row 0 : x_c, y_c, z_c            | 0-pad      ───► centroid
row 1 : x0, x1, x2, ...          | 0-pad      ─┐
  :                                            │─► per-axis point data
row k : ... z values ...         | 0-pad      ─┘   (= NN search output)
```

### Output

Identical to the **NN search output** (per-axis groups).

| | Input | Output |
|---|---|---|
| Packing | Row 0 = centroid; rows 1… = per-axis points | Per-axis (= NN output) |

## 4. Compute Covariance Matrix — *Matrix-Matrix Multiplication*

### Input

An interleaved, reordered version of the **center output**, carrying **both** point clouds (source `P` and nearest-target `Q`). Rows pair up the two clouds, one axis at a time, then repeat:

```
row 0 : P x values    | 0-pad
row 1 : Q x values    | 0-pad
row 2 : P y values    | 0-pad
row 3 : Q y values    | 0-pad
row 4 : P z values    | 0-pad
row 5 : Q z values    | 0-pad
row 6 : P x values    | 0-pad   (next block of points, back to x)
row 7 : Q x values    | 0-pad
  :
```

### Output

Each of the 9 scalars of the 3×3 matrix `H` occupies a **fixed 64-bit field**. This width is both a consequence of accumulation (the centered-coordinate products summed over the points need the headroom) and a deliberate interface choice: holding `H` at a fixed 64 bits — rather than a per-run bit width that tracks `n_coord_bits` and the point count — makes the subsequent int64→float32 promotion on the RISC-V unit a trivial fixed-width load. One matrix row per output row, zero-padded to 256 bits.

```
row 0 : h00, h01, h02   | 0-pad
row 1 : h10, h11, h12   | 0-pad
row 2 : h20, h21, h22   | 0-pad
```

| | Input | Output |
|---|---|---|
| Packing | Per-axis, interleaved P/Q, cycling x,y,z then repeating | One `H` row per output row |
| Field width | — | Fixed 64-bit per scalar (accumulation headroom + trivial RISC-V float32 promotion) |
| Layout | P→even rows, Q→odd rows within each x/y/z pair | row-major `H` (3 scalars/row) |

## 5. Compute Transformation Matrix — *RISC-V*

Not hardware-accelerated, so **no memory map is required.** Its layout is implicitly defined by the previous unit's output and the next unit's input.

## 6. Apply Transformation — *Vector Processor*

### Input

- **Rows 0–1:** transformation parameters — rotation matrix `R` (3×3, 9 scalars) and translation vector `t` (3×1, 3 scalars); 12 scalars total.
  - **Row 0:** `r00, r01, r02, t0, r10, r11, r12, t1`  (zero-pad to 256 if needed)
  - **Row 1:** `r20, r21, r22, t2`  (zero-pad to 256)
- **Rows 2…:** identical to the **NN search input** (interleaved points).

```
row 0 : r00, r01, r02, t0, r10, r11, r12, t1    | 0-pad
row 1 : r20, r21, r22, t2                       | 0-pad
row 2 : x0, y0, z0, x1, y1, z1, ...             | 0-pad   ─┐
row 3 : ...                                     | 0-pad    │─► interleaved points
  :                                                       ─┘   (= NN search input)
```

### Output

The transformed source point cloud, carried **twice** in two consecutive groups so it can fan out to its two downstream consumers, each of which expects a different layout:

1. **Interleaved group** — whole `(x, y, z)` points in sequence, `floor(256 / (3·b))` points per row. This is the **NN search input** form, routed to the next iteration's Nearest Neighbor Search.
2. **Per-axis group** — all `x`, then all `y`, then all `z`, `floor(256 / b)` values per row. This is the **NN search output / Average input** form, routed to Compute Centroid.

The per-axis group starts on a fresh row (the interleaved group zero-pads its final row), matching the fresh-row convention used for the X/Y/Z groups elsewhere. The two groups hold identical data in different packings; a consumer reads only the group matching its expected input layout.

```
--- interleaved group (= NN search input) ---
row 0 : x0, y0, z0, x1, y1, z1, ...     | 0-pad
row 1 : ...                             | 0-pad
  :
--- per-axis group (= Average input) ---
row a : x0, x1, x2, ...                 | 0-pad
  :
row b : y0, y1, y2, ...                 | 0-pad
  :
row c : z0, z1, z2, ...                 | 0-pad
  :
```

| | Input | Output |
|---|---|---|
| Rows 0–1 | Transform params: `R` (9) + `t` (3), split 8 scalars / 4 scalars | — |
| Rows 2… | Interleaved `(x,y,z)` points (= NN search input) | — |
| Packing | as above | Interleaved group **then** per-axis group (same data, both forms) |
| Consumers | — | Interleaved → NN search; per-axis → Average |