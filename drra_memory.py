"""
drra_memory.py

A Python writer that reproduces the binary memory-file format of the Vesyla
DRRA C++ ``Array`` template (and its default ``IO = Array<65536, 256>``
instantiation), with bit-packing semantics tailored for row-at-a-time hardware
loading.

Bit ordering (faithful to the C++ ``assemble`` + ``bitset::to_string()``):
  * A row is ``row_width`` bits, indexed LSB-first in ``[0, row_width)``.
  * A value's own bit 0 (LSB) goes to the lowest free position in the row.
  * The first value written sits at the low bits; later values extend upward.
  * The emitted string is MSB-first (bit ``row_width-1`` first, bit 0 last),
    so the first value written appears at the RIGHT end of the line.

Packing model (point-aligned, no splitting across rows):
  * Values are np.int64, each interpreted at a per-write ``width`` (only the low
    ``width`` bits are stored). Out-of-range values raise, never silently mask.
  * Values are grouped into "points" of ``point_size`` values (default 1).
    A point (``point_size * width`` bits) is atomic across row boundaries: if it
    does not fit in the current row's remaining bits, the rest of the row is
    zero-filled and the whole point starts on the next row.
  * Writing advances an internal cursor; ``seek`` repositions it, and ``write``
    accepts an optional ``(row, bit)`` override (both required together).

Output format matches ``io_to_file``: one ``"<address> <bitstring>"`` line per
active (non-empty) row, in ascending address order.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np


class DrraMemory:
    """Sparse, bit-addressable memory image with point-aligned packing."""

    def __init__(
        self,
        max_rows: int = 65536,
        row_width: int = 256,
        align_addresses: bool = True,
    ) -> None:
        if max_rows <= 0:
            raise ValueError(f"max_rows must be positive, got {max_rows}")
        if row_width <= 0:
            raise ValueError(f"row_width must be positive, got {row_width}")
        self.max_rows: int = int(max_rows)
        self.row_width: int = int(row_width)
        # When True, addresses are right-aligned (space-padded) to the width of
        # the largest possible address, so every binary column starts at the
        # same offset. When False, output is byte-identical to the C++
        # io_to_file format. The C++ reader tolerates either (regex \s*..\s+).
        self.align_addresses: bool = bool(align_addresses)
        self._addr_width: int = len(str(self.max_rows - 1))

        # Sparse storage: row index -> Python int holding that row's bits.
        # Bit i of the int is the LSB-first bit i of the row, so the int value
        # equals sum(bit_i << i) and rendering is just a zero-padded binary.
        self._rows: dict[int, int] = {}

        # Cursor: next free (row, bit) position.
        self._cur_row: int = 0
        self._cur_bit: int = 0

    # ------------------------------------------------------------------ #
    # Cursor
    # ------------------------------------------------------------------ #
    def seek(self, row: int, bit: int = 0) -> "DrraMemory":
        """Move the cursor to (row, bit). Returns self for chaining."""
        self._validate_position(row, bit)
        self._cur_row = int(row)
        self._cur_bit = int(bit)
        return self

    def tell(self) -> tuple[int, int]:
        """Return the current cursor position as (row, bit)."""
        return (self._cur_row, self._cur_bit)

    def _validate_position(self, row: int, bit: int) -> None:
        if not (0 <= row < self.max_rows):
            raise IndexError(
                f"row {row} out of range [0, {self.max_rows})"
            )
        if not (0 <= bit < self.row_width):
            raise IndexError(
                f"bit {bit} out of range [0, {self.row_width})"
            )

    # ------------------------------------------------------------------ #
    # Encoding helpers
    # ------------------------------------------------------------------ #
    def _encode_value(self, value: int, width: int) -> int:
        """Return the low `width` bits of `value` (two's-complement), as an int.

        Raises ValueError if `value` does not fit in a signed `width`-bit field
        or an unsigned `width`-bit field. Accepts both signed and unsigned
        ranges so e.g. width=8 admits both -128..127 and 0..255.
        """
        v = int(value)  # np.int64 -> Python int (exact, unbounded)
        signed_min = -(1 << (width - 1))
        signed_max = (1 << (width - 1)) - 1
        unsigned_max = (1 << width) - 1
        if not (signed_min <= v <= unsigned_max):
            raise ValueError(
                f"value {v} does not fit in {width} bits "
                f"(allowed signed [{signed_min}, {signed_max}] "
                f"or unsigned [0, {unsigned_max}])"
            )
        return v & unsigned_max  # low `width` bits, two's-complement for negatives

    def _place_bits(self, row: int, bit_offset: int, bits: int, nbits: int) -> None:
        """OR `nbits` of `bits` into `row` starting at LSB position `bit_offset`."""
        mask = (1 << nbits) - 1
        bits &= mask
        existing = self._rows.get(row, 0)
        self._rows[row] = existing | (bits << bit_offset)

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def write(
        self,
        values: Iterable[int],
        width: int,
        point_size: int = 1,
        row: Optional[int] = None,
        bit: Optional[int] = None,
    ) -> "DrraMemory":
        """Pack `values` into the buffer.

        Args:
            values: sequence of integers (np.int64 / Python int) to write.
            width: effective bit width per value (low `width` bits stored).
            point_size: number of consecutive values forming an atomic point.
            row, bit: optional start-position override. Pass BOTH or NEITHER;
                passing only one raises ValueError. When given, the cursor jumps
                there before writing. When omitted, writing resumes at the cursor.

        After the call the cursor points just past the last written value.
        """
        # --- argument validation ---
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if width > self.row_width:
            raise ValueError(
                f"width {width} exceeds row_width {self.row_width}"
            )
        if point_size <= 0:
            raise ValueError(f"point_size must be positive, got {point_size}")

        point_bits = point_size * width
        if point_bits > self.row_width:
            raise ValueError(
                f"point of {point_size} x {width} = {point_bits} bits "
                f"exceeds row_width {self.row_width}; a point can never fit"
            )

        # Position override must be all-or-nothing.
        if (row is None) != (bit is None):
            raise ValueError(
                "pass both `row` and `bit`, or neither "
                f"(got row={row}, bit={bit})"
            )
        if row is not None:
            self.seek(row, bit)  # type: ignore[arg-type]

        vals = [int(v) for v in values]
        if len(vals) % point_size != 0:
            raise ValueError(
                f"number of values ({len(vals)}) is not a multiple of "
                f"point_size ({point_size})"
            )

        # --- pack point by point ---
        for p in range(0, len(vals), point_size):
            point = vals[p : p + point_size]

            # If the point doesn't fit in the remaining bits of the current row,
            # zero-fill the rest of the row (implicit) and move to the next row.
            if self._cur_bit + point_bits > self.row_width:
                self._cur_row += 1
                self._cur_bit = 0

            if self._cur_row >= self.max_rows:
                raise IndexError(
                    f"row index {self._cur_row} reached max_rows "
                    f"({self.max_rows}); buffer overflow"
                )

            # Place each value in the point at consecutive positions.
            for value in point:
                encoded = self._encode_value(value, width)
                self._place_bits(self._cur_row, self._cur_bit, encoded, width)
                self._cur_bit += width

            # A point never lands exactly mid-row in a way that splits, but the
            # cursor may now sit at row_width; normalize to the next row start.
            if self._cur_bit == self.row_width:
                self._cur_row += 1
                self._cur_bit = 0

        return self

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def _row_string(self, row: int) -> str:
        """MSB-first binary string of `row`, matching bitset::to_string()."""
        value = self._rows.get(row, 0)
        return format(value, f"0{self.row_width}b")

    def active_rows(self) -> List[int]:
        """Sorted list of non-empty row addresses."""
        return sorted(self._rows.keys())

    def lines(self) -> List[str]:
        """One '<address> <bitstring>' line per active row, ascending.

        If `align_addresses` is set, addresses are right-aligned to the width of
        the largest possible address so the binary column is vertically aligned.
        """
        if self.align_addresses:
            return [
                f"{r:>{self._addr_width}} {self._row_string(r)}"
                for r in self.active_rows()
            ]
        return [f"{r} {self._row_string(r)}" for r in self.active_rows()]

    def to_string(self) -> str:
        """All lines joined with newlines (trailing newline included)."""
        out = "\n".join(self.lines())
        return out + "\n" if out else ""

    def to_file(self, path: str) -> None:
        """Write the memory image to `path` in the C++ io_to_file format."""
        with open(path, "w") as f:
            f.write(self.to_string())

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear all data and reset the cursor to (0, 0)."""
        self._rows.clear()
        self._cur_row = 0
        self._cur_bit = 0

    def get_slice(self, row: int) -> int:
        """Raw integer value of a row (0 if inactive)."""
        return self._rows.get(row, 0)

    def __repr__(self) -> str:
        return (
            f"DrraMemory(max_rows={self.max_rows}, row_width={self.row_width}, "
            f"active_rows={len(self._rows)}, cursor={self.tell()})"
        )