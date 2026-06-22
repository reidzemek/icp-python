"""Generate SVG diagrams for the ICP harness design specification.

Each function writes a self-contained SVG. Style is consistent across all
figures: a restrained palette, system sans-serif, rounded boxes, thin strokes.
"""

from pathlib import Path
import textwrap

OUT = Path("./figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared style tokens ───────────────────────────────────────────────────────
INK      = "#1f2933"   # near-black text
MUTED    = "#6b7280"   # secondary text
LINE     = "#9aa5b1"   # connectors
BORDER   = "#cbd2d9"   # box borders
BG_BLUE  = "#eef4fb"   # process / harness
ED_BLUE  = "#4a78b5"
BG_GREEN = "#eef7f0"   # config / data
ED_GREEN = "#4a9d6a"
BG_AMBER = "#fcf4e8"   # decisions / outputs
ED_AMBER = "#c8923a"
BG_GREY  = "#f3f4f6"   # neutral
ED_GREY  = "#9aa5b1"
FONT = ('font-family="ui-sans-serif,-apple-system,Segoe UI,Roboto,Helvetica,'
        'Arial,sans-serif"')


def esc(text: str) -> str:
    """Escape characters that are invalid in SVG/XML text content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, text, fill, edge, *, fs=13, bold_first=False, rx=8,
        text_fill=INK):
    """A rounded rectangle with (optionally multi-line) centered text."""
    lines = [esc(ln) for ln in text.split("\n")]
    line_h = fs + 4
    total = line_h * len(lines)
    start_y = y + h / 2 - total / 2 + fs
    spans = []
    for i, ln in enumerate(lines):
        weight = "600" if (bold_first and i == 0) else "400"
        col = text_fill if (bold_first and i == 0) else (text_fill if i == 0 else MUTED)
        spans.append(
            f'<text x="{x + w/2:.1f}" y="{start_y + i*line_h:.1f}" '
            f'text-anchor="middle" {FONT} font-size="{fs}" '
            f'font-weight="{weight}" fill="{col}">{ln}</text>'
        )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{edge}" stroke-width="1.5"/>\n' + "\n".join(spans)
    )


def arrow(x1, y1, x2, y2, *, label=None, dashed=False, color=LINE):
    dash = 'stroke-dasharray="5 4" ' if dashed else ""
    lbl = ""
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        lbl = (f'<rect x="{mx-len(label)*3.4-5:.1f}" y="{my-11:.1f}" '
               f'width="{len(label)*6.8+10:.1f}" height="18" rx="4" '
               f'fill="white" opacity="0.92"/>'
               f'<text x="{mx:.1f}" y="{my+3:.1f}" text-anchor="middle" '
               f'{FONT} font-size="11" fill="{MUTED}">{label}</text>')
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="1.6" {dash}marker-end="url(#arrow)"/>\n{lbl}'
    )


def header(svg_w, title, subtitle=None):
    s = (f'<text x="{svg_w/2}" y="30" text-anchor="middle" {FONT} '
         f'font-size="17" font-weight="600" fill="{INK}">{esc(title)}</text>')
    if subtitle:
        s += (f'\n<text x="{svg_w/2}" y="50" text-anchor="middle" {FONT} '
              f'font-size="12" fill="{MUTED}">{esc(subtitle)}</text>')
    return s


def svg_wrap(w, h, body):
    return (
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="sans-serif">\n'
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker></defs>\n'
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>\n'
        f'{body}\n</svg>\n'
    )


# ── Figure 1: System architecture ─────────────────────────────────────────────
def fig_architecture():
    w, h = 760, 470
    b = [header(w, "System Architecture",
                "Separation of exploration, configuration, and execution")]

    # config files (left)
    b.append(box(40, 90, 180, 60, "config.yaml\nflat parameters", BG_GREEN, ED_GREEN,
                 bold_first=True))
    b.append(box(40, 175, 180, 70, "sweep.yaml\nextends: config.yaml\n+ sweep axes",
                 BG_GREEN, ED_GREEN, bold_first=True))

    # harness (center)
    b.append(box(300, 90, 180, 155, "run_icp.py\n\nrun command\nsweep command\n\n"
                 "validation\nauto-range\nconfig resolution", BG_BLUE, ED_BLUE,
                 bold_first=True, fs=12))

    # core lib (center-bottom)
    b.append(box(300, 290, 180, 95, "Core library\n\nICP · PointCloud\nKDTree · TargetCloud",
                 BG_GREY, ED_GREY, bold_first=True, fs=12))

    # outputs (right)
    b.append(box(560, 90, 175, 60, "run_NNN/\nconfig_used.yaml", BG_AMBER, ED_AMBER,
                 bold_first=True))
    b.append(box(560, 175, 175, 60, "sweep_manifest.yaml\nindex → axis values", BG_AMBER,
                 ED_AMBER, bold_first=True))
    b.append(box(560, 290, 175, 95, "Trace artifacts\n\nCSV · .mem\n(point clouds, trees)",
                 BG_AMBER, ED_AMBER, bold_first=True, fs=12))

    # separate analysis notebook
    b.append(box(40, 290, 180, 95, "Analysis notebook\n(marimo / Jupyter)\n\n"
                 "distribution plots\n(separate concern)", BG_GREY, ED_GREY,
                 bold_first=True, fs=12))

    # arrows
    b.append(arrow(220, 120, 300, 140))
    b.append(arrow(220, 210, 300, 175))
    b.append(arrow(480, 150, 560, 130))
    b.append(arrow(480, 175, 560, 195))
    b.append(arrow(390, 245, 390, 290))
    b.append(arrow(480, 337, 560, 337))

    return svg_wrap(w, h, "\n".join(b))


# ── Figure 2: Config resolution & loading logic ──────────────────────────────
def fig_resolution():
    w, h = 720, 540
    b = [header(w, "Config Loading & Command Dispatch")]

    b.append(box(270, 70, 180, 46, "Load YAML file", BG_GREY, ED_GREY, bold_first=True))

    # decision: has sweep/extends?
    b.append(box(255, 150, 210, 56, "Has extends: / sweep: ?", BG_AMBER, ED_AMBER, fs=13))
    b.append(arrow(360, 116, 360, 150))

    # plain config branch (left)
    b.append(box(60, 250, 200, 60, "Plain config\nparameters at top level",
                 BG_GREEN, ED_GREEN, bold_first=True))
    b.append(arrow(290, 206, 160, 250, label="no"))

    # sweep file branch (right)
    b.append(box(460, 250, 210, 60, "Sweep file\nresolve extends: → base params",
                 BG_GREEN, ED_GREEN, bold_first=True, fs=12))
    b.append(arrow(430, 206, 565, 250, label="yes"))

    # command split
    b.append(box(60, 360, 200, 70, "run command\n\nrun single config\n(warn if sweep: ignored)",
                 BG_BLUE, ED_BLUE, bold_first=True, fs=12))
    b.append(arrow(160, 310, 160, 360))

    b.append(box(460, 360, 210, 70, "sweep command\n\nrequires extends: + sweep:\nvalidate axes vs allowlist",
                 BG_BLUE, ED_BLUE, bold_first=True, fs=12))
    b.append(arrow(565, 310, 565, 360))

    # selectors under sweep
    b.append(box(430, 460, 110, 50, "--index N\nselect point", BG_GREY, ED_GREY, fs=11))
    b.append(box(560, 460, 110, 50, "--list\nshow grid", BG_GREY, ED_GREY, fs=11))
    b.append(arrow(510, 430, 485, 460))
    b.append(arrow(610, 430, 615, 460))

    return svg_wrap(w, h, "\n".join(b))


# ── Figure 3: norm_range auto-computation ─────────────────────────────────────
def fig_normrange():
    w, h = 720, 430
    b = [header(w, "norm_range Resolution",
                "Explicit value, or global auto-computation over P ∪ Q")]

    b.append(box(265, 75, 190, 50, "norm_range in config", BG_GREEN, ED_GREEN,
                 bold_first=True))
    b.append(box(255, 160, 210, 50, "norm_range is null ?", BG_AMBER, ED_AMBER))
    b.append(arrow(360, 125, 360, 160))

    # explicit (left)
    b.append(box(70, 270, 200, 60, "Use [lo, hi] verbatim\nignore norm_margin_frac",
                 BG_GREY, ED_GREY, bold_first=True, fs=12))
    b.append(arrow(290, 210, 170, 270, label="no"))

    # auto (right)
    b.append(box(450, 250, 220, 110,
                 "Auto-compute (global)\n\n"
                 "scan P ∪ Q over processed pairs\n"
                 "min/max across all coords\n"
                 "pad ± margin_frac × span",
                 BG_BLUE, ED_BLUE, bold_first=True, fs=11))
    b.append(arrow(430, 210, 560, 250, label="yes"))

    b.append(f'<text x="{w/2}" y="405" text-anchor="middle" {FONT} font-size="11.5" '
             f'fill="{MUTED}">Computed once per run · reused across all sweep points · '
             f'recorded in config_used.yaml</text>')
    return svg_wrap(w, h, "\n".join(b))


# ── Figure 4: Sweep grid (Cartesian product) ──────────────────────────────────
def fig_grid():
    w, h = 700, 470
    b = [header(w, "Sweep Grid — Cartesian Product",
                "itertools.product · last axis varies fastest")]

    A = [8, 12, 16]          # n_coord_bits
    B = [5, 10, 20]          # n_icp_iters
    cell = 92
    x0, y0 = 240, 110

    # axis labels
    b.append(f'<text x="{x0 + len(B)*cell/2:.0f}" y="75" text-anchor="middle" '
             f'{FONT} font-size="12.5" font-weight="600" fill="{ED_GREEN}">'
             f'icp.n_icp_iters  (inner / fastest)</text>')
    b.append(f'<text x="185" y="{y0 + len(A)*cell/2:.0f}" text-anchor="middle" '
             f'{FONT} font-size="12.5" font-weight="600" fill="{ED_BLUE}" '
             f'transform="rotate(-90 185 {y0 + len(A)*cell/2:.0f})">'
             f'quantization.n_coord_bits  (outer)</text>')

    # column headers
    for j, bv in enumerate(B):
        b.append(f'<text x="{x0 + j*cell + cell/2:.0f}" y="{y0-12:.0f}" '
                 f'text-anchor="middle" {FONT} font-size="12" fill="{MUTED}">'
                 f'={bv}</text>')
    # row headers
    for i, av in enumerate(A):
        b.append(f'<text x="{x0-14:.0f}" y="{y0 + i*cell + cell/2 + 4:.0f}" '
                 f'text-anchor="end" {FONT} font-size="12" fill="{MUTED}">'
                 f'={av}</text>')

    idx = 0
    for i, av in enumerate(A):
        for j, bv in enumerate(B):
            cx, cy = x0 + j*cell, y0 + i*cell
            b.append(box(cx+6, cy+6, cell-12, cell-12,
                         f"run_{idx:03d}\n({av}, {bv})", BG_AMBER, ED_AMBER,
                         fs=12, bold_first=True))
            idx += 1

    b.append(f'<text x="{w/2}" y="445" text-anchor="middle" {FONT} font-size="11.5" '
             f'fill="{MUTED}">3 × 3 = 9 points · index 0→8 maps directly to '
             f'run_000 → run_008</text>')
    return svg_wrap(w, h, "\n".join(b))


# ── Figure 5: Sweep execution flow (two-pass) ─────────────────────────────────
def fig_execution():
    w, h = 320, 600
    b = [header(w, "Sweep Execution")]

    steps = [
        ("Load & validate sweep file", BG_GREY, ED_GREY,
         "extends: + sweep: · axes in allowlist"),
        ("Enumerate grid", BG_BLUE, ED_BLUE,
         "itertools.product over axes"),
        ("Pass 1: range discovery", BG_BLUE, ED_BLUE,
         "if norm_range null · scan P ∪ Q · once"),
        ("Pass 2: per-point loop", BG_BLUE, ED_BLUE,
         "resolve config · run ICP · write traces"),
        ("Write outputs", BG_AMBER, ED_AMBER,
         "run_NNN/ + config_used.yaml"),
        ("Write manifest", BG_AMBER, ED_AMBER,
         "sweep_manifest.yaml"),
    ]
    y = 70
    bh = 66
    gap = 26
    cx = 40
    cw = w - 80
    prev_cy = None
    for title, fill, edge, sub in steps:
        b.append(box(cx, y, cw, bh, f"{title}\n{sub}", fill, edge, bold_first=True, fs=12))
        if prev_cy is not None:
            b.append(arrow(w/2, prev_cy, w/2, y))
        prev_cy = y + bh
        y += bh + gap
    return svg_wrap(w, h, "\n".join(b))


def main():
    figs = {
        "01_architecture.svg": fig_architecture(),
        "02_config_resolution.svg": fig_resolution(),
        "03_norm_range.svg": fig_normrange(),
        "04_sweep_grid.svg": fig_grid(),
        "05_execution_flow.svg": fig_execution(),
    }
    for name, svg in figs.items():
        (OUT / name).write_text(svg)
        print(f"wrote {name} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()