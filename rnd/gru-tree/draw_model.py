"""Draws model.svg: the GRU-only tree as slot 147 ran it (gru-tree 8818093),
over 16 characters of text. Run: python3 rnd/gru-tree/draw_model.py

The SVG has its own light background and system fonts, so it reads the same
on GitHub in light or dark mode."""

from pathlib import Path

OUT = Path(__file__).with_name("model.svg")

INK, MUTED, RULE, ACCENT, SHUT = "#18212b", "#5a6573", "#d3d9df", "#0c7a72", "#9aa6b2"


def arrow(x1, y1, x2, y2, cls="ln", head="ah"):
    return f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#{head})"/>'


def line(x1, y1, x2, y2, cls="ln"):
    return f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'


def poly(points, cls="ln", head="ah"):
    pts = " ".join(f"{x},{y}" for x, y in points)
    end = f' marker-end="url(#{head})"' if head else ""
    return f'<polyline class="{cls}" points="{pts}"{end}/>'


def text(x, y, s, cls="lbl", anchor="start"):
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{s}</text>'


def box(x, y, w, h, cls="bx", rx=4):
    return f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/>'


def dot(x, y, cls="dot", r=3.5):
    return f'<circle class="{cls}" cx="{x}" cy="{y}" r="{r}"/>'


chars = list("a cat sat on the")  # 16 characters from mid-piece: places 1,025..1,040
BASE = 1024  # character n of the drawing is place BASE + n in its piece (p + 1)
X = {n: 246 + 44 * (n - 1) for n in range(1, 17)}
READ = 12  # the character whose read is drawn
COL = X[READ] + 22  # its read column, between characters 12 and 13
LEFT, RIGHT = 200, 932  # where earlier states enter, where the rows run on


def place(n):
    return f"{BASE + n:,}"

CY = 456  # the character GRU
LEVELS = [  # (level, y, every, width label, gate after epoch 3, open)
    (1, 372, 2, "every 2nd character", -0.2669, True),
    (2, 302, 4, "every 4th", 0.1482, True),
    (3, 232, 8, "every 8th", -0.0098, False),
    (4, 162, 16, "every 16th", 0.0011, False),
]
TOPY = 92  # levels 5-10, drawn as one row
BW, BH = 28, 24  # a level's box

f = []

# ---- row labels
for k, y, every, sub, _, _ in LEVELS:
    f.append(text(20, y - 1, f"level {k} · GRU 132"))
    f.append(text(20, y + 14, sub, "sub"))
f.append(text(20, TOPY - 1, "levels 5–10 · GRU 132"))
f.append(text(20, TOPY + 14, "every 32nd … 1,024th", "sub"))
f.append(text(20, CY - 1, "character GRU · 192"))
f.append(text(20, CY + 14, "every character", "sub"))

# ---- levels 5-10: one dashed row, no step inside these 16 characters
f.append(arrow(LEFT, TOPY, RIGHT, TOPY, "ln dash"))
f.append(text(LEFT + 10, TOPY - 8, f"no step here: all ten levels last stepped after {place(0)}; level 5 next after {BASE + 32:,}", "sub"))

# ---- the levels: recurrence along each row, one step each 2^k characters
for k, y, every, _, _, _ in LEVELS:
    steps = [n for n in range(1, 17) if n % every == 0]
    xs = [X[n] for n in steps]
    f.append(arrow(LEFT, y, xs[0] - BW / 2 - 1, y))
    for a, b in zip(xs, xs[1:]):
        f.append(arrow(a + BW / 2, y, b - BW / 2 - 1, y))
    f.append(arrow(xs[-1] + BW / 2, y, RIGHT, y))
    for n in steps:
        f.append(box(X[n] - BW / 2, y - BH / 2, BW, BH))

# ---- the feeds up: level k reads level k-1's state after every 2nd step of it
lower_y = {0: CY - 14}
for k, y, every, _, _, _ in LEVELS:
    lower_y[k] = y - BH / 2
for k, y, every, _, _, _ in LEVELS:
    for n in range(every, 17, every):
        f.append(arrow(X[n], lower_y[k - 1], X[n], y + BH / 2 + 1))
f.append(text(X[2] + 6, CY - 34, "its state", "sub"))
f.append(text(X[2] + 6, CY - 22, "after the pair", "sub"))

# ---- the character GRU, one step a character
f.append(arrow(LEFT, CY, X[1] - 15, CY))
for n, c in zip(range(1, 17), chars):
    x = X[n]
    f.append(box(x - 14, CY - 14, 28, 28, "bx strong" if n == READ else "bx"))
    f.append(text(x, CY + 5, "␣" if c == " " else c, "chr", "middle"))
    f.append(text(x, CY + 30, place(n), "num", "middle"))
    if n < 16:
        f.append(arrow(x + 14, CY, X[n + 1] - 15, CY))
f.append(arrow(X[16] + 14, CY, RIGHT, CY))
f.append(text(LEFT - 2, CY + 30, "place in the piece:", "num", "end"))

# ---- states from earlier in the piece
f.append(line(LEFT, TOPY + 12, LEFT, CY + 8, "ln faint"))
for i, s in enumerate(["Arrows in from the left: each row’s", "state from earlier in the piece, carried", "across training updates (the gradient", "stops at an update’s start)"]):
    f.append(text(20, 512 + 15 * i, s, "sub"))

# ---- the read at character 12: each level's line where it passes
f.append(line(COL, TOPY, COL, 540, "rd"))
for _, y, _, _, _, _ in LEVELS:
    f.append(dot(COL, y))
f.append(dot(COL, TOPY))
f.append(text(COL, 50, f"what character {place(READ)} reads:", "acc", "middle"))
f.append(text(COL, 66, "each level’s state after its latest step", "acc-sub", "middle"))
# where each read comes from
for txt, y in [(place(12), 372), (place(12), 302), (place(8), 232), (place(0), 162), (place(0), TOPY)]:
    f.append(text(COL + 8, y - 6, txt, "acc-sub"))

# ---- the output
PY = 540
f.append(f'<circle class="bx" cx="{X[READ]}" cy="{PY}" r="11"/>')
f.append(text(X[READ], PY + 5, "+", "lbl", "middle"))
f.append(arrow(X[READ], CY + 36, X[READ], PY - 12))
f.append(arrow(COL, PY, X[READ] + 12, PY, "rd", "aha"))
f.append(text(COL + 10, PY - 4, "each level’s state, projected 132 → 192,", "acc-sub"))
f.append(text(COL + 10, PY + 11, "times tanh of its gate; summed over all 10", "acc-sub"))
f.append(text(X[READ] - 18, PY + 5, "h", "lbl", "end"))
f.append(arrow(X[READ], PY + 11, X[READ], 573))
f.append(box(X[READ] - 135, 574, 270, 46, "bx", 5))
f.append(text(X[READ], 593, "output layer: Linear 192 → 96", "lbl", "middle"))
f.append(text(X[READ], 611, f"softmax → p(character {place(READ + 1)})", "sub", "middle"))

# ---- the gates after the screen
GX, AX = 985, 1072  # the column's left, its zero
f.append(text(GX, 40, "each level’s gate, tanh(g),", "sub"))
f.append(text(GX, 55, "after epoch 3 (slot 147)", "sub"))
f.append(line(AX, 142, AX, 392, "ln faint"))
for k, y, _, _, g, opened in LEVELS:
    w = abs(g) * 200
    x0 = AX if g >= 0 else AX - w
    f.append(f'<rect class="{"bar" if opened else "bar shut"}" x="{x0:.1f}" y="{y - 6}" width="{max(w, 1):.1f}" height="12"/>')
    tx = AX + 8 + w if g >= 0 else AX - 8 - w
    f.append(text(tx, y + 4, f"{g:+.2f}" if opened else f"{g:+.3f}", "val" if opened else "val shut-t", "start" if g >= 0 else "end"))
f.append(text(GX, TOPY + 4, "all six within ±0.003", "val shut-t"))
f.append(text(GX, 418, "levels 1–2 are read;", "sub"))
f.append(text(GX, 431, "3–10 stay shut", "sub"))

MARKERS = f"""<defs>
<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path fill="{INK}" d="M0,0 L10,5 L0,10 z"/></marker>
<marker id="aha" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path fill="{ACCENT}" d="M0,0 L10,5 L0,10 z"/></marker>
</defs>"""

STYLE = f"""<style>
text {{ font-family: -apple-system, "Segoe UI", "Source Sans 3", Helvetica, Arial, sans-serif; }}
.ln {{ stroke: {INK}; stroke-width: 1.2; fill: none; }}
.dash {{ stroke-dasharray: 5 4; stroke: {MUTED}; }}
.faint {{ stroke: {RULE}; stroke-dasharray: 3 3; }}
.bx {{ fill: #ffffff; stroke: {INK}; stroke-width: 1.2; }}
.strong {{ stroke: {ACCENT}; stroke-width: 2.2; }}
.rd {{ stroke: {ACCENT}; stroke-width: 1.6; fill: none; }}
.dot {{ fill: {ACCENT}; }}
.bar {{ fill: {ACCENT}; }}
.bar.shut {{ fill: {SHUT}; }}
.lbl {{ fill: {INK}; font-size: 14px; font-weight: 600; }}
.sub {{ fill: {MUTED}; font-size: 12.5px; }}
.acc {{ fill: {ACCENT}; font-size: 13.5px; font-weight: 600; }}
.acc-sub {{ fill: {ACCENT}; font-size: 12.5px; }}
.val {{ fill: {INK}; font-size: 13px; font-variant-numeric: tabular-nums; }}
.shut-t {{ fill: {MUTED}; }}
.chr {{ fill: {INK}; font: 15px ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}
.num {{ fill: {MUTED}; font-size: 11px; }}
.subs {{ font-size: 9px; }}
</style>"""

W, H = 1160, 640
svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
    'aria-label="The GRU-only tree: a character GRU steps at every character; level k, a GRU 132 wide, steps after '
    "every 2^k-th character, reading the state of the level below after its second step of each pair. Each character's "
    "prediction adds every level's latest state, projected and gated, to the character GRU's state before a linear "
    'output layer. After epoch 3 only levels 1 and 2 have open gates.">'
    + MARKERS + STYLE
    + f'<rect x="0" y="0" width="{W}" height="{H}" rx="8" fill="#ffffff" stroke="{RULE}"/>'
    + "".join(f) + "</svg>\n"
)
OUT.write_text(svg)
print(f"-> {OUT} ({len(svg):,} bytes)")
