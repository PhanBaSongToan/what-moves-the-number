"""Draw the single-task and metadata-fusion pipelines as slide images."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)

NAVY = "#0B2545"
BLUE = "#1D4E89"
LIGHT = "#E8F0FB"
GREY = "#5B6770"
GREEN = "#1E7B4F"
GREEN_FILL = "#E8F5EC"

plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, cx, cy, w, h, text, fill=LIGHT, edge=BLUE, size=11, weight="normal"):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=NAVY,
            weight=weight, zorder=3, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, style="-|>", color=GREY, lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, linewidth=lw, color=color,
                                 linestyle=ls, zorder=1,
                                 shrinkA=1, shrinkB=1))


def canvas(width, height, xlim=13.6):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, height * xlim / width)
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    return fig, ax


# ---------------------------------------------------------------- single task
fig, ax = canvas(13.2, 2.2)
y = 1.55
row = [
    (1.30, 2.20, "Review comment\n(văn bản thô)", LIGHT, BLUE),
    (4.05, 2.10, "Tokenizer\n≤ 100 token", LIGHT, BLUE),
    (6.85, 2.30, "PhoBERT-base\n135M tham số", LIGHT, BLUE),
    (9.70, 2.10, "Lớp linear\n4 chiều", LIGHT, BLUE),
    (12.30, 2.10, "Dự đoán\n4 lớp", GREEN_FILL, GREEN),
]
for cx, w, text, fill, edge in row:
    box(ax, cx, y, w, 0.95, text, fill=fill, edge=edge,
        weight="bold" if edge == GREEN else "normal", size=10.5)
for i in range(len(row) - 1):
    arrow(ax, row[i][0] + row[i][1] / 2 + 0.03, y, row[i + 1][0] - row[i + 1][1] / 2 - 0.03, y)
ax.text(6.8, 0.45, "Một encoder, một đầu ra: bốn lớp coi như ngang hàng",
        ha="center", fontsize=10.5, color=GREY, style="italic")
fig.savefig(os.path.join(OUT, "pipeline_single.png"), dpi=220, bbox_inches="tight",
            facecolor="white", pad_inches=0.05)
plt.close(fig)

# ---------------------------------------------------------------- fusion
fig, ax = canvas(13.2, 4.4)
top, bottom = 3.35, 1.15
col = [1.25, 3.75, 6.35, 9.05]
widths = [2.10, 1.95, 2.30, 1.80]

box(ax, col[0], top, widths[0], 0.95, "Review comment", size=10.5)
box(ax, col[1], top, widths[1], 0.95, "Tokenizer\n≤ 100 token", size=10.5)
box(ax, col[2], top, widths[2], 0.95, "PhoBERT\n(trọng số chia sẻ)", size=10.5)
box(ax, col[3], top, widths[3], 0.95, "Vector <s>\n768 chiều", size=10.5)

box(ax, col[0], bottom, widths[0], 0.95, "Mô tả\nsản phẩm", size=10.5)
box(ax, col[1], bottom, widths[1], 0.95, "Tokenizer\n≤ 100 token", size=10.5)
box(ax, col[2], bottom, widths[2], 0.95, "PhoBERT\n(cùng bộ trọng số)", size=10.5)
box(ax, col[3], bottom, widths[3], 0.95, "Mean-pool\n768 chiều", size=10.5)

for cy in (top, bottom):
    for i in range(3):
        arrow(ax, col[i] + widths[i] / 2 + 0.03, cy, col[i + 1] - widths[i + 1] / 2 - 0.03, cy)

ax.plot([col[2], col[2]], [bottom + 0.50, top - 0.50], linestyle=":", color=BLUE, linewidth=1.2)
ax.text(col[2] + 0.10, (top + bottom) / 2, "chia sẻ\ntrọng số", fontsize=8.5, color=BLUE, va="center")

merge_left, merge_right = 10.55, 13.45
merge_center = (merge_left + merge_right) / 2
arrow(ax, col[3] + widths[3] / 2 + 0.03, top, merge_left - 0.03, 2.72)
arrow(ax, col[3] + widths[3] / 2 + 0.03, bottom, merge_left - 0.03, 1.78)
box(ax, merge_center, 2.25, merge_right - merge_left, 1.85,
    "Nối 2 vector\n(1536 chiều)\n→ linear 4 chiều\n→ dự đoán 4 lớp",
    fill=GREEN_FILL, edge=GREEN, weight="bold", size=10.5)

ax.text(6.6, 0.35, "Hai đầu vào, hai lượt forward qua cùng một encoder, rồi nối vector",
        ha="center", fontsize=10.5, color=GREY, style="italic")
fig.savefig(os.path.join(OUT, "pipeline_fusion.png"), dpi=220, bbox_inches="tight",
            facecolor="white", pad_inches=0.05)
plt.close(fig)

print("written", OUT)
