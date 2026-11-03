"""Render the training objective and the cascade rule as a slide image."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)

NAVY = "#0B2545"
GREY = "#5B6770"
BLUE = "#1D4E89"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"

fig = plt.figure(figsize=(11.0, 3.6))
fig.patch.set_facecolor("white")

ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 11)
ax.set_ylim(0, 3.6)
ax.axis("off")

ax.text(0.35, 3.15, "Huấn luyện", fontsize=13, color=BLUE, weight="bold")
ax.text(0.35, 2.55,
        r"$\mathcal{L} \;=\; \mathcal{L}_{\mathrm{bin}}(b) \;+\; \mathbf{1}[\,y_{\mathrm{bin}}=1\,]\;\mathcal{L}_{\mathrm{type}}(t)$",
        fontsize=25, color=NAVY, va="center")
ax.text(0.35, 2.02,
        "chỉ tính loss loại rác khi dòng đó thực sự là rác",
        fontsize=12, color=GREY, style="italic")

ax.plot([0.35, 10.6], [1.72, 1.72], color="#DDDDDD", linewidth=1)

ax.text(0.35, 1.42, "Suy luận (cascade cứng)", fontsize=13, color=BLUE, weight="bold")
ax.text(0.35, 0.85,
        r"$p(c{=}0)=p(b{=}0)$      $p(c{=}k)=p(b{=}1)\;p(t{=}k{-}1)$",
        fontsize=22, color=NAVY, va="center")
ax.text(0.35, 0.30,
        "tích hai marginal chỉ dùng để xem độ tin cậy — dự đoán thật lấy argmax của cascade",
        fontsize=12, color=GREY, style="italic")

fig.savefig(os.path.join(OUT, "loss_equations.png"), dpi=220, bbox_inches="tight",
            facecolor="white", pad_inches=0.12)
plt.close(fig)
print("written", os.path.join(OUT, "loss_equations.png"))
