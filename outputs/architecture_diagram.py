"""
Architecture diagram generator for the Research Claim Auditor.
Run once to produce outputs/architecture_diagram.png.

Usage:
    python outputs/architecture_diagram.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless — no display required

# Extend font fallback chain so emoji glyphs resolve on macOS/Linux
import matplotlib.font_manager as _fm
_available = {f.name for f in _fm.fontManager.ttflist}
_emoji_candidates = ["Apple Color Emoji", "Noto Color Emoji", "Segoe UI Emoji"]
_emoji_font = next((f for f in _emoji_candidates if f in _available), None)
_family = ["DejaVu Sans"] + ([_emoji_font] if _emoji_font else [])
matplotlib.rcParams["font.family"] = _family

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
C_GRAY   = "#7f8c8d"
C_BLUE   = "#2980b9"
C_PURPLE = "#8e44ad"
C_GREEN  = "#27ae60"
C_ORANGE = "#e67e22"
C_BG     = "#f8f9fa"
C_ARROW  = "#2c3e50"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BOX_W = 1.45
BOX_H = 0.72


def _box(ax, cx: float, cy: float, label: str, color: str) -> None:
    patch = FancyBboxPatch(
        (cx - BOX_W / 2, cy - BOX_H / 2),
        BOX_W, BOX_H,
        boxstyle="round,pad=0.08",
        linewidth=1.8,
        edgecolor="white",
        facecolor=color,
        zorder=3,
        alpha=0.92,
    )
    ax.add_patch(patch)
    ax.text(
        cx, cy, label,
        ha="center", va="center",
        fontsize=8.5, fontweight="bold",
        color="white", zorder=4,
        multialignment="center",
    )


def _sublabel(ax, cx: float, cy_bottom: float, text: str) -> None:
    ax.text(
        cx, cy_bottom - 0.18, text,
        ha="center", va="top",
        fontsize=6.8, color="#4a4a4a",
        linespacing=1.55, multialignment="center",
        zorder=2,
    )


def _arrow(ax, x1, y1, x2, y2, style="->", color=C_ARROW,
           conn="arc3,rad=0", lw=1.6) -> None:
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            connectionstyle=conn,
        ),
        zorder=5,
    )


# ---------------------------------------------------------------------------
# Main diagram
# ---------------------------------------------------------------------------

def generate(out_path: str = "outputs/architecture_diagram.png") -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(-0.6, 10.4)
    ax.set_ylim(-3.6, 2.0)
    ax.axis("off")

    # ── Title & subtitle ────────────────────────────────────────────────────
    ax.text(
        5.0, 1.75,
        "Research Claim Auditor — System Architecture",
        ha="center", va="center",
        fontsize=15, fontweight="bold", color="#2c3e50",
    )
    ax.text(
        5.0, 1.38,
        "Agentic RAG Pipeline for Citation Integrity Detection",
        ha="center", va="center",
        fontsize=10, color="#7f8c8d", style="italic",
    )

    # ── Top-row boxes ────────────────────────────────────────────────────────
    # Box 1 — Input Paper
    _box(ax, 0.5, 0.0, "Input\nPaper", C_GRAY)

    # Box 2 — Claim Extractor
    _box(ax, 2.5, 0.0, "Claim\nExtractor", C_BLUE)
    _sublabel(ax, 2.5, -BOX_H / 2,
              "Claude Haiku\nHedging detection\nClaim strength")

    # Box 3 — Source Retriever
    _box(ax, 4.5, 0.0, "Source\nRetriever", C_BLUE)
    _sublabel(ax, 4.5, -BOX_H / 2,
              "sentence-transformers\nFAISS index\nTop-k retrieval")

    # Box 4 — Distortion Classifier
    _box(ax, 6.5, 0.0, "Distortion\nClassifier", C_PURPLE)
    _sublabel(ax, 6.5, -BOX_H / 2,
              "Claude Haiku\n5-type taxonomy\nSilent failure test")

    # Box 5 — Report Generator
    _box(ax, 8.5, 0.0, "Report\nGenerator", C_GREEN)
    _sublabel(ax, 8.5, -BOX_H / 2,
              "Integrity score\nJSON + text output")

    # ── Bottom box ────────────────────────────────────────────────────────────
    # Box 6 — Retraction Checker (below Box 3)
    _box(ax, 4.5, -1.9, "Retraction\nChecker", C_ORANGE)
    _sublabel(ax, 4.5, -1.9 - BOX_H / 2,
              "Retraction Watch CSV\nDOI + fuzzy match\nDeterministic")

    # ── Horizontal arrows (top row) ──────────────────────────────────────────
    gap = 0.08  # clearance from box edge
    right = BOX_W / 2 + gap
    left  = -(BOX_W / 2 + gap)

    _arrow(ax, 0.5 + right, 0.0,  2.5 + left,  0.0)   # 1 → 2
    _arrow(ax, 2.5 + right, 0.0,  4.5 + left,  0.0)   # 2 → 3
    _arrow(ax, 4.5 + right, 0.0,  6.5 + left,  0.0)   # 3 → 4
    _arrow(ax, 6.5 + right, 0.0,  8.5 + left,  0.0)   # 4 → 5

    # ── Vertical double-headed arrow: Box3 ↕ Box6 ───────────────────────────
    bot3 = -BOX_H / 2 - gap          # bottom of box 3  ≈ -0.44
    top6 = -1.9 + BOX_H / 2 + gap   # top of box 6     ≈ -1.54

    _arrow(ax, 4.5, bot3, 4.5, top6, style="<->")

    # ── Diagonal arrow: Box6 → Box4 ─────────────────────────────────────────
    # From right edge of Box6 to bottom-left of Box4
    x6r = 4.5 + BOX_W / 2 + gap       # ~5.33
    y6c = -1.9
    x4b = 6.5                          # bottom-centre of box 4
    y4b = -BOX_H / 2 - gap            # ~-0.44

    _arrow(ax, x6r, y6c, x4b, y4b, conn="arc3,rad=-0.25")

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_items = [
        (C_GRAY,   "User Input"),
        (C_BLUE,   "LLM Component"),
        (C_PURPLE, "LLM Component (classification)"),
        (C_ORANGE, "Deterministic"),
        (C_GREEN,  "Output"),
    ]
    lx, ly = 7.2, -2.5
    ax.text(lx, ly + 0.30, "Legend", fontsize=8.5, fontweight="bold",
            color="#2c3e50", va="bottom")
    for idx, (col, label) in enumerate(legend_items):
        y_pos = ly - idx * 0.32
        sq = FancyBboxPatch(
            (lx, y_pos - 0.10), 0.22, 0.22,
            boxstyle="round,pad=0.02",
            facecolor=col, edgecolor="white", lw=1.0, zorder=4,
        )
        ax.add_patch(sq)
        ax.text(lx + 0.30, y_pos + 0.01, label, fontsize=7.5,
                color="#2c3e50", va="center")

    # ── Step labels on arrows ────────────────────────────────────────────────
    step_labels = [
        (1.5, 0.12, "① Extract"),
        (3.5, 0.12, "② Retrieve"),
        (5.5, 0.12, "③ Classify"),
        (7.5, 0.12, "④ Report"),
    ]
    for lx2, ly2, ltxt in step_labels:
        ax.text(lx2, ly2, ltxt, ha="center", va="bottom",
                fontsize=6.5, color="#888888", style="italic")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.tight_layout(pad=0.4)
    plt.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=C_BG, edgecolor="none")
    plt.close(fig)


if __name__ == "__main__":
    out = "outputs/architecture_diagram.png"
    generate(out)
    print(f"Saved: {out}")
