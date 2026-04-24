"""
visual_report.py

Generates a matplotlib chart for audit reports.
"""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DISTORTION_COLORS = {
    "none":                "#10b981",
    "accurate":            "#10b981",
    "certainty_inflation": "#f59e0b",
    "causal_overclaim":    "#ef4444",
    "scope_inflation":     "#eab308",
    "cherry_picking":      "#8b5cf6",
    "unverifiable":        "#6b7280",
}


def generate_audit_infographic(report) -> bytes:
    entries = report.entries
    n = len(entries)

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.5 + 2)))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    labels = [f"Claim {e.claim_id}" for e in entries]
    values = [max(e.confidence, 0.05) for e in entries]
    colors = [DISTORTION_COLORS.get(e.distortion_type, "#6b7280") for e in entries]

    bars = ax.barh(labels, values, color=colors, height=0.6, edgecolor='none')

    for bar, entry in zip(bars, entries):
        label = entry.distortion_type.replace("_", " ")
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            label,
            va='center', ha='left', fontsize=8,
            color='white', alpha=0.85,
        )

    ax.set_xlim(0, 1.45)
    ax.set_xlabel("Classifier Confidence", color='white', fontsize=10)
    ax.set_title(
        f"Citation Audit — {report.paper_title[:50]}{'...' if len(report.paper_title) > 50 else ''}\n"
        f"Integrity Score: {report.integrity_score:.0f}/100  |  Risk: {report.overall_risk.upper()}",
        color='white', fontsize=11, pad=12,
    )
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    legend_items = [
        mpatches.Patch(color="#10b981", label="Accurate"),
        mpatches.Patch(color="#f59e0b", label="Certainty Inflation"),
        mpatches.Patch(color="#ef4444", label="Causal Overclaim"),
        mpatches.Patch(color="#eab308", label="Scope Inflation"),
        mpatches.Patch(color="#8b5cf6", label="Cherry Picking"),
        mpatches.Patch(color="#6b7280", label="Unverifiable"),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=7,
              facecolor='#1e293b', edgecolor='#334155', labelcolor='white')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return buf.getvalue()
