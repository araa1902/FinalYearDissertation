#!/usr/bin/env python3
"""
Figure 6.4 - Per-Edge Attention Weight Changes (Baseline to Stress)

Left panel  : Top-10 amplifications  (Delta alpha > 0)
Right panel : Top-10 attenuations    (Delta alpha < 0)

Usage:
    python plot_attention_deltas_refactored.py
    python plot_attention_deltas_refactored.py --input results/regime_attention_matrices_20260413_120000.pkl
"""

import sys
import glob
import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Palette - cool-neutral base with precise semantic colours
# ---------------------------------------------------------------------------

_PALETTE: Dict[str, str] = {
    "bg":         "#FFFFFF",
    "axes_bg":    "#F8F9FB",       # very slight blue-grey tint; cleaner than pure white
    "text":       "#1C1C2E",       # near-black with slight warmth
    "muted":      "#6B7280",       # subdued labels
    "grid":       "#E5E7EB",       # light neutral grid
    "border":     "#D1D5DB",
    "zero_line":  "#9CA3AF",       # visible but unobtrusive zero rule
    "amp":        "#1A6EBD",       # amplification - blue
    "amp_light":  "#EBF3FB",       # amp bar background band
    "att":        "#D6232A",       # attenuation - red
    "att_light":  "#FDECEA",       # att bar background band
    "rank_text":  "#FFFFFF",       # rank badge text
}

mpl.rcParams.update({
    "figure.facecolor":   _PALETTE["bg"],
    "axes.facecolor":     _PALETTE["axes_bg"],
    "axes.edgecolor":     _PALETTE["border"],
    "axes.labelcolor":    _PALETTE["text"],
    "axes.titlecolor":    _PALETTE["text"],
    "xtick.color":        _PALETTE["muted"],
    "ytick.color":        _PALETTE["text"],
    "text.color":         _PALETTE["text"],
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.labelsize":     10,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    10,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.18,
    "savefig.facecolor":  _PALETTE["bg"],
    "savefig.edgecolor":  "none",
    "axes.grid":          False,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ===========================================================================

class AttentionDeltasVisualizer:
    """Generates Figure 6.4: per-edge attention weight changes."""

    def __init__(self, attention_file: str) -> None:
        self.attention_file = attention_file
        self.data = self._load_attention_data()
        self.tickers: List[str] = self.data["tickers"]
        self.regimes: Dict[str, np.ndarray] = {}
        self._prepare_regimes()

    def _load_attention_data(self) -> dict:
        with open(self.attention_file, "rb") as fh:
            data = pickle.load(fh)
        print(f"Loaded: {self.attention_file}")
        return data

    def _prepare_regimes(self) -> None:
        regime_data = self.data["regime_attentions"]
        for name in ("Baseline", "Stress", "Rally"):
            if name in regime_data:
                attn = regime_data[name]["attention"]
                if attn.ndim > 2:
                    attn = attn.mean(axis=0)
                self.regimes[name] = attn.astype(float)
        print(f"Prepared {len(self.regimes)} regime matrices")

    # -----------------------------------------------------------------------

    def generate_figure_6_4(self, output_path: Optional[str] = None) -> str:
        if "Baseline" not in self.regimes or "Stress" not in self.regimes:
            raise ValueError("Baseline and Stress regimes required.")

        delta = self.regimes["Stress"] - self.regimes["Baseline"]
        edges = _extract_edges(delta, self.tickers)

        amps = sorted([e for e in edges if e["delta"] > 0],
                      key=lambda x: x["delta"], reverse=True)[:10]
        atts = sorted([e for e in edges if e["delta"] < 0],
                      key=lambda x: x["delta"])[:10]

        # --- Layout: slightly taller to give labels breathing room -----------
        fig, axes = plt.subplots(
            1, 2, figsize=(15, 6.8), facecolor=_PALETTE["bg"]
        )
        fig.subplots_adjust(wspace=0.42)

        self._plot_edge_panel(axes[0], amps, side="amp")
        self._plot_edge_panel(axes[1], atts, side="att")

        # Shared x-axis label, centred across both panels
        fig.text(
            0.5, 0.01,
            "Attention Weight Change (Delta alpha)",
            ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=_PALETTE["text"],
        )

        fig.tight_layout(rect=[0, 0.045, 1, 0.965])

        output_path = output_path or "results/figure_6_4_edge_attention_changes.png"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved: {output_path}")
        return output_path

    # -----------------------------------------------------------------------

    @staticmethod
    def _plot_edge_panel(ax: plt.Axes, edges: List[dict], side: str) -> None:
        """
        Render one ranked horizontal-bar panel.

        Improvements over original:
          - Uniform bar opacity (no fading) - fading obscures magnitudes
          - Thin coloured band behind each row for readability
          - Rank badge on bar end (styled circle with rank number)
          - Clean, minimal x-axis; zero rule slightly heavier
          - Panel sub-title uses a coloured pill badge
        """
        is_amp   = side == "amp"
        bar_col  = _PALETTE["amp"]  if is_amp else _PALETTE["att"]
        band_col = _PALETTE["amp_light"] if is_amp else _PALETTE["att_light"]
        sign_sym = "(+)" if is_amp else "(-)"
        panel_letter = "(a)" if is_amp else "(b)"
        panel_label  = "Top-10 Amplifications" if is_amp else "Top-10 Attenuations"

        labels = [f"{e['source']} -> {e['target']}" for e in edges]
        values = [e["delta"] for e in edges]
        y_pos  = np.arange(len(labels))

        if not values:
            ax.set_visible(False)
            return

        abs_max = max(abs(v) for v in values)

        # --- Alternating row bands for readability -------------------------
        for i, y in enumerate(y_pos):
            ax.axhspan(y - 0.45, y + 0.45,
                       color=band_col if i % 2 == 0 else _PALETTE["axes_bg"],
                       zorder=0, linewidth=0)

        # --- Bars ----------------------------------------------------------
        bars = ax.barh(
            y_pos, values,
            color=bar_col, alpha=0.82,
            edgecolor="none", height=0.62,
            zorder=3,
        )

        # --- Value annotations (right-end of bar) --------------------------
        offset = abs_max * 0.022
        for bar, val in zip(bars, values):
            ha = "left"  if val >= 0 else "right"
            x  = val + offset if val >= 0 else val - offset
            ax.text(
                x, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}",
                va="center", ha=ha,
                fontsize=12, fontweight="bold",
                color=bar_col, zorder=5,
            )

        # --- Rank badges removed for cleaner appearance ---

        # --- Axes config ---------------------------------------------------
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=13, fontweight="bold", color=_PALETTE["text"])
        ax.invert_yaxis()

        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.tick_params(axis="x", length=3, color=_PALETTE["border"])

        # Zero rule - clear, not distracting
        ax.axvline(0, color=_PALETTE["zero_line"], linewidth=1.1,
                   linestyle="-", zorder=2)

        # Subtle vertical grid only
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=_PALETTE["grid"],
                linewidth=0.6, linestyle="--", alpha=1.0, zorder=1)

        # Spines: keep only bottom
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(_PALETTE["border"])
        ax.spines["bottom"].set_linewidth(0.8)

        # X limits: just enough margin for annotations
        ax.set_xlim(
            -abs_max * 1.35 if not is_amp else -abs_max * 0.15,
             abs_max * 1.35 if is_amp     else  abs_max * 0.15,
        )
        ax.margins(y=0.03)

        # --- Panel title (inside axes, top) --------------------------------
        ax.set_title(
            f"{panel_letter}  {sign_sym}  {panel_label}\n"
            "Baseline -> Stress",
            fontsize=14, fontweight="bold",
            color=bar_col,
            pad=10, loc="left",
        )

    # -----------------------------------------------------------------------

    def print_analysis_summary(self) -> None:
        sep = "=" * 72
        print(f"\n{sep}\nATTENTION WEIGHT CHANGES (Baseline to Stress)\n{sep}")
        if "Baseline" in self.regimes and "Stress" in self.regimes:
            delta = self.regimes["Stress"] - self.regimes["Baseline"]
            np.fill_diagonal(delta, 0) # Exclude self-loops
            print(f"  Mean Delta alpha   : {delta.mean():.4f}")
            print(f"  Amplifications : {(delta > 0).sum()} edges")
            print(f"  Attenuations   : {(delta < 0).sum()} edges")
            print(f"  Max increase   : {delta.max():.4f}")
            print(f"  Max decrease   : {delta.min():.4f}")
        print(f"\n{sep}\n")


# ===========================================================================
# Helpers
# ===========================================================================

def _extract_edges(delta: np.ndarray, tickers: List[str]) -> List[dict]:
    n = len(tickers)
    return [
        {"source": tickers[i], "target": tickers[j], "delta": float(delta[i, j])}
        for i in range(n) for j in range(n) if i != j
    ]


def find_latest_attention_file() -> str:
    files = glob.glob("results/regime_attention_matrices_*.pkl")
    if files:
        return sorted(files)[-1]
    raise FileNotFoundError(
        "No regime_attention_matrices_*.pkl found in results/. "
        "Run extract_regime_attention.py first."
    )


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 6.4 - attention weight deltas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to regime attention matrices pickle (auto-detected if omitted).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory for saved figures.",
    )
    args = parser.parse_args()

    input_file = args.input
    if input_file is None:
        print("[*] Auto-detecting latest attention matrix file ...")
        input_file = find_latest_attention_file()
        print(f"Found: {input_file}")

    vis = AttentionDeltasVisualizer(input_file)

    try:
        path = vis.generate_figure_6_4(
            output_path=f"{args.output_dir}/figure_6_4_edge_attention_changes.png"
        )
        vis.print_analysis_summary()
        print("=" * 72)
        print("FIGURE GENERATION COMPLETE")
        print("=" * 72)
        print(f"  Figure 6.4: {path}")
    except Exception as exc:
        import traceback
        print(f"\n[!] Error: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
