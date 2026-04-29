#!/usr/bin/env python3
"""
Generate Figures 6.3 and 6.4 from Extracted Attention Weights
"""

import sys
import glob
import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Global style — dissertation-grade, IEEE/ACM compatible
# ---------------------------------------------------------------------------

# Use a font available in most LaTeX-adjacent environments; fall back gracefully
_FONT_FAMILY = "DejaVu Sans"  # reliable cross-platform; swap for "Times New Roman" if desired

_PALETTE: Dict[str, str] = {
    "bg":         "#FFFFFF",   # pure white — matches most document backgrounds
    "axes_bg":    "#FAFAFA",   # very light grey for axes interior
    "text":       "#1A1A2E",   # near-black for all labels
    "subtitle":   "#555577",
    "grid":       "#DDDDEE",
    "border":     "#CCCCDD",
    "increase":   "#C0392B",   # distinct red  (amplification)
    "decrease":   "#2471A3",   # distinct blue (attenuation)
}

# Shared rcParams — applied once at module level
mpl.rcParams.update({
    "figure.facecolor":       _PALETTE["bg"],
    "axes.facecolor":         _PALETTE["axes_bg"],
    "axes.edgecolor":         _PALETTE["border"],
    "axes.labelcolor":        _PALETTE["text"],
    "axes.titlecolor":        _PALETTE["text"],
    "xtick.color":            _PALETTE["text"],
    "ytick.color":            _PALETTE["text"],
    "text.color":             _PALETTE["text"],
    "font.family":            "sans-serif",
    "font.sans-serif":        [_FONT_FAMILY, "Arial", "Helvetica"],
    "font.size":              10,
    "axes.titlesize":         12,
    "axes.labelsize":         11,
    "xtick.labelsize":        9,
    "ytick.labelsize":        9,
    "figure.dpi":             150,
    "savefig.dpi":            300,
    "savefig.bbox":           "tight",
    "savefig.pad_inches":     0.15,
    "savefig.facecolor":      _PALETTE["bg"],
    "savefig.edgecolor":      "none",
    "axes.grid":              False,   # managed per-plot
    "pdf.fonttype":           42,      # embed fonts in PDF exports
    "ps.fonttype":            42,
})

# Diverging colormap — perceptually uniform, colourblind-safe
_CMAP_ATTN   = "YlOrRd"          # sequential: 0 to 1 attention weights
_CMAP_DELTA  = "RdBu_r"          # diverging: negative (blue) to positive (red)


_REGIME_PANELS = [
    {
        "key":             "Baseline",
        "label":           "Baseline Period",
        "subtitle":        "2021 Bull Market — Stable Regime",
        "title_color":     "#154360",   # dark blue
        "filename_suffix": "baseline",
    },
    {
        "key":             "Stress",
        "label":           "Market Stress Period",
        "subtitle":        "2022 Rate Shock — Crisis Regime",
        "title_color":     "#78281F",   # dark red
        "filename_suffix": "stress",
    },
    {
        "key":             "Rally",
        "label":           "Strong Rally Period",
        "subtitle":        "2024 Recovery — Recovery Regime",
        "title_color":     "#145A32",   # dark green
        "filename_suffix": "rally",
    },
]

_FIG_WIDTH  = 12 
_FIG_HEIGHT = 11.5


# ===========================================================================
class RegimeAttentionVisualiser:
    """Generates publication-ready figures from extracted GAT attention matrices."""

    # -----------------------------------------------------------------------
    # Construction & data loading
    # -----------------------------------------------------------------------

    def __init__(self, attention_file: str) -> None:
        self.attention_file = attention_file
        self.data    = self._load_attention_data()
        self.tickers: List[str] = self.data["tickers"]
        self.regimes: Dict[str, np.ndarray] = {}
        self._prepare_regimes()

    def _load_attention_data(self) -> dict:
        with open(self.attention_file, "rb") as fh:
            data = pickle.load(fh)
        print(f"Loaded attention data: {self.attention_file}")
        return data

    def _prepare_regimes(self) -> None:
        """Collapse multi-head attention to a single N×N matrix per regime."""
        regime_data = self.data["regime_attentions"]
        for name in ("Baseline", "Stress", "Rally"):
            if name in regime_data:
                attn = regime_data[name]["attention"]
                if attn.ndim > 2:
                    attn = attn.mean(axis=0)          # average over heads
                self.regimes[name] = attn.astype(float)
        print(f"Prepared {len(self.regimes)} regime matrices")

    # -----------------------------------------------------------------------
    # Figure 6.3 — individual heatmaps
    # -----------------------------------------------------------------------

    def generate_figure_6_3(self, output_path: Optional[str] = None) -> str:
        """
        Generate Figure 6.3: three separate full-page GAT attention heatmaps.

        Each figure uses a *shared* colour scale anchored at [0, global_vmax]
        so intensities are directly comparable across regimes.

        Returns:
            Base output path prefix (individual files suffixed per regime).
        """
        if len(self.regimes) < 3:
            raise ValueError(
                "All three regimes (Baseline, Stress, Rally) are required for Figure 6.3."
            )

        # Shared colour scale across all three panels for comparability
        global_vmax = max(m.max() for m in self.regimes.values())
        # Round up to nearest 0.05 to give a clean colourbar
        global_vmax = np.ceil(global_vmax / 0.05) * 0.05

        base_path = output_path or "results/figure_6_3_regime_attention_heatmaps.png"

        for panel in _REGIME_PANELS:
            key   = panel["key"]
            attn  = self.regimes[key]
            stats = _compute_matrix_stats(attn)

            fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT),
                                   facecolor=_PALETTE["bg"])

            sns.heatmap(
                attn,
                annot=True,
                fmt=".2f",
                cmap=_CMAP_ATTN,
                square=True,
                vmin=0.0,
                vmax=global_vmax,
                linewidths=0.8,
                linecolor="#CCCCCC",
                xticklabels=self.tickers,
                yticklabels=self.tickers,
                ax=ax,
                cbar=True,
                cbar_kws={
                    "label":  r"Attention Weight $\alpha_{ij}$",
                    "shrink": 0.68,
                    "pad":    0.02,
                },
                annot_kws={"size": 13, "weight": "bold", "color": "#111111"},
            )

            # ---- colourbar font ----
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.label.set_fontsize(10)
            cbar.ax.tick_params(labelsize=9)

            # ---- title (only metrics, no label/subtitle) ----
            ax.set_title(
                rf"max $\alpha$ = {stats['max']:.4f}  |  "
                f"Sparsity = {stats['sparsity']:.1f}%",
                fontsize=18,
                fontweight="bold",
                color=panel["title_color"],
                pad=15,
                loc="center",
            )

            # ---- axis labels ----
            ax.set_xlabel("Target Asset (Impacted)",
                          fontsize=20, fontweight="bold", labelpad=10)
            ax.set_ylabel("Source Asset (Influence)",
                          fontsize=20, fontweight="bold", labelpad=10)

            # ---- tick labels ----
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right", fontsize=12, fontweight="bold"
            )
            ax.set_yticklabels(
                ax.get_yticklabels(), rotation=0, fontsize=12, fontweight="bold"
            )

            ax.set_facecolor(_PALETTE["axes_bg"])
            fig.tight_layout(rect=[0, 0, 1, 0.97])

            out = _build_output_path(base_path, panel["filename_suffix"])
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            fig.savefig(out)
            plt.close(fig)
            print(f"Figure 6.3 ({key}): {out}")

        return base_path

    # -----------------------------------------------------------------------
    # Figure 6.4 — per-edge delta bar chart
    # -----------------------------------------------------------------------

    def generate_figure_6_4(self, output_path: Optional[str] = None) -> str:
        """
        Generate Figure 6.4: top-10 attention amplifications and attenuations
        (Δα_ij = α_ij[Stress] − α_ij[Baseline]).

        Returns:
            Path to the saved figure.
        """
        if "Baseline" not in self.regimes or "Stress" not in self.regimes:
            raise ValueError("Baseline and Stress regimes are required for Figure 6.4.")

        delta   = self.regimes["Stress"] - self.regimes["Baseline"]
        edges   = _extract_edges(delta, self.tickers)

        amps    = sorted([e for e in edges if e["delta"] > 0],
                         key=lambda x: x["delta"], reverse=True)[:10]
        atts    = sorted([e for e in edges if e["delta"] < 0],
                         key=lambda x: x["delta"])[:10]

        fig, (ax_amp, ax_att) = plt.subplots(
            1, 2, figsize=(16, 7.5), facecolor=_PALETTE["bg"]
        )
        fig.subplots_adjust(wspace=0.35)

        self._plot_edge_panel(
            ax_amp, amps,
            title=r"(a) Top-10 Amplifications" + "\n" + r"Baseline $\rightarrow$ Stress",
            color=_PALETTE["increase"],
        )
        self._plot_edge_panel(
            ax_att, atts,
            title=r"(b) Top-10 Attenuations" + "\n" + r"Baseline $\rightarrow$ Stress",
            color=_PALETTE["decrease"],
        )

        # Shared x-axis label via figure text
        fig.text(
            0.5, 0.01,
            r"Attention Weight Change  $\Delta\alpha_{ij}$",
            ha="center", va="bottom",
            fontsize=16, fontweight="bold", color=_PALETTE["text"],
        )

        fig.tight_layout(rect=[0, 0.04, 1, 0.95])

        output_path = output_path or "results/figure_6_4_edge_attention_changes.png"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Figure 6.4: {output_path}")
        return output_path

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _plot_edge_panel(
        ax: plt.Axes,
        edges: List[dict],
        title: str,
        color: str,
    ) -> None:
        """Render a single horizontal-bar panel for Figure 6.4."""
        labels = [f"{e['source']} to {e['target']}" for e in edges]
        values = [e["delta"] for e in edges]
        y_pos  = np.arange(len(labels))

        # Gradient alpha — strongest edge is most opaque
        alphas = np.linspace(0.95, 0.50, len(values))
        bars   = ax.barh(y_pos, values, color=color,
                         edgecolor="none", height=0.65)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)

        # Value annotations
        abs_max = max(abs(v) for v in values) if values else 1.0
        offset  = abs_max * 0.025
        for bar, val in zip(bars, values):
            ha  = "left"  if val >= 0 else "right"
            x   = val + offset if val >= 0 else val - offset
            ax.text(
                x, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}",
                va="center", ha=ha,
                fontsize=9, fontweight="bold",
                color=_PALETTE["text"],
            )

        # Axes configuration
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10, color=_PALETTE["text"])
        ax.invert_yaxis()   # rank-1 at the top

        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, symmetric=True))

        ax.set_title(title, fontsize=12, fontweight="bold",
                     color=_PALETTE["text"], pad=14, loc="center")

        # Zero reference line
        ax.axvline(0, color=_PALETTE["text"], linewidth=0.9,
                   linestyle="-", alpha=0.35)

        # Grid (x-axis only)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=_PALETTE["grid"],
                linewidth=0.7, linestyle="--", alpha=0.9)

        # Spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(_PALETTE["border"])
        ax.spines["bottom"].set_color(_PALETTE["border"])

        # X-axis limits — symmetric with margin
        ax.set_xlim(-abs_max * 1.25, abs_max * 1.25)
        ax.margins(y=0.02)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------

    def print_analysis_summary(self) -> None:
        """Print regime-level statistics and transition changes to stdout."""
        sep = "=" * 72
        print(f"\n{sep}\nREGIME-CONDITIONED ATTENTION ANALYSIS\n{sep}")

        for label, key in [
            ("Baseline Period (2021 Bull Market)", "Baseline"),
            ("Market Stress Period (2022 Rate Shock)", "Stress"),
            ("Strong Rally Period (2024 Recovery)", "Rally"),
        ]:
            attn = self.regimes.get(key)
            if attn is None:
                continue
            s = _compute_matrix_stats(attn)
            print(f"\n{label}:")
            print(f"  Mean α   : {s['mean']:.4f}")
            print(f"  Max  α   : {s['max']:.4f}")
            print(f"  Sparsity : {s['sparsity']:.1f}%")

        # Transition deltas
        for (src, tgt) in [("Baseline", "Stress"), ("Stress", "Rally")]:
            if src in self.regimes and tgt in self.regimes:
                delta = self.regimes[tgt] - self.regimes[src]
                print(f"\nAttention Changes ({src} to {tgt}):")
                print(f"  Mean Δα          : {delta.mean():.4f}")
                print(f"  Amplifications   : {(delta > 0).sum()} edges")
                print(f"  Attenuations     : {(delta < 0).sum()} edges")
                print(f"  Max increase     : {delta.max():.4f}")
                print(f"  Max decrease     : {delta.min():.4f}")
        print(f"\n{sep}\n")


# ===========================================================================
# Module-level helpers
# ===========================================================================

def _compute_matrix_stats(attn: np.ndarray) -> dict:
    """Return descriptive statistics for an attention matrix."""
    return {
        "mean":     float(attn.mean()),
        "max":      float(attn.max()),
        "sparsity": float((attn == 0).sum() / attn.size * 100),
    }


def _extract_edges(delta: np.ndarray, tickers: List[str]) -> List[dict]:
    """Flatten a delta matrix into a list of non-self-loop edge dicts."""
    n = len(tickers)
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append({
                    "source": tickers[i],
                    "target": tickers[j],
                    "delta":  float(delta[i, j]),
                })
    return edges


def _build_output_path(base_path: str, suffix: str) -> str:
    """Inject a per-regime suffix before the .png extension."""
    if base_path.endswith(".png"):
        return base_path[:-4] + f"_{suffix}.png"
    return f"{base_path}_{suffix}.png"


def find_latest_attention_file() -> str:
    """Return the most recently created attention matrix pickle in results/."""
    files = glob.glob("results/regime_attention_matrices_*.pkl")
    if files:
        return sorted(files)[-1]
    raise FileNotFoundError(
        "No regime_attention_matrices_*.pkl files found in results/. "
        "Run extract_regime_attention.py first."
    )


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dissertation Figures 6.3 and 6.4 from GAT attention weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to regime attention matrices pickle file (auto-detected if omitted).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory for saved figures.",
    )
    parser.add_argument(
        "--skip_6_3", action="store_true",
        help="Skip generation of Figure 6.3 (heatmaps).",
    )
    parser.add_argument(
        "--skip_6_4", action="store_true",
        help="Skip generation of Figure 6.4 (edge bar chart).",
    )
    args = parser.parse_args()

    input_file = args.input
    if input_file is None:
        print("[*] Auto-detecting latest attention matrix file …")
        input_file = find_latest_attention_file()
        print(f"Found: {input_file}")

    vis = RegimeAttentionVisualiser(input_file)

    generated: List[str] = []

    try:
        if not args.skip_6_3:
            path = vis.generate_figure_6_3(
                output_path=f"{args.output_dir}/figure_6_3_regime_attention_heatmaps.png"
            )
            generated.append(f"Figure 6.3  — {path}")

        if not args.skip_6_4:
            path = vis.generate_figure_6_4(
                output_path=f"{args.output_dir}/figure_6_4_edge_attention_changes.png"
            )
            generated.append(f"Figure 6.4  — {path}")

        vis.print_analysis_summary()

        print("=" * 72)
        print("FIGURE GENERATION COMPLETE")
        print("=" * 72)
        for entry in generated:
            print(f"  {entry}")

    except Exception as exc:
        import traceback
        print(f"\n[!] Visualisation error: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
