import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import Dict, List, Tuple, Optional


class AttentionAnalyser:
    """
    Analyses GAT attention weights captured during training/evaluation.
    """
    
    def __init__(self, attention_buffer_path: str, tickers: List[str]):
        """Load attention buffer from pickle file."""
        self.tickers = tickers
        self.buffer = self._load_attention_buffer(attention_buffer_path)
        # Convert timestamps to pandas DatetimeIndex for reliable date matching
        self.timestamps_pd = pd.to_datetime(self.buffer['timestamps'])
        
    def _load_attention_buffer(self, path: str) -> Dict:
        """Load attention buffer from pickle file."""
        with open(path, 'rb') as f:
            buffer = pickle.load(f)
        print(f" Loaded attention buffer from {path}")
        print(f"  Timesteps: {len(buffer['timestamps'])}")
        print(f"  Date range: {buffer['timestamps'][0]} to {buffer['timestamps'][-1]}")
        return buffer
    
    def get_timestamp_index(self, target_date: str) -> Optional[int]:
        """Find index of a specific date in the buffer."""
        target_ts = pd.Timestamp(target_date)
        # Find closest match with same date
        matches = (self.timestamps_pd.date == target_ts.date()).nonzero()[0]
        if len(matches) > 0:
            return int(matches[0])  # Return first match for the date
        return None
    
    def get_attention_for_date(self, target_date: str) -> Optional[np.ndarray]:
        """Get attention weights for a specific date."""
        idx = self.get_timestamp_index(target_date)
        if idx is None:
            print(f"Date {target_date} not found in buffer")
            return None
        
        attn = self.buffer['attention_weights'][idx]
        return np.array(attn) if isinstance(attn, list) else attn
    
    def aggregate_attention(self, indices: List[int]) -> np.ndarray:
        """Average attention weights across multiple timesteps."""
        attentions = [np.array(self.buffer['attention_weights'][i]) 
                      for i in indices if i < len(self.buffer['attention_weights'])]
        
        if not attentions:
            raise ValueError(f"No valid indices in {indices}")
        
        return np.mean(attentions, axis=0)
    
    def get_attention_for_period(self, start_date: str, end_date: str) -> np.ndarray:
        """Average attention across a date range."""
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        # Use pandas timestamp comparison for reliable period matching
        mask = (self.timestamps_pd.date >= start_ts.date()) & (self.timestamps_pd.date <= end_ts.date())
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            raise ValueError(f"Could not find dates in range: {start_date} to {end_date}")
        
        return self.aggregate_attention(indices.tolist())
    
    def plot_attention_heatmap(self, 
                               attention: np.ndarray, 
                               head_idx: int = 0,
                               title: str = "GAT Attention Weights",
                               figsize: Tuple[int, int] = (10, 8),
                               cmap: str = 'YlOrRd',
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot a single attention head as a heatmap."""
        # Extract single head if multi-head
        if attention.ndim == 3:
            attn_matrix = attention[head_idx]
        else:
            attn_matrix = attention
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            attn_matrix,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            square=True,
            cbar_kws={'label': 'Attention Weight'},
            xticklabels=self.tickers,
            yticklabels=self.tickers,
            ax=ax,
            vmin=0,
            vmax=1
        )
        
        ax.set_title(f"{title}\n(Head {head_idx})", fontsize=14, fontweight='bold')
        ax.set_xlabel("To (Target Asset)", fontsize=12)
        ax.set_ylabel("From (Source Asset)", fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved heatmap to {save_path}")
        
        return fig
    
    def plot_attention_comparison(self,
                                  attention_normal: np.ndarray,
                                  attention_stress: np.ndarray,
                                  normal_label: str = "Normal Period",
                                  stress_label: str = "Stress Period",
                                  head_idx: int = 0,
                                  figsize: Tuple[int, int] = (16, 5),
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Compare attention patterns between two periods side-by-side."""
        # Extract single head if needed
        if attention_normal.ndim == 3:
            attn_normal = attention_normal[head_idx]
        else:
            attn_normal = attention_normal
            
        if attention_stress.ndim == 3:
            attn_stress = attention_stress[head_idx]
        else:
            attn_stress = attention_stress
        
        # Compute difference
        diff = attn_stress - attn_normal
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Normal period
        sns.heatmap(attn_normal, annot=True, fmt='.2f', cmap='Blues',
                    square=True, cbar_kws={'label': 'Attention'}, 
                    xticklabels=self.tickers, yticklabels=self.tickers,
                    ax=axes[0], vmin=0, vmax=1)
        axes[0].set_title(f"{normal_label}\n(Head {head_idx})", fontweight='bold')
        axes[0].set_ylabel("From Asset")
        
        # Stress period
        sns.heatmap(attn_stress, annot=True, fmt='.2f', cmap='Reds',
                    square=True, cbar_kws={'label': 'Attention'},
                    xticklabels=self.tickers, yticklabels=self.tickers,
                    ax=axes[1], vmin=0, vmax=1)
        axes[1].set_title(f"{stress_label}\n(Head {head_idx})", fontweight='bold')
        
        # Difference (diverging colourmap to show both increase and decrease)
        sns.heatmap(diff, annot=True, fmt='.2f', cmap='RdBu_r',
                    square=True, cbar_kws={'label': 'Δ Attention'},
                    xticklabels=self.tickers, yticklabels=self.tickers,
                    ax=axes[2], center=0)
        axes[2].set_title(f"Difference: {stress_label} - {normal_label}\n(Head {head_idx})", fontweight='bold')
        axes[2].set_ylabel("")
        
        plt.suptitle("GAT Attention Pattern Comparison", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved comparison plot to {save_path}")
        
        return fig
    
    def plot_all_heads(self,
                       attention: np.ndarray,
                       title: str = "GAT Attention Weights - All Heads",
                       figsize_per_head: Tuple[int, int] = (6, 5),
                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot all attention heads in a grid."""
        if attention.ndim != 3:
            raise ValueError("attention must be 3D array (n_heads, N, N)")
        
        n_heads = attention.shape[0]
        n_cols = min(n_heads, 4)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        figsize = (figsize_per_head[0] * n_cols, figsize_per_head[1] * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_heads == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        
        for head_idx in range(n_heads):
            ax = axes[head_idx]
            sns.heatmap(attention[head_idx], annot=False, cmap='YlOrRd',
                        square=True, cbar_kws={'label': 'Attention'},
                        xticklabels=self.tickers, yticklabels=self.tickers,
                        ax=ax, vmin=0, vmax=1)
            ax.set_title(f"Head {head_idx}")
        
        # Hide unused subplots
        for idx in range(n_heads, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved all-heads plot to {save_path}")
        
        return fig
    
    def get_attention_statistics(self, attention: np.ndarray) -> Dict:
        """Compute statistics about attention patterns."""
        if attention.ndim == 3:
            attn = attention.mean(axis=0)  # Average across heads
        else:
            attn = attention
        
        # Remove self-attention for inter-asset analysis
        np.fill_diagonal(attn, 0)
        
        stats = {
            'mean_attention': float(attn.mean()),
            'max_attention': float(attn.max()),
            'min_attention': float(attn.min()),
            'std_attention': float(attn.std()),
            'sparsity': float((attn < 0.1).sum() / attn.size),  # % of weights < 0.1
            'concentration': float((attn > 0.2).sum() / attn.size),  # % of weights > 0.2
        }
        
        return stats
    
    def compare_statistics(self,
                          attention_normal: np.ndarray,
                          attention_stress: np.ndarray) -> pd.DataFrame:
        """Compare attention statistics between two periods."""
        stats_normal = self.get_attention_statistics(attention_normal)
        stats_stress = self.get_attention_statistics(attention_stress)
        
        df = pd.DataFrame({
            'Normal Period': stats_normal,
            'Stress Period': stats_stress,
            'Difference': {k: stats_stress[k] - stats_normal[k] for k in stats_normal.keys()},
            '% Change': {k: ((stats_stress[k] - stats_normal[k]) / stats_normal[k] * 100) 
                        if stats_normal[k] != 0 else 0 for k in stats_normal.keys()}
        })
        
        return df.round(4)
    
    def identify_focus_shifts(self,
                             attention: np.ndarray,
                             top_k: int = 5) -> Dict[int, List[Tuple[str, str, float]]]:
        """Identify the strongest connections (focus) for each head."""
        if attention.ndim != 3:
            raise ValueError("attention must be 3D array (n_heads, N, N)")
        
        focus_shifts = {}
        
        for head_idx in range(attention.shape[0]):
            attn_matrix = attention[head_idx]
            
            # Get top-k connections (excluding self-loops)
            connections = []
            for i in range(len(self.tickers)):
                for j in range(len(self.tickers)):
                    if i != j:  # Exclude self-attention
                        connections.append((
                            self.tickers[i],
                            self.tickers[j],
                            float(attn_matrix[i, j])
                        ))
            
            # Sort by weight and get top-k
            connections.sort(key=lambda x: x[2], reverse=True)
            focus_shifts[head_idx] = connections[:top_k]
        
        return focus_shifts
    
    def print_focus_analysis(self,
                            attention_normal: np.ndarray,
                            attention_stress: np.ndarray,
                            top_k: int = 5):
        """Pretty-print focus shift analysis."""
        focus_normal = self.identify_focus_shifts(attention_normal, top_k)
        focus_stress = self.identify_focus_shifts(attention_stress, top_k)
        
        print("\n" + "="*80)
        print("ATTENTION FOCUS SHIFT ANALYSIS")
        print("="*80)
        
        for head_idx in range(len(focus_normal)):
            print(f"\n{'─'*80}")
            print(f"HEAD {head_idx}")
            print(f"{'─'*80}")
            
            print(f"\nNormal Regime - Top {top_k} Connections:")
            for i, (from_ticker, to_ticker, weight) in enumerate(focus_normal[head_idx], 1):
                print(f"  {i}. {from_ticker:6s} to {to_ticker:6s}  {weight:.4f}")
            
            print(f"\nStress Regime - Top {top_k} Connections:")
            for i, (from_ticker, to_ticker, weight) in enumerate(focus_stress[head_idx], 1):
                print(f"  {i}. {from_ticker:6s} to {to_ticker:6s}  {weight:.4f}")
            
            print(f"\nKey Observations:")
            # Find connections that changed importance
            normal_dict = {(f, t): w for f, t, w in focus_normal[head_idx]}
            stress_dict = {(f, t): w for f, t, w in focus_stress[head_idx]}
            
            for (f, t), w_stress in stress_dict.items():
                w_normal = normal_dict.get((f, t), 0)
                if w_stress > w_normal:
                    print(f"   {f} to {t}: +{(w_stress - w_normal):.4f} (NEW FOCUS)")
        
        print("\n" + "="*80 + "\n")


def main_example():
    """Example usage of AttentionAnalyser."""
    # Load buffer
    analyser = AttentionAnalyser(
        attention_buffer_path='results/attention_logs/attention_buffer_20260208_225516.pkl',
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    )
    
    # Get attention for specific dates
    attn_normal = analyser.get_attention_for_date('2022-09-27')  # Day before crash
    attn_crash = analyser.get_attention_for_date('2022-09-28')   # Crash day
    
    if attn_normal is not None and attn_crash is not None:
        # Plot comparison
        fig = analyser.plot_attention_comparison(
            attn_normal, attn_crash,
            normal_label="Sept 27, 2022 (Normal)",
            crash_label="Sept 28, 2022 (Crash)",
            save_path='results/attention_comparison_sept2022.png'
        )
        
        # Print statistics
        stats_df = analyser.compare_statistics(attn_normal, attn_crash)
        print("\nAttention Statistics Comparison:")
        print(stats_df)
        
        # Focus shift analysis
        analyser.print_focus_analysis(attn_normal, attn_crash)


if __name__ == "__main__":
    main_example()
