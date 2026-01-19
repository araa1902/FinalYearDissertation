import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class GraphBuilder:
    def __init__(self, df: pd.DataFrame, lookback_window: int, threshold: float = 0.3, top_k: int = 10):
        self.df = df.copy()
        self.lookback_window = lookback_window
        self.threshold = threshold
        self.unique_tickers, self.unique_dates = self.df['ticker'].unique().tolist(), sorted(self.df['date'].unique().tolist())
        self.top_k = top_k
        self.METRICS = [
            'close',
            'macd',          # Moving Average Convergence Divergence
            'rsi',           # Relative Strength Index
            'cci',           # Commodity Channel Index
            'dx',            # Directional Movement Index
            'log_return',
            'open',
            'high',
            'low',
            'volume'
        ]

    def get_wide_returns(self) -> pd.DataFrame:
            '''
            Converts long format to wide format using log returns.
            '''
            wide_returns = self.df.pivot(index='date', columns='ticker', values='log_return')
            wide_returns = wide_returns.fillna(0)
            
            return wide_returns
    
    def build_graphs(self, sparsity_method: str = 'threshold') -> dict:
        """Builds correlation graphs using trailing windows [t-Tw, t-1]."""
        graphs = {}
        num_days = len(self.df['date'].unique())
        wide_returns = self.get_wide_returns()

        # Choice of sparsity method: 'knn' or 'threshold' - used in ablation studies
        for i in range(self.lookback_window, num_days):
            returns_window = wide_returns.iloc[i - self.lookback_window:i]
            corr_matrix = np.nan_to_num(returns_window.corr('pearson').values, nan=0.0)
            if sparsity_method == 'knn':
                # Keep only top_k strongest connections per node
                abs_corr = np.abs(corr_matrix)
                partition_index = np.argpartition(abs_corr, -self.top_k, axis=1)
                mask, rows = np.zeros_like(corr_matrix, dtype=bool), np.arange(corr_matrix.shape[0])[:, None]
                mask[rows, partition_index[:, -self.top_k:]] = True
                adj_matrix = np.where(mask, corr_matrix, 0)
            else:
                # Apply thresholding
                adj_matrix = np.where((corr_matrix >= self.threshold) | (corr_matrix <= -self.threshold), corr_matrix, 0)
            
            graphs[self.unique_dates[i]] = adj_matrix

        return graphs