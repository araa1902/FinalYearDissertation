import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GraphBuilder:
    def __init__(self, df: pd.DataFrame, lookback_window: int, threshold: float = 0.3):
        self.df = df.copy()
        self.lookback_window = lookback_window
        self.threshold = threshold
        self.unique_tickers, self.unique_dates = self.df['ticker'].unique().tolist(), self.df['date'].unique().tolist()
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
        '''Converts long format to wide format returns DataFrame.'''
        pivot_df = self.df.pivot(index='date', columns='ticker', values='close')
        returns = pivot_df.pct_change().fillna(0)
        return returns
    
    def build_graphs(self):
        """Builds correlation graphs using trailing windows [t-Tw, t-1]."""
        graphs = {}
        num_days = len(self.df['date'].unique())
        wide_returns = self.get_wide_returns()

        for i in range(self.lookback_window, num_days):
            returns_window = wide_returns.iloc[i - self.lookback_window:i]
            corr_matrix = np.nan_to_num(returns_window.corr('pearson').values, nan=0.0)
            adj_matrix = np.where((corr_matrix >= self.threshold) | (corr_matrix <= -self.threshold), corr_matrix, 0)
            graphs[self.unique_dates[i]] = adj_matrix

        return graphs

    def merge_graphs_to_dataframe(self, df: pd.DataFrame, graphs: dict) -> pd.DataFrame:
        """
        Merges pre-computed adjacency matrices into the long-format DataFrame.
        """
        df = df.copy()
        
        df['cov_list'] = None
        
        for date, matrix in graphs.items():
            mask = df['date'] == date
            for idx in df[mask].index:
                df.at[idx, 'cov_list'] = matrix
        
        # Remove rows where no graph was computed (dates before lookback_window)
        df = df.dropna(subset=['cov_list']).reset_index(drop=True)
        
        return df

    def convert_to_wide_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts LONG format to WIDE format (one row per date with aggregated ticker data)."""
        df_wide = df.copy()
        unique_dates = df_wide['date'].unique()
        wide_data = []
        
        for date in unique_dates:
            date_data = df_wide[df_wide['date'] == date]
            
            if len(date_data) != len(self.unique_tickers):
                continue
            
            row_dict = {'date': date}
            cov_matrix = date_data['cov_list'].iloc[0]
            
            for _, ticker_row in date_data.iterrows():
                ticker = ticker_row['ticker']
                for col in self.METRICS:
                    if col in ticker_row.index:
                        row_dict[f'{ticker}_{col}'] = ticker_row[col]
            
            row_dict['cov_list'] = cov_matrix
            wide_data.append(row_dict)
        
        df_wide_format = pd.DataFrame(wide_data).reset_index(drop=True)
        return df_wide_format
            

    def cov_matrix_visualiser_over_time(self, graphs: dict, num_matrices: int = 5):
        """Visualises a sample of covariance matrices over time."""
        sampled_dates = list(graphs.keys())[::max(1, len(graphs)//num_matrices)][:num_matrices]
        
        plt.figure(figsize=(15, 3 * num_matrices))
        for i, date in enumerate(sampled_dates):
            plt.subplot(num_matrices, 1, i + 1)
            plt.title(f'Covariance Matrix on {date}')
            plt.imshow(graphs[date], cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(label='Correlation Coefficient')
            plt.xticks(ticks=np.arange(len(self.unique_tickers)), labels=self.unique_tickers, rotation=90)
            plt.yticks(ticks=np.arange(len(self.unique_tickers)), labels=self.unique_tickers)
        
        plt.tight_layout()
        plt.show()