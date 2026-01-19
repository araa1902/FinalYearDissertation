import numpy as np
import pandas as pd
import ta

class FeatureEngineer:
    """Preprocesses data with technical indicators and rolling normalisation."""
    def __init__(self, 
                 use_technical_indicator=True, 
                 tech_indicator_list=None, 
                 use_turbulence=False, 
                 user_defined_feature=False,
                 normalisation_window=63):
        
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list or ["macd", "rsi", "cci", "dx"]
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature
        self.normalisation_window = normalisation_window

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(by=['date', 'ticker'])
        
        if self.use_technical_indicator:
            df = self._add_technical_indicators(df)

        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        cols_to_normalise = self.tech_indicator_list + ['volume', 'log_return']
        df = self._apply_rolling_normalisation(df, cols_to_normalise)

        df = df.dropna().reset_index(drop=True)
        
        print(f"Data preprocessed. Final shape: {df.shape}")
        return df

    def _add_technical_indicators(self, df):
        """Adds technical indicators (MACD, RSI, CCI, DX) per ticker."""
        processed_dfs = []
        for _, group in df.groupby('ticker'):
            group = group.copy()
            for col in ['close', 'high', 'low']:
                if col not in group.columns:
                    raise ValueError(f"Missing required column '{col}'.")
            if 'macd' in self.tech_indicator_list:
                group['macd'] = ta.trend.macd(group['close'], fillna=True)
            if 'rsi' in self.tech_indicator_list:
                group['rsi'] = ta.momentum.rsi(group['close'], fillna=True)
            if 'cci' in self.tech_indicator_list:
                group['cci'] = ta.trend.cci(group['high'], group['low'], group['close'], fillna=True)
            if 'dx' in self.tech_indicator_list:
                group['dx'] = ta.trend.adx(group['high'], group['low'], group['close'], fillna=True)
            processed_dfs.append(group)
        return pd.concat(processed_dfs)

    def _apply_rolling_normalisation(self, df, cols):
        """Applies Z-Score normalisation using trailing windows [t-Tw, t-1]."""
        for col in cols:
            df[col] = df.groupby('ticker')[col].transform(lambda x: (x - x.rolling(self.normalisation_window).mean().shift(1)) / (x.rolling(self.normalisation_window).std().shift(1) + 1e-8))
        return df