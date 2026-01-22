import numpy as np
import pandas as pd
import ta
import yaml

class FeatureEngineer:
    """Preprocesses data with technical indicators and rolling normalisation."""
    def __init__(self, use_technical_indicator: bool = True, tech_indicator_list: list = None, normalisation_window: int = 63):
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list if tech_indicator_list is not None else config['preprocessing']['tech_indicator_list']
        self.normalisation_window = normalisation_window

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(by=['date', 'ticker'])
        
        if self.use_technical_indicator:
            df = self.add_technical_indicators(df)
            
        df['log_return'] = df.groupby('ticker')['close'].transform(lambda x: np.log(x / x.shift(1)))
        
        # This prevents a crash if 'volume' is missing from the download
        potential_cols = self.tech_indicator_list + ['volume', 'log_return']
        cols_to_normalise = [c for c in potential_cols if c in df.columns]
        
        df = self.apply_rolling_normalisation(df, cols_to_normalise)

        df = df.dropna().reset_index(drop=True)
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds technical indicators (MACD, RSI, CCI, DX, bollinger bands) per ticker."""
        processed_dfs = []
        # Groupby preserves the index, so we concat at the end
        for _, group in df.groupby('ticker'):
            group = group.copy()
            for col in ['close', 'high', 'low']:
                if col not in group.columns:
                    continue
            
            if 'macd' in self.tech_indicator_list:
                group['macd'] = ta.trend.macd(group['close'], fillna=True)
            if 'rsi' in self.tech_indicator_list:
                group['rsi'] = ta.momentum.rsi(group['close'], fillna=True)
            if 'cci' in self.tech_indicator_list:
                group['cci'] = ta.trend.cci(group['high'], group['low'], group['close'], fillna=True)
            if 'dx' in self.tech_indicator_list:
                group['dx'] = ta.trend.adx(group['high'], group['low'], group['close'], fillna=True)
            if 'boll_ub' in self.tech_indicator_list and 'boll_lb' in self.tech_indicator_list:
                bollinger = ta.volatility.BollingerBands(group['close'], fillna=True)
                group['boll_ub'] = bollinger.bollinger_hband()
                group['boll_lb'] = bollinger.bollinger_lband()
            processed_dfs.append(group)
        
        return pd.concat(processed_dfs)

    def apply_rolling_normalisation(self, df : pd.DataFrame, cols: list) -> pd.DataFrame:
        """Applies Z-Score normalisation using trailing windows [t-Tw, t-1]. This is integral to avoid lookahead bias
        to maintain a realistic trading simulation."""
        for col in cols:
            df[col] = df.groupby('ticker')[col].transform(lambda x: (x - x.rolling(self.normalisation_window).mean().shift(1)) / (x.rolling(self.normalisation_window).std().shift(1) + 1e-8))
        return df