import numpy as np
import pandas as pd
import ta
import yaml

class FeatureEngineer:
    """Preprocesses data with technical indicators and rolling normalisation."""
    def __init__(self, use_technical_indicator: bool = True, tech_indicator_list: list = None, normalisation_window: int = 63):
        with open("config/config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
        
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list if tech_indicator_list is not None else self.config['preprocessing']['tech_indicator_list']
        self.normalisation_window = normalisation_window

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _apply_rolling_normalisation(self, df : pd.DataFrame, cols: list) -> pd.DataFrame:
        """Applies Z-Score normalisation using trailing windows [t-Tw, t-1]. This is integral to avoid lookahead bias
        to maintain a realistic trading simulation."""
        for col in cols:
            df[col] = df.groupby('ticker')[col].transform(lambda x: (x - x.rolling(self.normalisation_window).mean().shift(1)) / (x.rolling(self.normalisation_window).std().shift(1) + 1e-8))
        return df
    
    def _align_to_business_days(self, df, ticker_list, start_date, end_date):
        """Aligns the dataframe to a complete business day calendar for all tickers.
           This treats yfinance issue with missing days as non-trading days and fills them accordingly.
        """
        # 1. Create Full Business Day Date Range
        full_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 2. Create the Cartesian Product (Dates x Tickers)
        index = pd.MultiIndex.from_product(
            [full_dates, ticker_list], 
            names=['date', 'ticker']
        )
        
        # 3. Align the existing data to this grid
        df_clean = df.copy()
        if 'date' in df_clean.columns:
            df_clean.set_index(['date', 'ticker'], inplace=True)
        df_aligned = df_clean.reindex(index)
        
        # 4. Handle Missing Data (The "Causal" Way)
        #If data is missing today, assume price is same as yesterday.
        df_aligned = df_aligned.groupby(level='ticker').ffill()
        
        # Backward Fill (Edge Case): If data is missing at the very start (Day 0), 
        df_aligned = df_aligned.groupby(level='ticker').bfill()
        
        # 5. Handle Volume specifically since it should be zero on non-trading days
        if 'volume' in df_aligned.columns:
            df_aligned['volume'] = df_aligned['volume'].fillna(0)
            
        # Reset index to match standard format
        df_aligned = df_aligned.reset_index()
        
        # Filter out any weekends that might have slipped in (sanity check)
        df_aligned['day_of_week'] = df_aligned['date'].dt.dayofweek
        df_aligned = df_aligned[df_aligned['day_of_week'] < 5]
        df_aligned.drop(columns=['day_of_week'], inplace=True)
        
        return df_aligned
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(by=['date', 'ticker'])   

        df = self._align_to_business_days(
            df, 
            ticker_list=df['ticker'].unique().tolist(),
            start_date=df['date'].min(),
            end_date=df['date'].max()
        )
        
        if self.use_technical_indicator:
            df = self._add_technical_indicators(df)
            
        df['log_return'] = df.groupby('ticker')['close'].transform(lambda x: np.log(x / x.shift(1)))
        
        potential_cols = self.tech_indicator_list + ['volume', 'log_return']
        cols_to_normalise = [c for c in potential_cols if c in df.columns]
        
        df = self._apply_rolling_normalisation(df, cols_to_normalise)

        df = df.dropna().reset_index(drop=True)
        return df