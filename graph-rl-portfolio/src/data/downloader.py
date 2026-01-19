import yfinance as yf
import pandas as pd
import os

class YahooDataDownloader:
    """Downloads and caches daily stock data from Yahoo Finance in LONG format."""
    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.cache_dir = "data/raw"
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_data(self) -> pd.DataFrame:
        file_path = f"{self.cache_dir}/data_{self.start_date}_{self.end_date}.parquet"
        
        if os.path.exists(file_path):
            print(f"Loading data from cache: {file_path}")
            return pd.read_parquet(file_path)

        print(f"Downloading data for {len(self.ticker_list)} tickers...")
        try:
            data = yf.download(
                tickers=self.ticker_list,
                start=self.start_date,
                end=self.end_date,
                group_by='ticker',
                auto_adjust=True,
                threads=True
            )
        except Exception as e:
            raise RuntimeError(f"Yahoo Finance download failed: {e}")

        if data.empty:
            raise ValueError("No data downloaded")

        data = data.stack(level=0).rename_axis(['date', 'ticker']).reset_index()
        data.columns = [c.lower() for c in data.columns] 
        
        data.to_parquet(file_path)
        return data