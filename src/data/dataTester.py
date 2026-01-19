# main.py
import pandas as pd
from downloader import YahooDataDownloader
from preprocessor import FeatureEngineer
from graphbuilder import GraphBuilder
from src.env.portfolio_env import StockPortfolioEnv

# 1. Download
downloader = YahooDataDownloader(start_date="2020-01-01", end_date="2023-01-01", ticker_list=['AAPL', 'MSFT', 'GOOG'])
raw_df = downloader.fetch_data()

# 2. Preprocess (Calculates Log Returns + Z-Score)
# This drops the first 63 rows due to rolling window!
fe = FeatureEngineer()
processed_df = fe.preprocess_data(raw_df)

# 3. Build Graphs
# GraphBuilder needs raw log returns, so we can pass processed_df (it has log_returns)
# OR pass raw_df if you want the builder to do its own log calc.
# Passing processed_df is safer to ensure dates match.
gb = GraphBuilder(processed_df, lookback_window=30, top_k=5)
graphs = gb.build_graphs(sparsity_method='knn')

# 4. Align Data & Graphs
# The GraphBuilder starts at index 'lookback_window'. 
# Ensure env data matches available graph dates - treats issue from yfinance missing data.
common_dates = sorted(list(set(processed_df.date.unique()) & set(graphs.keys())))
final_df = processed_df[processed_df['date'].isin(common_dates)].reset_index(drop=True)

# 5. Init Environment
env = StockPortfolioEnv(
    df=final_df,
    graph_dict=graphs,
    stock_dim=len(final_df.ticker.unique()),
    hmax=100,
    initial_amount=1000000,
    transaction_cost_pct=0.001,
    reward_scaling=1,
    state_space=len(final_df.ticker.unique()),
    action_space=len(final_df.ticker.unique()),
    tech_indicator_list=['macd', 'rsi', 'cci', 'dx'],
    turbulence_threshold=None
)

# Test Run
obs = env.reset()
print("Environment Ready. State Shape:", obs.shape)