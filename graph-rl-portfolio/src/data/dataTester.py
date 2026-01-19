import pandas as pd
from src.data.downloader import YahooDataDownloader
from src.data.preprocessor import FeatureEngineer
from src.data.graphbuilder import GraphBuilder
from src.env.portfolio_env import StockPortfolioEnv


tickers = ['AAPL', 'MSFT', 'GOOGL']
downloader = YahooDataDownloader(start_date='2018-01-01', end_date='2021-01-01', ticker_list=tickers)
df = downloader.fetch_data()

feature_engineer = FeatureEngineer()
df = feature_engineer.preprocess_data(df)

graph_builder = GraphBuilder(df, lookback_window=63)
graphs = graph_builder.build_graphs()
df = graph_builder.merge_graphs_to_dataframe(df, graphs)
df_wide = graph_builder.convert_to_wide_format(df)
print(df_wide)
# env = StockPortfolioEnv(df_wide, initial_amount=1_000_000, transaction_cost_pct=0.001)

'''
PIPELINE:
1. Download raw stock data using YahooDataDownloader.
2. Preprocess data with FeatureEngineer to add technical indicators and normalise.
3. Build correlation graphs with GraphBuilder and merge into DataFrame.
4. Convert long-format DataFrame to wide-format for model input.
5. Use PortfolioEnv to create RL environment with processed data.
'''