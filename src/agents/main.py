from src.data.downloader import YahooDataDownloader
from src.data.preprocessor import FeatureEngineer
from src.data.graphbuilder import GraphBuilder
from src.env.portfolio_env import StockPortfolioEnv
from src.agents.PPOTrainer import PPOTrainer
import yaml

def load_config():
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    
    # Extract config sections
    data_config = config['data']
    preproc_config = config['preprocessing']
    graph_config = config['graph']
    env_config = config['env']
    
    print("=== Data Ingestion & Preprocessing ===")
    downloader = YahooDataDownloader(
        start_date=data_config['start_date'], 
        end_date=data_config['end_date'], 
        ticker_list=data_config['ticker_list']
    )
    df_raw = downloader.fetch_data()
    
    preprocessor = FeatureEngineer(
        use_technical_indicator=preproc_config['use_technical_indicators'], 
        tech_indicator_list=preproc_config['tech_indicator_list'],
        normalisation_window=preproc_config['normalisation_window']
    )
    df_processed = preprocessor.preprocess_data(df_raw)

    print("=== Graph Construction ===")
    builder = GraphBuilder(
        df_processed, 
        lookback_window=graph_config['lookback'],
        threshold=graph_config['threshold'],
        top_k=graph_config['top_k']
    )
    graphs = builder.build_graphs(sparsity_method=graph_config['sparsity_method'])
    
    common_dates = sorted(list(set(df_processed.date.unique()) & set(graphs.keys())))
    df_train = df_processed[df_processed['date'].isin(common_dates)].reset_index(drop=True)

    print("=== PPO Baseline Training ===")
    
    env_kwargs = {
        "stock_dim": len(data_config['ticker_list']),
        "hmax": env_config['hmax'], 
        "initial_amount": env_config['initial_amount'], 
        "transaction_cost_pct": env_config['transaction_cost_pct'], 
        "reward_scaling": env_config['reward_scaling'],
        "state_space": len(data_config['ticker_list']), 
        "action_space": len(data_config['ticker_list']),
        "tech_indicator_list": preprocessor.tech_indicator_list,
        "turbulence_threshold": env_config['turbulence_threshold'],
        "lookback": graph_config['lookback'],
        "graph_dict": graphs
    }

    env = StockPortfolioEnv(df=df_train, **env_kwargs)
    trainer = PPOTrainer(env, config)
    trainer.train()
    trainer.save("models/baseline_ppo")

if __name__ == "__main__":
    main()