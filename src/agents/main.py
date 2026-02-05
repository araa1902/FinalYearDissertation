from src.data.downloader import YahooDataDownloader
from src.data.preprocessor import FeatureEngineer
from src.data.graphbuilder import GraphBuilder
from src.env.portfolio_env import StockPortfolioEnv
from src.agents.PPOTrainer import PPOTrainer
from src.agents.evaluator import ModelEvaluator
import yaml
import pandas as pd
from datetime import datetime

def load_config():
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    data_config = config['data']
    preproc_config = config['preprocessing']
    graph_config = config['graph']
    env_config = config['env']
    config_ppo = config['ppo']

    print("=== Data Ingestion & Preprocessing ===")
    downloader = YahooDataDownloader(
        start_date=data_config['start_date'], 
        end_date=data_config['end_date'], 
        ticker_list=data_config['ticker_list']
    )
    df_raw = downloader.fetch_data()
    
    preprocessor = FeatureEngineer(
        use_technical_indicator=preproc_config['use_technical_indicator'], 
        tech_indicator_list=preproc_config['tech_indicator_list'],
        normalisation_window=preproc_config['normalisation_window']
    )
    df_processed = preprocessor.preprocess_data(df_raw)

    print("=== Graph Construction ===")
    builder = GraphBuilder(
        df_processed, 
        lookback_window=graph_config['lookback_window'],
        threshold=graph_config['threshold'],
        top_k=graph_config['top_k']
    )
    graphs = builder.build_graphs(sparsity_method=graph_config['sparsity_method'])
    
    if df_processed['date'].dtype == 'datetime64[ns]':
        graphs = {pd.Timestamp(k) if not isinstance(k, pd.Timestamp) else k: v for k, v in graphs.items()}
    
    # Get common dates between processed data and graphs
    common_dates = sorted(list(set(df_processed.date.unique()) & set(graphs.keys())))
    df_aligned = df_processed[df_processed['date'].isin(common_dates)].reset_index(drop=True)
    
    # Split data into train and test sets based on dates
    print("\n=== Splitting Data: Train/Test ===")
    train_end = pd.to_datetime(data_config['train_end_date'])
    test_start = pd.to_datetime(data_config['test_start_date'])
    
    df_train = df_aligned[df_aligned['date'] <= train_end].reset_index(drop=True)
    df_test = df_aligned[df_aligned['date'] >= test_start].reset_index(drop=True)
    
    print(f"Training period: {df_train['date'].min()} to {df_train['date'].max()}")
    print(f"Testing period: {df_test['date'].min()} to {df_test['date'].max()}")
    print(f"Training samples: {len(df_train.date.unique())} days")
    print(f"Testing samples: {len(df_test.date.unique())} days")

    print("\n=== PPO Training ===")
    
    env_kwargs = {
        "stock_dim": len(data_config['ticker_list']),
        "initial_amount": env_config['initial_amount'], 
        "transaction_cost_pct": env_config['transaction_cost_pct'], 
        "reward_scaling": env_config['reward_scaling'],
        "state_space": len(data_config['ticker_list']), 
        "action_space": len(data_config['ticker_list']),
        "tech_indicator_list": preprocessor.tech_indicator_list,
        "turbulence_threshold": env_config['turbulence_threshold'],
        "lookback": graph_config['lookback_window'],
        "graph_dict": graphs
    }

    # Train on training data only
    train_env = StockPortfolioEnv(df=df_train, **env_kwargs)
    trainer = PPOTrainer(train_env, config_ppo)
    trainer.train()
    
    # Save the trained model
    model_path = "models/trained_ppo"
    trainer.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Evaluate on unseen test data
    print("\n=== Evaluating on Unseen Test Data ===")
    evaluator = ModelEvaluator(
        model_path=config_ppo['best_model_path'],
        env_kwargs=env_kwargs,
        test_data=df_test
    )
    
    test_results = evaluator.evaluate()
    
    # Save test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"results/test_evaluation_{timestamp}.csv"
    evaluator.save_results(results_path)
    print(f"\nTest results saved to {results_path}")
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Total Return: {test_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {test_results['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {test_results['max_drawdown']:.2%}")
    print(f"Volatility: {test_results['volatility']:.2%}")

if __name__ == "__main__":
    main()