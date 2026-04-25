"""
Comparison script: PPO with Graph Embeddings vs PPO without Graph Embeddings
Trains both models on the same data and evaluates on the test set.
Uses the improved configuration with BIL ticker and tuned hyperparameters.
"""
from src.data.downloader import YahooDataDownloader
from src.data.preprocessor import FeatureEngineer
from src.data.graphbuilder import GraphBuilder
from src.env.portfolio_env import StockPortfolioEnv
from src.env.portfolio_env_baseline import StockPortfolioEnvBaseline
from src.agents.PPOTrainer import PPOTrainer
from src.utils.config_manager import load_config
from src.utils.seeding import Seed
from stable_baselines3 import PPO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def evaluate_model_on_test(model_path, env_class, env_kwargs, test_data):
    """
    Evaluate a trained model on test data.
    """
    # Load the best model
    model_file = os.path.join(model_path, "best_model.zip")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found at {model_file}")
    
    print(f"Loading model from {model_file}")
    
    # Set seed for deterministic evaluation
    Seed(42).set_all_seeds()
    
    model = PPO.load(model_file)
    
    # Create test environment
    test_env = env_class(df=test_data, **env_kwargs)
    
    print(f"Evaluating on {len(test_data.date.unique())} trading days...")
    
    # Run evaluation
    obs, _ = test_env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
    
    # Calculate metrics
    returns = test_env.portfolio_return_memory
    values = test_env.asset_memory
    
    df_returns = pd.DataFrame(returns, columns=['daily_return'])
    
    total_return = (values[-1] - values[0]) / values[0]
    sharpe = (252**0.5) * df_returns['daily_return'].mean() / df_returns['daily_return'].std()
    
    cumulative_returns = df_returns['daily_return'].cumsum()
    running_max = cumulative_returns.expanding().max()
    drawdown = cumulative_returns - running_max
    max_drawdown = drawdown.min()
    
    volatility = df_returns['daily_return'].std() * np.sqrt(252)
    
    results = {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'returns': returns,
        'portfolio_values': values,
        'actions': test_env.actions_memory,
        'dates': test_env.date_memory
    }
    
    return results

def save_test_results(results, csv_path):
    """Save detailed test results to CSV."""
    df = pd.DataFrame({
        'date': results['dates'],
        'portfolio_value': results['portfolio_values'],
        'daily_return': results['returns'],
        'cumulative_return': pd.Series(results['returns']).cumsum()
    })
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Test results saved to {csv_path}")

def train_and_evaluate():
    """Train both models and compare their performance on train and test sets"""
    config = load_config()
    data_config = config['data']
    preproc_config = config['preprocessing']
    graph_config = config['graph']
    env_config = config['env']
    config_ppo = config['ppo']

    print("\n" + "="*80)
    print("GRAPH-RL PORTFOLIO COMPARISON: WITH GRAPHS vs WITHOUT GRAPHS")
    print("="*80 + "\n")

    # ===== DATA PREPARATION (SHARED) =====
    print("=== Step 1: Data Ingestion & Preprocessing ===")
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

    print("\n=== Step 2: Graph Construction ===")
    builder = GraphBuilder(
        df_processed, 
        lookback_window=graph_config['lookback_window'],
        threshold=graph_config['threshold'],
        top_k=graph_config['top_k']
    )
    graphs = builder.build_graphs(sparsity_method=graph_config['sparsity_method'])
    
    # Ensure timestamp consistency
    if df_processed['date'].dtype == 'datetime64[ns]':
        graphs = {pd.Timestamp(k) if not isinstance(k, pd.Timestamp) else k: v for k, v in graphs.items()}
    
    # Get common dates between processed data and graphs
    common_dates = sorted(list(set(df_processed.date.unique()) & set(graphs.keys())))
    df_aligned = df_processed[df_processed['date'].isin(common_dates)].reset_index(drop=True)
    
    # Split data into train and test sets
    print("\n=== Step 3: Train/Test Split ===")
    train_end = pd.to_datetime(data_config['train_end_date'])
    test_start = pd.to_datetime(data_config['test_start_date'])
    
    df_train = df_aligned[df_aligned['date'] <= train_end].reset_index(drop=True)
    df_test = df_aligned[df_aligned['date'] >= test_start].reset_index(drop=True)
    
    print(f"Training period: {df_train['date'].min()} to {df_train['date'].max()}")
    print(f"Testing period: {df_test['date'].min()} to {df_test['date'].max()}")
    print(f"Training samples: {len(df_train.date.unique())} days")
    print(f"Testing samples: {len(df_test.date.unique())} days")
    print(f"Number of assets: {len(data_config['ticker_list'])} (including BIL)")

    # Timestamp for this comparison run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ===== MODEL 1: TRAIN WITH GRAPHS =====
    print("\n" + "="*80)
    print("MODEL 1: PPO WITH GRAPH EMBEDDINGS")
    print("="*80 + "\n")
    
    # Set seed for reproducibility of training
    print("Setting random seed to 42")
    Seed(42).set_all_seeds()
    
    env_kwargs_with = {
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

    print("Training on training set...")
    train_env_with = StockPortfolioEnv(df=df_train, **env_kwargs_with)
    trainer_with = PPOTrainer(train_env_with, config_ppo)
    trainer_with.train()
    
    # Save model
    model_path_with = f"models/ppo_with_graphs_{timestamp}"
    best_model_path_with = config_ppo['best_model_path']
    trainer_with.save(model_path_with)
    print(f"Model saved to {model_path_with}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results_with = evaluate_model_on_test(
        model_path=best_model_path_with,
        env_class=StockPortfolioEnv,
        env_kwargs=env_kwargs_with,
        test_data=df_test
    )
    
    # Save test results
    test_csv_with = f"results/test_with_graphs_{timestamp}.csv"
    save_test_results(test_results_with, test_csv_with)
    
    print("\n--- Test Results (WITH Graphs) ---")
    print(f"Total Return: {test_results_with['total_return']:.2%}")
    print(f"Sharpe Ratio: {test_results_with['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {test_results_with['max_drawdown']:.2%}")
    print(f"Volatility: {test_results_with['volatility']:.2%}")

    # ===== MODEL 2: TRAIN WITHOUT GRAPHS =====
    print("\n" + "="*80)
    print("MODEL 2: PPO WITHOUT GRAPH EMBEDDINGS (BASELINE)")
    print("="*80 + "\n")
    
    # Reset seed to same value for fair comparison
    print("Resetting random seed to 42 for fair comparison...")
    Seed(42).set_all_seeds()
    
    env_kwargs_without = {
        "stock_dim": len(data_config['ticker_list']),
        "initial_amount": env_config['initial_amount'], 
        "transaction_cost_pct": env_config['transaction_cost_pct'], 
        "reward_scaling": env_config['reward_scaling'],
        "state_space": len(data_config['ticker_list']), 
        "action_space": len(data_config['ticker_list']),
        "tech_indicator_list": preprocessor.tech_indicator_list,
        "turbulence_threshold": env_config['turbulence_threshold'],
        "lookback": graph_config['lookback_window']
    }

    print("Training on training set...")
    train_env_without = StockPortfolioEnvBaseline(df=df_train, **env_kwargs_without)
    trainer_without = PPOTrainer(train_env_without, config_ppo)
    trainer_without.train()
    
    # Save model
    model_path_without = f"models/ppo_without_graphs_{timestamp}"
    best_model_path_without = config_ppo['best_model_path']
    trainer_without.save(model_path_without)
    print(f"Model saved to {model_path_without}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results_without = evaluate_model_on_test(
        model_path=best_model_path_without,
        env_class=StockPortfolioEnvBaseline,
        env_kwargs=env_kwargs_without,
        test_data=df_test
    )
    
    # Save test results
    test_csv_without = f"results/test_without_graphs_{timestamp}.csv"
    save_test_results(test_results_without, test_csv_without)
    
    print("\n--- Test Results (WITHOUT Graphs) ---")
    print(f"Total Return: {test_results_without['total_return']:.2%}")
    print(f"Sharpe Ratio: {test_results_without['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {test_results_without['max_drawdown']:.2%}")
    print(f"Volatility: {test_results_without['volatility']:.2%}")

    # ===== COMPARISON & ANALYSIS =====
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON ANALYSIS")
    print("="*80 + "\n")
    
    generate_comparison_report(
        test_results_with, 
        test_results_without,
        test_csv_with,
        test_csv_without,
        timestamp
    )

def generate_comparison_report(results_with, results_without, csv_with, csv_without, timestamp):
    """Generate comprehensive comparison report and visualizations"""
    
    # Load detailed results
    df_with = pd.read_csv(csv_with)
    df_without = pd.read_csv(csv_without)
    
    # Print comparison table
    print("="*80)
    print("TEST SET PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Metric':<30} {'With Graphs':<20} {'Without Graphs':<20} {'Improvement':<15}")
    print("-"*80)
    
    # Total Return
    return_diff = results_with['total_return'] - results_without['total_return']
    return_pct = (return_diff / abs(results_without['total_return'])) * 100 if results_without['total_return'] != 0 else 0
    print(f"{'Total Return':<30} {results_with['total_return']:>19.2%} {results_without['total_return']:>19.2%} {return_pct:>+14.2f}%")
    
    # Sharpe Ratio
    sharpe_diff = results_with['sharpe_ratio'] - results_without['sharpe_ratio']
    sharpe_pct = (sharpe_diff / abs(results_without['sharpe_ratio'])) * 100 if results_without['sharpe_ratio'] != 0 else 0
    print(f"{'Sharpe Ratio':<30} {results_with['sharpe_ratio']:>19.4f} {results_without['sharpe_ratio']:>19.4f} {sharpe_pct:>+14.2f}%")
    
    # Max Drawdown (lower is better)
    drawdown_diff = results_without['max_drawdown'] - results_with['max_drawdown']
    drawdown_pct = (drawdown_diff / abs(results_without['max_drawdown'])) * 100 if results_without['max_drawdown'] != 0 else 0
    print(f"{'Max Drawdown (lower=better)':<30} {results_with['max_drawdown']:>19.2%} {results_without['max_drawdown']:>19.2%} {drawdown_pct:>+14.2f}%")
    
    # Volatility (lower is better)
    vol_diff = results_without['volatility'] - results_with['volatility']
    vol_pct = (vol_diff / abs(results_without['volatility'])) * 100 if results_without['volatility'] != 0 else 0
    print(f"{'Volatility (lower=better)':<30} {results_with['volatility']:>19.2%} {results_without['volatility']:>19.2%} {vol_pct:>+14.2f}%")
    
    print("="*80)
    
    # Key insights
    print("\nKEY INSIGHTS:")
    if results_with['total_return'] > results_without['total_return']:
        print(f"Graph embeddings IMPROVED returns by {return_pct:.2f}%")
    else:
        print(f"Graph embeddings DECREASED returns by {abs(return_pct):.2f}%")
    
    if results_with['sharpe_ratio'] > results_without['sharpe_ratio']:
        print(f"Graph embeddings IMPROVED Sharpe ratio by {sharpe_pct:.2f}%")
    else:
        print(f"Graph embeddings DECREASED Sharpe ratio by {abs(sharpe_pct):.2f}%")
    
    if results_with['max_drawdown'] < results_without['max_drawdown']:
        print(f"Graph embeddings REDUCED max drawdown by {drawdown_pct:.2f}%")
    else:
        print(f"Graph embeddings INCREASED max drawdown by {abs(drawdown_pct):.2f}%")
    
    print()
    
    # Generate visualizations
    os.makedirs('results/plots_data', exist_ok=True)
    
    # 1. Portfolio value over time comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Portfolio value
    axes[0, 0].plot(df_with['portfolio_value'], label='With Graphs', linewidth=2)
    axes[0, 0].plot(df_without['portfolio_value'], label='Without Graphs', linewidth=2)
    axes[0, 0].set_xlabel('Trading Day')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].set_title('Portfolio Value Over Time (Test Period)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative returns
    axes[0, 1].plot(df_with['cumulative_return'], label='With Graphs', linewidth=2)
    axes[0, 1].plot(df_without['cumulative_return'], label='Without Graphs', linewidth=2)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Trading Day')
    axes[0, 1].set_ylabel('Cumulative Return')
    axes[0, 1].set_title('Cumulative Returns (Test Period)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Daily returns distribution
    axes[1, 0].hist(df_with['daily_return'], bins=30, alpha=0.6, label='With Graphs', density=True)
    axes[1, 0].hist(df_without['daily_return'], bins=30, alpha=0.6, label='Without Graphs', density=True)
    axes[1, 0].set_xlabel('Daily Return')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Daily Returns Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance metrics bar chart
    metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown\n(neg)', 'Volatility\n(neg)']
    with_values = [
        results_with['total_return']*100, 
        results_with['sharpe_ratio'], 
        -results_with['max_drawdown']*100,
        -results_with['volatility']*100
    ]
    without_values = [
        results_without['total_return']*100, 
        results_without['sharpe_ratio'], 
        -results_without['max_drawdown']*100,
        -results_without['volatility']*100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, with_values, width, label='With Graphs', alpha=0.8)
    axes[1, 1].bar(x + width/2, without_values, width, label='Without Graphs', alpha=0.8)
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Performance Metrics Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics, fontsize=9)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plot_path = f'results/plots_data/comparison_plot_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {plot_path}")
    
    # Save summary CSV
    summary_df = pd.DataFrame({
        'Metric': ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Volatility (%)'],
        'With Graphs': [
            results_with['total_return']*100,
            results_with['sharpe_ratio'],
            results_with['max_drawdown']*100,
            results_with['volatility']*100
        ],
        'Without Graphs': [
            results_without['total_return']*100,
            results_without['sharpe_ratio'],
            results_without['max_drawdown']*100,
            results_without['volatility']*100
        ],
        'Improvement (%)': [
            return_pct,
            sharpe_pct,
            drawdown_pct,
            vol_pct
        ]
    })
    
    summary_path = f'results/comparison_summary_{timestamp}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80 + "\n")

if __name__ == "__main__":
    train_and_evaluate()
