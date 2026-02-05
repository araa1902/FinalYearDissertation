"""
Comparison script for PPO with and without graph embeddings.
This script trains two models side-by-side and compares their performance.
"""
from ..data.downloader import YahooDataDownloader
from ..data.preprocessor import FeatureEngineer
from ..data.graphbuilder import GraphBuilder
from ..env.portfolio_env import StockPortfolioEnv
from ..env.portfolio_env_baseline import StockPortfolioEnvBaseline
from ..agents.PPOTrainer import PPOTrainer
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def load_config():
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)

def compare_models():
    """Train PPO with and without graphs, then compare results"""
    config = load_config()
    data_config = config['data']
    preproc_config = config['preprocessing']
    graph_config = config['graph']
    env_config = config['env']
    config_ppo = config['ppo']

    print("\n" + "="*60)
    print("PPO COMPARISON: WITH GRAPHS vs WITHOUT GRAPHS")
    print("="*60 + "\n")

    # ===== DATA PREPARATION (SHARED) =====
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
    
    common_dates = sorted(list(set(df_processed.date.unique()) & set(graphs.keys())))
    df_train = df_processed[df_processed['date'].isin(common_dates)].reset_index(drop=True)

    # ===== TRAIN WITH GRAPHS =====
    print("\n" + "="*60)
    print("TRAINING PPO WITH GRAPH EMBEDDINGS")
    print("="*60 + "\n")
    
    timestamp_with_graphs = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv_with = f'results/comparison_with_graphs_{timestamp_with_graphs}.csv'
    
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
        "graph_dict": graphs,
        "results_csv_path": results_csv_with
    }

    env_with_graphs = StockPortfolioEnv(df=df_train, **env_kwargs_with)
    trainer_with = PPOTrainer(env_with_graphs, config_ppo)
    trainer_with.train()
    trainer_with.save("models/ppo_with_graphs")
    env_with_graphs.save_final_results()
    
    # Save metrics
    metrics_with = pd.read_csv(results_csv_with)

    # ===== TRAIN WITHOUT GRAPHS =====
    print("\n" + "="*60)
    print("TRAINING PPO WITHOUT GRAPH EMBEDDINGS (BASELINE)")
    print("="*60 + "\n")
    
    timestamp_without_graphs = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv_without = f'results/comparison_without_graphs_{timestamp_without_graphs}.csv'
    
    env_kwargs_without = {
        "stock_dim": len(data_config['ticker_list']),
        "initial_amount": env_config['initial_amount'], 
        "transaction_cost_pct": env_config['transaction_cost_pct'], 
        "reward_scaling": env_config['reward_scaling'],
        "state_space": len(data_config['ticker_list']), 
        "action_space": len(data_config['ticker_list']),
        "tech_indicator_list": preprocessor.tech_indicator_list,
        "turbulence_threshold": env_config['turbulence_threshold'],
        "results_csv_path": results_csv_without
    }

    env_without_graphs = StockPortfolioEnvBaseline(df=df_train, **env_kwargs_without)
    trainer_without = PPOTrainer(env_without_graphs, config_ppo)
    trainer_without.train()
    trainer_without.save("models/ppo_without_graphs")
    env_without_graphs.save_final_results()
    
    # Save metrics
    metrics_without = pd.read_csv(results_csv_without)

    # ===== COMPARISON & ANALYSIS =====
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60 + "\n")
    
    compare_and_plot_results(metrics_with, metrics_without, results_csv_with, results_csv_without)

def compare_and_plot_results(metrics_with, metrics_without, csv_with, csv_without):
    """Compare metrics and generate comparison plots"""
    
    # Calculate aggregate statistics
    stats_with = {
        'avg_return': metrics_with['Total_Return'].mean(),
        'std_return': metrics_with['Total_Return'].std(),
        'avg_sharpe': metrics_with['Sharpe_Ratio'].mean(),
        'avg_drawdown': metrics_with['Max_Drawdown'].mean(),
        'final_asset': metrics_with['End_Total_Asset'].iloc[-1]
    }
    
    stats_without = {
        'avg_return': metrics_without['Total_Return'].mean(),
        'std_return': metrics_without['Total_Return'].std(),
        'avg_sharpe': metrics_without['Sharpe_Ratio'].mean(),
        'avg_drawdown': metrics_without['Max_Drawdown'].mean(),
        'final_asset': metrics_without['End_Total_Asset'].iloc[-1]
    }
    
    # Print comparison table
    print("Performance Metrics Comparison:")
    print("-" * 60)
    print(f"{'Metric':<25} {'With Graphs':<20} {'Without Graphs':<20}")
    print("-" * 60)
    print(f"{'Avg Total Return':<25} {stats_with['avg_return']:>19.4f} {stats_without['avg_return']:>19.4f}")
    print(f"{'Std Dev Return':<25} {stats_with['std_return']:>19.4f} {stats_without['std_return']:>19.4f}")
    print(f"{'Avg Sharpe Ratio':<25} {stats_with['avg_sharpe']:>19.4f} {stats_without['avg_sharpe']:>19.4f}")
    print(f"{'Avg Max Drawdown':<25} {stats_with['avg_drawdown']:>19.4f} {stats_without['avg_drawdown']:>19.4f}")
    print(f"{'Final Portfolio Value':<25} ${stats_with['final_asset']:>18.2f} ${stats_without['final_asset']:>18.2f}")
    print("-" * 60)
    
    # Calculate improvements
    return_improvement = ((stats_with['avg_return'] - stats_without['avg_return']) / abs(stats_without['avg_return'])) * 100 if stats_without['avg_return'] != 0 else 0
    sharpe_improvement = ((stats_with['avg_sharpe'] - stats_without['avg_sharpe']) / abs(stats_without['avg_sharpe'])) * 100 if stats_without['avg_sharpe'] != 0 else 0
    
    print(f"\nImprovement with Graphs:")
    print(f"  Return improvement: {return_improvement:+.2f}%")
    print(f"  Sharpe ratio improvement: {sharpe_improvement:+.2f}%")
    print()
    
    # Generate comparison plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Total Return Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(metrics_with['Total_Return'].values, label='With Graphs', marker='o')
    axes[0, 0].plot(metrics_without['Total_Return'].values, label='Without Graphs', marker='s')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Return')
    axes[0, 0].set_title('Total Return per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Sharpe Ratio Comparison
    axes[0, 1].plot(metrics_with['Sharpe_Ratio'].values, label='With Graphs', marker='o')
    axes[0, 1].plot(metrics_without['Sharpe_Ratio'].values, label='Without Graphs', marker='s')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].set_title('Sharpe Ratio per Episode')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Max Drawdown Comparison
    axes[1, 0].plot(metrics_with['Max_Drawdown'].values, label='With Graphs', marker='o')
    axes[1, 0].plot(metrics_without['Max_Drawdown'].values, label='Without Graphs', marker='s')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Max Drawdown')
    axes[1, 0].set_title('Maximum Drawdown per Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Portfolio Value Over Time
    axes[1, 1].plot(metrics_with['End_Total_Asset'].values, label='With Graphs', marker='o')
    axes[1, 1].plot(metrics_without['End_Total_Asset'].values, label='Without Graphs', marker='s')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Portfolio Value ($)')
    axes[1, 1].set_title('Final Portfolio Value per Episode')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_plot = f'results/comparison_plot_{timestamp}.png'
    plt.savefig(comparison_plot, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {comparison_plot}")
    
    # 5. Distribution Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(metrics_with['Total_Return'].values, bins=10, alpha=0.6, label='With Graphs')
    axes[0].hist(metrics_without['Total_Return'].values, bins=10, alpha=0.6, label='Without Graphs')
    axes[0].set_xlabel('Total Return')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Total Returns')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(metrics_with['Sharpe_Ratio'].values, bins=10, alpha=0.6, label='With Graphs')
    axes[1].hist(metrics_without['Sharpe_Ratio'].values, bins=10, alpha=0.6, label='Without Graphs')
    axes[1].set_xlabel('Sharpe Ratio')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Sharpe Ratios')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    distribution_plot = f'results/distribution_comparison_{timestamp}.png'
    plt.savefig(distribution_plot, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution plot saved to: {distribution_plot}")
    
    # Save detailed comparison CSV
    comparison_df = pd.DataFrame({
        'Metric': ['Avg Total Return', 'Std Dev Return', 'Avg Sharpe Ratio', 'Avg Max Drawdown', 'Final Portfolio Value'],
        'With Graphs': [stats_with['avg_return'], stats_with['std_return'], stats_with['avg_sharpe'], stats_with['avg_drawdown'], stats_with['final_asset']],
        'Without Graphs': [stats_without['avg_return'], stats_without['std_return'], stats_without['avg_sharpe'], stats_without['avg_drawdown'], stats_without['final_asset']],
        'Improvement %': [return_improvement if 'return' in 'avg_return' else 0, 0, sharpe_improvement if 'sharpe' in 'avg_sharpe' else 0, 0, 0]
    })
    
    comparison_csv = f'results/detailed_comparison_{timestamp}.csv'
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"Detailed comparison saved to: {comparison_csv}")
    
    print("\n" + "="*60)
    print("Comparison Complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    compare_models()
