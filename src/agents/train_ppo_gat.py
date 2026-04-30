"""
Training script: PPO with Graph Attention Networks (GAT) for Portfolio Optimisation.
Features:
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO
import argparse
import pickle

# --- Project Modules ---
from ..data.downloader import YahooDataDownloader
from ..data.preprocessor import FeatureEngineer
from ..data.graphbuilder import GraphBuilder
from ..env.portfolio_env import StockPortfolioEnv
from .PPO_GAT_Trainer import PPOGATTrainer
from ..utils.seeding import Seed
from ..utils.config_manager import load_config


def evaluate_model_on_test(model_path, env_class, env_kwargs, test_data):
    """
    Evaluate a trained model on test data using deterministic predictions.
    SPRINT 1: Captures GAT attention weights during evaluation for explainability analysis.
    """
    # Load the best model - try both possible locations
    model_file = os.path.join(model_path, "best_model.zip")
    if not os.path.exists(model_file):
        # If not found in subdirectory, try the path directly as a .zip file
        model_file = f"{model_path}.zip"
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model not found at {model_path}/best_model.zip or {model_path}.zip")
    
    print(f"Loading model from {model_file}")
    
    # Set seed for deterministic evaluation
    Seed(42).set_all_seeds()
    
    model = PPO.load(model_file)
    
    # Create test environment
    # Note: graph_dict is None because GAT builds it internally
    test_env = env_class(df=test_data, **env_kwargs)
    
    print(f"Evaluating on {len(test_data.date.unique())} trading days...")
    print("Capturing attention weights during evaluation...")
    
    # Access feature extractor for attention capture
    feature_extractor = model.policy.features_extractor
    
    # Run evaluation
    obs, _ = test_env.reset()
    done = False
    step_count = 0
    
    while not done:
        # deterministic=True is crucial for valid testing (turns off exploration noise)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
        # Capture attention weights at each step
        if feature_extractor is not None:
            test_env.log_attention_weights(feature_extractor)
        
        step_count += 1
    
    # Save attention buffer for test period analysis
    test_env.save_final_results()
    print(f" Captured attention for {step_count} evaluation steps")
    
    # Calculate metrics from environment memory
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
    """Main pipeline: Data -> Train PPO+GAT -> Evaluate"""
    config = load_config()
    data_config = config['data']
    preproc_config = config['preprocessing']
    graph_config = config['graph']
    env_config = config['env']
    config_ppo = config['ppo']

    print("\n" + "="*80)
    print("PPO WITH GRAPH ATTENTION NETWORKS (GAT) - TRAINING & EVALUATION")
    print("="*80 + "\n")

    # ===== STEP 1: DATA PREPARATION =====
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

    # ===== STEP 1B: GRAPH CONSTRUCTION =====
    print("\n=== Step 1B: Graph Construction (Price Correlations) ===")
    print("Building adjacency matrices from historical price correlations...")
    builder = GraphBuilder(
        df_processed,
        lookback_window=graph_config['lookback_window'],
        threshold=graph_config['threshold'],
        top_k=graph_config['top_k']
    )
    graphs = builder.build_graphs(sparsity_method=graph_config['sparsity_method'])
    print(f" Built {len(graphs)} daily correlation graphs")
    print(f"  Sparsity method: {graph_config['sparsity_method']}")
    print(f"  Threshold: {graph_config['threshold']}")
    
    # Align data with graph dates
    common_dates = sorted(list(set(df_processed['date'].unique()) & set(graphs.keys())))
    df_processed = df_processed[df_processed['date'].isin(common_dates)].reset_index(drop=True)
    print(f" Aligned data: {len(common_dates)} dates with both features and graphs")

    # ===== STEP 2: TRAIN/TEST SPLIT =====
    print("\n=== Step 2: Train/Test Split ===")
    train_end = pd.to_datetime(data_config['train_end_date'])
    test_start = pd.to_datetime(data_config['test_start_date'])
    
    # Ensure correct data types
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    
    df_train = df_processed[df_processed['date'] <= train_end].reset_index(drop=True)
    df_test = df_processed[df_processed['date'] >= test_start].reset_index(drop=True)
    
    print(f"Training period: {df_train['date'].min()} to {df_train['date'].max()}")
    print(f"Testing period: {df_test['date'].min()} to {df_test['date'].max()}")
    print(f"Training samples: {len(df_train.date.unique())} days")
    print(f"Testing samples: {len(df_test.date.unique())} days")

    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ===== STEP 3: TRAINING =====
    print("\n" + "="*80)
    print("TRAINING: PPO WITH GAT")
    print("="*80 + "\n")
    
    # Set seed for reproducibility
    Seed(42).set_all_seeds()
    
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
        "sector_map": data_config['sector_map'],
        "max_sector_weight": env_config['max_sector_weight'],
        "graph_dict": graphs 
    }

    print("Creating training environment...")
    train_env = StockPortfolioEnv(df=df_train, **env_kwargs)
    
    print("Initialising PPO+GAT trainer...")
    trainer = PPOGATTrainer(train_env, config_ppo)
    
    print(f"Starting training for {config_ppo['total_timesteps']} timesteps...")
    trainer.train()
    
    # Save model
    model_path = f"models/ppo_gat_{timestamp}"
    os.makedirs(model_path, exist_ok=True)
    trainer.save(model_path)
    print(f"Model saved to {model_path}")

    # ===== STEP 4: EVALUATION =====
    print("\n" + "="*80)
    print("EVALUATION: TEST SET PERFORMANCE")
    print("="*80 + "\n")
    
    print("Evaluating trained model on test set...")
    test_results = evaluate_model_on_test(
        model_path=model_path,
        env_class=StockPortfolioEnv,
        env_kwargs=env_kwargs,
        test_data=df_test
    )
    
    # Save test results
    test_csv = f"results/gat_ppo_results/test_ppo_gat_{timestamp}.csv"
    save_test_results(test_results, test_csv)
    
    # ===== RESULTS SUMMARY =====
    print("\n" + "="*80)
    print("TEST SET PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Total Return:        {test_results['total_return']:>10.2%}")
    print(f"Sharpe Ratio:        {test_results['sharpe_ratio']:>10.4f}")
    print(f"Max Drawdown:        {test_results['max_drawdown']:>10.2%}")
    print(f"Volatility (Annual): {test_results['volatility']:>10.2%}")
    print("="*80 + "\n")

    # ===== VISUALISATION =====
    print("Generating visualisation...")
    generate_visualisation(test_results, test_csv, timestamp)

def generate_visualisation(results, csv_path, timestamp):
    """Generate visualisation of test performance"""
    
    # Load detailed results
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('PPO+GAT Portfolio Optimisation - Test Performance', fontsize=16, fontweight='bold')
    
    # 1. Portfolio value over time
    axes[0, 0].plot(df['portfolio_value'], linewidth=2, color='#2E86AB')
    axes[0, 0].fill_between(range(len(df)), df['portfolio_value'].min(), df['portfolio_value'], alpha=0.3, color='#2E86AB')
    axes[0, 0].set_xlabel('Trading Day', fontsize=11)
    axes[0, 0].set_ylabel('Portfolio Value ($)', fontsize=11)
    axes[0, 0].set_title('Portfolio Value Over Time (Test Period)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cumulative returns
    axes[0, 1].plot(df['cumulative_return'], linewidth=2, color='#A23B72')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].fill_between(range(len(df)), df['cumulative_return'], 0, alpha=0.3, color='#A23B72')
    axes[0, 1].set_xlabel('Trading Day', fontsize=11)
    axes[0, 1].set_ylabel('Cumulative Return', fontsize=11)
    axes[0, 1].set_title('Cumulative Returns (Test Period)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Daily returns distribution
    axes[1, 0].hist(df['daily_return'], bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=df['daily_return'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["daily_return"].mean():.4f}')
    axes[1, 0].set_xlabel('Daily Return', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Performance metrics
    metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown\n(Lower=Better)', 'Volatility\n(Lower=Better)']
    values = [
        results['total_return'] * 100,
        results['sharpe_ratio'],
        results['max_drawdown'] * 100,
        results['volatility'] * 100
    ]
    colours = ['#06A77D' if v > 0 else '#D62828' for v in values]
    
    bars = axes[1, 1].bar(metrics, values, color=colours, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_ylabel('Value', fontsize=11)
    axes[1, 1].set_title('Performance Metrics Summary', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}',
                       ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_path = f'results/gat_ppo_results/ppo_gat_performance_{timestamp}.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Performance plot saved to: {plot_path}")


def merge_attention_buffers():
    """
    Automatically merge training and evaluation attention buffers into a single unified buffer.
    This enables complete analysis across both in-sample and out-of-sample periods.
    """
    
    buffer_files = sorted(glob.glob('results/attention_logs/attention_buffer_*.pkl'))
    
    # Filter out already-merged buffers
    non_merged = [f for f in buffer_files if 'merged' not in f]
    
    if len(non_merged) < 2:
        print(f" Found only {len(non_merged)} non-merged buffer(s). Merge skipped.")
        return None
    
    # Get the two most recent non-merged buffers (training and evaluation)
    train_buffer_path = non_merged[-2]
    eval_buffer_path = non_merged[-1]
    
    print("\n" + "="*80)
    print("MERGING ATTENTION BUFFERS: Training + Evaluation")
    print("="*80)
    print(f"Loading training buffer: {train_buffer_path}")
    
    with open(train_buffer_path, 'rb') as f:
        train_buffer = pickle.load(f)
    
    print(f"Loading evaluation buffer: {eval_buffer_path}")
    with open(eval_buffer_path, 'rb') as f:
        eval_buffer = pickle.load(f)
    
    print(f"\n  Training timesteps:   {len(train_buffer['timestamps'])}")
    print(f"  Evaluation timesteps: {len(eval_buffer['timestamps'])}")
    
    # Merge buffers
    merged_buffer = {
        'timestamps': train_buffer['timestamps'] + eval_buffer['timestamps'],
        'dates': train_buffer['dates'] + eval_buffer['dates'],
        'attention_weights': train_buffer['attention_weights'] + eval_buffer['attention_weights'],
        'adjacency_matrices': train_buffer['adjacency_matrices'] + eval_buffer['adjacency_matrices'],
        'portfolio_values': train_buffer['portfolio_values'] + eval_buffer['portfolio_values'],
    }
    
    print(f"  Total timesteps:      {len(merged_buffer['timestamps'])}")
    
    # Save merged buffer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_path = f'results/attention_logs/attention_buffer_merged_{timestamp}.pkl'
    
    with open(merged_path, 'wb') as f:
        pickle.dump(merged_buffer, f)
    
    print(f"\n Merged buffer saved to: {merged_path}")
    print(f"  Date range: {merged_buffer['timestamps'][0]} to {merged_buffer['timestamps'][-1]}")
    print("="*80 + "\n")
    
    return merged_path


if __name__ == "__main__":
    train_and_evaluate()
    # Automatically merge buffers after training completes
    merge_attention_buffers()