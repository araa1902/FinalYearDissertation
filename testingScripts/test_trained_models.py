"""
Test Pre-trained Models: 1/N Baseline vs Vanilla PPO vs GCN vs GAT
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO

# Add workspace root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Project modules
from src.data.downloader import YahooDataDownloader
from src.data.preprocessor import FeatureEngineer
from src.data.graphbuilder import GraphBuilder
from src.env.portfolio_env import StockPortfolioEnv
from src.env.portfolio_env_baseline import StockPortfolioEnvBaseline
from src.utils.seeding import Seed
from src.utils.config_manager import ConfigManager


def evaluate_equal_weight_baseline(test_data, n_assets, initial_amount=100000, transaction_cost_pct=0.001):
    """
    Evaluate simple 1/N equal-weight portfolio strategy.
    
    CRITICAL: Calculates portfolio return correctly using log of weighted prices,
    NOT the weighted sum of log returns (which is mathematically incorrect).
    """
    equal_weight = 1.0 / n_assets
    portfolio_values = [initial_amount]
    daily_returns = [0.0]
    
    unique_dates = sorted(test_data['date'].unique())
    
    # Need prices to calculate correct portfolio return
    # Get a mapping of date -> ticker -> close price
    price_dict = {}
    for date in unique_dates:
        date_data = test_data[test_data['date'] == date]
        prices_by_ticker = dict(zip(date_data['ticker'], date_data['close']))
        if len(prices_by_ticker) == n_assets:
            price_dict[date] = prices_by_ticker
    
    previous_weights = np.array([equal_weight] * n_assets)
    
    for i, date in enumerate(unique_dates):
        if date not in price_dict:
            continue  # Skip if we don't have all stocks
        
        current_prices = price_dict[date]
        sorted_tickers = sorted(current_prices.keys())
        current_price_values = np.array([current_prices[t] for t in sorted_tickers])
        
        # Equal weight allocation
        new_weights = np.array([equal_weight] * n_assets)
        
        # Transaction cost (rebalancing from previous weights to new weights)
        weight_changes = np.abs(new_weights - previous_weights)
        transaction_cost_pct_of_portfolio = np.sum(weight_changes) * transaction_cost_pct
        
        # CORRECT METHOD: Portfolio return = log(weighted price ratio)
        # If we have previous prices, calculate correct log return
        if i > 0:
            prev_date = unique_dates[i - 1]
            if prev_date in price_dict:
                prev_prices = price_dict[prev_date]
                prev_price_values = np.array([prev_prices[t] for t in sorted_tickers])
                
                # Weighted average of prices
                prev_portfolio_price = np.dot(new_weights, prev_price_values)
                curr_portfolio_price = np.dot(new_weights, current_price_values)
                
                # Log return of weighted portfolio
                portfolio_log_return = np.log(curr_portfolio_price / prev_portfolio_price) - transaction_cost_pct_of_portfolio
                
                # Update portfolio value
                new_portfolio_value = portfolio_values[-1] * np.exp(portfolio_log_return)
                
                portfolio_values.append(new_portfolio_value)
                daily_returns.append(portfolio_log_return)
        
        previous_weights = new_weights.copy()
    
    # Calculate metrics
    df_returns = pd.DataFrame(daily_returns[1:], columns=['daily_return'])
    
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    
    n_days = len(portfolio_values) - 1
    years = n_days / 252.0
    annual_return = (portfolio_values[-1] / portfolio_values[0]) ** (1 / years) - 1 if years > 0 else 0
    
    mean_return = df_returns['daily_return'].mean()
    std_return = df_returns['daily_return'].std()
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    
    # Max drawdown
    portfolio_array = np.array(portfolio_values)
    running_max = np.maximum.accumulate(portfolio_array)
    drawdown = (portfolio_array - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    volatility = std_return * np.sqrt(252)
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
    
    print(f"1/N Equal-Weight Baseline evaluation complete ({len(portfolio_values)-1} days)")
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'returns': daily_returns,
        'portfolio_values': portfolio_values,
        'dates': unique_dates
    }


def evaluate_model(model_path, env_class, env_kwargs, test_data, model_name="Model", deterministic=True, is_zip=False):
    """Evaluate a trained model on test data"""
    if is_zip:
        model_file = model_path
    else:
        model_file = os.path.join(model_path, "best_model.zip")
    
    if not os.path.exists(model_file):
        print(f"ERROR: Model file not found: {model_file}")
        return None
    
    print(f"Loading {model_name} from {model_file}...")
    model = PPO.load(model_file)
    
    # For baseline (StockPortfolioEnvBaseline), don't pass graph_dict
    if env_class == StockPortfolioEnvBaseline:
        filtered_kwargs = {k: v for k, v in env_kwargs.items() if k != 'graph_dict'}
        test_env = env_class(df=test_data, **filtered_kwargs)
    else:
        test_env = env_class(df=test_data, **env_kwargs)
    
    print(f"Running evaluation on {len(test_data['date'].unique())} test days...")
    obs, _ = test_env.reset()
    done = False
    step_count = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        step_count += 1
    
    # Calculate metrics
    returns = test_env.portfolio_return_memory
    values = test_env.asset_memory
    
    df_returns = pd.DataFrame(returns, columns=['daily_return'])
    
    total_return = (values[-1] - values[0]) / values[0]
    
    # Annualized return for Calmar Ratio
    n_days = len(returns)
    years = n_days / 252.0
    annual_return = (values[-1] / values[0]) ** (1 / years) - 1 if years > 0 else 0
    
    sharpe = (252**0.5) * df_returns['daily_return'].mean() / df_returns['daily_return'].std()
    
    cumulative_returns = df_returns['daily_return'].cumsum()
    running_max = cumulative_returns.expanding().max()
    drawdown = cumulative_returns - running_max
    max_drawdown = drawdown.min()
    
    volatility = df_returns['daily_return'].std() * np.sqrt(252)
    
    # Calmar Ratio: Annual Return / Absolute Maximum Drawdown
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
    
    print(f" {model_name} evaluation complete ({step_count} steps)")
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'calmar_ratio': calmar,
        'annual_return': annual_return,
        'returns': returns,
        'portfolio_values': values,
        'dates': test_env.date_memory
    }


def test_trained_models():
    """Test pre-trained models"""
    # Model paths - LATEST TRAINED MODELS
    vanilla_path = "models/ppo_baseline_20260409_171727"
    gcn_path = "models/ppo_gcn_20260409_171727"
    gat_path = "models/best_model/best_model.zip"
    
    # Verify all models exist
    print("\n" + "="*80)
    print("VERIFYING MODEL PATHS")
    print("="*80 + "\n")
    
    vanilla_model_file = os.path.join(vanilla_path, "best_model.zip")
    gcn_model_file = os.path.join(gcn_path, "best_model.zip")
    gat_model_file = gat_path
    
    models_to_test = {
        "Vanilla PPO": vanilla_model_file,
        "GCN": gcn_model_file,
        "GAT": gat_model_file
    }
    
    for model_name, path in models_to_test.items():
        if os.path.exists(path):
            print(f"  {model_name}: {path}")
        else:
            print(f"  ERROR: {model_name}: NOT FOUND at {path}")
            return
    
    # Load config
    config = ConfigManager.load_config()
    data_config = config['data']
    preproc_config = config['preprocessing']
    graph_config = config['graph']
    env_config = config['env']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80 + "\n")
    
    # Download and preprocess data
    print("Downloading data...")
    downloader = YahooDataDownloader(
        start_date=data_config['start_date'],
        end_date=data_config['end_date'],
        ticker_list=data_config['ticker_list']
    )
    df_raw = downloader.fetch_data()
    
    print("Preprocessing data...")
    preprocessor = FeatureEngineer(
        use_technical_indicator=preproc_config['use_technical_indicator'],
        tech_indicator_list=preproc_config['tech_indicator_list'],
        normalisation_window=preproc_config['normalisation_window']
    )
    df_processed = preprocessor.preprocess_data(df_raw)
    
    # Build graphs for GAT
    print("Building graphs...")
    builder = GraphBuilder(
        df_processed,
        lookback_window=graph_config['lookback_window'],
        threshold=graph_config['threshold'],
        top_k=graph_config['top_k']
    )
    graphs = builder.build_graphs(sparsity_method=graph_config['sparsity_method'])
    print(f"Built {len(graphs)} correlation graphs")
    
    # Align data
    common_dates = sorted(list(set(df_processed['date'].unique()) & set(graphs.keys())))
    df_processed = df_processed[df_processed['date'].isin(common_dates)].reset_index(drop=True)
    print(f"Aligned data to {len(common_dates)} dates")
    
    # Train/test split
    print("\n=== Train/Test Split ===")
    train_end = pd.to_datetime(data_config['train_end_date'])
    test_start = pd.to_datetime(data_config['test_start_date'])
    
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    
    df_train = df_processed[df_processed['date'] <= train_end].reset_index(drop=True)
    df_test = df_processed[df_processed['date'] >= test_start].reset_index(drop=True)
    
    print(f"Training: {df_train['date'].min()} to {df_train['date'].max()}")
    print(f"Testing: {df_test['date'].min()} to {df_test['date'].max()}")
    print(f"Test samples: {len(df_test.date.unique())} days")
    
    # Prepare environment kwargs
    print("\n" + "="*80)
    print("PREPARING ENVIRONMENTS")
    print("="*80 + "\n")
    
    # Vanilla PPO env (no graphs)
    vanilla_env_kwargs = {
        "stock_dim": len(data_config['ticker_list']),
        "initial_amount": env_config['initial_amount'],
        "transaction_cost_pct": env_config['transaction_cost_pct'],
        "reward_scaling": env_config['reward_scaling'],
        "state_space": len(data_config['ticker_list']),
        "action_space": len(data_config['ticker_list']),
        "tech_indicator_list": preprocessor.tech_indicator_list,
        "turbulence_threshold": env_config['turbulence_threshold'],
    }
    
    # GCN env (static graphs from training)
    returns_matrix_train = df_train.pivot(
        index='date',
        columns='ticker',
        values='log_return'
    )
    training_correlations = returns_matrix_train.corr().values
    training_correlations = np.nan_to_num(training_correlations, nan=0.0)
    
    static_adj_gcn = np.abs(training_correlations)
    static_adj_gcn = (static_adj_gcn >= 0.3).astype(np.float32) * static_adj_gcn
    np.fill_diagonal(static_adj_gcn, 1.0)
    
    test_dates = sorted(df_test['date'].unique())
    static_graph_dict_test = {date: static_adj_gcn for date in test_dates}
    
    gcn_env_kwargs = {
        "stock_dim": len(data_config['ticker_list']),
        "initial_amount": env_config['initial_amount'],
        "transaction_cost_pct": env_config['transaction_cost_pct'],
        "reward_scaling": env_config['reward_scaling'],
        "state_space": len(data_config['ticker_list']),
        "action_space": len(data_config['ticker_list']),
        "tech_indicator_list": preprocessor.tech_indicator_list,
        "turbulence_threshold": env_config['turbulence_threshold'],
        "sector_map": data_config['sector_map'],
        "max_sector_weight": env_config['max_sector_weight'],
        "graph_dict": static_graph_dict_test
    }
    
    # GAT env (dynamic graphs)
    gat_env_kwargs = {
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
    
    # Evaluate models
    print("\n" + "="*80)
    print("EVALUATION PHASE")
    print("="*80 + "\n")
    
    Seed(42).set_all_seeds()
    
    # Evaluate 1/N Equal-Weight Baseline
    print("Evaluating 1/N Equal-Weight Baseline...")
    equal_weight_results = evaluate_equal_weight_baseline(
        test_data=df_test,
        n_assets=len(data_config['ticker_list']),
        initial_amount=env_config['initial_amount'],
        transaction_cost_pct=env_config['transaction_cost_pct']
    )
    
    print("\nEvaluating Vanilla PPO (no graphs)...")
    vanilla_results = evaluate_model(vanilla_path, StockPortfolioEnvBaseline, vanilla_env_kwargs, df_test, "Vanilla PPO", is_zip=False)
    
    print("\nEvaluating GCN (static graphs)...")
    gcn_results = evaluate_model(gcn_path, StockPortfolioEnv, gcn_env_kwargs, df_test, "GCN", is_zip=False)
    
    print("\nEvaluating GAT (dynamic graphs/attention)...")
    gat_results = evaluate_model(gat_path, StockPortfolioEnv, gat_env_kwargs, df_test, "GAT", is_zip=True)
    
    # Comparison results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80 + "\n")
    
    comparison_data = {
        'Metric': ['Total Return', 'Annual Return', 'Sharpe Ratio', 'Calmar Ratio', 'Max Drawdown', 'Volatility'],
        '1/N Baseline': [
            f"{equal_weight_results['total_return']:.6f}",
            f"{equal_weight_results['annual_return']:.6f}",
            f"{equal_weight_results['sharpe_ratio']:.4f}",
            f"{equal_weight_results['calmar_ratio']:.4f}",
            f"{equal_weight_results['max_drawdown']:.6f}",
            f"{equal_weight_results['volatility']:.6f}"
        ],
        'Vanilla PPO': [
            f"{vanilla_results['total_return']:.6f}",
            f"{vanilla_results['annual_return']:.6f}",
            f"{vanilla_results['sharpe_ratio']:.4f}",
            f"{vanilla_results['calmar_ratio']:.4f}",
            f"{vanilla_results['max_drawdown']:.6f}",
            f"{vanilla_results['volatility']:.6f}"
        ],
        'GCN (Static)': [
            f"{gcn_results['total_return']:.6f}",
            f"{gcn_results['annual_return']:.6f}",
            f"{gcn_results['sharpe_ratio']:.4f}",
            f"{gcn_results['calmar_ratio']:.4f}",
            f"{gcn_results['max_drawdown']:.6f}",
            f"{gcn_results['volatility']:.6f}"
        ],
        'GAT (Dynamic)': [
            f"{gat_results['total_return']:.6f}",
            f"{gat_results['annual_return']:.6f}",
            f"{gat_results['sharpe_ratio']:.4f}",
            f"{gat_results['calmar_ratio']:.4f}",
            f"{gat_results['max_drawdown']:.6f}",
            f"{gat_results['volatility']:.6f}"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Save comparison summary
    comparison_csv = f"results/trained_models_comparison_{timestamp}.csv"
    os.makedirs(os.path.dirname(comparison_csv), exist_ok=True)
    df_comparison.to_csv(comparison_csv, index=False)
    print(f"\nComparison summary saved to {comparison_csv}")
    
    # Generate visualization
    print("\nGenerating visualizations...")
    h1_plot = generate_h1_evidence_figure(equal_weight_results, vanilla_results, gcn_results, gat_results, timestamp)
    
    # Generate regime analysis (H2 Evidence)
    print("\nGenerating regime-specific analysis for H2 verification...")
    regime_metrics, degradation_gaps, h2_verdict = generate_regime_analysis_figure(
        equal_weight_results, vanilla_results, gcn_results, gat_results, timestamp
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")
    
    return df_comparison, equal_weight_results, vanilla_results, gcn_results, gat_results


def generate_h1_evidence_figure(equal_weight_results, vanilla_results, gcn_results, gat_results, timestamp):
    """
    Generate Figure 6.1: H1 Evidence - Impact of Dynamic Topology
    Equity curve with annotated drawdown troughs and key inflection points
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Colour scheme
    baseline_color = '#1f77b4'  # Blue
    vanilla_color = '#D62828'   # Red
    gcn_color = '#F77F00'       # Orange
    gat_color = '#06A77D'       # Green
    
    # Plot portfolio values (scaled to percentage for better visualization)
    baseline_pct = [(v / equal_weight_results['portfolio_values'][0] - 1) * 100 
                    for v in equal_weight_results['portfolio_values']]
    vanilla_pct = [(v / vanilla_results['portfolio_values'][0] - 1) * 100 
                   for v in vanilla_results['portfolio_values']]
    gcn_pct = [(v / gcn_results['portfolio_values'][0] - 1) * 100 
               for v in gcn_results['portfolio_values']]
    gat_pct = [(v / gat_results['portfolio_values'][0] - 1) * 100 
               for v in gat_results['portfolio_values']]
    
    trading_days = range(len(baseline_pct))
    
    ax.plot(trading_days, baseline_pct, linewidth=3, label='1/N Baseline', 
            color=baseline_color, alpha=0.8, linestyle='--')
    ax.plot(trading_days, vanilla_pct, linewidth=2.5, label='Vanilla PPO', 
            color=vanilla_color, alpha=0.75)
    ax.plot(trading_days, gcn_pct, linewidth=2.5, label='GCN (Static)', 
            color=gcn_color, alpha=0.75)
    ax.plot(trading_days, gat_pct, linewidth=2.5, label='GAT (Dynamic)', 
            color=gat_color, alpha=0.85)
    
    # Find and annotate drawdown troughs
    def find_drawdown_troughs(portfolio_values):
        """Find the two largest drawdown troughs"""
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - running_max) / running_max
        
        # Find local minima (troughs)
        troughs = []
        for i in range(1, len(drawdown) - 1):
            if drawdown[i] < drawdown[i-1] and drawdown[i] < drawdown[i+1]:
                troughs.append((i, drawdown[i]))
        
        # Get two deepest troughs
        troughs.sort(key=lambda x: x[1])
        return troughs[:2] if len(troughs) >= 2 else troughs
    
    troughs = find_drawdown_troughs(gat_results['portfolio_values'])
    
    # ===== POINTER 1: Trough 1 (2022 Rate Shock) =====
    # Find the deepest drawdown index (approximately Sept 2022)
    vanilla_drawdown = np.array(vanilla_pct)
    gat_drawdown = np.array(gat_pct)
    
    # Find minimum around Sept 2022 (roughly 60% into the test period)
    test_length = len(vanilla_pct)
    approximate_crisis_start = int(0.35 * test_length)
    approximate_crisis_end = int(0.65 * test_length)
    crisis_period = vanilla_drawdown[approximate_crisis_start:approximate_crisis_end]
    crisis_min_idx = approximate_crisis_start + np.argmin(crisis_period)
    
    # Draw shaded region for rate shock
    ax.axvspan(crisis_min_idx - 30, crisis_min_idx + 30, alpha=0.15, color='red', 
               label='2022 Rate Shock Period')
    
    # ===== POINTER 2: Find actual GAT vs GCN Divergence Point =====
    # Place divergence point at the end of the rate shock period (Nov 2022)
    divergence_idx = crisis_min_idx + 30
    
    ax.axvline(x=divergence_idx, color='purple', linestyle='--', linewidth=3, 
               alpha=0.8, label='Recovery Inflection Point')
    
    # ===== POINTER 3: Terminal Delta (Dec 2024 - End of period) =====
    final_idx = len(gat_pct) - 1
    
    # ===== IMPROVED DATE AXIS =====
    # Create more dates for clarity - show every ~3 months
    num_ticks = 16  # More ticks for clarity
    date_indices = np.linspace(0, len(gat_results['dates']) - 1, num_ticks, dtype=int)
    date_labels = [pd.to_datetime(gat_results['dates'][i]).strftime('%b %Y') for i in date_indices]
    
    ax.set_xticks(date_indices)
    ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=13, fontweight='bold')
    
    # Add minor ticks for better readability
    ax.set_xticks(np.linspace(0, len(gat_pct) - 1, 64), minor=True)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':')
    ax.grid(True, which='major', alpha=0.3, linestyle='--')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
    ax.set_xlabel('Date', fontsize=18, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=18, fontweight='bold')
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=14, loc='upper left', framealpha=0.97, edgecolor='black', fancybox=True)
    ax.set_facecolor('#fafafa')
    
    # Improve layout
    plt.tight_layout()
    h1_plot_path = f'results/h1_evidence_equity_curve_{timestamp}.png'
    os.makedirs(os.path.dirname(h1_plot_path), exist_ok=True)
    plt.savefig(h1_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"H1 Evidence figure saved to: {h1_plot_path}")
    return h1_plot_path


def calculate_regime_metrics(portfolio_values, daily_returns, dates, regime_start_date, regime_end_date, regime_name):
    """
    Calculate performance metrics for a specific regime (time period).
    
    CRITICAL: Resets portfolio value to starting amount on first day of regime
    to calculate drawdowns specific to that period.
    
    Args:
        portfolio_values: Full list of portfolio values across entire test period
        daily_returns: Full list of daily returns across entire test period
        dates: Full list of dates corresponding to portfolio values/returns
        regime_start_date: Start date for regime (pandas Timestamp or datetime)
        regime_end_date: End date for regime (pandas Timestamp or datetime)
        regime_name: String name of regime (e.g., 'Bull Market')
    
    Returns:
        dict with keys: 'sharpe_ratio', 'max_drawdown', 'annual_return', 'calmar_ratio'
    """
    dates_pd = pd.to_datetime(dates)
    
    # Filter to regime dates
    mask = (dates_pd >= regime_start_date) & (dates_pd <= regime_end_date)
    regime_indices = np.where(mask)[0]
    
    if len(regime_indices) == 0:
        print(f"WARNING: No data found for {regime_name} ({regime_start_date} to {regime_end_date})")
        return {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'annual_return': 0.0,
            'calmar_ratio': 0.0,
            'num_days': 0
        }
    
    # Extract regime-specific values and returns
    regime_start_idx = regime_indices[0]
    regime_end_idx = regime_indices[-1]
    
    # Get portfolio values for this regime
    regime_portfolio_values = np.array(portfolio_values[regime_start_idx:regime_end_idx + 1])
    
    # CRITICAL: Normalize to starting value of regime (reset to 100 on first day)
    regime_start_value = regime_portfolio_values[0]
    normalized_values = regime_portfolio_values / regime_start_value
    
    # Get daily returns for this regime (skip first day since we have no previous return)
    if regime_start_idx == 0:
        regime_returns = np.array(daily_returns[1:regime_end_idx + 1])
    else:
        regime_returns = np.array(daily_returns[regime_start_idx + 1:regime_end_idx + 1])
    
    # Calculate metrics
    num_days = len(regime_portfolio_values) - 1
    years = num_days / 252.0
    
    # Annual return based on normalized values
    annual_return = (normalized_values[-1] / normalized_values[0]) ** (1 / years) - 1 if years > 0 else 0
    
    # Sharpe Ratio
    if len(regime_returns) > 0:
        mean_return = np.mean(regime_returns)
        std_return = np.std(regime_returns)
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
    else:
        sharpe = 0.0
    
    # Max Drawdown (calculated on normalized values for regime-specific drawdown)
    running_max = np.maximum.accumulate(normalized_values)
    drawdowns = (normalized_values - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    # Calmar Ratio: Annual Return / |Max Drawdown|
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
    
    return {
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'annual_return': annual_return,
        'calmar_ratio': calmar,
        'num_days': num_days
    }


def calculate_degradation_gap(bull_sharpe, bear_sharpe):
    """
    Calculate Sharpe degradation from Bull to Bear regime.
    Positive value = degradation (drop in Sharpe).
    """
    return bull_sharpe - bear_sharpe


def generate_regime_analysis_figure(equal_weight_results, vanilla_results, gcn_results, gat_results, timestamp):
    """
    Generate Figure 6.2: H2 Evidence - Regime-Specific Performance Analysis
    
    This function implements the exact 4-step process:
    Step 1: Extract daily equity curves
    Step 2: Apply regime time-masks (Bull, Bear, High-Vol)
    Step 3: Recalculate localized metrics for each regime
    Step 4: Calculate degradation gaps to verify H2
    
    Produces:
    - Grouped bar chart (Sharpe Ratio by regime and model)
    - Table 6.2 (Max Drawdown metrics)
    - Degradation gap analysis (H2 verification)
    """
    
    # ===== STEP 1: Extract Daily Equity Curves =====
    print("\n" + "="*80)
    print("STEP 1: EXTRACTING DAILY EQUITY CURVES")
    print("="*80)
    
    models = {
        '1/N Baseline': equal_weight_results,
        'Vanilla PPO': vanilla_results,
        'GCN (Static)': gcn_results,
        'GAT (Dynamic)': gat_results
    }
    
    # ===== STEP 2: Define Regime Time-Masks =====
    print("\n" + "="*80)
    print("STEP 2: APPLYING REGIME TIME-MASKS")
    print("="*80)
    
    # Convert dates to pandas format for comparison
    sample_dates = pd.to_datetime(gat_results['dates'])
    min_date = sample_dates.min()
    max_date = sample_dates.max()
    
    print(f"\nFull test period: {min_date.date()} to {max_date.date()}")
    
    # Define regimes based on the provided criteria
    regimes = {
        'Bull Market': {
            'periods': [
                (pd.Timestamp('2021-01-01'), pd.Timestamp('2021-12-31')),
                (pd.Timestamp('2023-06-01'), pd.Timestamp('2023-12-31')),  # post-SVB recovery
                (pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31'))
            ],
            'description': '2021, Jun-Dec 2023 & 2024 (Bull years)'
        },
        'Bear Market': {
            'periods': [(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'))],
            'description': '2022 (Rate shock & downturn)'
        },
        'High-Volatility Crisis': {
            'periods': [(pd.Timestamp('2023-03-01'), pd.Timestamp('2023-05-31'))],
            'description': 'Mar-May 2023 (SVB Crisis)'
        }
    }
    
    print("\nDefined Regimes:")
    for regime_name, regime_info in regimes.items():
        for start, end in regime_info['periods']:
            print(f"  {regime_name}: {start.date()} to {end.date()} ({regime_info['description']})")
    
    # ===== STEP 3: Recalculate Localized Metrics =====
    print("\n" + "="*80)
    print("STEP 3: RECALCULATING LOCALIZED METRICS FOR EACH REGIME")
    print("="*80)
    
    regime_metrics = {}
    
    for model_name, model_results in models.items():
        print(f"\n{model_name}:")
        regime_metrics[model_name] = {}
        
        for regime_name, regime_info in regimes.items():
            # Combine all periods for this regime
            all_returns = []
            all_portfolio_values = []
            all_dates = []
            
            for period_start, period_end in regime_info['periods']:
                metrics = calculate_regime_metrics(
                    model_results['portfolio_values'],
                    model_results['returns'],
                    model_results['dates'],
                    period_start,
                    period_end,
                    f"{model_name} - {regime_name}"
                )
                
                # Accumulate metrics
                if metrics['num_days'] > 0:
                    all_returns.append(metrics)
            
            # Average metrics across all periods in this regime
            if all_returns:
                avg_sharpe = np.mean([m['sharpe_ratio'] for m in all_returns])
                avg_mdd = np.min([m['max_drawdown'] for m in all_returns])  # Worst case
                avg_annual_return = np.mean([m['annual_return'] for m in all_returns])
                avg_calmar = np.mean([m['calmar_ratio'] for m in all_returns if m['calmar_ratio'] != np.inf])
            else:
                avg_sharpe = 0.0
                avg_mdd = 0.0
                avg_annual_return = 0.0
                avg_calmar = 0.0
            
            regime_metrics[model_name][regime_name] = {
                'sharpe_ratio': avg_sharpe,
                'max_drawdown': avg_mdd,
                'annual_return': avg_annual_return,
                'calmar_ratio': avg_calmar
            }
            
            print(f"  {regime_name:20s} | Sharpe: {avg_sharpe:7.4f} | MDD: {avg_mdd:7.4f} | Annual Ret: {avg_annual_return:7.4f}")
    
    # ===== STEP 4: Calculate Degradation Gaps (H2 Verification) =====
    print("\n" + "="*80)
    print("STEP 4: CALCULATING DEGRADATION GAPS (H2 VERIFICATION)")
    print("="*80)
    
    print("\nSharpe Ratio Degradation Analysis (Bull to Bear):")
    print("-" * 70)
    
    degradation_gaps = {}
    for model_name in models.keys():
        bull_sharpe = regime_metrics[model_name]['Bull Market']['sharpe_ratio']
        bear_sharpe = regime_metrics[model_name]['Bear Market']['sharpe_ratio']
        degradation = calculate_degradation_gap(bull_sharpe, bear_sharpe)
        
        degradation_gaps[model_name] = degradation
        
        print(f"{model_name:20s} | Bull: {bull_sharpe:7.4f} | Bear: {bear_sharpe:7.4f} | Degradation: {degradation:7.4f}")
    
    # Verify H2: GAT degradation < Vanilla PPO degradation
    gat_degradation = degradation_gaps['GAT (Dynamic)']
    vanilla_degradation = degradation_gaps['Vanilla PPO']
    gcn_degradation = degradation_gaps['GCN (Static)']
    baseline_degradation = degradation_gaps['1/N Baseline']
    
    print("\n" + "="*80)
    print("H2 HYPOTHESIS VERIFICATION")
    print("="*80)
    print(f"\nH2: 'Dynamic topology (GAT) limits capital degradation better than static approaches'")
    print(f"\n1/N Baseline Degradation:  {baseline_degradation:.4f}")
    print(f"Vanilla PPO Degradation:   {vanilla_degradation:.4f}")
    print(f"GCN Degradation:           {gcn_degradation:.4f}")
    print(f"GAT Degradation:           {gat_degradation:.4f}")
    
    # Check if GAT is best among learning approaches (vs Vanilla and GCN)
    gat_beats_vanilla = gat_degradation < vanilla_degradation
    gat_beats_gcn = gat_degradation < gcn_degradation
    gat_vs_vanilla_gap = vanilla_degradation - gat_degradation
    gat_vs_gcn_gap = gcn_degradation - gat_degradation
    
    print(f"\nGAT vs Vanilla PPO:  {gat_vs_vanilla_gap:+.4f} Sharpe points (Better if gat_beats_vanilla else Worse)")
    print(f"GAT vs GCN:          {gat_vs_gcn_gap:+.4f} Sharpe points (Better if gat_beats_gcn else Worse)")
    
    # H2 is verified if GAT shows smallest degradation among learned models
    if gat_beats_vanilla and gat_beats_gcn:
        h2_verdict = "VERIFIED"
        print(f"\nH2 VERIFIED: GAT shows SMALLEST degradation gap among learned models")
        if gat_vs_vanilla_gap > 0:
            print(f"  - Dynamic topology limits degradation by {gat_vs_vanilla_gap:.4f} Sharpe points vs Vanilla")
        if gat_vs_gcn_gap > 0:
            print(f"  - Dynamic topology improves on static graphs by {gat_vs_gcn_gap:.4f} Sharpe points")
    elif gat_beats_vanilla:
        h2_verdict = "PARTIALLY VERIFIED"
        print(f"\nH2 PARTIALLY VERIFIED: GAT beats Vanilla PPO but not GCN")
        print(f"  - GAT advantage over Vanilla: {gat_vs_vanilla_gap:.4f} Sharpe points")
        print(f"  - But GCN slightly outperforms GAT: {abs(gat_vs_gcn_gap):.4f} Sharpe points")
    else:
        h2_verdict = "NOT VERIFIED"
        print(f"\nH2 NOT VERIFIED: GAT does not show smallest degradation gap")
        print(f"  - Vanilla advantage over GAT: {abs(gat_vs_vanilla_gap):.4f} Sharpe points")
        print(f"  - GCN advantage over GAT:     {abs(gat_vs_gcn_gap):.4f} Sharpe points")
    
    # ===== CREATE VISUALIZATIONS =====
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Colour scheme
    colors = {
        '1/N Baseline': '#1f77b4',   # Blue
        'Vanilla PPO': '#D62828',    # Red
        'GCN (Static)': '#F77F00',   # Orange
        'GAT (Dynamic)': '#06A77D'   # Green
    }
    
    # ===== FIGURE 6.2: PRIMARY H2 VISUALIZATION - GROUPED BAR CHART =====
    # This is the main H2 evidence figure showing Sharpe Ratio across regimes
    fig_6_2, ax_main = plt.subplots(figsize=(16, 9))
    
    regime_names = list(regimes.keys())
    model_names_list = list(models.keys())
    x = np.arange(len(regime_names))
    width = 0.2
    
    # Plot bars for each model
    for i, model_name in enumerate(model_names_list):
        sharpe_values = [regime_metrics[model_name][regime]['sharpe_ratio'] for regime in regime_names]
        ax_main.bar(x + i*width - 1.5*width, sharpe_values, width, 
                   label=model_name, color=colors[model_name], alpha=0.85, 
                   edgecolor='black', linewidth=2)
    
    # Formatting
    ax_main.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax_main.set_xlabel('Market Regime', fontsize=15, fontweight='bold')
    ax_main.set_ylabel('Sharpe Ratio', fontsize=15, fontweight='bold')
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(regime_names, fontsize=13, fontweight='bold')
    ax_main.legend(fontsize=12, loc='lower right', framealpha=0.97, edgecolor='black', 
                  fancybox=True, shadow=True)
    ax_main.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax_main.set_facecolor('#fafafa')
    
    # Add value labels on each bar. Nudge small negative Bear-regime labels below the bars and
    # colour them to match the bar so they remain legible near the zero line.
    for i, model_name in enumerate(model_names_list):
        sharpe_values = [regime_metrics[model_name][regime]['sharpe_ratio'] for regime in regime_names]
        for j, v in enumerate(sharpe_values):
            if v != 0:
                regime = regime_names[j]
                # Defaults
                if v > 0:
                    y_offset = 0.08
                    va = 'bottom'
                    txt_color = 'black'
                else:
                    y_offset = -0.12
                    va = 'top'
                    txt_color = 'black'

                # If this is the Bear Market and the magnitude is small (labels sit near 0),
                # nudge them slightly below the bar and colour the label to match the bar.
                if regime == 'Bear Market' and abs(v) < 0.15:
                    y_offset = -0.06
                    va = 'top'
                    txt_color = colors.get(model_name, 'black')

                ax_main.text(j + i*width - 1.5*width, v + y_offset, f'{v:.3f}',
                           ha='center', va=va, color=txt_color,
                           fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig_6_2_path = f'results/figure_6_2_h2_evidence_regime_sharpe_{timestamp}.png'
    os.makedirs(os.path.dirname(fig_6_2_path), exist_ok=True)
    plt.savefig(fig_6_2_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nFigure 6.2 (PRIMARY H2 VISUALIZATION) saved to: {fig_6_2_path}")
    
    # ===== TABLE 6.2: Max Drawdown Comparison (standalone visualization) =====
    # Create a standalone figure for Table 6.2 data
    fig_table_6_2, ax_table = plt.subplots(figsize=(14, 7))
    
    x_mdd = np.arange(len(regime_names))
    
    for i, model_name in enumerate(model_names_list):
        mdd_values = [regime_metrics[model_name][regime]['max_drawdown'] for regime in regime_names]
        ax_table.bar(x_mdd + i*width - 1.5*width, mdd_values, width,
                    label=model_name, color=colors[model_name], alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax_table.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax_table.set_xlabel('Market Regime', fontsize=14, fontweight='bold')
    ax_table.set_ylabel('Maximum Drawdown', fontsize=14, fontweight='bold')
    ax_table.set_xticks(x_mdd)
    ax_table.set_xticklabels(regime_names, fontsize=12)
    ax_table.legend(fontsize=11, loc='lower right', framealpha=0.95, edgecolor='black')
    ax_table.grid(True, alpha=0.3, axis='y')
    ax_table.set_facecolor('#fafafa')
    
    plt.tight_layout()
    fig_table_6_2_path = f'results/table_6_2_max_drawdown_figure_{timestamp}.png'
    plt.savefig(fig_table_6_2_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Table 6.2 (Max Drawdown) visualization saved to: {fig_table_6_2_path}")
    
    # ===== Generate Table 6.2: Max Drawdown Summary Table =====
    print("\n" + "="*80)
    print("TABLE 6.2: MAXIMUM DRAWDOWN BY REGIME")
    print("="*80)
    
    table_6_2_data = []
    for model_name in models.keys():
        row = {'Model': model_name}
        for regime in regime_names:
            mdd = regime_metrics[model_name][regime]['max_drawdown']
            row[regime] = f"{mdd:.6f}"
        table_6_2_data.append(row)
    
    df_table_6_2 = pd.DataFrame(table_6_2_data)
    print("\n" + df_table_6_2.to_string(index=False))
    
    # Save Table 6.2
    table_6_2_path = f'results/table_6_2_max_drawdown_{timestamp}.csv'
    df_table_6_2.to_csv(table_6_2_path, index=False)
    print(f"\nTable 6.2 saved to: {table_6_2_path}")
    
    # ===== Generate Summary Report =====
    print("\n" + "="*80)
    print("REGIME ANALYSIS SUMMARY REPORT")
    print("="*80)
    
    summary_data = []
    for model_name in models.keys():
        for regime in regime_names:
            summary_data.append({
                'Model': model_name,
                'Regime': regime,
                'Sharpe Ratio': f"{regime_metrics[model_name][regime]['sharpe_ratio']:.6f}",
                'Max Drawdown': f"{regime_metrics[model_name][regime]['max_drawdown']:.6f}",
                'Annual Return': f"{regime_metrics[model_name][regime]['annual_return']:.6f}",
                'Calmar Ratio': f"{regime_metrics[model_name][regime]['calmar_ratio']:.6f}"
            })
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = f'results/regime_analysis_summary_{timestamp}.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"\nFull summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("REGIME ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    return regime_metrics, degradation_gaps, h2_verdict


def generate_comparison_plot(equal_weight_results, vanilla_results, gcn_results, gat_results, timestamp):
    """Create comparison visualizations for four models (1/N Baseline + 3 RL agents)"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Four-Way Comparison: 1/N Baseline vs Vanilla PPO vs GCN vs GAT Portfolio Performance', 
                 fontsize=16, fontweight='bold')
    
    # Colour scheme for four models
    baseline_color = '#1f77b4'  # Blue
    vanilla_color = '#D62828'   # Red
    gcn_color = '#F77F00'       # Orange
    gat_color = '#06A77D'       # Green
    
    # 1. Portfolio value comparison (ALL FOUR MODELS)
    axes[0, 0].plot(equal_weight_results['portfolio_values'], linewidth=2.5, 
                    label='1/N Baseline', color=baseline_color, alpha=0.8, linestyle='--')
    axes[0, 0].plot(vanilla_results['portfolio_values'], linewidth=2, 
                    label='Vanilla PPO (No Graphs)', color=vanilla_color, alpha=0.8)
    axes[0, 0].plot(gcn_results['portfolio_values'], linewidth=2, 
                    label='GCN (Static Graph)', color=gcn_color, alpha=0.8)
    axes[0, 0].plot(gat_results['portfolio_values'], linewidth=2, 
                    label='GAT (Dynamic Graph)', color=gat_color, alpha=0.8)
    axes[0, 0].set_xlabel('Trading Day', fontsize=11)
    axes[0, 0].set_ylabel('Portfolio Value ($)', fontsize=11)
    axes[0, 0].set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cumulative returns comparison
    baseline_cumret = pd.Series(equal_weight_results['returns']).cumsum()
    vanilla_cumret = pd.Series(vanilla_results['returns']).cumsum()
    gcn_cumret = pd.Series(gcn_results['returns']).cumsum()
    gat_cumret = pd.Series(gat_results['returns']).cumsum()
    
    axes[0, 1].plot(baseline_cumret, linewidth=2.5, 
                    label='1/N Baseline', color=baseline_color, alpha=0.8, linestyle='--')
    axes[0, 1].plot(vanilla_cumret, linewidth=2, 
                    label='Vanilla PPO (No Graphs)', color=vanilla_color, alpha=0.8)
    axes[0, 1].plot(gcn_cumret, linewidth=2, 
                    label='GCN (Static Graph)', color=gcn_color, alpha=0.8)
    axes[0, 1].plot(gat_cumret, linewidth=2, 
                    label='GAT (Dynamic Graph)', color=gat_color, alpha=0.8)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Trading Day', fontsize=11)
    axes[0, 1].set_ylabel('Cumulative Return', fontsize=11)
    axes[0, 1].set_title('Cumulative Returns Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Daily returns distribution
    axes[1, 0].hist(equal_weight_results['returns'][1:], bins=50, alpha=0.4, 
                    label='1/N Baseline', color=baseline_color, edgecolor='black')
    axes[1, 0].hist(vanilla_results['returns'], bins=50, alpha=0.4, 
                    label='Vanilla PPO', color=vanilla_color, edgecolor='black')
    axes[1, 0].hist(gcn_results['returns'], bins=50, alpha=0.4, 
                    label='GCN', color=gcn_color, edgecolor='black')
    axes[1, 0].hist(gat_results['returns'], bins=50, alpha=0.4, 
                    label='GAT', color=gat_color, edgecolor='black')
    axes[1, 0].set_xlabel('Daily Return', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Metrics comparison (bar chart) - Sharpe Ratio & Calmar Ratio
    metrics = ['Sharpe Ratio', 'Calmar Ratio']
    baseline_values = [
        equal_weight_results['sharpe_ratio'],
        equal_weight_results['calmar_ratio']
    ]
    vanilla_values = [
        vanilla_results['sharpe_ratio'],
        vanilla_results['calmar_ratio']
    ]
    gcn_values = [
        gcn_results['sharpe_ratio'],
        gcn_results['calmar_ratio']
    ]
    gat_values = [
        gat_results['sharpe_ratio'],
        gat_results['calmar_ratio']
    ]
    
    x = np.arange(len(metrics))
    width = 0.2
    
    axes[1, 1].bar(x - 1.5*width, baseline_values, width, label='1/N Baseline', 
                   color=baseline_color, alpha=0.8, edgecolor='black')
    axes[1, 1].bar(x - 0.5*width, vanilla_values, width, label='Vanilla PPO', 
                   color=vanilla_color, alpha=0.8, edgecolor='black')
    axes[1, 1].bar(x + 0.5*width, gcn_values, width, label='GCN', 
                   color=gcn_color, alpha=0.8, edgecolor='black')
    axes[1, 1].bar(x + 1.5*width, gat_values, width, label='GAT', 
                   color=gat_color, alpha=0.8, edgecolor='black')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_ylabel('Value', fontsize=11)
    axes[1, 1].set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics, fontsize=10)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = f'results/trained_models_comparison_{timestamp}.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" Comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    test_trained_models()
