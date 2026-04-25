"""
Exact Edge Ablation
This script demonstrates how to use the ExactEdgeAblationExplainer to explain
the latent embeddings of a trained GAT model for portfolio optimisation.
"""

import torch
import pandas as pd
from pathlib import Path

# Import your modules
from src.env.portfolio_env import StockPortfolioEnv
from src.explainability.posthoc.exact_edge_ablation_explainer import DenseGNNExplainer
from src.data.downloader import YahooDataDownloader
from src.data.preprocessor import FeatureEngineer
from src.data.graphbuilder import GraphBuilder
from stable_baselines3 import PPO
from src.utils.config_manager import load_config


def prepare_data_for_explanation(env, model, target_date=None):
    """
    Prepare data for explanation by stepping through the environment.
    """
    obs, info = env.reset()
    done = False
    
    print(f" Extracting initial state from market environment...")
    step_count = 0
    
    # If target_date specified, fast-forward to that date
    if target_date:
        while not done:
            current_date = getattr(env.unwrapped, 'current_date_str', None) or env.unwrapped.unique_dates[env.unwrapped.day]
            if str(current_date)[:10] == target_date:
                print(f"  Target Critical State Reached: {target_date}")
                break
                
            # Step the environment
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
    else:
        # Use initial state (no fast-forward)
        print(f"  Using initial test state")
    
    if isinstance(obs, dict):
        x_np = obs['features']
        adj_np = obs['adjacency']
    else:
        stock_dim = env.unwrapped.stock_dim
        adj_np = obs[:stock_dim, :]  
        x_np = obs[stock_dim:, :]     
    
    # Convert to tensors with batch dimension
    x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # (1, F, N)
    adj_tensor = torch.tensor(adj_np, dtype=torch.float32).unsqueeze(0)  # (1, N, N)
    
    # Transpose to (1, N, F) format for GAT
    x_tensor = x_tensor.transpose(1, 2)
    
    return x_tensor, adj_tensor, obs


def run_explanation_pipeline(model_path=None,
                            target_node=0,
                            target_date=None,
                            save_plots=True,
                            output_dir='results/explainability'):
    """
    Complete explanation pipeline for portfolio optimisation model.
    """
    
    # ========== Step 1: Load Configuration ==========
    print("\n" + "="*80)
    print("STEP 1: Loading Configuration")
    print("="*80)
    config = load_config()
    tickers_list = config['data']['ticker_list']
    num_tickers = len(tickers_list)
    
    print(f" Loaded config for {num_tickers} tickers")
    print(f"  Tickers: {', '.join(tickers_list[:5])}..." if len(tickers_list) > 5 else f"  Tickers: {', '.join(tickers_list)}")
    
    # ========== Step 2: Load Model and Environment ==========
    print("\n" + "="*80)
    print("STEP 2: Loading Trained Model")
    print("="*80)
    
    # Use provided model path or default to most recent GAT model
    if model_path is None:
        model_path = 'models/ppo_gat_20260222_113032.zip'
    
    model = PPO.load(model_path)
    print(f" Loaded PPO model from {model_path}")
    
    # ========== Step 3: Load and Preprocess Data ==========
    print("\n" + "="*80)
    print("STEP 3: Loading and Preprocessing Data")
    print("="*80)
    
    data_config = config['data']
    preproc_config = config['preprocessing']
    env_config = config['env']
    graph_config = config['graph']
    
    # Download data
    downloader = YahooDataDownloader(
        start_date=data_config['start_date'], 
        end_date=data_config['end_date'], 
        ticker_list=data_config['ticker_list']
    )
    df = downloader.fetch_data()
    print(f" Downloaded data: {len(df)} records")
    
    # Preprocess features
    preprocessor = FeatureEngineer(
        use_technical_indicator=preproc_config['use_technical_indicator'], 
        tech_indicator_list=preproc_config['tech_indicator_list'],
        normalisation_window=preproc_config['normalisation_window']
    )
    df_processed = preprocessor.preprocess_data(df)
    print(f" Added technical indicators")
    
    # Build graphs
    builder = GraphBuilder(
        df_processed,
        lookback_window=graph_config['lookback_window'],
        threshold=graph_config['threshold'],
        top_k=graph_config['top_k']
    )
    graphs = builder.build_graphs(sparsity_method=graph_config['sparsity_method'])
    print(f" Built {len(graphs)} daily correlation graphs")
    
    # Align data with graph dates
    common_dates = sorted(list(set(df_processed['date'].unique()) & set(graphs.keys())))
    df_processed = df_processed[df_processed['date'].isin(common_dates)].reset_index(drop=True)
    print(f" Aligned data: {len(common_dates)} dates with both features and graphs")
    
    # Use test data for explanation
    test_start = pd.to_datetime(config['data']['test_start_date'])
    df_test = df_processed[df_processed['date'] >= test_start].reset_index(drop=True)
    print(f" Using test period: {df_test['date'].min()} to {df_test['date'].max()}")
    
    # ========== Step 4: Initialise Environment ==========
    print("\n" + "="*80)
    print("STEP 4: Initialising Portfolio Environment")
    print("="*80)
    
    env_kwargs = {
        "stock_dim": len(data_config['ticker_list']),
        "initial_amount": env_config['initial_amount'], 
        "transaction_cost_pct": env_config['transaction_cost_pct'], 
        "reward_scaling": env_config['reward_scaling'],
        "state_space": len(data_config['ticker_list']), 
        "action_space": len(data_config['ticker_list']),
        "tech_indicator_list": preproc_config['tech_indicator_list'],
        "turbulence_threshold": env_config['turbulence_threshold'],
        "lookback": graph_config['lookback_window'],
        "sector_map": data_config['sector_map'],
        "max_sector_weight": env_config['max_sector_weight'],
        "graph_dict": graphs 
    }
    
    env = StockPortfolioEnv(df=df_test, **env_kwargs)
    print(f" Initialised portfolio environment in test mode")
    
    x, adj, obs = prepare_data_for_explanation(env, model, target_date=target_date)
    print(f" Extracted data shapes:")
    print(f"  - Node features (x): {x.shape}")
    print(f"  - Adjacency matrix (adj): {adj.shape}")
    
    # ========== Step 5: Initialise Explainer ==========
    print("\n" + "="*80)
    print("STEP 5: Initialising Exact Ablation Explainer")
    print("="*80)
    
    # Extract the FULL feature extractor to preserve the Sector Bias logic
    feature_extractor = model.policy.features_extractor
    
    explainer = DenseGNNExplainer(feature_extractor, config=config, device='cpu')
    print(f"Initialised DenseGNNExplainer (Ablation Mode)")
    
    # ========== Step 6: Run Explanation ==========
    print("\n" + "="*80)
    print("STEP 6: Running Explanation")
    print("="*80)
    
    target_ticker = tickers_list[target_node]
    print(f"Target Node: {target_node} ({target_ticker})")
    
    final_mask, important_edges, explanation_dict = explainer.explain(
        x, adj, 
        target_node_idx=target_node,
        tickers_list=tickers_list
    )
    
    # ========== Step 7: Analyse Results ==========
    print("\n" + "="*80)
    print("STEP 7: Analysis Results")
    print("="*80)
    
    sparsity = explanation_dict['sparsity']
    subgraph_drop = explanation_dict['subgraph_fidelity_drop']
    single_edge_drop = explanation_dict['max_single_edge_fidelity_drop']
    num_important_edges = len(important_edges)
    
    print(f"\n Explanation Quality Metrics:")
    print(f"  - Sparsity: {sparsity:.4f} (Removed {sparsity*100:.1f}% of edges)")
    print(f"  - Subgraph Fidelity Drop: {subgraph_drop:.6f} (Impact of removing ALL unimportant edges)")
    print(f"  - Max Single-Edge Drop: {single_edge_drop:.6f} (Impact of removing the single strongest driver)")
    print(f"  - Important Edges: {num_important_edges}")
    
    print(f"\n Top Causal Edges:")
    for i, edge in enumerate(important_edges[:10], 1):
        print(f"  {i}. {edge['source_ticker']} to {edge['target_ticker']}: {edge['score']:.4f}")
    
    # ========== Step 8: Save Results ==========
    if save_plots:
        print("\n" + "="*80)
        print("STEP 8: Saving Results")
        print("="*80)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save IMPORTANT edges to CSV
        edges_df = pd.DataFrame(important_edges)
        edges_path = output_path / f"causal_edges_{target_ticker}.csv"
        edges_df.to_csv(edges_path, index=False)
        print(f" Saved important causal edges to {edges_path}")
        
        # Save ALL edges (important + filtered) for debugging
        all_edges = explanation_dict.get('all_edges', [])
        all_edges_df = pd.DataFrame(all_edges)
        all_edges_path = output_path / f"all_edges_ablation_{target_ticker}.csv"
        all_edges_df.to_csv(all_edges_path, index=False)
        print(f" Saved all ablation edges to {all_edges_path}")
        
        # Save comprehensive metrics
        metrics_path = output_path / f"explanation_metrics_{target_ticker}.txt"
        with open(metrics_path, 'w') as f:
            f.write(f"Explanation for: {target_ticker}\n")
            f.write(f"Target Node Index: {target_node}\n")
            f.write(f"\nQuality Metrics:\n")
            f.write(f"  Sparsity: {sparsity:.4f}\n")
            f.write(f"  Subgraph Fidelity Drop: {subgraph_drop:.6f}\n")
            f.write(f"  Max Single-Edge Drop: {single_edge_drop:.6f}\n")
            f.write(f"  Important Edges Count: {num_important_edges}\n")
            f.write(f"  Total Edges Tested: {len(all_edges)}\n")
            f.write(f"  Mask Threshold: 0.8 (only edges with score >= 0.8 are important)\n")
            f.write(f"\nTop 5 Edges by Raw Fidelity Drop:\n")
            for i, edge in enumerate(sorted(all_edges, key=lambda e: e['raw_fidelity_drop'], reverse=True)[:5], 1):
                f.write(f"  {i}. {edge['source_ticker']} to {edge['target_ticker']}: ")
                f.write(f"raw_drop={edge['raw_fidelity_drop']:.6f}, score={edge['score']:.4f}\n")
        print(f" Saved metrics to {metrics_path}")
    
    # ========== Summary ==========
    print("\n" + "="*80)
    print("EXPLANATION PIPELINE COMPLETE")
    print("="*80)
    
    results = {
        'target_node': target_node,
        'target_ticker': target_ticker,
        'mask': final_mask,
        'important_edges': important_edges,
        'sparsity': sparsity,
        'fidelity_drop': subgraph_drop,
        'explanation_dict': explanation_dict
    }
    
    return results


def explain_multiple_nodes(config_path='config/config.yaml',
                          model_path=None,
                          node_indices=None,
                          save_plots=True,
                          output_dir='results/explainability'):
    """
    Explain multiple nodes in the portfolio.
    """
    
    config = load_config()
    tickers_list = config['data']['ticker_list']
    
    if node_indices is None:
        node_indices = list(range(min(5, len(tickers_list))))
    
    results = {}
    
    for node_idx in node_indices:
        print(f"\n\n{'#'*80}")
        print(f"# Explaining Node {node_idx}: {tickers_list[node_idx]}")
        print(f"{'#'*80}")
        
        result = run_explanation_pipeline(
            model_path=model_path,
            target_node=node_idx,
            save_plots=save_plots,
            output_dir=output_dir
        )
        
        results[node_idx] = result
    
    return results


if __name__ == "__main__":
    # Load config to get ticker list
    config = load_config()
    tickers_list = config['data']['ticker_list']
    
    # Display available tickers
    print("\n" + "="*80)
    print("AVAILABLE TICKERS")
    print("="*80)
    for i, ticker in enumerate(tickers_list):
        sector = config['data']['sector_map'].get(ticker, 'N/A')
        print(f"  {i:2d}. {ticker} ({sector})")
    
    # Prompt user for ticker selection
    print("\n" + "="*80)
    print("TICKER SELECTION")
    print("="*80)
    while True:
        try:
            user_input = input("\nEnter ticker index (0-{}) or ticker name (e.g., 'AAPL'): ".format(len(tickers_list)-1)).strip()
            
            # Try to parse as integer index
            if user_input.isdigit():
                target_node = int(user_input)
                if 0 <= target_node < len(tickers_list):
                    target_ticker = tickers_list[target_node]
                    break
                else:
                    print(f" Index {target_node} out of range. Please enter 0-{len(tickers_list)-1}")
            # Try to match ticker name
            elif user_input.upper() in tickers_list:
                target_node = tickers_list.index(user_input.upper())
                target_ticker = user_input.upper()
                break
            else:
                print(f" '{user_input}' not found. Please enter a valid index or ticker name.")
        except ValueError:
            print(" Invalid input. Please enter a number or ticker name.")
    
    print(f"\n Selected: {target_ticker} (Index {target_node})")
    
    # Example 1: Explain a single node
    print("\n" + "="*80)
    print("EXAMPLE 1: Explaining Single Node")
    print("="*80)
    
    result = run_explanation_pipeline(
        target_node=target_node,
        save_plots=True,
        output_dir='results/explainability'
    )
    
    # Explain multiple nodes (uncomment to run)
    # print("\n" + "="*80)
    # print("EXAMPLE 2: Explaining Multiple Nodes")
    # print("="*80)
    # 
    # results = explain_multiple_nodes(
    #     config_path='config/config.yaml',
    #     node_indices=[0, 1, 2, 3],
    #     save_plots=True,
    #     output_dir='results/explainability'
    # )
