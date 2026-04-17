"""
PPO Trainer with Static GCN Feature Extractor (Baseline 3)

Trains PPO with a Graph Convolutional Network that uses a FIXED adjacency matrix
computed once from the full training period (2015-2021 in the dissertation context).

This is the critical baseline for isolating the contribution of dynamic topology 
recomputation (H2) vs. static graph structure (H1).
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from ..env.portfolio_env import StockPortfolioEnv
from src.gcn.static_gcn_feature_extractor import StaticGCNFeatureExtractor
from datetime import datetime
import numpy as np
import pandas as pd


class PPOStaticGCNTrainer:
    """
    Trains PPO agent with Static GCN feature extractor.
    
    Key differences from dynamic (GAT) trainer:
    - Uses StaticGCNFeatureExtractor instead of GATFeatureExtractor
    - Requires pre-computed training period correlations for static adjacency
    - Adjacency matrix is fixed throughout all train/val/test splits
    - Same hyperparameter configuration as dynamic model for fair comparison
    """
    
    def __init__(self, env: StockPortfolioEnv, config: dict, 
                 training_correlations: np.ndarray = None):
        """Initialise Static GCN trainer."""
        self.env = DummyVecEnv([lambda: env])
        self.config = config
        self.model = None
        self.training_correlations = training_correlations
        
        print("[PPOStaticGCNTrainer] Initialised trainer for Static GCN baseline")
    
    def compute_training_correlations(self, training_data: pd.DataFrame) -> np.ndarray:
        """Compute correlation matrix from full training period data."""
        # Pivot returns by ticker (rows = dates, columns = tickers)
        returns_matrix = training_data.pivot(
            index='date',
            columns='ticker',
            values='log_return'
        )
        
        # Compute correlation
        correlation_matrix = returns_matrix.corr().values
        
        # Handle NaN values
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        print(f"[PPOStaticGCNTrainer] Computed training period correlations")
        print(f"  Shape: {correlation_matrix.shape}")
        print(f"  Mean abs correlation: {np.mean(np.abs(correlation_matrix)):.4f}")
        print(f"  Sparsity (corr < 0.3): {(np.abs(correlation_matrix) < 0.3).sum() / correlation_matrix.size:.2%}")
        
        return correlation_matrix
    
    def train(self):
        """Train the PPO agent with Static GCN feature extractor."""
        print("\n" + "="*80)
        print("BASELINE 3: Static GCN Trainer")
        print("="*80)
        
        # Compute or use provided training correlations
        if self.training_correlations is None:
            # Try to get training data from environment
            if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
                env_obj = self.env.envs[0]
                if hasattr(env_obj, 'df'):
                    print("[PPOStaticGCNTrainer] Computing training correlations from environment data...")
                    self.training_correlations = self.compute_training_correlations(env_obj.df)
            
            if self.training_correlations is None:
                print("[PPOStaticGCNTrainer] WARNING: No training correlations provided.")
                print("  Using default identity matrix. Results may be suboptimal.")
                self.training_correlations = np.eye(self.config['data']['n_stocks'])
        
        print("\n[PPOStaticGCNTrainer] Initialising PPO with Static GCN Feature Extractor...")
        
        # Define policy kwargs with static GCN feature extractor
        policy_kwargs = dict(
            features_extractor_class=StaticGCNFeatureExtractor,
            features_extractor_kwargs=dict(
                config_path="config/config.yaml",
                training_correlations=self.training_correlations
            )
        )
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            ent_coef=self.config["ent_coef"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"],
            seed=42  # Ensure reproducibility
        )
        
        # Set up evaluation callback
        self.eval_callback = EvalCallback(
            self.env,
            best_model_save_path=self.config['best_model_path'],
            log_path=self.config['log_path'],
            eval_freq=self.config['eval_freq'],
            deterministic=True,
            render=False
        )
        
        print(f"\n[PPOStaticGCNTrainer] Training for {self.config['total_timesteps']} timesteps...")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Gamma (discount): {self.config['gamma']}")
        print(f"  Clip range (epsilon): {self.config['clip_range']}")
        
        # Train the model
        self.model.learn(
            total_timesteps=self.config["total_timesteps"],
            callback=self.eval_callback
        )
        
        print("[PPOStaticGCNTrainer] Training complete.")
        print(f"  Best model saved to: {self.config['best_model_path']}")
        
        # Print static adjacency info
        if hasattr(self.model.policy, 'features_extractor'):
            extractor = self.model.policy.features_extractor
            if hasattr(extractor, 'static_adjacency'):
                adj = extractor.static_adjacency.cpu().numpy()
                sparsity = (adj == 0).sum() / adj.size
                print(f"\n[PPOStaticGCNTrainer] Static Adjacency Statistics:")
                print(f"  Density (non-zero entries): {1 - sparsity:.2%}")
                print(f"  Mean weight: {np.mean(adj[adj > 0]):.4f}")
    
    def save(self, path: str = "models/baseline_static_gcn"):
        """
        Saves the trained model to the specified path.
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.model.save(path)
        print(f"[PPOStaticGCNTrainer] Model saved to {path}")
    
    def get_feature_extractor(self):
        """
        Get the feature extractor from the trained model.
        Useful for inspection and visualization.
        """
        if self.model is None:
            raise ValueError("No model. Train first.")
        return self.model.policy.features_extractor


class StaticGCNEvaluator:
    """
    Evaluates a trained Static GCN model on test data.
    """
    
    def __init__(self, model_path: str, env_kwargs: dict, test_data: pd.DataFrame):
        """Initialise evaluator."""
        self.model_path = model_path
        self.env_kwargs = env_kwargs
        self.test_data = test_data
        self.model = None
        self.test_env = None
        self.results = None
    
    def load_model(self):
        """Load the trained PPO model."""
        model_file = os.path.join(self.model_path, "best_model.zip")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model not found at {model_file}")
        
        print(f"[StaticGCNEvaluator] Loading model from {model_file}")
        self.model = PPO.load(model_file)
    
    def evaluate(self, num_episodes: int = 1, deterministic: bool = True) -> dict:
        """
        Evaluate the Static GCN model on test data.
        """
        if self.model is None:
            self.load_model()
        
        self.test_env = StockPortfolioEnv(df=self.test_data, **self.env_kwargs)
        
        print(f"[StaticGCNEvaluator] Evaluating on {len(self.test_data['date'].unique())} trading days...")
        
        all_returns = []
        all_values = []
        all_actions = []
        all_dates = []
        
        for episode in range(num_episodes):
            obs, _ = self.test_env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.test_env.step(action)
                done = terminated or truncated
            
            all_returns.extend(self.test_env.portfolio_return_memory)
            all_values.extend(self.test_env.asset_memory)
            all_actions.extend(self.test_env.actions_memory)
            all_dates.extend(self.test_env.date_memory)
        
        self.results = self._calculate_metrics(all_returns, all_values)
        self.results['returns'] = all_returns
        self.results['portfolio_values'] = all_values
        self.results['actions'] = all_actions
        self.results['dates'] = all_dates
        
        return self.results
    
    def _calculate_metrics(self, returns: list, portfolio_values: list) -> dict:
        """Calculate performance metrics."""
        df_returns = pd.DataFrame(returns, columns=['daily_return'])
        
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        mean_return = df_returns['daily_return'].mean()
        std_return = df_returns['daily_return'].std()
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        n_days = len(returns)
        years = n_days / 252.0
        annualised_return = (portfolio_values[-1] / portfolio_values[0]) ** (1 / years) - 1 if years > 0 else 0
        
        portfolio_values_array = np.array(portfolio_values)
        cumulative_max = np.maximum.accumulate(portfolio_values_array)
        drawdown = (portfolio_values_array - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdown)
        
        # Calmar Ratio (Annualised Return / Absolute Max Drawdown)
        calmar_ratio = annualised_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        volatility = std_return * np.sqrt(252)
        win_rate = (df_returns['daily_return'] > 0).sum() / len(df_returns)
        
        return {
            'total_return': total_return,
            'annualised_return': annualised_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'volatility': volatility,
            'win_rate': win_rate,
            'final_portfolio_value': portfolio_values[-1],
            'initial_portfolio_value': portfolio_values[0]
        }
    
    def save_results(self, output_path: str):
        """Save evaluation results to CSV."""
        if self.results is None:
            raise ValueError("No results to save. Run evaluate() first.")
        
        df_results = pd.DataFrame({
            'date': self.results['dates'],
            'daily_return': self.results['returns'],
            'portfolio_value': self.results['portfolio_values']
        })
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
