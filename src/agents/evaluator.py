import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from ..env.portfolio_env import StockPortfolioEnv
import os


class ModelEvaluator:
    """
    Evaluates a trained PPO model on unseen test data.
    """
    
    def __init__(self, model_path, env_kwargs, test_data):
        """
        Initialise the evaluator.
        """
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
        
        print(f"Loading model from {model_file}")
        self.model = PPO.load(model_file)
        
    def evaluate(self, num_episodes=1, deterministic=True):
        """
        Evaluate the model on test data.
        
        Args:
            num_episodes: Number of episodes to run (default 1 for full test period)
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Create test environment
        self.test_env = StockPortfolioEnv(df=self.test_data, **self.env_kwargs)
        
        print(f"Evaluating on {len(self.test_data.date.unique())} trading days...")
        
        all_returns = []
        all_values = []
        all_actions = []
        all_dates = []
        
        for episode in range(num_episodes):
            obs, _ = self.test_env.reset()
            done = False
            episode_rewards = []
            
            while not done:
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=deterministic)
                
                # Take step in environment
                obs, reward, terminated, truncated, info = self.test_env.step(action)
                done = terminated or truncated
                
                episode_rewards.append(reward)
            
            # Collect results from this episode
            all_returns.extend(self.test_env.portfolio_return_memory)
            all_values.extend(self.test_env.asset_memory)
            all_actions.extend(self.test_env.actions_memory)
            all_dates.extend(self.test_env.date_memory)
        
        # Calculate metrics
        self.results = self._calculate_metrics(all_returns, all_values)
        
        # Store detailed results
        self.results['returns'] = all_returns
        self.results['portfolio_values'] = all_values
        self.results['actions'] = all_actions
        self.results['dates'] = all_dates
        
        return self.results
    
    def _calculate_metrics(self, returns, portfolio_values):
        """Calculate performance metrics from test results."""
        df_returns = pd.DataFrame(returns, columns=['daily_return'])
        
        # Total return
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        mean_return = df_returns['daily_return'].mean()
        std_return = df_returns['daily_return'].std()
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Maximum drawdown
        portfolio_values_array = np.array(portfolio_values)
        cumulative_max = np.maximum.accumulate(portfolio_values_array)
        drawdown = (portfolio_values_array - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdown)
        
        # Volatility (annualized)
        volatility = std_return * np.sqrt(252)
        
        # Win rate
        win_rate = (df_returns['daily_return'] > 0).sum() / len(df_returns)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'final_portfolio_value': portfolio_values[-1],
            'initial_portfolio_value': portfolio_values[0]
        }
    
    def save_results(self, output_path):
        """
        Save evaluation results to CSV.
        """
        if self.results is None:
            raise ValueError("No results to save. Run evaluate() first.")
        
        # Create DataFrame with daily results
        df_results = pd.DataFrame({
            'date': self.results['dates'],
            'daily_return': self.results['returns'],
            'portfolio_value': self.results['portfolio_values']
        })
        
        # Add actions as separate columns
        actions_df = pd.DataFrame(
            self.results['actions'],
            columns=[f'weight_stock_{i}' for i in range(len(self.results['actions'][0]))]
        )
        
        df_results = pd.concat([df_results, actions_df], axis=1)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df_results.to_csv(output_path, index=False)
        print(f"Detailed results saved to {output_path}")
        
        # Also save summary metrics
        summary_path = output_path.replace('.csv', '_summary.csv')
        df_summary = pd.DataFrame([{
            'total_return': self.results['total_return'],
            'sharpe_ratio': self.results['sharpe_ratio'],
            'max_drawdown': self.results['max_drawdown'],
            'volatility': self.results['volatility'],
            'win_rate': self.results['win_rate'],
            'initial_value': self.results['initial_portfolio_value'],
            'final_value': self.results['final_portfolio_value']
        }])
        df_summary.to_csv(summary_path, index=False)
        print(f"Summary metrics saved to {summary_path}")
    
    def plot_results(self, output_path=None):
        """
        Plot the portfolio performance over time.
        """
        if self.results is None:
            raise ValueError("No results to plot. Run evaluate() first.")
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot portfolio value
        axes[0].plot(self.results['portfolio_values'])
        axes[0].set_title('Portfolio Value Over Time (Test Period)')
        axes[0].set_xlabel('Trading Day')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot daily returns
        axes[1].plot(self.results['returns'])
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_title('Daily Returns (Test Period)')
        axes[1].set_xlabel('Trading Day')
        axes[1].set_ylabel('Daily Return')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
