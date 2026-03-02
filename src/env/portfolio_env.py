# This environment is adapted based on FinRL's Portfolio Allocation Environment implementation. 
# Due to the nature of my research, I have modified the environment to incorporate graph structures and portfolio management specifics.
# Link: https://finrl.readthedocs.io/en/latest/tutorial/Introduction/PortfolioAllocation.html
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime
from matplotlib.lines import Line2D
import pickle

class StockPortfolioEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self,
                df,
                graph_dict,
                stock_dim,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold,
                lookback=252,
                day=0,
                results_csv_path=None,
                sector_map=None,
                max_sector_weight=0.4):
        
        self.day = day
        self.lookback = lookback
        self.df = df
        self.graph_dict = graph_dict # Store the graph dict
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.tech_indicator_list = tech_indicator_list
        self.sector_map = sector_map or {}
        self.max_sector_weight = max_sector_weight  # Max allocation per sector (0.4 = 40%)
        
        # Build sector-to-indices mapping for constraint checking
        self.sector_indices = self._build_sector_indices()
        
        # Generate unique filename with timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_csv_path = results_csv_path or f'results/episodes_metrics_with_graphs_{timestamp}.csv'
        self.all_episode_returns = []  # Track returns across all episodes

        # Get unique dates from the dataframe to index our steps
        self.unique_dates = sorted(self.df['date'].unique().tolist())
        
        # Action space: Portfolio weights (must sum to 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_dim,))
        
        # State space: Matrix of shape (N_assets + N_indicators + 2, N_assets)
        # Where N_indicators + 2 = len(tech_indicator_list) + volume + log_return
        # Row 0 to N-1: The Adjacency Matrix (Graph)
        # Row N to End: The Feature Vectors (Tech Indicators + Volume + Log Return)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(self.stock_dim + len(self.tech_indicator_list) + 2, self.stock_dim))

        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1/self.stock_dim]*self.stock_dim]
        self.episode_count = 0  # Track episode number
        
        # Risk-aware reward parameters
        self.volatility_lookback = 20  # 20-day rolling window for volatility
        self.recent_returns = []  # Track recent returns for volatility calculation
        self.max_drawdown_in_episode = 0  # Track max drawdown for this episode
        self.peak_value = self.initial_amount  # Track peak portfolio value
        
        #Attention Logging Buffer for Intrinsic Explainability
        self.attention_buffer = {
            'timestamps': [],           # List of dates
            'attention_weights': [],    # List of attention matrices (batch_size, n_heads, N, N)
            'adjacency_matrices': [],   # List of processed adjacency matrices
            'portfolio_values': [],     # Portfolio value at each step
            'dates': []                 # ISO format timestamps
        }
        
        # Initialise State
        self.data, self.current_date_str = self.get_daily_data(self.day)
        self.state = self.get_state(self.data, self.current_date_str)
        self.date_memory = [self.current_date_str]
    
    def _build_sector_indices(self):
        """Build mapping from sector -> list of stock indices for constraint checking."""
        sector_to_indices = {}
        tickers = sorted(self.df['ticker'].unique().tolist())
        
        for sector, ticker in self.sector_map.items():
            if ticker in tickers:
                idx = tickers.index(ticker)
                if sector not in sector_to_indices:
                    sector_to_indices[sector] = []
                sector_to_indices[sector].append(idx)
        
        return sector_to_indices

    def get_daily_data(self, day_index : int):
        """Helper to get all ticker data for a specific calendar day."""
        if day_index >= len(self.unique_dates):
            # End of data
            return None, None
            
        date = self.unique_dates[day_index]
        day_data = self.df[self.df['date'] == date].sort_values('ticker')
        return day_data, date

    def get_state(self, day_data : pd.DataFrame, date: str) -> np.ndarray:
        """Constructs the state matrix: Graph + Features"""
        covs = self.graph_dict.get(date, np.eye(self.stock_dim))
        
        #Get Technical Features + Volume + Log_Return [N_features, N_stocks]
        # N_features = len(tech_indicator_list) + 2 (volume + log_return)
        tech_features = []
        for tech in self.tech_indicator_list:
            tech_features.append(day_data[tech].values.tolist())
        # Add volume and log_return to complete the 8-dimensional feature set
        tech_features.append(day_data['volume'].values.tolist())
        tech_features.append(day_data['log_return'].values.tolist())
        state = np.vstack((covs, np.array(tech_features)))
        return state

    def calculate_episode_metrics(self):
        """Calculate all metrics for the completed episode"""
        df_result = pd.DataFrame(self.portfolio_return_memory)
        df_result.columns = ['daily_return']
        
        total_return = df_result['daily_return'].sum()
        sharpe = (252**0.5) * df_result['daily_return'].mean() / df_result['daily_return'].std()
        max_drawdown = (df_result.daily_return.cumsum().expanding().max() - df_result.daily_return.cumsum()).max()
        
        metrics = {
            'episode': self.episode_count + 1,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'end_total_asset': self.portfolio_value,
            'daily_returns': df_result['daily_return'].cumsum().tolist()
        }
        return metrics

    def log_episode_metrics(self, metrics):
        """Log episode metrics to console"""
        print("=================================")
        print(f"Episode {metrics['episode']} Complete")
        print(f"End Total Asset: {metrics['end_total_asset']:.2f}")
        print(f"Total Return: {metrics['total_return']:.4f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
        print("=================================")

    def save_episode_metrics(self, metrics):
        """Save episode metrics to CSV"""
        os.makedirs('results', exist_ok=True)
        
        metrics_data = {
            'Episode': metrics['episode'],
            'Timestamp': datetime.now().isoformat(),
            'End_Total_Asset': metrics['end_total_asset'],
            'Total_Return': metrics['total_return'],
            'Sharpe_Ratio': metrics['sharpe_ratio'],
            'Max_Drawdown': metrics['max_drawdown']
        }
        
        metrics_df = pd.DataFrame([metrics_data])
        
        # Append to CSV file
        if os.path.exists(self.results_csv_path):
            metrics_df.to_csv(self.results_csv_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(self.results_csv_path, mode='w', header=True, index=False)

    def track_episode_returns(self, metrics):
        """Track episode returns for final plotting"""
        self.all_episode_returns.append({
            'episode': metrics['episode'],
            'returns': metrics['daily_returns']
        })

    def handle_episode_logging(self, metrics):
        """Consolidated logging function for episode completion"""
        self.log_episode_metrics(metrics)
        self.save_episode_metrics(metrics)
        self.track_episode_returns(metrics)

    def log_attention_weights(self, feature_extractor):
        """
        Capture attention weights from the GAT feature extractor.
        Called after each environment step to maintain temporal alignment.
        """
        if hasattr(feature_extractor, 'latest_attention_weights') and feature_extractor.latest_attention_weights is not None:
            self.attention_buffer['timestamps'].append(self.current_date_str)
            self.attention_buffer['dates'].append(datetime.now().isoformat())
            
            # Store attention weights (shape: batch_size, n_heads, N, N)
            attn = feature_extractor.latest_attention_weights
            if attn.dim() == 4:  # (batch, heads, N, N)
                attn = attn.mean(dim=0)  # Average across batch -> (heads, N, N)
            
            self.attention_buffer['attention_weights'].append(attn.numpy())
            
            # Store processed adjacency matrix for context
            if hasattr(feature_extractor, 'latest_adjacency') and feature_extractor.latest_adjacency is not None:
                adj = feature_extractor.latest_adjacency
                if adj.dim() == 3: adj = adj.mean(dim=0)  # Average across batch -> (N, N)
                self.attention_buffer['adjacency_matrices'].append(adj.numpy())
            
            # Track portfolio value
            self.attention_buffer['portfolio_values'].append(self.portfolio_value)

    def step(self, actions):
        self.terminal = self.day >= len(self.unique_dates) - 1

        if self.terminal:
            # Calculate metrics
            metrics = self.calculate_episode_metrics()
            
            # Handle all logging
            self.handle_episode_logging(metrics)
            
            self.episode_count += 1
            
            return self.state, self.reward, True, False, {}
        else:
            weights = self.process_actions(actions)
            trans_cost = self.calculate_transaction_cost(weights)
            last_day_data = self.data
            self.day += 1
            self.data, self.current_date_str = self.get_daily_data(self.day)
            self.state = self.get_state(self.data, self.current_date_str)
            
            # Calculate returns and update portfolio
            portfolio_return = self.calculate_portfolio_return(weights, last_day_data, trans_cost)
            self.update_portfolio(weights, portfolio_return)
            
            self.reward = self.calculate_reward()

            return self.state, self.reward, False, False, {}

    def process_actions(self, actions):
        """Convert raw actions to normalised portfolio weights with sector constraints"""
        actions = np.array(actions)
        exp_values = np.exp(actions - np.max(actions))
        weights = exp_values / np.sum(exp_values)
        
        weights[weights < 0.01] = 0.0
        w_sum = np.sum(weights)
        if w_sum > 0:
            weights /= w_sum
        else:
            weights = np.ones_like(weights) / self.stock_dim
        
        # Apply sector constraints: cap allocation per sector at max_sector_weight
        if self.sector_indices:
            for sector, indices in self.sector_indices.items():
                sector_weight = np.sum(weights[indices])
                if sector_weight > self.max_sector_weight:
                    # Scale down sector weights proportionally
                    excess = sector_weight - self.max_sector_weight
                    weights[indices] *= (1 - excess / sector_weight)
            
            # Renormalise to ensure weights sum to 1
            total_weight = np.sum(weights)
            if total_weight > 0:
                weights /= total_weight
        
        return weights

    def calculate_transaction_cost(self, weights):
        """Calculate transaction costs based on portfolio rebalancing"""
        prev_weights = self.actions_memory[-1]
        turnover = np.sum(np.abs(weights - prev_weights))
        trans_cost = turnover * self.transaction_cost_pct
        
        self.actions_memory.append(weights)
        return trans_cost

    def calculate_portfolio_return(self, weights, last_day_data, trans_cost):
        """Calculate daily portfolio return"""
        portfolio_return = np.sum(((self.data.close.values / last_day_data.close.values) - 1) * weights)
        portfolio_return -= trans_cost
        return portfolio_return

    def update_portfolio(self, weights, portfolio_return):
        """Update portfolio value and tracking variables"""
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_return_memory.append(portfolio_return)
        self.date_memory.append(self.current_date_str)
        self.asset_memory.append(self.portfolio_value)

    def calculate_reward(self):
        """
        Calculate risk-adjusted reward:
        reward = log_return - volatility_penalty - drawdown_penalty
        
        This incentivises:
        - Returns (positive log_return)
        - Smooth returns (low volatility)
        - Drawdown avoidance (protect capital)
        """
        # 1. Basic log return
        log_return = np.log(self.portfolio_value / self.asset_memory[-2])
        
        # 2. Track recent returns for volatility calculation
        self.recent_returns.append(log_return)
        if len(self.recent_returns) > self.volatility_lookback:
            self.recent_returns.pop(0)
        
        # 3. Calculate rolling volatility (penalises erratic behavior)
        if len(self.recent_returns) > 1:
            volatility = np.std(self.recent_returns)
        else:
            volatility = 0.0
        
        # 4. Track maximum drawdown in episode (penalises losing capital)
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        self.max_drawdown_in_episode = max(self.max_drawdown_in_episode, drawdown)
        
        # 5. Construct risk-aware reward
        # Weights: focus on returns but penalise volatility and drawdowns
        volatility_penalty = 0.5 * volatility  # Penalise erratic returns
        drawdown_penalty = 2.0 * drawdown      # Penalise current drawdown state
        
        reward = log_return - volatility_penalty - drawdown_penalty
        
        return reward * self.reward_scaling
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data, self.current_date_str = self.get_daily_data(self.day)
        self.state = self.get_state(self.data, self.current_date_str)
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        
        # Reset risk tracking for new episode
        self.recent_returns = []
        self.max_drawdown_in_episode = 0
        self.peak_value = self.initial_amount
        self.actions_memory = [[1/self.stock_dim]*self.stock_dim]
        self.date_memory = [self.current_date_str]
        
        # SPRINT 1: Reset attention buffer for new episode
        self.attention_buffer = {
            'timestamps': [],
            'attention_weights': [],
            'adjacency_matrices': [],
            'portfolio_values': [],
            'dates': []
        }
        
        return self.state, {}
    
    def save_final_results(self):
        """Call this after training completes to save the final graph"""
        if not self.all_episode_returns:
            print("No episodes to plot")
            return
        
        os.makedirs('results', exist_ok=True)
        
        graph_path = self.plot_all_episodes()
        self.print_training_summary(graph_path)
        
        # Save attention buffer for explainability analysis
        self.save_attention_buffer()

    def save_attention_buffer(self):
        """
        Save the attention weights buffer to disk for later analysis.
        Creates a pickle file with all captured attention data.
        """
        
        if not self.attention_buffer['timestamps']:
            print("No attention data to save")
            return
        
        os.makedirs('results/attention_logs', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        attention_log_path = f'results/attention_logs/attention_buffer_{timestamp}.pkl'
        buffer_to_save = {
            'timestamps': self.attention_buffer['timestamps'],
            'dates': self.attention_buffer['dates'],
            'attention_weights': [attn.tolist() if hasattr(attn, 'tolist') else attn 
                                  for attn in self.attention_buffer['attention_weights']],
            'adjacency_matrices': [adj.tolist() if hasattr(adj, 'tolist') else adj 
                                   for adj in self.attention_buffer['adjacency_matrices']],
            'portfolio_values': self.attention_buffer['portfolio_values']
        }
        
        with open(attention_log_path, 'wb') as f:
            pickle.dump(buffer_to_save, f)
        
        print(f"\nAttention buffer saved: {attention_log_path}")
        print(f"  Timesteps captured: {len(self.attention_buffer['timestamps'])}")


    def plot_all_episodes(self):
        """Generate and save the cumulative reward plot for all episodes"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.figure(figsize=(14, 7))
        
        for ep_data in self.all_episode_returns:
            plt.plot(ep_data['returns'], alpha=0.6, linewidth=1.5)
        
        # Add dates to x-axis
        x_ticks = np.linspace(0, len(self.all_episode_returns[0]['returns']) - 1, 10, dtype=int) if self.all_episode_returns else []
        x_labels = [pd.to_datetime(self.unique_dates[min(i, len(self.unique_dates) - 1)]).strftime('%b %Y') for i in x_ticks]
        plt.xticks(x_ticks, x_labels, rotation=45, ha='right')
        
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Portfolio Cumulative Return Over Time - All Episodes')
        plt.grid(True, alpha=0.3)
        custom_lines = [Line2D([0], [0], color='C0', lw=1.5, alpha=0.6)]
        plt.legend(custom_lines, [f'Episodes: {len(self.all_episode_returns)}'], loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        graph_path = f'results/cumulative_reward_{timestamp}.png'
        plt.savefig(graph_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return graph_path

    def print_training_summary(self, graph_path):
        """Print training completion summary"""
        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"Results saved:")
        print(f"  - Episodes Metrics CSV: {self.results_csv_path}")
        print(f"  - Cumulative Reward Graph: {graph_path}")
        print(f"{'='*50}\n")

    def render(self, mode='human'):
        return self.state