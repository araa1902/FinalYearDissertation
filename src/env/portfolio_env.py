# This code is adapted based on Stable Baselines3 PPO implementation. Due to the nature of our research, we have modified the environment to incorporate graph structures and portfolio management specifics.
# Link: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
import numpy as np
import pandas as pd
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class StockPortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                df,
                graph_dict,
                stock_dim,
                hmax,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold,
                lookback=252,
                day=0):
        
        self.day = day
        self.lookback = lookback
        self.df = df
        self.graph_dict = graph_dict # Store the graph dict
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.tech_indicator_list = tech_indicator_list

        # Get unique dates from the dataframe to index our steps
        self.unique_dates = sorted(self.df['date'].unique().tolist())
        
        # Action space: Portfolio weights (must sum to 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_dim,))
        
        # State space: Matrix of shape (N_assets + N_indicators, N_assets)
        # Row 0 to N-1: The Adjacency Matrix (Graph)
        # Row N to End: The Feature Vectors (Price, Tech Indicators)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(self.stock_dim + len(self.tech_indicator_list), self.stock_dim))

        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1/self.stock_dim]*self.stock_dim]
        
        # Initialise State
        self.data, self.current_date_str = self._get_daily_data(self.day)
        self.state = self._get_state(self.data, self.current_date_str)
        self.date_memory = [self.current_date_str]

    def _get_daily_data(self, day_index : int):
        """Helper to get all ticker data for a specific calendar day."""
        if day_index >= len(self.unique_dates):
            # End of data
            return None, None
            
        date = self.unique_dates[day_index]
        day_data = self.df[self.df['date'] == date].sort_values('ticker')
        return day_data, date

    def _get_state(self, day_data : pd.DataFrame, date: str) -> np.ndarray:
        """Constructs the state matrix: Graph + Features"""
        covs = self.graph_dict.get(date, np.eye(self.stock_dim))
        
        #Get Technical Features [N_features, N_stocks]
        tech_features = []
        for tech in self.tech_indicator_list:
            tech_features.append(day_data[tech].values.tolist())
            
        #Stack them: Top rows = Graph, Bottom rows = Features
        state = np.vstack((covs, np.array(tech_features)))
        return state

    def step(self, actions):
        self.terminal = self.day >= len(self.unique_dates) - 1

        if self.terminal:
            df_result = pd.DataFrame(self.portfolio_return_memory)
            df_result.columns = ['daily_return']
            plt.plot(df_result.daily_return.cumsum(), 'r')
            plt.savefig('results/cumulative_reward.png')
            plt.close()
            
            print("=================================")
            print(f"End Total Asset: {self.portfolio_value}")
            sharpe = (252**0.5) * df_result['daily_return'].mean() / df_result['daily_return'].std()
            print(f"Sharpe Ratio: {sharpe}")
            print("=================================")
            
            return self.state, self.reward, self.terminal, {}

        else:
            actions = np.array(actions)
            exp_values = np.exp(actions - np.max(actions))
            weights = exp_values / np.sum(exp_values)
            
            weights[weights < 0.01] = 0.0
            w_sum = np.sum(weights)
            if w_sum > 0:
                weights /= w_sum
            else:
                weights = np.ones_like(weights) / self.stock_dim
            
            # Calculate Transaction Costs
            prev_weights = self.actions_memory[-1]
            turnover = np.sum(np.abs(weights - prev_weights))
            trans_cost = turnover * self.transaction_cost_pct
            
            self.actions_memory.append(weights)
            last_day_data = self.data
            self.day += 1
            self.data, self.current_date_str = self._get_daily_data(self.day)
            
            # Update State
            self.state = self._get_state(self.data, self.current_date_str)
            
            # Calculate Return
            portfolio_return = np.sum(((self.data.close.values / last_day_data.close.values) - 1) * weights)
            portfolio_return -= trans_cost

            # Update Value
            self.portfolio_value *= (1 + portfolio_return)
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.current_date_str)
            self.asset_memory.append(self.portfolio_value)
            
            # Reward: Log Return
            self.reward = np.log(self.portfolio_value / self.asset_memory[-2]) 
            self.reward = self.reward * self.reward_scaling

            return self.state, self.reward, self.terminal, {}
    
    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data, self.current_date_str = self._get_daily_data(self.day)
        self.state = self._get_state(self.data, self.current_date_str)
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1/self.stock_dim]*self.stock_dim]
        self.date_memory = [self.current_date_str]
        return self.state

    def render(self, mode='human'):
        return self.state