# Majority of code is referenced using FinRL documentation (https://finrl.readthedocs.io/en/latest/tutorial/Introduction/PortfolioAllocation.html) and adapted for our graph RL stock trading environment
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym
    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date
    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step

    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                df,
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
                day = 0):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.lookback=lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalisation and shape is self.stock_dim
        self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,))
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(low=0,
                                            high=np.inf,
                                            shape = (self.state_space+len(self.tech_indicator_list),
                                                    self.state_space))

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.covs = self.data['cov_list'].values[0]
        self.state =  np.append(np.array(self.covs),
                    [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initialise state: initial portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorise portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorise portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]


    def step(self, actions):
        # Check if we have reached the end of the dataset
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            # --- Reporting & Visualisation ---
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            try:
                import os
                os.makedirs('results', exist_ok=True)
                
                plt.plot(df.daily_return.cumsum(), 'r')
                plt.savefig('results/cumulative_reward.png')
                plt.close()

                plt.plot(self.portfolio_return_memory, 'r')
                plt.savefig('results/rewards.png')
                plt.close()
            except Exception as e:
                plt.close('all')
                pass

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.portfolio_value))

            if df['daily_return'].std() != 0:
                sharpe = (252**0.5) * df['daily_return'].mean() / df['daily_return'].std()
                print("Sharpe: ", sharpe)
            else:
                print("Sharpe: 0.0 (Flatline)")
            print("=================================")

            return self.state, self.reward, self.terminal, {}

        else:
            # 1. Standard Softmax normalisation
            actions = np.array(actions)
            exp_values = np.exp(actions - np.max(actions))
            weights = exp_values / np.sum(exp_values)
            
            # Force small weights to exactly 0.0 to allow fully exiting positions.
            # This prevents paying transaction costs for negligible positions (e.g., 0.0001%).
            truncation_threshold = 0.01  # epsilon = 1%
            weights[weights < truncation_threshold] = 0.0
            
            # Renormalise: Ensure weights sum to 1.0 after truncation
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights /= weight_sum
            else:
                weights = np.ones_like(weights) / self.stock_dim
            
            # Calculate turnover: sum of absolute difference between new weights and previous weights
            prev_weights = self.actions_memory[-1]
            turnover = np.sum(np.abs(weights - prev_weights))
            trans_cost = turnover * self.transaction_cost_pct
            
            # Log weights for memory
            self.actions_memory.append(weights)
            last_day_memory = self.data

            self.day += 1
            self.data = self.df.loc[self.day, :]
            
            # Construct the state (Features + Adjacency Info)
            # This 'covs' variable is critical for your Graph RL integration
            self.covs = self.data['cov_list'].values[0]
            self.state = np.append(np.array(self.covs), 
                                   [self.data[tech].values.tolist() for tech in self.tech_indicator_list], 
                                   axis=0)

            gross_portfolio_return = sum(((self.data.close.values / last_day_memory.close.values) - 1) * weights)
            
            # Net Return: Subtract transaction costs
            portfolio_return = gross_portfolio_return - trans_cost

            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)
            self.reward = np.log(new_portfolio_value / self.asset_memory[-2]) 
            self.reward = self.reward * self.reward_scaling

        return self.state, self.reward, self.terminal, {}
    
    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        # load states
        self.covs = self.data['cov_list'].values[0]
        self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        self.portfolio_value = self.initial_amount
        #self.cost = 0
        #self.trades = 0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]
        return self.state

    def render(self, mode='human'):
        return self.state

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]