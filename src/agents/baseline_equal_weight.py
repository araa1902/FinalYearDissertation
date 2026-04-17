"""
Baseline 2: Equal-Weight Portfolio (1/N Strategy)
This baseline is rebalanced daily to match the RL agent's decision frequency.
"""

# British English note: Uses standard financial terminology (portfolio, benchmark, etc.)

import numpy as np
import pandas as pd
from datetime import datetime
import os


class EqualWeightEvaluator:
    """
    Evaluates the equal-weight (1/N) portfolio strategy on test data.
    """
    
    def __init__(self, test_data, n_assets, initial_amount=100000, transaction_cost_pct=0.001):
        """Initialise the evaluator."""
        self.test_data = test_data
        self.n_assets = n_assets
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.equal_weight = 1.0 / n_assets
        
        self.results = None
        self.unique_dates = sorted(test_data['date'].unique())
        
    def evaluate(self):
        """Evaluate the 1/N strategy over the test period."""
        portfolio_values = [self.initial_amount]
        daily_returns = [0.0]
        actions = []
        dates_list = []
        
        # Initial allocation
        previous_weights = np.array([self.equal_weight] * self.n_assets)
        previous_prices = None
        
        for date_idx, date in enumerate(self.unique_dates):
            # Get daily data for all stocks
            day_data = self.test_data[self.test_data['date'] == date].sort_values('ticker')
            
            if len(day_data) != self.n_assets:
                continue  # Skip if we don't have all stocks for this day
            
            current_prices = day_data['close'].values
            
            # New weights (always equal-weight)
            new_weights = np.array([self.equal_weight] * self.n_assets)
            
            # Calculate portfolio return
            if previous_prices is None:
                portfolio_return_raw = 0.0
            else:
                # CORRECT METHOD: Calculate portfolio return using log of weighted price ratio
                # NOT the weighted sum of simple returns
                prev_portfolio_price = np.dot(previous_weights, previous_prices)
                curr_portfolio_price = np.dot(new_weights, current_prices)
                portfolio_return_raw = np.log(curr_portfolio_price / prev_portfolio_price)
                
            previous_prices = current_prices.copy()
            # ----------------------------------------------------------
            
            # Calculate transaction costs
            weight_changes = np.abs(new_weights - previous_weights)
            transaction_cost = np.sum(weight_changes) * self.transaction_cost_pct * portfolio_values[-1]
            
            # Net return after transaction costs
            portfolio_return_net = portfolio_return_raw - (transaction_cost / portfolio_values[-1])
            
            # Update portfolio value
            new_portfolio_value = portfolio_values[-1] * (1 + portfolio_return_net)
            
            portfolio_values.append(new_portfolio_value)
            daily_returns.append(portfolio_return_net)
            actions.append(new_weights.copy())
            dates_list.append(date)
            
            previous_weights = new_weights.copy()
        
        # Calculate comprehensive metrics
        self.results = self._calculate_metrics(daily_returns, portfolio_values, dates_list)
        self.results['portfolio_values'] = portfolio_values
        self.results['daily_returns'] = daily_returns
        self.results['actions'] = actions
        self.results['dates'] = dates_list
        
        return self.results
    
    def _calculate_metrics(self, returns, portfolio_values, dates):
        """Calculate performance metrics from backtest results."""
        df_returns = pd.DataFrame(returns[1:], columns=['daily_return'])  # Skip initial 0
        
        # Total return
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Annualised return
        n_days = len(dates) if len(dates) > 0 else 1
        years = n_days / 252.0
        annualised_return = (portfolio_values[-1] / portfolio_values[0]) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe ratio (annualised, assuming 252 trading days)
        mean_return = df_returns['daily_return'].mean()
        std_return = df_returns['daily_return'].std()
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Sortino ratio (only penalise downside volatility)
        downside_returns = df_returns[df_returns['daily_return'] < 0]['daily_return']
        downside_std = downside_returns.std()
        sortino_ratio = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # Maximum drawdown
        portfolio_values_array = np.array(portfolio_values)
        cumulative_max = np.maximum.accumulate(portfolio_values_array)
        drawdown = (portfolio_values_array - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdown)
        
        # Volatility (annualised)
        volatility = std_return * np.sqrt(252)
        

        calmar_ratio = annualised_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        # Win rate (% of positive days)
        win_rate = (df_returns['daily_return'] > 0).sum() / len(df_returns) if len(df_returns) > 0 else 0
        
        # Cumulative return
        cumulative_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        return {
            'strategy_name': 'Equal-Weight (1/N)',
            'total_return': total_return,
            'cumulative_return': cumulative_return,
            'annualised_return': annualised_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'volatility': volatility,
            'win_rate': win_rate,
            'final_portfolio_value': portfolio_values[-1],
            'initial_portfolio_value': portfolio_values[0],
            'n_trading_days': len(dates),
            'transaction_cost_pct': self.transaction_cost_pct
        }
    
    def save_results(self, output_path):
        """Save evaluation results to CSV."""
        if self.results is None:
            raise ValueError("No results to save. Run evaluate() first.")
        
        # Create DataFrame with daily results
        # Skip the initial 0 values to align with dates and actions
        df_results = pd.DataFrame({
            'date': self.results['dates'],
            'daily_return': self.results['daily_returns'][1:],  # Skip initial 0
            'portfolio_value': self.results['portfolio_values'][1:]  # Skip initial value
        })
        
        # Add actions as separate columns
        actions_df = pd.DataFrame(
            self.results['actions'],
            columns=[f'weight_stock_{i}' for i in range(self.n_assets)]
        )
        
        df_results = pd.concat([df_results, actions_df], axis=1)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save detailed daily results
        df_results.to_csv(output_path, index=False)
        print(f"[1/N Baseline] Detailed results saved to {output_path}")
        
        # Also save summary metrics
        summary_path = output_path.replace('.csv', '_summary.csv')
        df_summary = pd.DataFrame([{
            'strategy': self.results['strategy_name'],
            'total_return': self.results['total_return'],
            'annualised_return': self.results['annualised_return'],
            'sharpe_ratio': self.results['sharpe_ratio'],
            'sortino_ratio': self.results['sortino_ratio'],
            'max_drawdown': self.results['max_drawdown'],
            'volatility': self.results['volatility'],
            'win_rate': self.results['win_rate'],
            'initial_value': self.results['initial_portfolio_value'],
            'final_value': self.results['final_portfolio_value'],
            'n_trading_days': self.results['n_trading_days']
        }])
        df_summary.to_csv(summary_path, index=False)
        print(f"[1/N Baseline] Summary metrics saved to {summary_path}")
    
    def print_summary(self):
        """Print performance summary to console."""
        if self.results is None:
            print("No results to print. Run evaluate() first.")
            return
        
        print("\n" + "="*70)
        print("EQUAL-WEIGHT (1/N) BASELINE PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Strategy:               {self.results['strategy_name']}")
        print(f"Test Period:            {len(self.results['dates'])} trading days")
        print(f"Initial Value:          ${self.results['initial_portfolio_value']:,.2f}")
        print(f"Final Value:            ${self.results['final_portfolio_value']:,.2f}")
        print(f"Total Return:           {self.results['total_return']:.4f} ({self.results['total_return']*100:.2f}%)")
        print(f"Annualised Return:      {self.results['annualised_return']:.4f} ({self.results['annualised_return']*100:.2f}%)")
        print(f"Sharpe Ratio:           {self.results['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio:          {self.results['sortino_ratio']:.4f}")
        print(f"Max Drawdown:           {self.results['max_drawdown']:.4f} ({self.results['max_drawdown']*100:.2f}%)")
        print(f"Volatility (Annual):    {self.results['volatility']:.4f} ({self.results['volatility']*100:.2f}%)")
        print(f"Win Rate:               {self.results['win_rate']:.4f} ({self.results['win_rate']*100:.2f}%)")
        print("="*70 + "\n")
