
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from ..env.portfolio_env import StockPortfolioEnv

class PPOTrainer:
    def __init__(self, env: StockPortfolioEnv, config: dict):
        self.env = DummyVecEnv([lambda: env])  # SB3 requires a vectorised env
        self.config = config
        self.model = None

    def train(self):
        """
        Trains the PPO agent on the provided environment using configurations from the YAML file.
        """
        print("Initialising PPO Baseline (MlpPolicy)...")
        self.model = PPO(
            "MlpPolicy", 
            self.env,
            verbose=1,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            ent_coef=self.config["ent_coef"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"]
        )

        # Set up evaluation callback to monitor performance during training and save best model
        self.eval_callback = EvalCallback(
            self.env,
            best_model_save_path=self.config['best_model_path'],
            log_path=self.config['log_path'],
            eval_freq=self.config['eval_freq'],
            deterministic=True,
            render=False
        )
        
        print(f"Training for {self.config['total_timesteps']} timesteps...")
        self.model.learn(total_timesteps=self.config["total_timesteps"], callback=self.eval_callback)
        print("Training complete.")

    def save(self, path="models/baseline_ppo"):
        """
        Saves the trained model to the specified path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")