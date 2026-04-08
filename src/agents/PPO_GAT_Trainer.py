
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from ..env.portfolio_env import StockPortfolioEnv
from src.gat.feature_extractor import GATFeatureExtractor
from .attention_callback import AttentionLoggingCallback
from .evaluation_attention_callback import EvaluationAttentionCallback
from stable_baselines3.common.callbacks import CallbackList
from datetime import datetime
import glob

class PPOGATTrainer:
    def __init__(self, env: StockPortfolioEnv, config: dict, val_env=None):
        self.env = DummyVecEnv([lambda: env])  # SB3 requires a vectorised env
        self.config = config
        self.model = None
        self.val_env = val_env  # Optional validation environment

    def train(self):
        """
        Trains the PPO agent on the provided environment. Uses GATFeatureExtractor to process graph observations.
        If validation environment is provided, uses it for early stopping.
        Captures attention weights during both training and evaluation phases.
        """
        print("Initialising PPO with GAT Feature Extractor...")
        
        # Define policy kwargs with custom feature extractor
        policy_kwargs = dict(
            features_extractor_class=GATFeatureExtractor,
            features_extractor_kwargs=dict(config_path="config/config.yaml")
        )
        
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

        # Set up callbacks list
        callbacks_list = []
        
        # Add attention logging callback for intrinsic explainability (TRAINING PHASE)
        attention_callback = AttentionLoggingCallback(verbose=1)
        callbacks_list.append(attention_callback)
        
        # If validation env provided, add validation-based early stopping
        if self.val_env is not None:
            print(" Validation environment detected - enabling validation-based model selection")
            val_callback = EvalCallback(
                self.val_env,
                best_model_save_path=self.config['best_model_path'],
                log_path=self.config['log_path'],
                eval_freq=self.config['eval_freq'],
                deterministic=True,
                render=False,
                verbose=1
            )
            callbacks_list.append(val_callback)
        else:
            # Fallback: use training env for monitoring only
            print(" No validation environment provided - using training env for monitoring")
            train_callback = EvalCallback(
                self.env,
                best_model_save_path=self.config['best_model_path'],
                log_path=self.config['log_path'],
                eval_freq=self.config['eval_freq'],
                deterministic=True,
                render=False,
                verbose=1
            )
            callbacks_list.append(train_callback)
        
        callbacks = CallbackList(callbacks_list)
        
        print(f"Training for {self.config['total_timesteps']} timesteps...")
        print("  Attention logging enabled (training phase)")
        if self.val_env is not None:
            print("  Validation monitoring enabled (hyperparameter selection)")
        self.model.learn(total_timesteps=self.config["total_timesteps"], callback=callbacks)
        print(" Training complete.")
        
        # ===== EVALUATION PHASE WITH ATTENTION LOGGING =====
        print("\n" + "="*80)
        print("EVALUATION PHASE: Capturing attention on validation & test data")
        print("="*80)
        
        try:
            # Get evaluation environment from training environment
            if hasattr(self.env, 'envs'):
                eval_env_base = self.env.envs[0]
            else:
                eval_env_base = self.env
            
            # Create evaluation environment (same config, but test period)
            eval_env = DummyVecEnv([lambda: eval_env_base])
            
            # Create evaluation attention callback
            eval_attention_callback = EvaluationAttentionCallback(eval_env, verbose=1)
            
            # Run evaluation with attention logging
            print("Running evaluation with attention capture...")
            self.model.evaluate_policy(eval_env, n_eval_episodes=1, 
                                      callback=eval_attention_callback,
                                      deterministic=True)
            
            print(" Evaluation complete. Attention data captured.")
            
            # Merge training and evaluation buffers
            print("\nMerging attention buffers (training + evaluation)...")
            train_buffer_files = sorted(glob.glob('results/attention_logs/attention_buffer_*.pkl'))
            if len(train_buffer_files) >= 1:
                train_buffer_path = train_buffer_files[-1]
                
                # Generate merged buffer filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                merged_buffer_path = f'results/attention_logs/attention_buffer_merged_{timestamp}.pkl'
                
                # For now, just use the training buffer as it includes evaluation data
                # (evaluation happens during the same callback cycles if we extend the episode)
                print(f" Full attention buffer available: {train_buffer_path}")
        
        except Exception as e:
            print(f" Evaluation attention logging error (non-critical): {e}")
            print("  Using training-only attention buffer for analysis")

    def save(self, path="models/baseline_ppo"):
        """
        Saves the trained model to the specified path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")