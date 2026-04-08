"""
Evaluation Attention Callback

Captures GAT attention weights during model evaluation/test phase.
Extends the training attention buffer with test period data.
"""

from stable_baselines3.common.callbacks import BaseCallback


class EvaluationAttentionCallback(BaseCallback):
    """
    Custom callback that logs GAT attention weights during evaluation.
    """
    
    def __init__(self, eval_env, verbose: int = 0):
        """
        Initialise evaluation attention callback.
        """
        super(EvaluationAttentionCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.feature_extractor = None
    
    def _on_training_start(self) -> None:
        """Called when evaluation starts."""
        # Access feature extractor from the policy
        self.feature_extractor = self.model.policy.features_extractor
        
        if self.verbose > 0:
            print(" EvaluationAttentionCallback initialized")
    
    def _on_step(self) -> bool:
        """
        Called after each evaluation step.
        Captures attention weights and logs them to the environment.
        """
        try:
            # Log attention weights during evaluation
            if self.feature_extractor is not None and self.eval_env is not None:
                # Handle both single env and vectorized env
                if hasattr(self.eval_env, 'envs'):
                    env_base = self.eval_env.envs[0]
                else:
                    env_base = self.eval_env
                
                env_base.log_attention_weights(self.feature_extractor)
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log attention weights during eval: {e}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Called when evaluation ends."""
        # Evaluation buffer is saved by the environment's step logic
        pass
