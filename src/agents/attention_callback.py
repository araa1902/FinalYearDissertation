"""
This module provides a custom callback for Stable Baselines3 that captures
GAT attention weights during training.
"""

from stable_baselines3.common.callbacks import BaseCallback


class AttentionLoggingCallback(BaseCallback):
    """
    Custom callback that logs GAT attention weights during training.
    
    Captures attention weights from the feature extractor after each step
    and stores them in the environment's attention buffer.
    """
    
    def __init__(self, verbose: int = 0):
        super(AttentionLoggingCallback, self).__init__(verbose)
        self.env_base = None
        self.feature_extractor = None
    
    def _on_training_start(self) -> None:
        """Called when training starts."""
        # Access the underlying environment
        if hasattr(self.model.env, 'envs'):  # DummyVecEnv
            self.env_base = self.model.env.envs[0]
        else:
            self.env_base = self.model.env
        
        # Access feature extractor
        self.feature_extractor = self.model.policy.features_extractor
        
        if self.verbose > 0:
            print(" AttentionLoggingCallback initialized")
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        Captures attention weights from the feature extractor and logs them to the environment's attention buffer.
        Returns True to continue training, False to stop.
        """
        try:
            # Log attention weights from the feature extractor
            if self.feature_extractor is not None and self.env_base is not None:
                self.env_base.log_attention_weights(self.feature_extractor)
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not log attention weights: {e}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        try:
            # Save attention buffer
            if self.env_base is not None:
                self.env_base.save_final_results()
                if self.verbose > 0:
                    print(" Attention buffer saved")
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not save attention buffer: {e}")
