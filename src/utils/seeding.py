import random
import numpy as np
from stable_baselines3.common.utils import set_random_seed

class Seed:
    def __init__(self, seed=42):
        self.seed = seed
        self.set_all_seeds()
    
    def set_all_seeds(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        set_random_seed(self.seed)