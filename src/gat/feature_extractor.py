import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.gat.gat import GAT
import yaml

"""
----------------------------------------------
This module extracts features for the PPO agent using a Graph Attention Network.
This decoupling prevents circular logic and allows the GAT to learn 
relationships between 'Market Structure' and 'Price Action'.
"""
class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, config_path="config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # 1. Load GAT Config
        self.n_hidden = self.config['gat']['n_hidden']
        self.n_heads = self.config['gat']['n_heads']
        self.threshold = self.config['gat']['threshold']
        self.dropout = self.config['gat']['dropout']
        self.alpha = self.config['gat']['alpha']
        
        # 2. Calculate Input Dimensions
        self.num_tickers = len(self.config['data']['ticker_list'])
        
        # Features = Tech Indicators (6) + Volume (1) + Log Return (1) = 8
        self.n_features = len(self.config['preprocessing']['tech_indicator_list']) + 2 

        # 3. Initialise Parent Class
        super(GATFeatureExtractor, self).__init__(observation_space, features_dim=self.n_hidden * self.n_heads)

        # 4. Learnable Sector Bias (The "Anchor")
        # This parameter allows the model to learn how much to trust 
        # Fundamental Sectors vs. Correlation Data.
        self.sector_weight = nn.Parameter(torch.tensor(self.config['gat']['sector_weight'], dtype=torch.float32))

        # 5. Define the GAT
        self.gat = GAT(
            n_features=self.n_features,  # Now correctly matches input x
            n_hidden=self.n_hidden,
            n_output=self.n_hidden * self.n_heads,
            dropout=self.dropout,
            alpha=self.alpha,
            n_heads=self.n_heads
        )

        # 6. Sector Mask Buffer (Static Domain Knowledge)
        self._create_sector_matrix()
        
        # Store for visualisation
        self.latest_attention_weights = None

    def _create_sector_matrix(self):
        """Creates a static boolean mask where 1 = Same Sector."""
        tickers = self.config['data']['ticker_list']
        sector_map = self.config['data']['sector_map']
    
        sector_matrix = torch.zeros((self.num_tickers, self.num_tickers), dtype=torch.float32)

        for i in range(self.num_tickers):
            for j in range(self.num_tickers):
                # Safe .get() in case of missing tickers
                if sector_map.get(tickers[i]) == sector_map.get(tickers[j]):
                    sector_matrix[i, j] = 1.0
                
        self.register_buffer('sector_mask', sector_matrix)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Observations shape: (Batch, N + Features, N)
        # We split the graph (first N rows) from the features (remaining rows)
        
        # 1. Extract The Graph (Adjacency)
        adj_from_env = observations[:, :self.num_tickers, :] 

        # 2. Extract The Features
        # Source: Technical Indicators (MACD, RSI, etc.)
        features_from_env = observations[:, self.num_tickers:, :]
        
        # Transpose to (Batch, Tickers, Features)
        x = features_from_env.transpose(1, 2)
        x = F.normalize(x, p=2, dim=-1)

        # 3. Prepare Adjacency Matrix
        adj = adj_from_env
        
        # 4. Apply Sector Bias (The "Fusion" Step)
        # Blends Historical Correlation (adj) with Fundamental Truth (sector_mask)
        sector_weight = torch.sigmoid(self.sector_weight)
        adj = (1 - sector_weight) * adj + (sector_weight * self.sector_mask)

        # 5. Sparsify (Thresholding)
        adj = torch.abs(adj)
        mask = (adj >= self.threshold).float()
        adj = adj * mask
        
        # 6. Add Self-Loops (Critical for GAT message passing)
        identity = torch.eye(self.num_tickers, device=adj.device).unsqueeze(0)
        adj = torch.max(adj, identity)

        # 7. GAT Forward Pass
        gat_output, attention_weights = self.gat(x, adj)
        
        #Store multi-head attention weights for explainability
        self.latest_attention_weights = attention_weights.detach().cpu()
        
        # Also store the processed adjacency matrix for context
        self.latest_adjacency = adj.detach().cpu()

        # 8. Pooling (Graph Level Embedding)
        # Aggregate node features to create a single vector for PPO
        return torch.mean(gat_output, dim=1)