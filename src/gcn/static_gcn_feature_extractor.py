"""
Static GCN Feature Extractor for Baseline 3
This allows us to distinguish graph structure generally from dynamic adaptation specifically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import yaml
import numpy as np


class StaticGCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer (static adjacency variant).
    Based on Kipf & Welling (2017): "Semi-Supervised Classification with Graph Convolutional Networks"
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super(StaticGCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        
        # Xavier initialisation
        self.weight = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass for GCN layer."""
        # Apply dropout
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Linear transformation: h' = h * W
        out = torch.matmul(x, self.weight)
        
        # Graph convolution: h'' = A * h'
        # Expand adjacency to batch dimension for broadcasting
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)  # (1, n_nodes, n_nodes)
        
        out = torch.matmul(adj, out)  # (batch, n_nodes, out_features)
        
        # Add bias
        out = out + self.bias
        
        return out


class StaticGCN(nn.Module):
    """Multi-layer Graph Convolutional Network with static adjacency."""
    
    def __init__(self, n_features: int, n_hidden: int, n_output: int, 
                 dropout: float = 0.5, n_layers: int = 2):
        super(StaticGCN, self).__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        
        # Layer 1: input -> hidden
        self.gc1 = StaticGCNLayer(n_features, n_hidden, dropout=dropout)
        
        # Additional hidden layers if specified
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - 2):
            self.hidden_layers.append(StaticGCNLayer(n_hidden, n_hidden, dropout=dropout))
        
        # Output layer: hidden -> output
        self.gc_out = StaticGCNLayer(n_hidden, n_output, dropout=dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass through GCN."""
        # First layer with ReLU activation
        x = self.gc1(x, adj)
        x = F.relu(x)
        
        # Hidden layers
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x, adj)
            x = F.relu(x)
        
        # Output layer (no activation - let downstream handle it)
        x = self.gc_out(x, adj)
        
        return x


class StaticGCNFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for PPO using Static GCN.
    This is the critical baseline for comparing dynamic vs. static graph topology.
    """
    
    def __init__(self, observation_space, config_path="config/config.yaml", 
                 static_adjacency=None, training_correlations=None):
        """
        Initialise the Static GCN feature extractor.
        """
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        # Load configuration
        self.n_hidden = self.config['gat']['n_hidden']
        self.n_heads = self.config['gat']['n_heads']
        self.threshold = self.config['gat']['threshold']
        self.dropout = self.config['gat']['dropout']
        
        # GCN specific: number of layers (not used in GAT with attention)
        self.n_layers = 2  # Standard 2-layer GCN
        
        # Calculate dimensions
        self.num_tickers = len(self.config['data']['ticker_list'])
        self.n_features = len(self.config['preprocessing']['tech_indicator_list']) + 2
        
        # Initialise parent class with output dimension
        # For GCN: output = n_hidden (1 head equivalent vs GAT's n_hidden * n_heads)
        super(StaticGCNFeatureExtractor, self).__init__(
            observation_space, 
            features_dim=self.n_hidden
        )
        
        # Create or load static adjacency matrix
        if static_adjacency is not None:
            self.register_buffer('static_adjacency', 
                                torch.tensor(static_adjacency, dtype=torch.float32))
        elif training_correlations is not None:
            # Build static adjacency from training correlations
            adj = self._build_static_adjacency(training_correlations)
            self.register_buffer('static_adjacency', adj)
        else:
            # Default: identity matrix (will be identity until set properly)
            self.register_buffer('static_adjacency',
                                torch.eye(self.num_tickers, dtype=torch.float32))
        
        # Learnable sector bias (same as dynamic model for fair comparison)
        self.sector_weight = nn.Parameter(
            torch.tensor(self.config['gat']['sector_weight'], dtype=torch.float32)
        )
        
        # Create static sector mask
        self._create_sector_matrix()
        
        # Define the GCN (not GAT, critical for baseline)
        self.gcn = StaticGCN(
            n_features=self.n_features,
            n_hidden=self.n_hidden,
            n_output=self.n_hidden,  # Output dimension matches feature extractor
            dropout=self.dropout,
            n_layers=self.n_layers
        )
        
        print(f"[StaticGCN Baseline] Initialised with fixed adjacency matrix")
        print(f"  - n_features: {self.n_features}")
        print(f"  - n_hidden: {self.n_hidden}")
        print(f"  - n_layers: {self.n_layers}")
        print(f"  - dropout: {self.dropout}")
        print(f"  - sector_bias: {self.config['gat']['sector_weight']}")
    
    def _build_static_adjacency(self, correlation_matrix: np.ndarray) -> torch.Tensor:
        """Build static adjacency from training period correlation matrix."""
        # Take absolute value of correlations
        adj = np.abs(correlation_matrix)
        
        # Apply threshold
        adj = (adj >= self.threshold).astype(np.float32) * adj
        
        # Add self-loops
        np.fill_diagonal(adj, 1.0)
        
        return torch.tensor(adj, dtype=torch.float32)
    
    def _create_sector_matrix(self):
        """Create static sector mask (same fundamental structure for all stocks)."""
        tickers = self.config['data']['ticker_list']
        sector_map = self.config['data']['sector_map']
        
        sector_matrix = torch.zeros((self.num_tickers, self.num_tickers), dtype=torch.float32)
        
        for i in range(self.num_tickers):
            for j in range(self.num_tickers):
                if sector_map.get(tickers[i]) == sector_map.get(tickers[j]):
                    sector_matrix[i, j] = 1.0
        
        self.register_buffer('sector_mask', sector_matrix)
    
    def set_static_adjacency(self, training_correlations: np.ndarray):
        """Set the static adjacency matrix from training correlations."""
        adj = self._build_static_adjacency(training_correlations)
        self.register_buffer('static_adjacency', adj)
        print(f"[StaticGCN Baseline] Static adjacency matrix set from training period")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through static GCN feature extractor."""
        # Extract features (ignore adjacency from env since we use static version)
        # Observations shape: (Batch, N + Features, N)
        features_from_env = observations[:, self.num_tickers:, :]
        
        # Transpose to (Batch, Tickers, Features)
        x = features_from_env.transpose(1, 2)
        x = F.normalize(x, p=2, dim=-1)
        
        # Prepare static adjacency with sector bias
        # The static adjacency is fixed but we still apply sector weighting
        adj = self.static_adjacency.unsqueeze(0)  # Add batch dimension
        
        # Apply sector bias (blends historical correlation with fundamental structure)
        sector_weight = torch.sigmoid(self.sector_weight)
        adj = (1 - sector_weight) * adj + (sector_weight * self.sector_mask.unsqueeze(0))
        
        # Apply sparsification threshold
        adj = torch.abs(adj)
        mask = (adj >= self.threshold).float()
        adj = adj * mask
        
        # Ensure self-loops
        identity = torch.eye(self.num_tickers, device=adj.device).unsqueeze(0)
        adj = torch.max(adj, identity)
        
        # Add row-normalization to mimic GAT Softmax scaling
        # This ensures weights coming into each node sum to 1, making it mathematically comparable to GAT
        row_sum = adj.sum(dim=-1, keepdim=True)
        adj = adj / (row_sum + 1e-8)
        
        # GCN forward pass
        gcn_output = self.gcn(x, adj)
        
        # Pooling: aggregate node features for PPO policy input
        # Shape: (batch, n_hidden)
        pooled = torch.mean(gcn_output, dim=1)
        
        return pooled
