import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
   """
   Singular Graph Attention Layer as adapted from the paper:
   "Graph Attention Networks" by Velickovic et al. (2018)
   """
   def __init__(self, in_features: int, out_features: int, dropout: float, alpha: float, concat=True):
       super(GATLayer, self).__init__()
       self.in_features = in_features
       self.out_features = out_features
       self.dropout = dropout
       self.alpha = alpha
       self.concat = concat


       # Xavier Initialisation for stability
       self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
       nn.init.xavier_uniform_(self.W.data, gain=1.414)
      
       self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
       nn.init.xavier_uniform_(self.a.data, gain=1.414)


       self.leakyrelu = nn.LeakyReLU(self.alpha)


   def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
       # 1. Linear Transformation
       Wh = torch.matmul(h, self.W)  # Shape: (Batch, N, Out_Features)
      
       # 2. Attention Mechanism - compute attention scores for each node pair
       B, N, _ = Wh.size()
       a_input = torch.cat([Wh.repeat(1, 1, N).view(B, N * N, -1), Wh.repeat(1, N, 1)], dim=2).view(B, N, N, 2 * self.out_features)
       e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) # unormalised attention scores


       # 3. Masking (Graph Topology Injection) - this is done to ignore non-neighbour nodes so that they do not contribute to the attention mechanism
       zero_vec = -9e15 * torch.ones_like(e)
       attention = torch.where(adj > 0, e, zero_vec)
      
       # 4. Softmax Normalisation - converts node pair attention scores into probabilities to provide contextual importance
       attention = F.softmax(attention, dim=2)
       attention = F.dropout(attention, self.dropout, training=self.training)
      
       # 5. Aggregation - weighted sum of neighbour features based on attention scores
       h_prime = torch.matmul(attention, Wh)


       if self.concat:
           return F.elu(h_prime), attention # exponential linear unit activation to introduce non-linearity
       else:
           return h_prime, attention


class GAT(nn.Module):
   """
   n_features: Number of input features per node
   n_hidden: Number of hidden units in each attention head
   n_output: Number of output classes
   dropout: Dropout rate
   alpha: Negative slope for LeakyReLU
   n_heads: Number of attention heads
   """
   def __init__(self, n_features: int, n_hidden: int, n_output: int, dropout: float, alpha: float, n_heads: int):
       super(GAT, self).__init__()
       self.dropout = dropout


       # Multi-Head Attention
       self.attentions = nn.ModuleList([GATLayer(n_features, n_hidden, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)])
      
       # Final Output Layer (aggregates heads)
       self.out_att = GATLayer(n_hidden * n_heads, n_output, dropout=dropout, alpha=alpha, concat=False)

   def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
       x = F.dropout(x, self.dropout, training=self.training)
       
       # Forward pass through all heads - capture multi-head attention
       head_results = [att(x, adj) for att in self.attentions]
       multi_head_attentions = torch.stack([res[1] for res in head_results], dim=1)  # Shape: (batch, n_heads, N, N)
       x_cat = torch.cat([res[0] for res in head_results], dim=2)
      
       x_cat = F.dropout(x_cat, self.dropout, training=self.training)
      
       # return final output layer along with multi-head attention weights
       x, final_attn_weights = self.out_att(x_cat, adj)
       # Average attention across heads for final layer as well
       return x, multi_head_attentions
   
