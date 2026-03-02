#GAT isolated training script for Phase 5
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.data.downloader import YahooDataDownloader
from src.data.preprocessor import FeatureEngineer
from src.data.graphbuilder import GraphBuilder
from src.gat.gat import GAT
from src.utils.config_manager import load_config

# 1. Setup & Config
config = load_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Data Pipeline
downloader = YahooDataDownloader(
    start_date=config['data']['start_date'], 
    end_date=config['data']['end_date'], 
    ticker_list=config['data']['ticker_list']
)
df_raw = downloader.fetch_data()
preprocessor = FeatureEngineer(use_technical_indicator=True)
df = preprocessor.preprocess_data(df_raw)

# 3. Build Graph
print("Building Graphs...")
builder = GraphBuilder(df, lookback_window=config['graph']['lookback_window'], threshold=config['graph']['threshold'])
graphs = builder.build_graphs(sparsity_method=config['graph']['sparsity_method'])

# 4. Prepare Tensors (Supervised Learning Task)
# X: Features at time t, Adj: Graph at time t
# Y: Returns at time t+1 (Target)
unique_dates = sorted(list(set(df.date.unique()) & set(graphs.keys())))
tickers = df.ticker.unique().tolist()
N = len(tickers)

X_list, Adj_list, Y_list = [], [], []

print("Aligning Tensors...")
for i in range(len(unique_dates) - 1): # Stop 1 day early for target
    date_t = unique_dates[i]
    date_next = unique_dates[i+1]
    
    # Feature Matrix X_t
    day_data = df[df['date'] == date_t].set_index('ticker').reindex(tickers).fillna(0)
    features = day_data[preprocessor.tech_indicator_list + ['volume', 'log_return']].values
    
    # Adjacency Matrix A_t
    adj = graphs[date_t]
    
    # Target Y_{t+1} (Next Day Return)
    next_day_data = df[df['date'] == date_next].set_index('ticker').reindex(tickers).fillna(0)
    target = next_day_data['log_return'].values.reshape(-1, 1) # (N, 1)
    
    X_list.append(features)
    Adj_list.append(adj)
    Y_list.append(target)

# Convert to Tensor
X_tensor = torch.FloatTensor(np.array(X_list)).to(device)    # (T, N, F)
Adj_tensor = torch.FloatTensor(np.array(Adj_list)).to(device) # (T, N, N)
Y_tensor = torch.FloatTensor(np.array(Y_list)).to(device)    # (T, N, 1)

print(f"Dataset Shape: {X_tensor.shape}")

# 5. Initialise Model
model = GAT(
    n_features=X_tensor.shape[2], 
    n_hidden=config['gat']['n_hidden'], 
    n_output=1,
    dropout=config['gat']['dropout'], 
    alpha=config['gat']['alpha'], 
    n_heads=config['gat']['n_heads']
).to(device)

optimiser = optim.Adam(model.parameters(), lr=config['gat']['lr'], weight_decay=config['gat']['weight_decay'])
criterion = nn.MSELoss()

# 6. Training Loop
print("\nStarting Training...")
model.train()
losses = []
for epoch in range(config['gat']['epochs']):
    optimiser.zero_grad()
    
    # Forward Pass
    output, attn_weights = model(X_tensor, Adj_tensor)
    
    # Loss Calculation
    loss = criterion(output, Y_tensor)
    
    # Backward
    loss.backward()
    optimiser.step()
    losses.append(loss.item())
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

print("\nTraining Complete.")


# 7. Verification (Success Criteria)
model.eval()
with torch.no_grad():
    # Extract Attention Weights for the last day
    attn = attn_weights.detach().cpu().numpy()
    last_attn = attn[-1]  # Last timestep attention
    print(attn)
    print("\n--- Phase 5 Verification ---")
    print(f"Attention Weights Shape: {attn.shape} (Heads merged or final layer)")
    print("Sample Attention Row (AAPL attending to others):")
    print(np.round(attn[0], 3))

# 8. Save Model and Visualization Data
import os
os.makedirs("results/gat_models", exist_ok=True)
os.makedirs("results/plots_data", exist_ok=True)

torch.save(model.state_dict(), "results/gat_models/gat_model_phase5.pth")
print("Trained model saved to 'results/gat_models/gat_model_phase5.pth'")
np.save("results/plots_data/training_loss.npy", np.array(losses)) 
np.save("results/plots_data/attention_weights.npy", last_attn)
np.save("results/plots_data/tickers.npy", np.array(tickers))

print("Visualisation data saved to 'results/plots_data/'")