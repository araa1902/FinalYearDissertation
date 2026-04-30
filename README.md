# Graph Reinforcement Learning for Portfolio Optimisation with Explainability

## 1. Project Overview and Structure

### 1.1 Brief Description

This project implements **GAT-PPO**, a novel framework for multi-asset portfolio optimisation that leverages graph neural networks to model inter-asset relationships and attention mechanisms to explain investment decisions. The framework is designed to:

- **Learn dynamic trading policies** using PPO on a 15-asset portfolio spanning multiple sectors (Tech, Finance, Healthcare, Consumer, Energy, Industrials, and hedging assets)
- **Capture market regimes** (2015-2024 period includes 2020 COVID, 2022 interest rate shock, 2024 recovery) through a custom Gymnasium environment
- **Provide explainability** via:
  - **Intrinsic explainability**: Real-time attention heatmaps showing which asset relationships drive decisions
  - **Post-hoc explainability**: Exact Edge Ablation audit to measure causal impact of graph edges on portfolio actions

### 1.2 Repository Map

```
graph-rl-portfolio/
├── src/                           # Main source code
│   ├── agents/                    # RL training pipelines
│   │   ├── train_ppo_gat.py       # Primary training script for GAT-PPO
│   │   ├── PPO_GAT_Trainer.py     # GAT policy wrapper for stable-baselines3
│   │   ├── PPO_StaticGCN_Trainer.py
│   │   ├── PPOTrainer.py          # Baseline PPO (no graph)
│   │   └── cross_validation.py    # K-fold cross-validation utilities
│   │
│   ├── gat/                       # Graph Attention Network implementation
│   │   ├── gat.py                 # GAT layer & full GAT model with sector fusion
│   │   ├── feature_extractor.py   # SB3-compatible feature extractor
│   │   └── sector_fusion.py       # Sector-level graph construction
│   │
│   ├── gcn/                       # Graph Convolutional Network (alternative)
│   │   ├── gcn.py                 # GCN implementation
│   │   └── feature_extractor.py
│   │
│   ├── env/                       # Custom Gymnasium environments
│   │   ├── portfolio_env.py       # Main RL environment (GAT-compatible, attention capture)
│   │   └── portfolio_env_baseline.py  # Baseline environment (no graph)
│   │
│   ├── data/                      # Data pipeline
│   │   ├── downloader.py          # YahooDataDownloader: fetches 10-year data from yfinance
│   │   ├── preprocessor.py        # FeatureEngineer: OHLCV + technical indicators
│   │   └── graphbuilder.py        # GraphBuilder: constructs dynamic correlation graphs
│   │
│   ├── ablation/                  # Ablation study framework
│   │   └── [Runs stored in logs/ and models/]
│   │
│   ├── explainability/            # Explainability pipeline
│   │   ├── intrinsic/
│   │   │   ├── attention_analyser.py        # Extract & analyse attention weights
│   │   │   ├── network_visualisation.py     # Draw network graphs with attention
│   │   │   ├── plot_regime_attention_figures.py  # Generate Figure 6.3 & 6.4 (heatmaps & deltas)
│   │   │   └── plot_attention_deltas.py
│   │   │
│   │   └── posthoc/
│   │       ├── exact_edge_ablation_explainer.py  # DenseGNNExplainer: causal edge ablation
│   │       └── explain_portfolio_decisions.py    # Interpret action explanations
│   │
│   ├── models/                    # Neural network definitions
│   │   └── [Policy networks and feature extractors]
│   │
│   └── utils/
│       ├── config_manager.py      # Load config.yaml
│       ├── seeding.py             # Deterministic seeds across torch/numpy/gym
│       └── [Utility functions]
│
├── config/
│   └── config.yaml                # Central configuration: tickers, sectors, train/test split
│
├── data/
│   └── raw/                       # Store downloaded .csv files from yfinance (optional)
│
├── models/                        # Saved trained models
│   ├── best_model/                # Latest best GAT-PPO checkpoint
│   ├── best_model_backup/
│   ├── ppo_gat_YYYYMMDD_HHMMSS/   # Timestamped training runs
│   ├── ppo_gcn_YYYYMMDD_HHMMSS/
│   ├── ppo_baseline_YYYYMMDD_HHMMSS/
│   ├── cv_fold_[1-5]_gat/         # Cross-validation fold checkpoints
│   ├── cv_fold_[1-5]_gcn/
│   └── ablation_*/                # Ablation study model variants
│
├── logs/                          # Training & evaluation logs
│   ├── cv_fold_[1-5]_gat/         # K-fold training logs
│   ├── cv_fold_[1-5]_gcn/
│   ├── ablation_full_YYYYMMDD_HHMMSS/         # Full model ablations
│   ├── ablation_no_sector_blending_YYYYMMDD_HHMMSS/
│   ├── ablation_no_signal_independence_YYYYMMDD_HHMMSS/
│   └── ablation_no_both_YYYYMMDD_HHMMSS/
│
├── results/                       # Evaluation outputs
│   ├── episodes_metrics_*.csv     # Per-episode returns, sharpe, drawdown
│   ├── ablation_study_*.csv       # Ablation study performance comparison
│   ├── ablation_models_*.txt      # Model hyperparameters
│   └── regime_attention_matrices_*.pkl  # Attention weights for visualisation
│
├── docs/                          # Documentation
├── scripts/
│   ├── visualisation/             # Visualisation utilities
│   └── [Helper scripts]
│
├── testingScripts/                # Ad-hoc testing & debugging
│
├── requirements.txt               # Python dependencies
├── package.json                   # Node.js dependencies (for data viz tools)
├── config.yaml                    # [Primary config file]
└── README.md                      # This file
```

## 2. Computational Environment

### 2.1 Software Dependencies

**Python 3.9+** (tested on 3.11)

Core packages:

| Package             | Version | Purpose                                |
| ------------------- | ------- | -------------------------------------- |
| `torch`             | 2.0+    | Deep learning framework                |
| `torch-geometric`   | 2.3+    | Graph neural network ops               |
| `stable-baselines3` | 2.0+    | PPO implementation & RL utilities      |
| `gymnasium`         | 0.29+   | Standardized RL environment API        |
| `pandas`            | 2.0+    | Data manipulation                      |
| `numpy`             | 1.24+   | Numerical computing                    |
| `yfinance`          | 0.2.28+ | Download stock data from Yahoo Finance |
| `networkx`          | 3.0+    | Graph construction & analysis          |
| `scikit-learn`      | 1.3+    | Preprocessing & metrics                |
| `ta`                | 0.10+   | Technical analysis indicators          |
| `matplotlib`        | 3.7+    | Plotting                               |
| `seaborn`           | 0.12+   | Statistical visualisation              |
| `streamlit`         | 1.28+   | Interactive dashboard (optional)       |
| `pyyaml`            | 6.0+    | Config file parsing                    |
| `shap`              | 0.42+   | SHAP explanations (optional)           |

Install all dependencies:

```bash
pip install -r requirements.txt
```

### 2.2 Hardware Requirements

- **Minimum**: CPU-based training (slow, ~24-48 hours for 1M timesteps)
- **Recommended**: NVIDIA GPU (RTX 3080 or equivalent)
  - Tested on: NVIDIA RTX 3090, NVIDIA A100 (40 GB)
  - Memory: 8 GB+ VRAM for standard training
  - Training time: ~4-6 hours for 1M timesteps

## 3. Setup and Installation

### 3.1 Clone Repository

```bash
git clone https://github.com/araa1902/FinalYearDissertation.git
cd graph-rl-portfolio
```

### 3.2 Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate

# Or using conda
conda create -n grl-portfolio python=3.11
conda activate grl-portfolio
```

### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.4 Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
```

### 3.5 Data Acquisition

#### Option A: Automatic Download (Recommended)

The training pipeline automatically downloads data from Yahoo Finance via `yfinance`:

```python
# config.yaml controls the ticker list and date range
# Default: 2015-01-01 to 2024-12-31 (10 years)
# Split: Train (2015-2020), Test (2021-2024)
```

#### Option B: Manual Download (Pre-cached)

If you have a `.csv` file with OHLCV data, place it in `data/raw/`:

```
data/raw/
├── AAPL_2015_2024.csv
├── MSFT_2015_2024.csv
└── ...
```

**Expected CSV format**:

```csv
Date,Open,High,Low,Close,Volume
2015-01-02,111.39,111.44,107.41,107.84,53248600
...
```

## 4. Reproduction Instructions

### 4.1 Training: GAT-PPO Agent

Train the full model on 2015-2020 data with deterministic seed:

```bash
cd src/agents
python train_ppo_gat.py \
  --seed 42 \
  --total_timesteps 1000000 \
  --learning_rate 3e-4 \
  --n_steps 2048 \
  --output_dir ../../models/ppo_gat_$(date +%Y%m%d_%H%M%S)
```

**Expected outputs:**

- Model checkpoint: `models/best_model.zip`
- Training logs: `logs/training_metrics.csv`
- Plots: `results/episodes_metrics_*.csv`

**Key hyperparameters:**

- `total_timesteps`: 1M for quick validation, 2-5M for full training
- `learning_rate`: 3e-4 (standard for PPO)
- `n_steps`: 2048 (GAT is compute-intensive; reduce if OOM)
- `seed`: 42 (or any fixed value for reproducibility)

### 4.2 Evaluation & Backtesting

Test the trained model on 2021-2024 (out-of-sample):

```bash
cd src/agents
python -c "
from train_ppo_gat import evaluate_model_on_test
from src.env.portfolio_env import StockPortfolioEnv
from src.data.downloader import YahooDataDownloader
from src.data.preprocessor import FeatureEngineer
import pandas as pd

# Download data
downloader = YahooDataDownloader(start_date='2015-01-01', end_date='2024-12-31')
df = downloader.download(['AAPL', 'MSFT', ..., 'TLT'])

# Preprocess
fe = FeatureEngineer()
df_processed = fe.engineer(df)

# Split into test
test_data = df_processed[df_processed['Date'] >= '2021-01-01']

# Evaluate
results = evaluate_model_on_test(
    model_path='../../models/best_model',
    env_class=StockPortfolioEnv,
    env_kwargs={'initial_balance': 100000},
    test_data=test_data
)

print(f\"Total Return: {results['total_return']:.2%}\")
print(f\"Sharpe Ratio: {results['sharpe_ratio']:.2f}\")
print(f\"Max Drawdown: {results['max_drawdown']:.2%}\")
"
```

**Outputs:**

- `results/evaluation_results_*.csv`: Daily returns, portfolio values, actions
- Attention weights logged during evaluation (see explainability section)

### 4.3 Explainability Pipeline

#### 4.3.1 Generate Attention Heatmaps (Figure 6.3 & 6.4)

Extract attention weights from test period and generate publication-ready figures:

```bash
cd src/explainability/intrinsic

# Step 1: Extract attention weights from evaluation run
python attention_analyser.py \
  --model_path ../../models/best_model/best_model.zip \
  --test_start 2021-01-01 \
  --test_end 2024-12-31 \
  --output_pkl results/regime_attention_matrices.pkl

# Step 2: Generate regime-conditioned heatmaps & delta plots
python plot_regime_attention_figures.py \
  --input results/regime_attention_matrices.pkl \
  --output_dir ../../results
```

**Generated figures:**

- `Figure_6_3_Baseline_Heatmap.pdf` - 2021 Bull Market attention
- `Figure_6_3_Stress_Heatmap.pdf` - 2022 Interest Rate Shock attention
- `Figure_6_3_Rally_Heatmap.pdf` - 2024 Recovery attention
- `Figure_6_4_Attention_Deltas.pdf` - Top 10 amplifications and attenuations

#### 4.3.2 Run Exact Edge Ablation (Silicon Valley Bank Case Study)

Perform causal analysis on how graph edges affect portfolio embeddings during SVB crisis (March 2023):

```bash
cd src/explainability/posthoc

python -c "
from exact_edge_ablation_explainer import DenseGNNExplainer
from src.env.portfolio_env import StockPortfolioEnv
from src.utils.config_manager import load_config
import torch

# Load trained model
config = load_config()
feature_extractor = torch.load('../../models/best_model/feature_extractor.pt')
explainer = DenseGNNExplainer(feature_extractor, config=config, device='cuda')

# Get SVB event window (March 2023)
# x: [1, num_assets, feature_dim] at SVB crisis date
# adj: correlation adjacency matrix

# Run edge ablation for each asset
for target_asset_idx in range(len(config['data']['ticker_list'])):
    impacts = explainer.explain(x, adj, target_node_idx=target_asset_idx)
    print(f\"Asset {config['data']['ticker_list'][target_asset_idx]}: {impacts}\")
"
```

**Outputs:**

- `ablation_output.log` - Edge impact scores for each asset pair
- Identifies which connections were most critical during financial stress

## 5. Key Mechanisms

### 5.1 Custom Gymnasium Environment

- **State**: 15-asset OHLCV + technical indicators (RSI, Bollinger Bands, MACD)
- **Action space**: Continuous portfolio weights (simplex constraint via softmax)
- **Reward**: Sharpe ratio of returns and penalty for transaction costs
- **Graph construction**: Dynamic correlation-based adjacency matrix updated daily

### 5.2 Graph Attention Network (GAT)

- **Multi-head attention**: 4 heads to capture different relationship types
- **Sector fusion**: Blends inter-sector and intra-sector correlation matrices
- **Signal independence**: Decorrelated graph inputs via sector-level decomposition
- **Explainability**: Stores raw attention weights for each head & layer

### 5.3 Ablation Studies

Four variants tested:

- **Full model**: All components active
- **No sector blending**: Correlation-based graphs only
- **No signal independence**: No sector decomposition
- **No both**: Baseline graph construction

### 5.4 Explainability Framework

| Method                    | Scope     | Output                                                   |
| ------------------------- | --------- | -------------------------------------------------------- |
| **Attention heatmaps**    | Intrinsic | Regime-specific edge importance maps                     |
| **Attention deltas**      | Intrinsic | What changed from baseline to stress?                    |
| **Edge ablation**         | Post-hoc  | Exact causal impact of each edge                         |
| **Network visualisation** | Both      | Animated or static graph drawings with attention overlay |

## 6. Configuration

Edit `config/config.yaml` to customise:

```yaml
data:
  start_date: "2015-01-01" # Data start
  end_date: "2024-12-31" # Data end
  train_end_date: "2020-12-31" # Train/test split
  ticker_list:
    - "AAPL"
    - "MSFT"
    # ... 15 assets total
  sector_map:
    "AAPL": "TECH"
    # ... maps each ticker to sector

training:
  total_timesteps: 1000000
  learning_rate: 3e-4
  batch_size: 128
  n_steps: 2048

explainability:
  mask_threshold: 0.8 # Edge ablation threshold
  attention_aggregation: "mean" # How to pool attention heads
```

## 7. Troubleshooting

| Issue                              | Solution                                                       |
| ---------------------------------- | -------------------------------------------------------------- |
| **CUDA out of memory**             | Reduce `n_steps` or `batch_size` in config                     |
| **Slow data download**             | Pre-download and cache in `data/raw/`                          |
| **Model not found**                | Ensure `models/best_model/best_model.zip` exists               |
| **Attention weights not captured** | Verify `StockPortfolioEnv.log_attention_weights()` is called   |
| **Ablation script crashes**        | Check `config.yaml` ticker list matches attention matrix shape |

**Last updated**: April 2026  
**Python version**: 3.11+  
**PyTorch version**: 2.0+  
**Stable-Baselines3 version**: 2.0+
