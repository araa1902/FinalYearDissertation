import torch
import torch.nn.functional as F
from src.utils.config_manager import load_config

class DenseGNNExplainer:
    """
    Exact Edge-Ablation Explainer for Constructed graph
    systematically ablating (deleting) edges one-by-one to measure the exact causal impact on the latent embedding.
    """
    
    def __init__(self, feature_extractor, config=None, device='cpu'):
        self.config = config if config is not None else load_config()
        self.feature_extractor = feature_extractor.to(device)
        self.feature_extractor.eval()
        self.device = device
        
        self.mask_threshold = self.config['explainability'].get('mask_threshold', 0.8)
        print(f"Initialised. Mask Threshold: {self.mask_threshold}")
    
    def explain(self, x, raw_adj, target_node_idx, tickers_list=None):
        num_tickers = x.shape[1]
        
        if tickers_list is None:
            tickers_list = [f"Node_{i}" for i in range(num_tickers)]
        
        # 1. Replicate exact Sector Fusion preprocessing
        with torch.no_grad():
            sector_weight = torch.sigmoid(self.feature_extractor.sector_weight)
            adj = (1 - sector_weight) * raw_adj + (sector_weight * self.feature_extractor.sector_mask)
            
            # Apply your hard threshold
            adj = torch.abs(adj)
            mask = (adj >= self.feature_extractor.threshold).float()
            processed_adj = adj * mask
            
            # Add self-loops
            identity = torch.eye(num_tickers, device=self.device).unsqueeze(0)
            true_adj = torch.max(processed_adj, identity)
            
            # 2. Get the baseline (original) embedding for the target node
            z_baseline_full, _ = self.feature_extractor.gat(x, true_adj)
            z_baseline = z_baseline_full[0, target_node_idx].clone()
            
        # 3. Exact Ablation Loop
        print(f"\n[Ablation] Testing {num_tickers - 1} incoming edges for {tickers_list[target_node_idx]}...")
        edge_impacts = []
        
        for src in range(num_tickers):
            if src == target_node_idx:
                continue # Ignore self-loop
                
            original_weight = true_adj[0, src, target_node_idx].item()
            
            if original_weight > 0:
                # Ablate (delete) the edge
                ablated_adj = true_adj.clone()
                ablated_adj[0, src, target_node_idx] = 0.0
                
                with torch.no_grad():
                    z_ablated, _ = self.feature_extractor.gat(x, ablated_adj)
                    z_ablated_target = z_ablated[0, target_node_idx]
                    
                # Measure EXACT Fidelity Drop (Cosine Distance)
                cosine_sim = F.cosine_similarity(z_baseline.unsqueeze(0), z_ablated_target.unsqueeze(0)).item()
                fidelity_drop = 1.0 - cosine_sim
                
                edge_impacts.append({
                    'source': src,
                    'target': target_node_idx,
                    'weight': original_weight,
                    'raw_fidelity_drop': fidelity_drop,
                    'source_ticker': tickers_list[src],
                    'target_ticker': tickers_list[target_node_idx]
                })

        # 4. Normalise scores to [0, 1] for thresholding (WITH NOISE FILTER)
        max_drop = max([e['raw_fidelity_drop'] for e in edge_impacts]) if edge_impacts else 0
        
        # Absolute significance threshold to prevent "best of a bad bunch" inflation
        MIN_SIGNIFICANCE = self.config['explainability'].get('min_significance', 0.001)
        
        for e in edge_impacts:
            if max_drop > MIN_SIGNIFICANCE:
                # Normalise relative to the most impactful edge
                e['score'] = e['raw_fidelity_drop'] / max_drop
            else:
                e['score'] = 0.0

        # Sort and filter by your config threshold
        edge_impacts.sort(key=lambda e: e['score'], reverse=True)
        important_edges = [e for e in edge_impacts if e['score'] >= self.mask_threshold]
        
        # 5. Subgraph-Level Fidelity Check (Simultaneous Ablation of non-important edges)
        subgraph_adj = true_adj.clone()
        non_important_srcs = {e['source'] for e in edge_impacts if e['score'] < self.mask_threshold}
        for src in non_important_srcs:
            subgraph_adj[0, src, target_node_idx] = 0.0
            
        with torch.no_grad():
            z_subgraph_full, _ = self.feature_extractor.gat(x, subgraph_adj)
            z_subgraph_target = z_subgraph_full[0, target_node_idx]
            
        subgraph_cosine_sim = F.cosine_similarity(z_baseline.unsqueeze(0), z_subgraph_target.unsqueeze(0)).item()
        subgraph_fidelity_drop = 1.0 - subgraph_cosine_sim

        # 6. Calculate Final Metrics (Sparsity of the causal subgraph)
        active_edges = len(important_edges)
        total_possible = num_tickers - 1
        sparsity = 1.0 - (active_edges / total_possible)
        
        print(f"\n--- True Causal Edges to {tickers_list[target_node_idx]} ---")
        for edge in important_edges:
            print(f"  {edge['source_ticker']} -> {edge['target_ticker']} | Normalised Impact: {edge['score']:.4f} (Raw Drop: {edge['raw_fidelity_drop']:.6f})")
            
        print(f"\n[Metrics] Sparsity: {sparsity:.4f}")
        print(f"[Metrics] Subgraph Fidelity Drop: {subgraph_fidelity_drop:.6f}")
        
        explanation_dict = {
            'sparsity': sparsity,
            'max_single_edge_fidelity_drop': max_drop, 
            'subgraph_fidelity_drop': subgraph_fidelity_drop, 
            'important_edges': important_edges
        }
        
        # Return none for mask as we are using direct edge lists now
        return None, important_edges, explanation_dict
