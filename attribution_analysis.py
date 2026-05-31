#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attribution Analyzer - Integrated Gradients Attribution (logits, all-N baseline)
X-axis shows position numbers (no bases displayed), no specific position labels
Supports random selection of specified number of sequences from FASTA, with automatic filtering
of sequences exceeding length threshold
"""

import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from Bio import SeqIO
import argparse
import warnings
import random

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

warnings.filterwarnings("ignore")

FONT_SIZE = 14

# Import your model (make sure model.py is in the same directory)
from model import NucleotideMambaModel, DualPathMambaBlock

# ========================== Analyzer Class ==========================
class AttributionAnalyzer:
    def __init__(self, model, output_dir="attribution_results", label_interval=10):
        self.model = model
        self.device = next(model.parameters()).device
        self.output_dir = output_dir
        self.label_interval = label_interval
        os.makedirs(output_dir, exist_ok=True)
        self.max_seq_len = getattr(model, 'max_seq_len', 2048)
        print(f"🔧 Attribution Analyzer (Integrated Gradients Attribution)")
        print(f"  Device: {self.device}")
        print(f"  Model max sequence length: {self.max_seq_len}")
        print(f"  X-axis position display interval: {label_interval} bp")
        print(f"  Note: Sequences exceeding {self.max_seq_len} bp will be truncated to first {self.max_seq_len} bases")

    # ---------- Helper: Predict probability ----------
    def _predict_prob(self, sequence: str) -> float:
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
        with torch.no_grad():
            class_pred, _, _, _ = self.model.forward_sequence([sequence], training_mode=False)
            prob = torch.sigmoid(class_pred[0]).item()
            return prob

    # ---------- Integrated Gradients Attribution (logits, all-N baseline) ----------
    def _forward_to_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = self.model.embedding_projection(embeddings)
        for block in self.model.mamba_blocks:
            if isinstance(block, DualPathMambaBlock):
                x, _, _ = block(x, variant_type=None, training_mode=False)
            else:
                x = block(x)
        pooled = self.model.sequence_pooler(x.transpose(1, 2))
        pooled = self.model.output_norm(pooled)
        logits = self.model.classifier(pooled)
        return logits.squeeze()

    def integrated_gradients_logits(self, sequence: str, steps: int = 150) -> np.ndarray:
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
        L = len(sequence)

        with torch.no_grad():
            emb_actual = self.model.nucleotide_embedding([sequence], training_mode=False)
            if emb_actual.dim() == 2:
                emb_actual = emb_actual.unsqueeze(0)
            emb_actual = emb_actual[:, :L, :].contiguous().to(self.device)

        baseline_seq = 'N' * L
        with torch.no_grad():
            emb_baseline = self.model.nucleotide_embedding([baseline_seq], training_mode=False)
            if emb_baseline.dim() == 2:
                emb_baseline = emb_baseline.unsqueeze(0)
            emb_baseline = emb_baseline[:, :L, :].contiguous().to(self.device)

        total_grad = torch.zeros_like(emb_actual)
        for alpha in np.linspace(0, 1, steps):
            interp = emb_baseline + alpha * (emb_actual - emb_baseline)
            interp.requires_grad_(True)
            logits = self._forward_to_logits(interp)
            self.model.zero_grad()
            logits.backward()
            grad = interp.grad.detach().clone()
            total_grad += grad
            interp.grad = None

        avg_grad = total_grad / steps
        attr = (emb_actual - emb_baseline) * avg_grad
        attr_sum = attr.squeeze(0).sum(dim=-1).cpu().numpy()

        # Outlier smoothing
        vals, counts = np.unique(attr_sum, return_counts=True)
        for v, c in zip(vals, counts):
            if v != 0 and c / len(attr_sum) > 0.05:
                mask = attr_sum == v
                indices = np.where(mask)[0]
                for idx in indices:
                    if 0 < idx < len(attr_sum)-1:
                        attr_sum[idx] = (attr_sum[idx-1] + attr_sum[idx+1]) / 2
                    elif idx == 0:
                        attr_sum[idx] = attr_sum[idx+1]
                    else:
                        attr_sum[idx] = attr_sum[idx-1]
                print(f"  Fixed outlier value {v:.6f}, total {c} occurrences")

        # Dynamic clipping of extreme values
        clip_limit = np.percentile(np.abs(attr_sum), 99) if len(attr_sum) > 0 else 0.5
        if clip_limit > 0.1:
            attr_sum = np.clip(attr_sum, -clip_limit, clip_limit)
        return attr_sum

    # ---------- Plotting and Saving ----------
    def _save_plot(self, importance, sequence, name, base_pred, suffix, color, ylabel):
        L = len(importance)
        fig_width = min(20, max(10, L * 0.05))
        plt.figure(figsize=(fig_width, 5))
        
        # ========== Modification: Hide top and right spines ==========
        ax = plt.gca()
        ax.spines['top'].set_visible(False)      # Hide top spine
        ax.spines['right'].set_visible(False)    # Hide right spine
        ax.spines['bottom'].set_linewidth(1.5)   # Bottom spine width
        ax.spines['left'].set_linewidth(1.5)     # Left spine width
        
        # ========== X-axis tick interval fixed at 25 ==========
        # Generate tick positions starting from 0 with interval 25
        ticks_positions = list(range(25, L, 25))
        # Corresponding labels (position numbers starting from 1)
        ticks_labels = [str(i) for i in ticks_positions]
        
        plt.bar(range(L), importance, width=1.0, color=color, alpha=0.7)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.xlabel('Position (bp)', fontsize=FONT_SIZE)
        plt.ylabel(ylabel, fontsize=FONT_SIZE)
        plt.title(f'{name} | Pred={base_pred:.4f}', fontsize=FONT_SIZE)
        # plt.grid(axis='y', alpha=0.3)
        
        # ========== Modification: Set Y-axis ticks to 2 decimal places ==========
        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # ========== Modification: Set axis tick width and label font size ==========
        ax.tick_params(axis='both', which='major', width=1.5, labelsize=FONT_SIZE)
        ax.tick_params(axis='both', which='minor', width=1.5, labelsize=FONT_SIZE)

        # X-axis tick settings
        plt.xticks(ticks=ticks_positions, labels=ticks_labels, rotation=0, ha='center', fontsize=FONT_SIZE)
        plt.subplots_adjust(bottom=0.1)
        
        pos_vals = importance[importance > 0]
        neg_vals = importance[importance < 0]
        
        if len(pos_vals) > 0:
            thresh_pos = np.percentile(pos_vals, 95)
            plt.axhline(thresh_pos, color='green', linestyle='--', alpha=0.5, label=f'95th pos ({thresh_pos:.4f})')
        if len(neg_vals) > 0:
            thresh_neg = np.percentile(neg_vals, 5)
            plt.axhline(thresh_neg, color='red', linestyle='--', alpha=0.5, label=f'5th neg ({thresh_neg:.4f})')
        
        plt.legend(fontsize=FONT_SIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{name}_{suffix}.png"), dpi=300)
        plt.close()

        high_indices = []
        low_indices = []
        if len(pos_vals) > 0:
            thresh_pos = np.percentile(pos_vals, 95)
            high_indices = np.where(importance > thresh_pos)[0].tolist()
        if len(neg_vals) > 0:
            thresh_neg = np.percentile(neg_vals, 5)
            low_indices = np.where(importance < thresh_neg)[0].tolist()

        json_path = os.path.join(self.output_dir, f"{name}_{suffix}.json")
        with open(json_path, 'w') as f:
            json.dump({
                'sequence_name': name,
                'sequence': sequence,
                'base_prediction': base_pred,
                'importance': importance.tolist(),
                'method': suffix,
                'effective_length': L,
                'high_importance_positions': high_indices,
                'low_importance_positions': low_indices
            }, f, indent=2)

        print(f"  {suffix} plot saved to: {self.output_dir}/{name}_{suffix}.png (length {L})")
        top5 = np.argsort(np.abs(importance))[-5:][::-1]
        print(f"    Top 5 important positions: {top5} | Bases: {[sequence[i] for i in top5]} | Values: {importance[top5]}")
        if high_indices:
            print(f"    Positions above 95th percentile: {high_indices} | Values: {importance[high_indices]}")
        if low_indices:
            print(f"    Positions below 5th percentile: {low_indices} | Values: {importance[low_indices]}")

    def analyze_sequence(self, sequence: str, name: str):
        print(f"\n{'='*70}\nSequence: {name}\nOriginal length: {len(sequence)} bp")
        if len(sequence) > self.max_seq_len:
            print(f"  Note: Sequence exceeds model max length {self.max_seq_len}, will be truncated to first {self.max_seq_len} bases")
            sequence = sequence[:self.max_seq_len]
        base = self._predict_prob(sequence)
        print(f"  Predicted probability: {base:.4f}")

        imp_ig = self.integrated_gradients_logits(sequence, steps=150)
        self._save_plot(imp_ig, sequence, name, base, "integrated_gradients_logits", color='steelblue', ylabel='Importance (IG on logits)')
        return base

# ========================== Model Loading ==========================
def load_model(model_path: str, freeze_transformer: bool = False, override_max_len: int = 2048):
    print(f"📥 Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint.get('model_config', {})
    if not model_config:
        train_config = checkpoint.get('config', {})
        if train_config:
            model_config = {
                'transformer_model_repo': train_config.get('transformer_model_repo', 'InstaDeepAI/NTv3_8M_pre'),
                'embedding_dim': train_config.get('embedding_dim', 256),
                'd_model': train_config.get('d_model', 256),
                'n_layer': train_config.get('n_layer', 2),
                'projection_dim': train_config.get('projection_dim', 128),
                'num_classes': train_config.get('num_classes', 1),
                'block_type': train_config.get('block_type', 'dual_path'),
                'dropout_rate': train_config.get('dropout_rate', 0.1),
                'use_path_selection': train_config.get('use_path_selection', True),
                'path_selection_weight': train_config.get('path_selection_weight', 0.1),
                'max_seq_len': override_max_len,
                'use_local_global_attn': train_config.get('use_local_global_attn', True),
                'use_global_invariance': train_config.get('use_global_invariance', True),
                'attn_num_heads': train_config.get('attn_num_heads', 4),
                'use_flash_attention': train_config.get('use_flash_attention', True),
                'use_adaptive_scales': train_config.get('use_adaptive_scales', True),
                'freeze_transformer': freeze_transformer,
                'use_caching': False,
                'trust_remote_code': True,
            }
    else:
        model_config['max_seq_len'] = override_max_len

    model = NucleotideMambaModel(**model_config)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    try:
        model.load_state_dict(state_dict, strict=False)
        print("  ✅ Model weights loaded successfully (non-strict mode)")
    except Exception as e:
        print(f"  ⚠️ Weight loading warning: {e}")
    if not freeze_transformer:
        for param in model.parameters():
            param.requires_grad = True
        print("  🔓 All model parameters unfrozen")
    model.eval()
    return model

# ========================== Main Function ==========================
def main():
    parser = argparse.ArgumentParser(description="Attribution Analysis - Integrated Gradients only, X-axis shows position numbers, no base labels, no position annotations")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path (.pt)")
    parser.add_argument("--fasta", type=str, required=True, help="FASTA file path")
    parser.add_argument("--num-sequences", type=int, default=100, help="Number of randomly selected sequences (default: 100)")
    parser.add_argument("--output-dir", type=str, default="attribution_results", help="Output directory")
    parser.add_argument("--freeze-transformer", action="store_true", help="Freeze Transformer part")
    parser.add_argument("--max-len", type=int, default=2048, help="Model maximum sequence length (default: 2048)")
    parser.add_argument("--label-interval", type=int, default=10, help="X-axis position number display interval (e.g., 10 means show a number every 10 positions, default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--max-length-filter", type=int, default=2048, help="Only analyze sequences with length <= this value (default: 500 bp)")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("\n🚀 Starting attribution analysis (Integrated Gradients only, X-axis shows position numbers, no base labels)")
    print(f"🎲 Random seed: {args.seed}")
    print(f"✂️  Sequence length filter threshold: ≤ {args.max_length_filter} bp")

    model = load_model(args.model, freeze_transformer=args.freeze_transformer, override_max_len=args.max_len)

    # Read all sequences from FASTA
    print(f"\n📖 Loading FASTA: {args.fasta}")
    all_records = list(SeqIO.parse(args.fasta, "fasta"))
    total = len(all_records)
    print(f"  Total sequences in file: {total}")

    # Filter by length
    filtered_records = [r for r in all_records if len(r.seq) <= args.max_length_filter]
    filtered_total = len(filtered_records)
    print(f"  Sequences with length ≤ {args.max_length_filter} bp: {filtered_total}")

    if filtered_total == 0:
        print(f"❌ Error: No sequences meet the criteria (length ≤ {args.max_length_filter} bp). Please adjust --max-length-filter parameter.")
        return

    # Random sampling
    n_sample = min(args.num_sequences, filtered_total)
    if n_sample < args.num_sequences:
        print(f"  ⚠️ Requested {args.num_sequences} sequences, but only {filtered_total} meet the criteria. Using all available.")
    selected_records = random.sample(filtered_records, n_sample)

    sequences = [str(record.seq).upper() for record in selected_records]
    names = [record.id for record in selected_records]

    print(f"  Selected {len(sequences)} sequences for analysis:")
    for i, name in enumerate(names):
        print(f"    [{i+1}] {name} ({len(sequences[i])} bp)")

    analyzer = AttributionAnalyzer(model, output_dir=args.output_dir, label_interval=args.label_interval)

    # Batch analysis with progress bar
    if HAS_TQDM:
        iterator = tqdm(zip(sequences, names), total=len(sequences), desc="Analysis progress")
    else:
        iterator = zip(sequences, names)
        print(f"\nStarting analysis of {len(sequences)} sequences...")

    for seq, name in iterator:
        try:
            analyzer.analyze_sequence(seq, name)
        except Exception as e:
            print(f"\n  ❌ Analysis failed for sequence {name}: {e}")
            if not HAS_TQDM:
                import traceback
                traceback.print_exc()

    print(f"\n✅ Batch analysis completed! Results saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
