#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
归因分析器 - 积分梯度归因（logits，全N基线）
横坐标显示位置编号（不显示碱基），不标注具体位置
支持从 FASTA 中随机挑选指定数量的序列进行分析，并自动过滤长度超过阈值的序列
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

# 可选进度条
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

warnings.filterwarnings("ignore")

FONT_SIZE = 14

# 导入您的模型（请确保 model.py 在同一目录）
from model import NucleotideMambaModel, DualPathMambaBlock

# ========================== 分析器类 ==========================
class AttributionAnalyzer:
    def __init__(self, model, output_dir="attribution_results", label_interval=10):
        self.model = model
        self.device = next(model.parameters()).device
        self.output_dir = output_dir
        self.label_interval = label_interval
        os.makedirs(output_dir, exist_ok=True)
        self.max_seq_len = getattr(model, 'max_seq_len', 2048)
        print(f"🔧 归因分析器 (积分梯度归因)")
        print(f"  设备: {self.device}")
        print(f"  模型最大序列长度: {self.max_seq_len}")
        print(f"  横坐标位置显示间隔: {label_interval} bp")
        print(f"  注意: 超过 {self.max_seq_len} bp 的序列将被截断为前 {self.max_seq_len} 个碱基")

    # ---------- 辅助：预测概率 ----------
    def _predict_prob(self, sequence: str) -> float:
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
        with torch.no_grad():
            class_pred, _, _, _ = self.model.forward_sequence([sequence], training_mode=False)
            prob = torch.sigmoid(class_pred[0]).item()
            return prob

    # ---------- 积分梯度归因（logits，全N基线）----------
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

        # 异常值平滑
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
                print(f"  已修复异常值 {v:.6f}，共 {c} 处")

        # 动态裁剪极端值
        clip_limit = np.percentile(np.abs(attr_sum), 99) if len(attr_sum) > 0 else 0.5
        if clip_limit > 0.1:
            attr_sum = np.clip(attr_sum, -clip_limit, clip_limit)
        return attr_sum

    # ---------- 绘图与保存 ----------
    def _save_plot(self, importance, sequence, name, base_pred, suffix, color, ylabel):
        L = len(importance)
        fig_width = min(20, max(10, L * 0.05))
        plt.figure(figsize=(fig_width, 5))
        
        # ========== 修改：隐藏右端和顶部的框线 ==========
        ax = plt.gca()
        ax.spines['top'].set_visible(False)      # 隐藏顶部框线
        ax.spines['right'].set_visible(False)    # 隐藏右侧框线
        ax.spines['bottom'].set_linewidth(1.5)   # 底部框线宽度
        ax.spines['left'].set_linewidth(1.5)     # 左侧框线宽度
        
        # ========== X轴刻度间隔固定为25 ==========
        # 生成从0开始，间隔为25的刻度位置
        ticks_positions = list(range(25, L, 25))
        # 对应的标签（位置编号从1开始）
        ticks_labels = [str(i) for i in ticks_positions]
        
        plt.bar(range(L), importance, width=1.0, color=color, alpha=0.7)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.xlabel('Position (bp)', fontsize=FONT_SIZE)
        plt.ylabel(ylabel, fontsize=FONT_SIZE)
        plt.title(f'{name} | Pred={base_pred:.4f}', fontsize=FONT_SIZE)
        # plt.grid(axis='y', alpha=0.3)
        
        # ========== 修改：设置Y轴刻度小数点后两位 ==========
        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # ========== 修改：设置坐标轴刻度线宽度和标签字体大小 ==========
        ax.tick_params(axis='both', which='major', width=1.5, labelsize=FONT_SIZE)
        ax.tick_params(axis='both', which='minor', width=1.5, labelsize=FONT_SIZE)

        # X轴刻度设置
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

        print(f"  {suffix} 图保存至: {self.output_dir}/{name}_{suffix}.png (长度 {L})")
        top5 = np.argsort(np.abs(importance))[-5:][::-1]
        print(f"    最重要位置 (前5): {top5} | 碱基: {[sequence[i] for i in top5]} | 值: {importance[top5]}")
        if high_indices:
            print(f"    超过95th pos的位置: {high_indices} | 值: {importance[high_indices]}")
        if low_indices:
            print(f"    低于5th neg的位置: {low_indices} | 值: {importance[low_indices]}")

    def analyze_sequence(self, sequence: str, name: str):
        print(f"\n{'='*70}\n序列: {name}\n原始长度: {len(sequence)} bp")
        if len(sequence) > self.max_seq_len:
            print(f"  注意: 序列超过模型最大长度 {self.max_seq_len}，将被截断为前 {self.max_seq_len} 个碱基")
            sequence = sequence[:self.max_seq_len]
        base = self._predict_prob(sequence)
        print(f"  预测概率: {base:.4f}")

        imp_ig = self.integrated_gradients_logits(sequence, steps=150)
        self._save_plot(imp_ig, sequence, name, base, "integrated_gradients_logits", color='steelblue', ylabel='Importance (IG on logits)')
        return base

# ========================== 模型加载 ==========================
def load_model(model_path: str, freeze_transformer: bool = False, override_max_len: int = 2048):
    print(f"📥 加载模型: {model_path}")
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
        print("  ✅ 模型权重加载成功（非严格模式）")
    except Exception as e:
        print(f"  ⚠️ 权重加载警告: {e}")
    if not freeze_transformer:
        for param in model.parameters():
            param.requires_grad = True
        print("  🔓 已解冻所有模型参数")
    model.eval()
    return model

# ========================== 主函数 ==========================
def main():
    parser = argparse.ArgumentParser(description="归因分析 - 仅积分梯度归因，横坐标显示位置编号，不显示碱基标签，不标注重要位置")
    parser.add_argument("--model", type=str, required=True, help="模型检查点路径 (.pt)")
    parser.add_argument("--fasta", type=str, required=True, help="FASTA文件路径")
    parser.add_argument("--num-sequences", type=int, default=100, help="随机选取的序列数量（默认100）")
    parser.add_argument("--output-dir", type=str, default="attribution_results", help="输出目录")
    parser.add_argument("--freeze-transformer", action="store_true", help="冻结Transformer部分")
    parser.add_argument("--max-len", type=int, default=2048, help="模型最大序列长度（默认2048）")
    parser.add_argument("--label-interval", type=int, default=10, help="横坐标位置编号显示间隔（例如10表示每隔10个位置显示一个数字，默认10）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于可复现的抽样（默认42）")
    parser.add_argument("--max-length-filter", type=int, default=2048, help="只分析长度 <= 该值的序列（默认500 bp）")
    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("\n🚀 启动归因分析（仅积分梯度归因，横坐标显示位置编号，不显示碱基标签）")
    print(f"🎲 随机种子: {args.seed}")
    print(f"✂️  序列长度过滤阈值: ≤ {args.max_length_filter} bp")

    model = load_model(args.model, freeze_transformer=args.freeze_transformer, override_max_len=args.max_len)

    # 读取 FASTA 中的所有序列
    print(f"\n📖 加载FASTA: {args.fasta}")
    all_records = list(SeqIO.parse(args.fasta, "fasta"))
    total = len(all_records)
    print(f"  文件中总序列数: {total}")

    # 过滤长度
    filtered_records = [r for r in all_records if len(r.seq) <= args.max_length_filter]
    filtered_total = len(filtered_records)
    print(f"  长度 ≤ {args.max_length_filter} bp 的序列数: {filtered_total}")

    if filtered_total == 0:
        print(f"❌ 错误: 没有符合条件的序列（长度 ≤ {args.max_length_filter} bp），请调整 --max-length-filter 参数。")
        return

    # 随机抽取
    n_sample = min(args.num_sequences, filtered_total)
    if n_sample < args.num_sequences:
        print(f"  ⚠️ 请求 {args.num_sequences} 条序列，但符合条件的仅有 {filtered_total} 条，将全部使用")
    selected_records = random.sample(filtered_records, n_sample)

    sequences = [str(record.seq).upper() for record in selected_records]
    names = [record.id for record in selected_records]

    print(f"  最终选取 {len(sequences)} 条序列进行分析:")
    for i, name in enumerate(names):
        print(f"    [{i+1}] {name} ({len(sequences[i])} bp)")

    analyzer = AttributionAnalyzer(model, output_dir=args.output_dir, label_interval=args.label_interval)

    # 批量分析，带进度条
    if HAS_TQDM:
        iterator = tqdm(zip(sequences, names), total=len(sequences), desc="分析进度")
    else:
        iterator = zip(sequences, names)
        print(f"\n开始分析 {len(sequences)} 条序列...")

    for seq, name in iterator:
        try:
            analyzer.analyze_sequence(seq, name)
        except Exception as e:
            print(f"\n  ❌ 序列 {name} 分析失败: {e}")
            if not HAS_TQDM:
                import traceback
                traceback.print_exc()

    print(f"\n✅ 批量分析完成！结果保存在: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
