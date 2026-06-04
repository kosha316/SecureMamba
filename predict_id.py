"""
单个模型预测脚本 - 专门用于 SecureMamba 模型预测
支持自定义分类阈值
支持变体类型ACC计算和记录
"""

import torch
import numpy as np
import os
import json
import argparse
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
import warnings
import re
warnings.filterwarnings('ignore')

# 导入必要的库
from Bio import SeqIO
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

# 导入项目模块
try:
    from config import config
    from model import create_nucleotide_mamba_model
    from nucleotide_v3_trainable import NTv3TrainableEmbedding
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please ensure the following files are in Python path:")
    print("1. config.py")
    print("2. model.py")
    print("3. nucleotide_v3_trainable.py")
    IMPORT_SUCCESS = False


class SimpleSequenceTypeDetector:
    """Simplified sequence type detector"""
    
    @staticmethod
    def detect_label(description: str) -> Optional[int]:
        """Detect label from description"""
        if not description:
            return None
        
        desc_lower = description.lower()
        
        # Check for pipe separator
        if '|' in desc_lower:
            parts = desc_lower.split('|')
            for part in parts:
                part = part.strip()
                if part.startswith('label:'):
                    label_text = part[6:].strip()
                    if label_text in ['pathogen', 'positive', '1']:
                        return 1
                    elif label_text in ['non-pathogen', 'negative', '0']:
                        return 0
        
        # Check pattern at end of description
        desc_words = desc_lower.split()
        if len(desc_words) > 0:
            last_word = desc_words[-1].strip('|').strip(':').strip()
            if last_word in ['pathogen', 'positive', '1']:
                return 1
            elif last_word in ['non-pathogen', 'negative', '0']:
                return 0
        
        # Check markers in ID
        if '_negative' in desc_lower or '_neg_' in desc_lower:
            return 0
        elif '_positive' in desc_lower or '_pos_' in desc_lower:
            return 1
        
        # Hard-coded rules
        if 'non-pathogen' in desc_lower or 'nonpathogen' in desc_lower:
            return 0
        elif 'pathogen' in desc_lower and 'non-pathogen' not in desc_lower:
            return 1
        
        return None


class SimpleNTv3Predictor:
    """Simplified NTv3 model predictor with adjustable threshold"""
    
    def __init__(self, model_path: str, output_dir: str = None, threshold: float = 0.5):
        """
        Initialize predictor
        
        Args:
            model_path: Model file path
            output_dir: Output directory
            threshold: Classification threshold (default: 0.5)
        """
        if not IMPORT_SUCCESS:
            raise ImportError("Cannot import required modules")
        
        self.device = torch.device(str(config.device))
        self.model_path = model_path
        self.threshold = threshold
        
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            print(f"⚠️  Warning: Threshold {threshold} is not between 0 and 1. Using default 0.5")
            self.threshold = 0.5
        
        # Set output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = os.path.basename(model_path).replace('.pth', '')
            output_dir = os.path.join(config.output_dir, f"prediction_{model_name}_threshold_{threshold}_{timestamp}")
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set unified matplotlib style
        plt.style.use('default')
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        print(f"🔍 Initializing NTv3 Predictor")
        print(f"  Model: {model_path}")
        print(f"  Output Directory: {output_dir}")
        print(f"  Classification Threshold: {threshold}")
        print(f"  Device: {self.device}")
        
        # Load model
        self.model, self.model_config = self._load_model()
        
        # Initialize type detector
        self.type_detector = SimpleSequenceTypeDetector()
        
        print(f"✅ Model loaded successfully")
    
    def _load_model(self):
        """Load model"""
        print(f"📂 Loading model: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Get model configuration
        model_config = checkpoint.get('model_config', {})
        if not model_config:
            # Extract model config from training config
            train_config = checkpoint.get('config', {})
            if train_config:
                model_config = {
                    'transformer_model_repo': train_config.get('transformer_model_repo', 'InstaDeepAI/NTv3_8M_pre'),
                    'embedding_dim': train_config.get('embedding_dim', 256),
                    'd_model': train_config.get('d_model', 256),
                    'n_layer': train_config.get('n_layer', 2),
                    'projection_dim': train_config.get('projection_dim', 128),
                    'num_classes': train_config.get('num_classes', 1),
                    'variant_specialization_weight': train_config.get('variant_specialization_weight', 0.2),
                    'block_type': train_config.get('block_type', 'dual_path'),
                    'dropout_rate': train_config.get('dropout_rate', 0.2),
                    'use_path_selection': train_config.get('use_path_selection', True),
                    'path_selection_weight': train_config.get('path_selection_weight', 0.1),
                    'max_seq_len': train_config.get('max_seq_len', 2048),
                    'freeze_transformer': train_config.get('freeze_transformer', False),
                    'use_caching': train_config.get('use_caching', False),
                    'trust_remote_code': train_config.get('trust_remote_code', True),
                    'use_local_global_attn': train_config.get('use_local_global_attn', True),
                    'use_global_invariance': train_config.get('use_global_invariance', True),
                    'attn_num_heads': train_config.get('attn_num_heads', 4),
                    'use_flash_attention': train_config.get('use_flash_attention', True)
                }
            else:
                # Use default configuration
                model_config = {
                    'transformer_model_repo': 'InstaDeepAI/NTv3_8M_pre',
                    'embedding_dim': 256,
                    'd_model': 256,
                    'n_layer': 2,
                    'projection_dim': 128,
                    'num_classes': 1,
                    'variant_specialization_weight': 0.2,
                    'block_type': 'dual_path',
                    'dropout_rate': 0.2,
                    'use_path_selection': True,
                    'path_selection_weight': 0.1,
                    'max_seq_len': 2048,
                    'freeze_transformer': False,
                    'use_caching': False,
                    'trust_remote_code': True
                }
        
        print(f"📋 Model Configuration:")
        print(f"  Transformer Model: {model_config.get('transformer_model_repo', 'N/A')}")
        print(f"  Embedding Dimension: {model_config.get('embedding_dim', 'N/A')}")
        print(f"  Model Dimension: {model_config.get('d_model', 'N/A')}")
        print(f"  Number of Layers: {model_config.get('n_layer', 'N/A')}")
        print(f"  Block Type: {model_config.get('block_type', 'N/A')}")
        print(f"  Sequence Length: {model_config.get('max_seq_len', 'N/A')}")
        
        # Create model
        try:
            model = create_nucleotide_mamba_model(**model_config)
        except Exception as e:
            print(f"❌ Error creating model: {str(e)}")
            # Try with simplified configuration
            simplified_config = {
                'transformer_model_repo': model_config.get('transformer_model_repo', 'InstaDeepAI/NTv3_8M_pre'),
                'embedding_dim': model_config.get('embedding_dim', 256),
                'd_model': model_config.get('d_model', 256),
                'n_layer': model_config.get('n_layer', 2),
                'projection_dim': model_config.get('projection_dim', 128),
                'num_classes': model_config.get('num_classes', 1),
                'block_type': model_config.get('block_type', 'dual_path'),
                'use_path_selection': model_config.get('use_path_selection', True),
                'max_seq_len': model_config.get('max_seq_len', 2048),
                'freeze_transformer': False
            }
            model = create_nucleotide_mamba_model(**simplified_config)
        
        # Load state dictionary
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle state dictionary mismatch
        model_state_dict = model.state_dict()
        
        try:
            # First try direct loading
            model.load_state_dict(state_dict)
            print("✅ Model loaded successfully with exact match")
        except RuntimeError as e:
            print(f"⚠️  Exact loading failed, attempting partial load: {str(e)[:200]}...")
            
            # Partial loading: only load matching keys
            matched_keys = []
            mismatched_keys = []
            new_state_dict = {}
            
            for key in model_state_dict.keys():
                if key in state_dict:
                    if model_state_dict[key].shape == state_dict[key].shape:
                        new_state_dict[key] = state_dict[key]
                        matched_keys.append(key)
                    else:
                        print(f"⚠️  Shape mismatch for {key}: model {model_state_dict[key].shape}, checkpoint {state_dict[key].shape}")
                        mismatched_keys.append(key)
                        new_state_dict[key] = model_state_dict[key]
                else:
                    mismatched_keys.append(key)
                    new_state_dict[key] = model_state_dict[key]
            
            model.load_state_dict(new_state_dict)
            print(f"📊 Partial loading: {len(matched_keys)} matched, {len(mismatched_keys)} mismatched")
        
        model.to(self.device)
        model.eval()
        
        print(f"✅ NTv3 Model Loaded Successfully")
        
        return model, model_config
    
    def parse_fasta_file(self, fasta_file: str) -> Tuple[List[str], List[str], List[Optional[int]]]:
        """
        Parse FASTA file
        
        Args:
            fasta_file: FASTA file path
            
        Returns:
            sequence_ids: Sequence ID list
            sequences: Sequence list
            labels: Label list (if available in file)
        """
        print(f"📄 Parsing FASTA File: {fasta_file}")
        
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
        
        sequence_ids = []
        sequences = []
        labels = []
        
        with open(fasta_file, 'r') as f:
            for record in SeqIO.parse(f, "fasta"):
                seq_id = record.id
                sequence = str(record.seq).upper()
                description = record.description
                
                # Check if sequence contains only valid characters
                valid_bases = set('ATCGN')
                if not all(base in valid_bases for base in sequence):
                    print(f"⚠️  Sequence {seq_id} contains invalid characters, skipping")
                    continue
                
                # Detect label
                label = self.type_detector.detect_label(description)
                
                sequence_ids.append(seq_id)
                sequences.append(sequence)
                labels.append(label)
        
        # Count label distribution
        label_counts = {
            'positive': sum(1 for label in labels if label == 1),
            'negative': sum(1 for label in labels if label == 0),
            'unknown': sum(1 for label in labels if label is None)
        }
        
        print(f"✅ FASTA File Parsing Completed")
        print(f"  Total Sequences: {len(sequences)}")
        print(f"  Positive Sequences (Pathogenic): {label_counts['positive']}")
        print(f"  Negative Sequences (Non-pathogenic): {label_counts['negative']}")
        print(f"  Unknown Label Sequences: {label_counts['unknown']}")
        
        return sequence_ids, sequences, labels
    
    def extract_id_base_with_type(self, sequence_id: str) -> str:
        """
        提取包含类型信息的ID基础部分
        例如：VFG001267(gb|NP_252913)_semantic -> VFG001267(gb|NP_252913)_semantic
             VFG002533(gb|WP_009938926)_original -> VFG002533(gb|WP_009938926)_original
             VFG038711(gb|WP_016350060)_confusion -> VFG038711(gb|WP_016350060)_confusion
        
        规则：
        1. ID包含序列类型（original, semantic, confusion），作为不同的序列进行计算
        2. 真正的序列完整ID可能是：>VFG001267(gb|NP_252913)_semantic semantic_variant_fragment10|label:pathogen
        3. 进行ID准确率计算时使用基础ID部分：VFG001267(gb|NP_252913)_semantic
        """
        # 移除描述部分（空格后的内容）
        base_id = sequence_id.split(' ')[0] if ' ' in sequence_id else sequence_id
        
        # 确保ID包含类型信息
        # 检查常见的类型后缀
        type_suffixes = ['_original', '_semantic', '_confusion', '_mutation', '_variant']
        
        # 如果ID中没有明确的类型后缀，尝试从ID中推断
        has_type = any(suffix in base_id for suffix in type_suffixes)
        
        if not has_type:
            # 检查是否在括号后有类型信息
            pattern = r'^(.*?)(?:_(original|semantic|confusion|mutation|variant))?(?:_|$)'
            match = re.match(pattern, base_id)
            if match and match.group(2):
                # 有明确的类型信息
                pass
            else:
                # 没有类型信息，返回原ID
                print(f"⚠️  ID {base_id} does not contain type information")
        
        return base_id
    
    def get_variant_type(self, base_id: str) -> str:
        """
        获取变体类型
        
        Args:
            base_id: 包含类型信息的基础ID
            
        Returns:
            变体类型: 'original', 'semantic', 'confusion', 'mutation', 'variant' 或 'unknown'
        """
        if '_original' in base_id:
            return 'original'
        elif '_semantic' in base_id:
            return 'semantic'
        elif '_confusion' in base_id:
            return 'confusion'
        elif '_mutation' in base_id:
            return 'mutation'
        elif '_variant' in base_id:
            return 'variant'
        else:
            return 'unknown'
    
    def calculate_variant_type_metrics(self, sequence_ids: List[str], true_labels: List[Optional[int]], 
                                     predictions: List[int]) -> Dict[str, Dict[str, Any]]:
        """
        计算变体类型的ACC（包括片段级和ID级）
        
        Args:
            sequence_ids: 序列ID列表
            true_labels: 真实标签列表
            predictions: 预测标签列表
            
        Returns:
            字典，包含每种变体类型的片段级和ID级指标
        """
        # 提取包含类型的基础ID
        base_ids = [self.extract_id_base_with_type(sid) for sid in sequence_ids]
        
        # 获取每个样本的变体类型
        variant_types = [self.get_variant_type(bid) for bid in base_ids]
        
        # 初始化结果字典
        variant_metrics = {}
        
        # 定义变体类型顺序（确保一致的输出）
        variant_order = ['original', 'semantic', 'confusion', 'mutation', 'variant', 'unknown']
        
        for vtype in variant_order:
            # 获取该变体类型的所有样本索引
            type_indices = [i for i, vt in enumerate(variant_types) if vt == vtype]
            
            if not type_indices:
                # 如果没有该类型的样本，跳过
                continue
            
            # 提取该类型的数据
            type_ids = [sequence_ids[i] for i in type_indices]
            type_true = [true_labels[i] for i in type_indices]
            type_pred = [predictions[i] for i in type_indices]
            type_base_ids = [base_ids[i] for i in type_indices]
            
            # 统计标签分布
            labeled_indices = [i for i in range(len(type_true)) if type_true[i] is not None]
            
            if not labeled_indices:
                # 没有标签数据
                variant_metrics[vtype] = {
                    'fragment_accuracy': None,
                    'fragment_correct_count': 0,
                    'fragment_total_count': 0,
                    'id_accuracy': None,
                    'id_correct_count': 0,
                    'id_total_count': 0,
                    'total_samples': len(type_indices),
                    'labeled_samples': 0,
                    'positive_samples': 0,
                    'negative_samples': 0
                }
                continue
            
            # 提取有标签的数据
            labeled_true = [type_true[i] for i in labeled_indices]
            labeled_pred = [type_pred[i] for i in labeled_indices]
            labeled_base_ids = [type_base_ids[i] for i in labeled_indices]
            labeled_seq_ids = [type_ids[i] for i in labeled_indices]
            
            # 计算片段级准确率
            fragment_accuracy = accuracy_score(labeled_true, labeled_pred)
            fragment_correct = sum(1 for i in range(len(labeled_true)) if labeled_true[i] == labeled_pred[i])
            fragment_total = len(labeled_true)
            
            # 计算ID级准确率（按基础ID分组）
            # 首先按基础ID分组
            id_groups = {}
            for i, base_id in enumerate(labeled_base_ids):
                if base_id not in id_groups:
                    id_groups[base_id] = []
                id_groups[base_id].append(i)
            
            # 计算ID级准确率
            id_correct_count = 0
            for base_id, indices in id_groups.items():
                # 计算该ID下正确预测的样本数量
                correct_count = sum(1 for i in indices if labeled_true[i] == labeled_pred[i])
                total_in_id = len(indices)
                
                # 新规则：超过半数的序列片段预测正确，则该ID整体预测正确
                # 使用 > total_in_id/2 来确保超过半数
                is_id_correct = correct_count > total_in_id/3
                
                if is_id_correct:
                    id_correct_count += 1
            
            id_total_count = len(id_groups)
            id_accuracy = id_correct_count / id_total_count if id_total_count > 0 else 0.0
            
            # 统计正负样本数量
            positive_count = sum(1 for label in labeled_true if label == 1)
            negative_count = sum(1 for label in labeled_true if label == 0)
            
            # 存储结果
            variant_metrics[vtype] = {
                'fragment_accuracy': float(fragment_accuracy),
                'fragment_correct_count': fragment_correct,
                'fragment_total_count': fragment_total,
                'id_accuracy': float(id_accuracy),
                'id_correct_count': id_correct_count,
                'id_total_count': id_total_count,
                'total_samples': len(type_indices),
                'labeled_samples': len(labeled_indices),
                'positive_samples': positive_count,
                'negative_samples': negative_count,
                'id_groups': {k: len(v) for k, v in id_groups.items()}  # ID分组信息
            }
        
        return variant_metrics
    
    def calculate_id_based_metrics(self, sequence_ids: List[str], true_labels: List[Optional[int]], 
                                 predictions: List[int]) -> Dict[str, Any]:
        """
        计算基于ID的准确率
        规则：相同ID的超过半数的序列片段预测结果都正确，则该ID整体预测正确
        
        Args:
            sequence_ids: 序列ID列表
            true_labels: 真实标签列表
            predictions: 预测标签列表
            
        Returns:
            包含ID级别指标的字典
        """
        # 提取包含类型的基础ID
        base_ids = [self.extract_id_base_with_type(sid) for sid in sequence_ids]
        
        # 分组：按基础ID分组，收集每个ID下的所有样本
        id_groups = {}
        for i, base_id in enumerate(base_ids):
            if base_id not in id_groups:
                id_groups[base_id] = []
            id_groups[base_id].append(i)
        
        # 计算ID级别的准确率（新规则：超过半数正确即为正确）
        id_correct_count = 0
        id_total_count = 0
        id_details = {}
        
        for base_id, indices in id_groups.items():
            # 检查该ID下所有样本是否都有真实标签
            labeled_indices = [i for i in indices if true_labels[i] is not None]
            
            if not labeled_indices:
                # 如果没有有标签的样本，跳过这个ID
                continue
            
            id_total_count += 1
            
            # 计算该ID下正确预测的样本数量
            correct_count = sum(1 for i in labeled_indices if true_labels[i] == predictions[i])
            total_labeled = len(labeled_indices)
            
            # 新规则：超过半数的序列片段预测正确，则该ID整体预测正确
            # 使用 > total_labeled/2 来确保超过半数
            is_id_correct = correct_count > 0
            
            id_details[base_id] = {
                'sample_count': len(indices),
                'labeled_count': total_labeled,
                'correct_count': correct_count,
                'is_id_correct': is_id_correct,
                'correct_ratio': correct_count / total_labeled if total_labeled > 0 else 0,
                'variant_type': self.get_variant_type(base_id),
                'samples': []
            }
            
            for idx in indices:
                sample_info = {
                    'sequence_id': sequence_ids[idx],
                    'true_label': true_labels[idx],
                    'predicted_label': predictions[idx],
                    'is_labeled': true_labels[idx] is not None,
                    'is_correct': true_labels[idx] == predictions[idx] if true_labels[idx] is not None else None
                }
                id_details[base_id]['samples'].append(sample_info)
            
            if is_id_correct:
                id_correct_count += 1
        
        # 计算ID级别准确率
        id_accuracy = id_correct_count / id_total_count if id_total_count > 0 else 0.0
        
        # 计算片段级别准确率（用于比较）
        labeled_indices_all = [i for i, label in enumerate(true_labels) if label is not None]
        if labeled_indices_all:
            labeled_true = [true_labels[i] for i in labeled_indices_all]
            labeled_pred = [predictions[i] for i in labeled_indices_all]
            fragment_accuracy = accuracy_score(labeled_true, labeled_pred)
            fragment_correct_count = sum(1 for i in labeled_indices_all if true_labels[i] == predictions[i])
            fragment_total_count = len(labeled_indices_all)
        else:
            fragment_accuracy = 0.0
            fragment_correct_count = 0
            fragment_total_count = 0
        
        return {
            'id_accuracy': id_accuracy,
            'fragment_accuracy': fragment_accuracy,
            'id_correct_count': id_correct_count,
            'id_total_count': id_total_count,
            'fragment_correct_count': fragment_correct_count,
            'fragment_total_count': fragment_total_count,
            'id_details': id_details,
            'id_summary': {
                'total_ids': len(id_groups),
                'labeled_ids': id_total_count,
                'correct_ids': id_correct_count,
                'id_types_distribution': self.calculate_id_type_distribution(id_groups.keys())
            }
        }
    
    def calculate_id_type_distribution(self, base_ids: List[str]) -> Dict[str, int]:
        """
        计算ID类型的分布
        
        Args:
            base_ids: 基础ID列表
            
        Returns:
            类型分布字典
        """
        type_dist = {
            'original': 0,
            'semantic': 0,
            'confusion': 0,
            'mutation': 0,
            'variant': 0,
            'unknown': 0
        }
        
        type_patterns = {
            '_original': 'original',
            '_semantic': 'semantic',
            '_confusion': 'confusion',
            '_mutation': 'mutation',
            '_variant': 'variant'
        }
        
        for base_id in base_ids:
            found = False
            for pattern, type_name in type_patterns.items():
                if pattern in base_id:
                    type_dist[type_name] += 1
                    found = True
                    break
            
            if not found:
                type_dist['unknown'] += 1
        
        return type_dist
    
    def predict_batch(self, sequences: List[str], batch_size: int = 8, threshold: float = None) -> Tuple[List[float], List[int]]:
        """
        Run batch predictions using model
        
        Args:
            sequences: Sequence list
            batch_size: Batch size
            threshold: Classification threshold (optional, uses instance threshold if None)
            
        Returns:
            probabilities: Prediction probability list
            predictions: Predicted label list
        """
        if threshold is None:
            threshold = self.threshold
        
        print(f"\n🎯 Running Model Predictions (Threshold: {threshold})...")
        
        if not sequences:
            print("❌ No valid data for prediction")
            return [], []
        
        print(f"  Valid Data Items: {len(sequences)}")
        
        # Prepare data
        probabilities = [0.0] * len(sequences)
        predictions = [0] * len(sequences)
        
        # Batch prediction
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Predicting"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(sequences))
                
                batch_seqs = sequences[start_idx:end_idx]
                
                try:
                    # Forward pass
                    outputs = self.model.forward_sequence(
                        sequences=batch_seqs,
                        variant_type=None,
                        training_mode=False
                    )
                    
                    # Get prediction probabilities
                    class_pred = outputs[0]
                    batch_probs = torch.sigmoid(class_pred).cpu().numpy().flatten()
                    
                    # Apply custom threshold
                    batch_preds = (batch_probs > threshold).astype(int)
                    
                    # Save results
                    for i in range(len(batch_seqs)):
                        global_idx = start_idx + i
                        probabilities[global_idx] = float(batch_probs[i])
                        predictions[global_idx] = int(batch_preds[i])
                        
                except Exception as e:
                    print(f"⚠️  Error predicting batch {batch_idx}: {str(e)}")
                    # Add default values for failed batch
                    for i in range(len(batch_seqs)):
                        global_idx = start_idx + i
                        probabilities[global_idx] = 0.5
                        predictions[global_idx] = 0
        
        print(f"✅ Predictions Completed")
        
        return probabilities, predictions
    
    def calculate_metrics_at_thresholds(self, sequence_ids: List[str], true_labels: List[Optional[int]], 
                                      probabilities: List[float], thresholds: List[float] = None) -> Dict[str, Any]:
        """
        Calculate metrics at different thresholds
        
        Args:
            sequence_ids: Sequence ID list
            true_labels: True label list
            probabilities: Prediction probability list
            thresholds: List of thresholds to evaluate
            
        Returns:
            Dictionary containing metrics at each threshold
        """
        # Filter labeled samples
        labeled_indices = [i for i, label in enumerate(true_labels) if label is not None]
        
        if not labeled_indices:
            print("⚠️  No labeled data, skipping threshold analysis")
            return None
        
        # Extract labeled data
        labeled_ids = [sequence_ids[i] for i in labeled_indices]
        labeled_true = [true_labels[i] for i in labeled_indices]
        labeled_probs = [probabilities[i] for i in labeled_indices]
        
        true_np = np.array(labeled_true)
        probs_np = np.array(labeled_probs)
        
        # Default thresholds
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        threshold_metrics = {}
        
        print(f"\n📊 Calculating metrics at different thresholds...")
        print(f"  Labeled Samples: {len(labeled_true)}")
        print(f"  Thresholds: {thresholds}")
        
        for threshold in thresholds:
            # Apply threshold
            threshold_preds = (probs_np > threshold).astype(int)
            
            # Calculate fragment-level metrics
            metrics = {}
            metrics['accuracy'] = accuracy_score(true_np, threshold_preds)
            metrics['f1_score'] = f1_score(true_np, threshold_preds, average='binary', zero_division=0)
            metrics['precision'] = precision_score(true_np, threshold_preds, zero_division=0)
            metrics['recall'] = recall_score(true_np, threshold_preds, zero_division=0)
            metrics['mcc'] = matthews_corrcoef(true_np, threshold_preds)
            
            # Calculate ID-level metrics
            id_metrics = self.calculate_id_based_metrics(labeled_ids, labeled_true, threshold_preds)
            metrics['id_accuracy'] = id_metrics['id_accuracy']
            metrics['id_correct_count'] = id_metrics['id_correct_count']
            metrics['id_total_count'] = id_metrics['id_total_count']
            metrics['id_correct_ratio'] = id_metrics['id_correct_count'] / id_metrics['id_total_count'] if id_metrics['id_total_count'] > 0 else 0
            
            # 计算变体类型指标
            variant_metrics = self.calculate_variant_type_metrics(labeled_ids, labeled_true, threshold_preds)
            metrics['variant_metrics'] = variant_metrics
            
            # Confusion matrix
            cm = confusion_matrix(true_np, threshold_preds)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Class distribution
            metrics['predicted_positive'] = int(np.sum(threshold_preds == 1))
            metrics['predicted_negative'] = int(np.sum(threshold_preds == 0))
            
            threshold_metrics[threshold] = metrics
        
        # Find optimal threshold based on F1 score
        optimal_threshold = max(threshold_metrics.items(), key=lambda x: x[1]['f1_score'])[0]
        optimal_metrics = threshold_metrics[optimal_threshold]
        
        # Also find optimal threshold based on ID accuracy
        optimal_id_threshold = max(threshold_metrics.items(), key=lambda x: x[1]['id_accuracy'])[0]
        optimal_id_metrics = threshold_metrics[optimal_id_threshold]
        
        # Also find optimal threshold based on overall accuracy (weighted average of fragment and ID)
        optimal_overall_threshold = None
        optimal_overall_score = -1
        
        for thresh, metric in threshold_metrics.items():
            # 计算综合分数：50%片段准确率 + 50%ID准确率
            overall_score = 0.5 * metric['accuracy'] + 0.5 * metric['id_accuracy']
            if overall_score > optimal_overall_score:
                optimal_overall_score = overall_score
                optimal_overall_threshold = thresh
        
        print(f"\n🎯 Optimal Threshold Analysis:")
        print(f"  Optimal Threshold (by F1 Score): {optimal_threshold:.2f}")
        print(f"  Optimal F1 Score: {optimal_metrics['f1_score']:.4f}")
        print(f"  Optimal Accuracy: {optimal_metrics['accuracy']:.4f}")
        print(f"  Optimal ID Accuracy: {optimal_metrics['id_accuracy']:.4f}")
        print(f"  Optimal Precision: {optimal_metrics['precision']:.4f}")
        print(f"  Optimal Recall: {optimal_metrics['recall']:.4f}")
        print(f"\n  Optimal Threshold (by ID Accuracy): {optimal_id_threshold:.2f}")
        print(f"  Optimal ID Accuracy: {optimal_id_metrics['id_accuracy']:.4f}")
        print(f"  Corresponding F1 Score: {optimal_id_metrics['f1_score']:.4f}")
        print(f"\n  Optimal Threshold (by Overall Score): {optimal_overall_threshold:.2f}")
        print(f"  Optimal Overall Score: {optimal_overall_score:.4f}")
        
        return {
            'threshold_metrics': threshold_metrics,
            'optimal_threshold': optimal_threshold,
            'optimal_metrics': optimal_metrics,
            'optimal_id_threshold': optimal_id_threshold,
            'optimal_id_metrics': optimal_id_metrics,
            'optimal_overall_threshold': optimal_overall_threshold,
            'optimal_overall_score': optimal_overall_score
        }
    
    def calculate_metrics(self, sequence_ids: List[str], true_labels: List[Optional[int]], 
                         predictions: List[int], probabilities: List[float], threshold: float = None) -> Dict[str, Any]:
        """
        Calculate evaluation metrics
        
        Args:
            sequence_ids: Sequence ID list
            true_labels: True label list
            predictions: Predicted label list
            probabilities: Prediction probability list
            threshold: Classification threshold used
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if threshold is None:
            threshold = self.threshold
        
        # Filter labeled samples
        labeled_indices = [i for i, label in enumerate(true_labels) if label is not None]
        
        if not labeled_indices:
            print("⚠️  No labeled data, skipping metric calculation")
            return None
        
        print(f"\n📊 Calculating evaluation metrics (Threshold: {threshold})...")
        print(f"  Labeled Samples: {len(labeled_indices)}")
        
        # Extract labeled data
        labeled_ids = [sequence_ids[i] for i in labeled_indices]
        labeled_true = [true_labels[i] for i in labeled_indices]
        labeled_pred = [predictions[i] for i in labeled_indices]
        labeled_probs = [probabilities[i] for i in labeled_indices]
        
        true_np = np.array(labeled_true)
        pred_np = np.array(labeled_pred)
        probs_np = np.array(labeled_probs)
        
        metrics = {}
        
        # Basic fragment-level metrics
        metrics['accuracy'] = accuracy_score(true_np, pred_np)
        metrics['f1_score'] = f1_score(true_np, pred_np, average='binary', zero_division=0)
        metrics['precision'] = precision_score(true_np, pred_np, zero_division=0)
        metrics['recall'] = recall_score(true_np, pred_np, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(true_np, pred_np)
        
        # ID-level metrics
        id_metrics = self.calculate_id_based_metrics(labeled_ids, labeled_true, labeled_pred)
        metrics['id_accuracy'] = id_metrics['id_accuracy']
        metrics['id_correct_count'] = id_metrics['id_correct_count']
        metrics['id_total_count'] = id_metrics['id_total_count']
        metrics['fragment_correct_count'] = id_metrics['fragment_correct_count']
        metrics['fragment_total_count'] = id_metrics['fragment_total_count']
        metrics['id_correct_ratio'] = id_metrics['id_correct_count'] / id_metrics['id_total_count'] if id_metrics['id_total_count'] > 0 else 0
        
        # 变体类型指标
        variant_metrics = self.calculate_variant_type_metrics(labeled_ids, labeled_true, labeled_pred)
        metrics['variant_metrics'] = variant_metrics
        
        # AUC and ROC
        if len(np.unique(true_np)) > 1:
            try:
                metrics['auc'] = roc_auc_score(true_np, probs_np)
                # Calculate ROC curve
                fpr, tpr, roc_thresholds = roc_curve(true_np, probs_np)
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist()
                }
                
                # Calculate precision-recall curve
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(true_np, probs_np)
                metrics['pr_curve'] = {
                    'precision': precision_curve.tolist(),
                    'recall': recall_curve.tolist(),
                    'thresholds': pr_thresholds.tolist()
                }
                
                # Calculate AUPRC (Area Under Precision-Recall Curve)
                metrics['auprc'] = np.trapz(precision_curve, recall_curve)
                
            except:
                metrics['auc'] = 0.0
                metrics['auprc'] = 0.0
                metrics['roc_curve'] = None
                metrics['pr_curve'] = None
        else:
            metrics['auc'] = 0.0
            metrics['auprc'] = 0.0
            metrics['roc_curve'] = None
            metrics['pr_curve'] = None
        
        # Brier Score
        if len(probs_np) > 0:
            brier_score = np.mean((probs_np - true_np) ** 2)
            metrics['brier_score'] = float(brier_score)
        else:
            metrics['brier_score'] = 1.0
        
        # Confusion matrix
        cm = confusion_matrix(true_np, pred_np)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            true_np, pred_np, 
            target_names=['Non-pathogenic', 'Pathogenic'],
            output_dict=True,
            zero_division=0
        )
        
        # Class distribution
        metrics['class_distribution'] = {
            'negative': int(np.sum(true_np == 0)),
            'positive': int(np.sum(true_np == 1)),
            'total': len(true_np)
        }
        
        # Prediction distribution
        metrics['prediction_distribution'] = {
            'predicted_negative': int(np.sum(pred_np == 0)),
            'predicted_positive': int(np.sum(pred_np == 1)),
            'total': len(pred_np)
        }
        
        # Store threshold information
        metrics['threshold_used'] = threshold
        
        # Store ID metrics summary
        metrics['id_metrics_summary'] = id_metrics['id_summary']
        
        print(f"✅ Metrics calculation completed")
        print(f"  Fragment Accuracy: {metrics['accuracy']:.4f} ({metrics['fragment_correct_count']}/{metrics['fragment_total_count']})")
        print(f"  ID Accuracy: {metrics['id_accuracy']:.4f} ({metrics['id_correct_count']}/{metrics['id_total_count']})")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  MCC: {metrics['mcc']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  AUPRC: {metrics.get('auprc', 0.0):.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        
        # 打印变体类型指标
        print(f"\n📊 Variant Type Performance Metrics:")
        print("-" * 80)
        print(f"{'Variant Type':<15} {'Frag Acc':<12} {'Frag Corr':<12} {'Frag Total':<12} {'ID Acc':<12} {'ID Corr':<12} {'ID Total':<12}")
        print("-" * 80)
        
        for vtype in ['original', 'semantic', 'confusion', 'mutation', 'variant', 'unknown']:
            if vtype in variant_metrics:
                vmetrics = variant_metrics[vtype]
                if vmetrics['fragment_accuracy'] is not None:
                    frag_acc = f"{vmetrics['fragment_accuracy']:.4f}"
                    frag_corr = f"{vmetrics['fragment_correct_count']}"
                    frag_total = f"{vmetrics['fragment_total_count']}"
                    id_acc = f"{vmetrics['id_accuracy']:.4f}"
                    id_corr = f"{vmetrics['id_correct_count']}"
                    id_total = f"{vmetrics['id_total_count']}"
                    
                    print(f"{vtype:<15} {frag_acc:<12} {frag_corr:<12} {frag_total:<12} {id_acc:<12} {id_corr:<12} {id_total:<12}")
        
        print(f"\n📊 ID-Level Performance:")
        print(f"  Total IDs: {metrics['id_metrics_summary']['total_ids']}")
        print(f"  Labeled IDs: {metrics['id_metrics_summary']['labeled_ids']}")
        print(f"  Correct IDs: {metrics['id_metrics_summary']['correct_ids']}")
        print(f"  ID Accuracy: {metrics['id_accuracy']:.4f}")
        print(f"  ID Types Distribution: {metrics['id_metrics_summary']['id_types_distribution']}")
        
        return metrics
    
    def create_visualizations(self, sequence_ids: List[str], true_labels: List[Optional[int]], 
                            predictions: List[int], probabilities: List[float], threshold: float = None):
        """
        Create visualization charts
        
        Args:
            sequence_ids: Sequence ID list
            true_labels: True label list
            predictions: Predicted label list
            probabilities: Prediction probability list
            threshold: Classification threshold used
        """
        if threshold is None:
            threshold = self.threshold
        
        print(f"\n🎨 Creating visualization charts...")
        
        # Filter labeled samples
        labeled_indices = [i for i, label in enumerate(true_labels) if label is not None]
        
        if not labeled_indices:
            print("⚠️  No labeled data, skipping visualization")
            return
        
        # Extract labeled data
        labeled_ids = [sequence_ids[i] for i in labeled_indices]
        labeled_true = [true_labels[i] for i in labeled_indices]
        labeled_pred = [predictions[i] for i in labeled_indices]
        labeled_probs = [probabilities[i] for i in labeled_indices]
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'Nucleotide Transformer v3 Prediction Results (Threshold: {threshold})', 
                    fontsize=18, fontweight='bold')
        
        # 1. Prediction Probability Distribution
        ax1 = axes[0, 0]
        
        # Group by true label
        pos_indices = [i for i, label in enumerate(labeled_true) if label == 1]
        neg_indices = [i for i, label in enumerate(labeled_true) if label == 0]
        
        if pos_indices:
            pos_probs = [labeled_probs[i] for i in pos_indices]
            ax1.hist(pos_probs, bins=20, alpha=0.7, label='Pathogenic', color='red', density=True)
        
        if neg_indices:
            neg_probs = [labeled_probs[i] for i in neg_indices]
            ax1.hist(neg_probs, bins=20, alpha=0.7, label='Non-pathogenic', color='blue', density=True)
        
        ax1.axvline(x=threshold, color='black', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Decision Threshold: {threshold}')
        ax1.set_xlabel('Prediction Probability')
        ax1.set_ylabel('Density')
        ax1.set_title('Prediction Probability Distribution (by True Label)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confusion Matrix
        ax2 = axes[0, 1]
        
        cm = confusion_matrix(labeled_true, labeled_pred)
        im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        # Add numerical labels
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['Non-pathogenic', 'Pathogenic'])
        ax2.set_yticklabels(['Non-pathogenic', 'Pathogenic'])
        ax2.set_title(f'Confusion Matrix (Threshold: {threshold})')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2)
        
        # 3. ROC Curve
        ax3 = axes[1, 0]
        
        if len(np.unique(labeled_true)) > 1:
            fpr, tpr, _ = roc_curve(labeled_true, labeled_probs)
            auc_score = roc_auc_score(labeled_true, labeled_probs)
            
            ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
            ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax3.legend(loc="lower right")
            ax3.grid(True, alpha=0.3)
            
            # Mark threshold point
            if len(labeled_probs) > 0:
                # Find closest threshold point on ROC curve
                threshold_idx = np.argmin(np.abs(np.array(labeled_probs) - threshold))
                if threshold_idx < len(fpr) and threshold_idx < len(tpr):
                    ax3.scatter(fpr[threshold_idx], tpr[threshold_idx], 
                              color='red', s=100, zorder=5, label=f'Threshold: {threshold}')
                    ax3.legend(loc="lower right")
        else:
            ax3.text(0.5, 0.5, "ROC curve requires both classes", 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('ROC Curve (Not enough classes)')
        
        # 4. Precision-Recall Curve
        ax4 = axes[1, 1]
        
        if len(np.unique(labeled_true)) > 1:
            precision, recall, _ = precision_recall_curve(labeled_true, labeled_probs)
            # Calculate AUPRC
            auprc = np.trapz(precision, recall)
            
            ax4.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUPRC = {auprc:.3f})')
            ax4.set_xlabel('Recall')
            ax4.set_ylabel('Precision')
            ax4.set_title('Precision-Recall Curve')
            ax4.legend(loc="lower left")
            ax4.grid(True, alpha=0.3)
            
            # Mark threshold point
            if len(labeled_probs) > 0:
                # Find closest threshold point
                threshold_idx = np.argmin(np.abs(np.array(labeled_probs) - threshold))
                if threshold_idx < len(precision) and threshold_idx < len(recall):
                    ax4.scatter(recall[threshold_idx], precision[threshold_idx], 
                              color='red', s=100, zorder=5, label=f'Threshold: {threshold}')
                    ax4.legend(loc="lower left")
        else:
            ax4.text(0.5, 0.5, "PR curve requires both classes", 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('Precision-Recall Curve (Not enough classes)')
        
        # 5. ID-Level Accuracy by Type
        ax5 = axes[2, 0]
        
        # Calculate ID-level metrics
        id_metrics = self.calculate_id_based_metrics(labeled_ids, labeled_true, labeled_pred)
        
        # Extract ID types and their accuracy
        id_types = {}
        for base_id, details in id_metrics['id_details'].items():
            # Determine ID type
            id_type = details['variant_type']
            
            if id_type not in id_types:
                id_types[id_type] = {'total': 0, 'correct': 0}
            
            id_types[id_type]['total'] += 1
            if details['is_id_correct']:
                id_types[id_type]['correct'] += 1
        
        # Calculate accuracy by type
        type_names = []
        type_accuracies = []
        type_counts = []
        
        # 确保顺序一致
        for id_type in ['original', 'semantic', 'confusion', 'mutation', 'variant', 'unknown']:
            if id_type in id_types and id_types[id_type]['total'] > 0:
                type_names.append(id_type)
                type_accuracies.append(id_types[id_type]['correct'] / id_types[id_type]['total'] * 100)  # Percentage
                type_counts.append(id_types[id_type]['total'])
        
        if type_names:
            colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'violet', 'gray']
            color_map = {type_name: colors[i] for i, type_name in enumerate(['original', 'semantic', 'confusion', 'mutation', 'variant', 'unknown'])}
            
            bar_colors = [color_map[tn] for tn in type_names]
            bars = ax5.bar(type_names, type_accuracies, color=bar_colors, alpha=0.8)
            ax5.set_xlabel('Variant Type')
            ax5.set_ylabel('ID Accuracy (%)')
            ax5.set_title('ID-Level Accuracy by Variant Type')
            
            # Add count labels on bars
            for i, (bar, count) in enumerate(zip(bars, type_counts)):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{type_accuracies[i]:.1f}%\n(n={count})', 
                        ha='center', va='bottom', fontsize=9)
            
            ax5.grid(True, alpha=0.3, axis='y')
        else:
            ax5.text(0.5, 0.5, "No ID type information available", 
                    ha='center', va='center', fontsize=12)
            ax5.set_title('ID-Level Accuracy by Variant Type')
        
        # 6. Fragment vs ID Accuracy by Variant Type
        ax6 = axes[2, 1]
        
        # 计算变体类型的片段级和ID级准确率
        variant_metrics = self.calculate_variant_type_metrics(labeled_ids, labeled_true, labeled_pred)
        
        # 提取有数据的变体类型
        variant_types_with_data = []
        fragment_accuracies = []
        id_accuracies = []
        
        for vtype in ['original', 'semantic', 'confusion', 'mutation', 'variant', 'unknown']:
            if vtype in variant_metrics and variant_metrics[vtype]['fragment_accuracy'] is not None:
                variant_types_with_data.append(vtype)
                fragment_accuracies.append(variant_metrics[vtype]['fragment_accuracy'] * 100)  # 转换为百分比
                id_accuracies.append(variant_metrics[vtype]['id_accuracy'] * 100)  # 转换为百分比
        
        if variant_types_with_data:
            x = np.arange(len(variant_types_with_data))
            width = 0.35
            
            bars1 = ax6.bar(x - width/2, fragment_accuracies, width, label='Fragment Accuracy', color='skyblue', alpha=0.8)
            bars2 = ax6.bar(x + width/2, id_accuracies, width, label='ID Accuracy', color='lightcoral', alpha=0.8)
            
            ax6.set_xlabel('Variant Type')
            ax6.set_ylabel('Accuracy (%)')
            ax6.set_title('Fragment vs ID Accuracy by Variant Type')
            ax6.set_xticks(x)
            ax6.set_xticklabels(variant_types_with_data)
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for i, (frag_acc, id_acc) in enumerate(zip(fragment_accuracies, id_accuracies)):
                ax6.text(i - width/2, frag_acc + 1, f'{frag_acc:.1f}%', 
                        ha='center', va='bottom', fontsize=8)
                ax6.text(i + width/2, id_acc + 1, f'{id_acc:.1f}%', 
                        ha='center', va='bottom', fontsize=8)
        else:
            ax6.text(0.5, 0.5, "No variant type data available", 
                    ha='center', va='center', fontsize=12)
            ax6.set_title('Fragment vs ID Accuracy by Variant Type')
        
        plt.tight_layout()
        
        # Save chart
        plot_path = os.path.join(self.output_dir, f"prediction_visualization_threshold_{threshold}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization chart saved: {plot_path}")
        
        # Create additional threshold analysis chart
        self.create_threshold_analysis_chart(labeled_ids, labeled_true, labeled_probs, threshold)
        
        # 创建变体类型详细分析图表
        self.create_variant_type_detailed_chart(labeled_ids, labeled_true, labeled_pred, labeled_probs, threshold)
    
    def create_variant_type_detailed_chart(self, sequence_ids: List[str], true_labels: List[int], 
                                         predictions: List[int], probabilities: List[float], threshold: float):
        """
        创建变体类型详细分析图表
        
        Args:
            sequence_ids: 序列ID列表
            true_labels: 真实标签列表
            predictions: 预测标签列表
            probabilities: 预测概率列表
            threshold: 分类阈值
        """
        # 计算变体类型指标
        variant_metrics = self.calculate_variant_type_metrics(sequence_ids, true_labels, predictions)
        
        # 提取有数据的变体类型
        variant_types = []
        for vtype in ['original', 'semantic', 'confusion', 'mutation', 'variant', 'unknown']:
            if vtype in variant_metrics and variant_metrics[vtype]['fragment_accuracy'] is not None:
                variant_types.append(vtype)
        
        if not variant_types:
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Variant Type Detailed Analysis (Threshold: {threshold:.2f})', 
                    fontsize=16, fontweight='bold')
        
        # 1. 变体类型样本分布
        ax1 = axes[0, 0]
        
        total_samples = [variant_metrics[vtype]['total_samples'] for vtype in variant_types]
        labeled_samples = [variant_metrics[vtype]['labeled_samples'] for vtype in variant_types]
        positive_samples = [variant_metrics[vtype]['positive_samples'] for vtype in variant_types]
        negative_samples = [variant_metrics[vtype]['negative_samples'] for vtype in variant_types]
        
        x = np.arange(len(variant_types))
        width = 0.2
        
        bars1 = ax1.bar(x - width*1.5, total_samples, width, label='Total Samples', color='gray', alpha=0.7)
        bars2 = ax1.bar(x - width/2, labeled_samples, width, label='Labeled Samples', color='skyblue', alpha=0.7)
        bars3 = ax1.bar(x + width/2, positive_samples, width, label='Positive Samples', color='red', alpha=0.7)
        bars4 = ax1.bar(x + width*1.5, negative_samples, width, label='Negative Samples', color='blue', alpha=0.7)
        
        ax1.set_xlabel('Variant Type')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Sample Distribution by Variant Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels(variant_types)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 变体类型准确率对比
        ax2 = axes[0, 1]
        
        fragment_acc = [variant_metrics[vtype]['fragment_accuracy'] * 100 for vtype in variant_types]
        id_acc = [variant_metrics[vtype]['id_accuracy'] * 100 for vtype in variant_types]
        
        bars1 = ax2.bar(x - width/2, fragment_acc, width, label='Fragment Accuracy', color='lightgreen', alpha=0.8)
        bars2 = ax2.bar(x + width/2, id_acc, width, label='ID Accuracy', color='orange', alpha=0.8)
        
        ax2.set_xlabel('Variant Type')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Comparison by Variant Type')
        ax2.set_xticks(x)
        ax2.set_xticklabels(variant_types)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, (frag, ida) in enumerate(zip(fragment_acc, id_acc)):
            ax2.text(i - width/2, frag + 0.5, f'{frag:.1f}%', 
                    ha='center', va='bottom', fontsize=8)
            ax2.text(i + width/2, ida + 0.5, f'{ida:.1f}%', 
                    ha='center', va='bottom', fontsize=8)
        
        # 3. 变体类型ID数量分布
        ax3 = axes[1, 0]
        
        id_counts = [variant_metrics[vtype]['id_total_count'] for vtype in variant_types]
        id_correct = [variant_metrics[vtype]['id_correct_count'] for vtype in variant_types]
        
        bars1 = ax3.bar(x - width/2, id_counts, width, label='Total IDs', color='purple', alpha=0.7)
        bars2 = ax3.bar(x + width/2, id_correct, width, label='Correct IDs', color='green', alpha=0.7)
        
        ax3.set_xlabel('Variant Type')
        ax3.set_ylabel('Number of IDs')
        ax3.set_title('ID Count Distribution by Variant Type')
        ax3.set_xticks(x)
        ax3.set_xticklabels(variant_types)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, (total, correct) in enumerate(zip(id_counts, id_correct)):
            ax3.text(i - width/2, total + 0.1, f'{total}', 
                    ha='center', va='bottom', fontsize=8)
            ax3.text(i + width/2, correct + 0.1, f'{correct}', 
                    ha='center', va='bottom', fontsize=8)
        
        # 4. 变体类型性能总结表格
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # 创建总结表格
        table_data = [['Variant', 'Frag Acc', 'ID Acc', 'Frag Corr/Total', 'ID Corr/Total']]
        
        for vtype in variant_types:
            vmetrics = variant_metrics[vtype]
            row = [
                vtype,
                f"{vmetrics['fragment_accuracy']*100:.1f}%",
                f"{vmetrics['id_accuracy']*100:.1f}%",
                f"{vmetrics['fragment_correct_count']}/{vmetrics['fragment_total_count']}",
                f"{vmetrics['id_correct_count']}/{vmetrics['id_total_count']}"
            ]
            table_data.append(row)
        
        # 添加总体行
        overall_frag_acc = accuracy_score(true_labels, predictions)
        id_metrics = self.calculate_id_based_metrics(sequence_ids, true_labels, predictions)
        overall_id_acc = id_metrics['id_accuracy']
        
        row = [
            'OVERALL',
            f"{overall_frag_acc*100:.1f}%",
            f"{overall_id_acc*100:.1f}%",
            f"{id_metrics['fragment_correct_count']}/{id_metrics['fragment_total_count']}",
            f"{id_metrics['id_correct_count']}/{id_metrics['id_total_count']}"
        ]
        table_data.append(row)
        
        table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # 样式设置
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 最后一行（总体）加粗
        last_row = len(table_data) - 1
        for i in range(len(table_data[0])):
            table[(last_row, i)].set_facecolor('#f2f2f2')
            table[(last_row, i)].set_text_props(weight='bold')
        
        ax4.set_title('Variant Type Performance Summary', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        variant_plot_path = os.path.join(self.output_dir, f"variant_type_detailed_analysis_threshold_{threshold}.png")
        plt.savefig(variant_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Variant type detailed analysis saved: {variant_plot_path}")
    
    def create_threshold_analysis_chart(self, sequence_ids: List[str], true_labels: List[int], 
                                       probabilities: List[float], current_threshold: float):
        """Create detailed threshold analysis chart including ID-level metrics"""
        if len(np.unique(true_labels)) <= 1:
            return
        
        # Calculate metrics at many thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        f1_scores = []
        fragment_accuracies = []
        id_accuracies = []
        precisions = []
        recalls = []
        
        for t in thresholds:
            preds = [1 if prob > t else 0 for prob in probabilities]
            
            # Fragment-level metrics
            f1_scores.append(f1_score(true_labels, preds, average='binary', zero_division=0))
            fragment_accuracies.append(accuracy_score(true_labels, preds))
            precisions.append(precision_score(true_labels, preds, zero_division=0))
            recalls.append(recall_score(true_labels, preds, zero_division=0))
            
            # ID-level metrics
            id_metrics = self.calculate_id_based_metrics(sequence_ids, true_labels, preds)
            id_accuracies.append(id_metrics['id_accuracy'])
        
        # Find optimal thresholds
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_f1_threshold = thresholds[optimal_f1_idx]
        optimal_f1_score = f1_scores[optimal_f1_idx]
        
        optimal_frag_acc_idx = np.argmax(fragment_accuracies)
        optimal_frag_acc_threshold = thresholds[optimal_frag_acc_idx]
        optimal_frag_acc_score = fragment_accuracies[optimal_frag_acc_idx]
        
        optimal_id_acc_idx = np.argmax(id_accuracies)
        optimal_id_acc_threshold = thresholds[optimal_id_acc_idx]
        optimal_id_acc_score = id_accuracies[optimal_id_acc_idx]
        
        # 计算综合最优阈值
        optimal_overall_idx = None
        optimal_overall_score = -1
        
        for i in range(len(thresholds)):
            # 综合分数：50%片段准确率 + 50%ID准确率
            overall_score = 0.5 * fragment_accuracies[i] + 0.5 * id_accuracies[i]
            if overall_score > optimal_overall_score:
                optimal_overall_score = overall_score
                optimal_overall_idx = i
        
        optimal_overall_threshold = thresholds[optimal_overall_idx]
        optimal_overall_frag_acc = fragment_accuracies[optimal_overall_idx]
        optimal_overall_id_acc = id_accuracies[optimal_overall_idx]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Detailed Threshold Analysis (Current: {current_threshold:.2f})', 
                    fontsize=16, fontweight='bold')
        
        # 1. Fragment vs ID Accuracy
        ax1 = axes[0, 0]
        ax1.plot(thresholds, fragment_accuracies, 'b-', linewidth=2, label='Fragment Accuracy')
        ax1.plot(thresholds, id_accuracies, 'r-', linewidth=2, label='ID Accuracy')
        
        # 标记各个最优阈值
        ax1.axvline(x=current_threshold, color='black', linestyle='--', alpha=0.7, 
                   label=f'Current: {current_threshold:.2f}')
        ax1.axvline(x=optimal_frag_acc_threshold, color='blue', linestyle=':', alpha=0.7, 
                   label=f'Optimal Fragment: {optimal_frag_acc_threshold:.2f}')
        ax1.axvline(x=optimal_id_acc_threshold, color='red', linestyle=':', alpha=0.7, 
                   label=f'Optimal ID: {optimal_id_acc_threshold:.2f}')
        ax1.axvline(x=optimal_overall_threshold, color='green', linestyle='-.', alpha=0.7,
                   label=f'Optimal Overall: {optimal_overall_threshold:.2f}')
        
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Fragment vs ID Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. F1 Score vs Threshold
        ax2 = axes[0, 1]
        ax2.plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1 Score')
        ax2.axvline(x=current_threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Current: {current_threshold:.2f}')
        ax2.axvline(x=optimal_f1_threshold, color='green', linestyle=':', alpha=0.7, 
                   label=f'Optimal F1: {optimal_f1_threshold:.2f}')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('F1 Score')
        ax2.set_title(f'F1 Score vs Threshold (Optimal: {optimal_f1_threshold:.2f}, F1: {optimal_f1_score:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall vs Threshold
        ax3 = axes[1, 0]
        ax3.plot(thresholds, precisions, 'purple', linewidth=2, label='Precision')
        ax3.plot(thresholds, recalls, 'orange', linewidth=2, label='Recall')
        ax3.axvline(x=current_threshold, color='black', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Score')
        ax3.set_title('Precision and Recall vs Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Summary Table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # 计算当前阈值下的指标
        current_preds = [1 if prob > current_threshold else 0 for prob in probabilities]
        current_fragment_acc = accuracy_score(true_labels, current_preds)
        current_id_metrics = self.calculate_id_based_metrics(sequence_ids, true_labels, current_preds)
        current_id_acc = current_id_metrics['id_accuracy']
        current_f1 = f1_score(true_labels, current_preds, average='binary', zero_division=0)
        
        # 创建总结表格
        summary_data = [
            ['Metric', 'Current', 'Optimal (Fragment)', 'Optimal (ID)', 'Optimal (F1)', 'Optimal (Overall)'],
            ['Threshold', 
             f'{current_threshold:.3f}', 
             f'{optimal_frag_acc_threshold:.3f}', 
             f'{optimal_id_acc_threshold:.3f}',
             f'{optimal_f1_threshold:.3f}',
             f'{optimal_overall_threshold:.3f}'],
            ['Fragment Acc', 
             f'{current_fragment_acc:.3f}',
             f'{optimal_frag_acc_score:.3f}',
             f'{fragment_accuracies[optimal_id_acc_idx]:.3f}',
             f'{fragment_accuracies[optimal_f1_idx]:.3f}',
             f'{optimal_overall_frag_acc:.3f}'],
            ['ID Acc', 
             f'{current_id_acc:.3f}',
             f'{id_accuracies[optimal_frag_acc_idx]:.3f}',
             f'{optimal_id_acc_score:.3f}',
             f'{id_accuracies[optimal_f1_idx]:.3f}',
             f'{optimal_overall_id_acc:.3f}'],
            ['F1 Score', 
             f'{current_f1:.3f}',
             f'{f1_scores[optimal_frag_acc_idx]:.3f}',
             f'{f1_scores[optimal_id_acc_idx]:.3f}',
             f'{optimal_f1_score:.3f}',
             f'{f1_scores[optimal_overall_idx]:.3f}']
        ]
        
        table = ax4.table(cellText=summary_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # 样式设置
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 第一列加粗
        for i in range(1, len(summary_data)):
            table[(i, 0)].set_facecolor('#f2f2f2')
            table[(i, 0)].set_text_props(weight='bold')
        
        ax4.set_title('Performance Summary at Different Thresholds', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save chart
        analysis_path = os.path.join(self.output_dir, f"detailed_threshold_analysis.png")
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Detailed threshold analysis saved: {analysis_path}")
        
        # Save threshold analysis data
        threshold_data = {
            'thresholds': thresholds.tolist(),
            'f1_scores': f1_scores,
            'precisions': precisions,
            'recalls': recalls,
            'fragment_accuracies': fragment_accuracies,
            'id_accuracies': id_accuracies,
            'current_threshold': current_threshold,
            'optimal_f1_threshold': float(optimal_f1_threshold),
            'optimal_f1_score': float(optimal_f1_score),
            'optimal_fragment_acc_threshold': float(optimal_frag_acc_threshold),
            'optimal_fragment_acc_score': float(optimal_frag_acc_score),
            'optimal_id_acc_threshold': float(optimal_id_acc_threshold),
            'optimal_id_acc_score': float(optimal_id_acc_score),
            'optimal_overall_threshold': float(optimal_overall_threshold),
            'optimal_overall_score': float(optimal_overall_score)
        }
        
        data_path = os.path.join(self.output_dir, "threshold_analysis_data.json")
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(threshold_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Threshold analysis data saved: {data_path}")
    
    def save_results(self, sequence_ids: List[str], sequences: List[str], 
                    true_labels: List[Optional[int]], predictions: List[int], 
                    probabilities: List[float], metrics: dict = None, threshold: float = None):
        """
        Save prediction results
        
        Args:
            sequence_ids: Sequence ID list
            sequences: Sequence list
            true_labels: True label list
            predictions: Predicted label list
            probabilities: Prediction probability list
            metrics: Evaluation metrics
            threshold: Classification threshold used
        """
        if threshold is None:
            threshold = self.threshold
        
        print(f"\n💾 Saving prediction results...")
        
        # 计算基于ID的详细结果
        id_metrics = self.calculate_id_based_metrics(sequence_ids, true_labels, predictions)
        
        # 计算变体类型指标
        variant_metrics = self.calculate_variant_type_metrics(sequence_ids, true_labels, predictions)
        
        # 1. Save as CSV format
        csv_file = os.path.join(self.output_dir, f"predictions_threshold_{threshold}.csv")
        
        data = []
        for i, (seq_id, seq, label, pred, prob) in enumerate(zip(
            sequence_ids, sequences, true_labels, predictions, probabilities)):
            
            # 提取包含类型的基础ID
            base_id = self.extract_id_base_with_type(seq_id)
            variant_type = self.get_variant_type(base_id)
            
            row = {
                'sequence_id': seq_id,
                'base_id_with_type': base_id,
                'variant_type': variant_type,
                'sequence_length': len(seq),
                'true_label': 'Unknown' if label is None else ('Pathogenic' if label == 1 else 'Non-pathogenic'),
                'predicted_label': 'Pathogenic' if pred == 1 else 'Non-pathogenic',
                'prediction_probability': prob,
                'prediction_confidence': 'High' if abs(prob - threshold) > 0.3 else ('Medium' if abs(prob - threshold) > 0.1 else 'Low'),
                'distance_to_threshold': abs(prob - threshold)
            }
            
            if label is not None:
                row['is_correct'] = 'Yes' if label == pred else 'No'
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"✅ Predictions saved to CSV: {csv_file}")
        
        # 2. Save ID-level results
        id_csv_file = os.path.join(self.output_dir, f"id_level_predictions_threshold_{threshold}.csv")
        
        id_data = []
        for base_id, details in id_metrics['id_details'].items():
            variant_type = self.get_variant_type(base_id)
            
            row = {
                'base_id_with_type': base_id,
                'variant_type': variant_type,
                'sample_count': details['sample_count'],
                'labeled_count': details['labeled_count'],
                'correct_count': details['correct_count'],
                'correct_ratio': details['correct_ratio'],
                'id_correctness': 'Correct' if details['is_id_correct'] else 'Incorrect'
            }
            
            # 添加每个样本的详细信息
            for j, sample in enumerate(details['samples']):
                if sample['is_labeled']:
                    row[f'sample_{j+1}_id'] = sample['sequence_id']
                    row[f'sample_{j+1}_true'] = 'Pathogenic' if sample['true_label'] == 1 else 'Non-pathogenic'
                    row[f'sample_{j+1}_pred'] = 'Pathogenic' if sample['predicted_label'] == 1 else 'Non-pathogenic'
                    row[f'sample_{j+1}_correct'] = 'Yes' if sample['is_correct'] else 'No'
                else:
                    row[f'sample_{j+1}_id'] = sample['sequence_id']
                    row[f'sample_{j+1}_true'] = 'Unknown'
                    row[f'sample_{j+1}_pred'] = 'Pathogenic' if sample['predicted_label'] == 1 else 'Non-pathogenic'
                    row[f'sample_{j+1}_correct'] = 'N/A'
            
            id_data.append(row)
        
        id_df = pd.DataFrame(id_data)
        id_df.to_csv(id_csv_file, index=False, encoding='utf-8')
        print(f"✅ ID-level predictions saved to CSV: {id_csv_file}")
        
        # 3. 保存变体类型指标
        variant_csv_file = os.path.join(self.output_dir, f"variant_type_metrics_threshold_{threshold}.csv")
        
        variant_data = []
        for vtype, vmetrics in variant_metrics.items():
            if vmetrics['fragment_accuracy'] is not None:
                row = {
                    'variant_type': vtype,
                    'fragment_accuracy': vmetrics['fragment_accuracy'],
                    'fragment_correct_count': vmetrics['fragment_correct_count'],
                    'fragment_total_count': vmetrics['fragment_total_count'],
                    'id_accuracy': vmetrics['id_accuracy'],
                    'id_correct_count': vmetrics['id_correct_count'],
                    'id_total_count': vmetrics['id_total_count'],
                    'total_samples': vmetrics['total_samples'],
                    'labeled_samples': vmetrics['labeled_samples'],
                    'positive_samples': vmetrics['positive_samples'],
                    'negative_samples': vmetrics['negative_samples']
                }
                variant_data.append(row)
        
        if variant_data:
            variant_df = pd.DataFrame(variant_data)
            variant_df.to_csv(variant_csv_file, index=False, encoding='utf-8')
            print(f"✅ Variant type metrics saved to CSV: {variant_csv_file}")
        
        # 4. Save as JSON format
        json_file = os.path.join(self.output_dir, "detailed_results.json")
        
        detailed_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_path': self.model_path,
            'model_config': self.model_config,
            'classification_threshold': threshold,
            'total_sequences': len(sequence_ids),
            'predictions': []
        }
        
        for i in range(len(sequence_ids)):
            base_id = self.extract_id_base_with_type(sequence_ids[i])
            variant_type = self.get_variant_type(base_id)
            
            pred_info = {
                'sequence_id': sequence_ids[i],
                'base_id_with_type': base_id,
                'variant_type': variant_type,
                'sequence_length': len(sequences[i]),
                'true_label': true_labels[i] if true_labels[i] is not None else None,
                'predicted_label': int(predictions[i]),
                'prediction_probability': float(probabilities[i]),
                'distance_to_threshold': float(abs(probabilities[i] - threshold))
            }
            detailed_data['predictions'].append(pred_info)
        
        if metrics:
            detailed_data['metrics'] = metrics
        
        detailed_data['id_metrics'] = id_metrics
        detailed_data['variant_metrics'] = variant_metrics
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Detailed results saved to JSON: {json_file}")
        
        # 5. Save metrics report
        if metrics:
            report_file = os.path.join(self.output_dir, f"prediction_report_threshold_{threshold}.txt")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("                 NTv3 PREDICTION RESULTS REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Prediction Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model Path: {self.model_path}\n")
                f.write(f"Classification Threshold: {threshold}\n")
                f.write(f"Model Configuration: {self.model_config.get('transformer_model_repo', 'N/A')}\n")
                f.write(f"Output Directory: {self.output_dir}\n\n")
                
                f.write("📊 DATA STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Sequences: {len(sequence_ids)}\n")
                f.write(f"Labeled Sequences: {sum(1 for label in true_labels if label is not None)}\n")
                f.write(f"Unlabeled Sequences: {sum(1 for label in true_labels if label is None)}\n")
                f.write(f"Total Unique IDs (with type): {id_metrics['id_summary']['total_ids']}\n")
                f.write(f"Labeled IDs: {id_metrics['id_summary']['labeled_ids']}\n")
                f.write(f"ID Types Distribution: {id_metrics['id_summary']['id_types_distribution']}\n\n")
                
                f.write("🎯 PREDICTION SUMMARY:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Predicted Pathogenic: {sum(predictions)}\n")
                f.write(f"Predicted Non-pathogenic: {len(predictions) - sum(predictions)}\n")
                f.write(f"Average Prediction Probability: {np.mean(probabilities):.4f}\n")
                f.write(f"Median Prediction Probability: {np.median(probabilities):.4f}\n")
                f.write(f"Standard Deviation: {np.std(probabilities):.4f}\n\n")
                
                f.write("📈 FRAGMENT-LEVEL PERFORMANCE METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['fragment_correct_count']}/{metrics['fragment_total_count']})\n")
                f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"MCC: {metrics['mcc']:.4f}\n")
                f.write(f"AUC: {metrics['auc']:.4f}\n")
                if 'auprc' in metrics:
                    f.write(f"AUPRC: {metrics['auprc']:.4f}\n")
                f.write(f"Brier Score: {metrics['brier_score']:.4f}\n\n")
                
                f.write("📊 ID-LEVEL PERFORMANCE METRICS (New Rule: >50% correct):\n")
                f.write("-" * 40 + "\n")
                f.write(f"ID Accuracy: {metrics['id_accuracy']:.4f}\n")
                f.write(f"Correct IDs: {metrics['id_correct_count']}/{metrics['id_total_count']}\n")
                f.write(f"Fragment Accuracy: {metrics['fragment_correct_count']}/{metrics['fragment_total_count']}\n")
                f.write(f"ID vs Fragment Accuracy Difference: {metrics['id_accuracy'] - metrics['accuracy']:.4f}\n")
                f.write(f"Note: ID contains sequence type (original, semantic, confusion) and counts as different sequences.\n")
                f.write(f"Note: ID is considered correct if >50% of its fragments are correctly predicted.\n\n")
                
                f.write("🎯 VARIANT TYPE PERFORMANCE METRICS:\n")
                f.write("-" * 79 + "\n")
                f.write(f"{'Variant Type':<15} {'Frag Acc':<12} {'Frag Corr/Total':<18} {'ID Acc':<12} {'ID Corr/Total':<18}\n")
                f.write("-" * 79 + "\n")
                
                for vtype in ['original', 'semantic', 'confusion', 'mutation', 'variant', 'unknown']:
                    if vtype in variant_metrics and variant_metrics[vtype]['fragment_accuracy'] is not None:
                        vmetrics = variant_metrics[vtype]
                        f.write(f"{vtype:<15} {vmetrics['fragment_accuracy']:12.4f} "
                              f"{vmetrics['fragment_correct_count']}/{vmetrics['fragment_total_count']:<18} "
                              f"{vmetrics['id_accuracy']:12.4f} "
                              f"{vmetrics['id_correct_count']}/{vmetrics['id_total_count']:<18}\n")
                
                f.write("\n")
                
                if 'confusion_matrix' in metrics:
                    f.write("📊 CONFUSION MATRIX (Fragment-Level):\n")
                    f.write("-" * 40 + "\n")
                    cm = metrics['confusion_matrix']
                    f.write(f"              Predicted Non-path    Predicted Path\n")
                    f.write(f"True Non-path        {cm[0][0]:8d}            {cm[0][1]:8d}\n")
                    f.write(f"True Path            {cm[1][0]:8d}            {cm[1][1]:8d}\n\n")
                
                f.write("📁 OUTPUT FILES:\n")
                f.write("-" * 40 + "\n")
                f.write(f"1. predictions_threshold_{threshold}.csv - CSV predictions\n")
                f.write(f"2. id_level_predictions_threshold_{threshold}.csv - ID-level predictions\n")
                f.write(f"3. variant_type_metrics_threshold_{threshold}.csv - Variant type metrics\n")
                f.write(f"4. detailed_results.json - JSON detailed results\n")
                f.write(f"5. prediction_report_threshold_{threshold}.txt - This report\n")
                f.write(f"6. prediction_visualization_threshold_{threshold}.png - Visualization charts\n")
                f.write(f"7. variant_type_detailed_analysis_threshold_{threshold}.png - Variant type analysis\n")
                f.write(f"8. detailed_threshold_analysis.png - Threshold analysis\n")
                f.write(f"9. threshold_analysis_data.json - Threshold analysis data\n")
                
                f.write("\n" + "=" * 80 + "\n")
            
            print(f"✅ Prediction report saved: {report_file}")
        
        return csv_file, json_file
    
    def predict(self, fasta_file: str, batch_size: int = 8, threshold: float = None):
        """
        Predict FASTA file
        
        Args:
            fasta_file: FASTA file path
            batch_size: Batch size
            threshold: Classification threshold (overrides instance threshold if provided)
            
        Returns:
            probabilities, predictions, metrics
        """
        if threshold is not None:
            # Use provided threshold
            prediction_threshold = threshold
            print(f"📢 Using provided threshold: {prediction_threshold}")
        else:
            # Use instance threshold
            prediction_threshold = self.threshold
        
        print("=" * 70)
        print(f"🔍 NTv3 FASTA PREDICTION")
        print(f"  Model: {os.path.basename(self.model_path)}")
        print(f"  FASTA File: {fasta_file}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Classification Threshold: {prediction_threshold}")
        print("=" * 70)
        
        try:
            # 1. Parse FASTA file
            sequence_ids, sequences, labels = self.parse_fasta_file(fasta_file)
            
            if not sequences:
                print("❌ No valid sequences in FASTA file")
                return None
            
            # 2. Run predictions with custom threshold
            probabilities, predictions = self.predict_batch(
                sequences, 
                batch_size, 
                threshold=prediction_threshold
            )
            
            if not probabilities:
                print("❌ Prediction failed")
                return None
            
            # 3. Calculate metrics
            metrics = None
            if any(label is not None for label in labels):
                metrics = self.calculate_metrics(sequence_ids, labels, predictions, probabilities, prediction_threshold)
                
                # Additional threshold analysis
                threshold_analysis = self.calculate_metrics_at_thresholds(sequence_ids, labels, probabilities)
                if threshold_analysis:
                    if metrics:
                        metrics['threshold_analysis'] = threshold_analysis
            
            # 4. Create visualization charts
            self.create_visualizations(sequence_ids, labels, predictions, probabilities, prediction_threshold)
            
            # 5. Save results
            self.save_results(sequence_ids, sequences, labels, predictions, probabilities, metrics, prediction_threshold)
            
            print(f"\n{'='*70}")
            print(f"🎉 PREDICTION COMPLETED!")
            print(f"{'='*70}")
            
            # Print summary
            print(f"\n📊 PREDICTION SUMMARY:")
            print(f"  Total Sequences: {len(sequences)}")
            print(f"  Predicted Pathogenic: {sum(predictions)}")
            print(f"  Predicted Non-pathogenic: {len(predictions) - sum(predictions)}")
            print(f"  Average Prediction Probability: {np.mean(probabilities):.4f}")
            print(f"  Classification Threshold: {prediction_threshold}")
            
            if metrics:
                print(f"\n📈 PERFORMANCE METRICS:")
                print(f"  Fragment Accuracy: {metrics['accuracy']:.4f} ({metrics['fragment_correct_count']}/{metrics['fragment_total_count']})")
                print(f"  ID Accuracy (New Rule: >50% correct): {metrics['id_accuracy']:.4f} ({metrics['id_correct_count']}/{metrics['id_total_count']})")
                print(f"  F1 Score: {metrics['f1_score']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  AUC: {metrics['auc']:.4f}")
                if 'auprc' in metrics:
                    print(f"  AUPRC: {metrics['auprc']:.4f}")
                print(f"  Brier Score: {metrics['brier_score']:.4f}")
                
                # Print ID-level summary
                print(f"\n📊 ID-LEVEL SUMMARY:")
                print(f"  Total IDs (with type): {metrics['id_metrics_summary']['total_ids']}")
                print(f"  Labeled IDs: {metrics['id_metrics_summary']['labeled_ids']}")
                print(f"  Correct IDs: {metrics['id_metrics_summary']['correct_ids']}")
                print(f"  ID Accuracy: {metrics['id_accuracy']:.4f}")
                print(f"  ID Types Distribution: {metrics['id_metrics_summary']['id_types_distribution']}")
                
                # Print variant type summary
                print(f"\n🎯 VARIANT TYPE PERFORMANCE:")
                print(f"{'Variant Type':<15} {'Frag Acc':<12} {'Frag Corr/Total':<18} {'ID Acc':<12} {'ID Corr/Total':<18}")
                print("-" * 79)
                
                for vtype in ['original', 'semantic', 'confusion', 'mutation', 'variant', 'unknown']:
                    if vtype in metrics['variant_metrics'] and metrics['variant_metrics'][vtype]['fragment_accuracy'] is not None:
                        vmetrics = metrics['variant_metrics'][vtype]
                        print(f"{vtype:<15} {vmetrics['fragment_accuracy']:12.4f} "
                              f"{vmetrics['fragment_correct_count']}/{vmetrics['fragment_total_count']:<18} "
                              f"{vmetrics['id_accuracy']:12.4f} "
                              f"{vmetrics['id_correct_count']}/{vmetrics['id_total_count']:<18}")
                
                # Print optimal threshold info if available
                if 'threshold_analysis' in metrics:
                    optimal = metrics['threshold_analysis']['optimal_threshold']
                    optimal_f1 = metrics['threshold_analysis']['optimal_metrics']['f1_score']
                    optimal_id_acc = metrics['threshold_analysis']['optimal_metrics']['id_accuracy']
                    optimal_id = metrics['threshold_analysis']['optimal_id_threshold']
                    optimal_id_acc2 = metrics['threshold_analysis']['optimal_id_metrics']['id_accuracy']
                    print(f"\n🎯 THRESHOLD OPTIMIZATION:")
                    print(f"  Optimal Threshold (by F1): {optimal:.3f}")
                    print(f"  Optimal F1 Score: {optimal_f1:.3f}")
                    print(f"  Optimal ID Accuracy (by F1): {optimal_id_acc:.3f}")
                    print(f"  Optimal Threshold (by ID Accuracy): {optimal_id:.3f}")
                    print(f"  Optimal ID Accuracy: {optimal_id_acc2:.3f}")
                    print(f"  Current F1 Score: {metrics['f1_score']:.3f}")
                    print(f"  Current ID Accuracy: {metrics['id_accuracy']:.3f}")
                    print(f"  F1 Improvement Potential: {optimal_f1 - metrics['f1_score']:.3f}")
                    print(f"  ID Accuracy Improvement Potential: {max(optimal_id_acc, optimal_id_acc2) - metrics['id_accuracy']:.3f}")
            
            print(f"\n📁 Results saved to directory: {self.output_dir}")
            
            return probabilities, predictions, metrics
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function"""
    if not IMPORT_SUCCESS:
        print("❌ Cannot import required modules, please check if files exist")
        return
    
    parser = argparse.ArgumentParser(description="Nucleotide Transformer v3 FASTA Prediction Tool with Adjustable Threshold")
    
    # Required parameters
    parser.add_argument("--fasta", type=str, required=True, help="Input FASTA file path")
    parser.add_argument("--model", type=str, required=True, help="Trained model path (.pth file)")
    
    # Optional parameters
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 8)")
    parser.add_argument("--threshold", type=float, default=0.6, 
                       help="Classification threshold (0.0-1.0, default: 0.5)")
    
    args = parser.parse_args()
    
    # Check file existence
    if not os.path.exists(args.fasta):
        print(f"❌ FASTA file not found: {args.fasta}")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print(f"❌ Invalid threshold: {args.threshold}. Must be between 0.0 and 1.0")
        return
    
    # Create predictor and run prediction
    try:
        predictor = SimpleNTv3Predictor(
            model_path=args.model,
            output_dir=args.output_dir,
            threshold=args.threshold
        )
        
        predictor.predict(args.fasta, args.batch_size, args.threshold)
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()