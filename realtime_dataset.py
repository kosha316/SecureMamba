"""
实时处理数据集 - 完全使用Nucleotide Transformer v3
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from typing import List, Dict, Optional, Any
import random
from concurrent.futures import ThreadPoolExecutor
import threading


class RealtimeSequenceDataset(Dataset):
    """实时序列数据集 - 完全使用Nucleotide Transformer v3"""
    
    def __init__(
        self, 
        segments: List[Dict], 
        max_seq_len: int = 512,
        use_cache: bool = True
    ):
        """
        初始化实时数据集
        
        Args:
            segments: 片段数据列表
            max_seq_len: 最大序列长度（DNA碱基数）
            use_cache: 是否使用缓存（为False，实现实时处理）
        """
        self.segments = segments
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache
        
        # 缓存系统（可选）
        self.sequence_cache = {} if use_cache else None
        self.cache_lock = threading.Lock()
        
        print(f"📊 Nucleotide Transformer v3数据集初始化")
        print(f"  总片段数: {len(segments):,}")
        print(f"  实时处理: {not use_cache}")
        print(f"  最大序列长度(DNA碱基): {max_seq_len}")
    
    def __len__(self) -> int:
        return len(self.segments)
        
    # 在 RealtimeSequenceDataset 类的 _get_sequence 方法中，更新序列处理：

    def _get_sequence(self, seq: str, cache_key: str = None) -> str:
        """获取序列（支持缓存）- 仅清理字符，不进行填充"""
        if self.use_cache and cache_key:
            with self.cache_lock:
                if cache_key in self.sequence_cache:
                    return self.sequence_cache[cache_key]
        
        # 确保序列只包含有效字符
        valid_chars = set('ACGTNacgtn')
        seq = ''.join([c for c in seq if c in valid_chars])
        
        # 转换为大写
        seq = seq.upper()
        
        # 如果序列太短，添加一些N使其有基本长度
        if len(seq) < 50:
            seq = seq + 'N' * (50 - len(seq))
        
        # 缓存结果
        if self.use_cache and cache_key:
            with self.cache_lock:
                self.sequence_cache[cache_key] = seq
        
        return seq

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seg = self.segments[idx]
        
        # 获取原始序列
        original_seq = seg.get('original_seq', '')
        if not original_seq:
            raise ValueError(f"样本 {idx} 缺少原始序列")
        
        # 获取标签
        label = seg.get('original_label', seg.get('label', 0.0))
        
        # 处理序列长度
        original_seq = self._get_sequence(original_seq, f"orig_{idx}")
        
        # 构建返回项
        item = {
            'sequence': original_seq,  # 原始DNA序列
            'labels': torch.FloatTensor([float(label)]),
            'segment_id': seg.get('segment_id', f"seq_{idx}"),
        }
        
        # 处理变体序列（如果存在）
        # 语义变体
        if 'positive_views' in seg and seg['positive_views']:
            semantic_view = None
            confusion_view = None
            
            # 查找语义和混淆变体
            for i, view_type in enumerate(seg.get('view_types', [])):
                if i < len(seg['positive_views']):
                    if view_type == 'semantic':
                        semantic_view = seg['positive_views'][i]
                    elif view_type == 'confusion':
                        confusion_view = seg['positive_views'][i]
            
            if semantic_view:
                semantic_seq = self._get_sequence(semantic_view, f"semantic_{idx}")
                item['semantic_sequence'] = semantic_seq
            
            if confusion_view:
                confusion_seq = self._get_sequence(confusion_view, f"confusion_{idx}")
                item['confusion_sequence'] = confusion_seq
        
        # 随机突变变体
        if 'contrastive_negative_views' in seg and seg['contrastive_negative_views']:
            random_view = random.choice(seg['contrastive_negative_views'])
            random_seq = self._get_sequence(random_view, f"random_{idx}_{hash(random_view)}")
            item['random_mutation_sequence'] = random_seq
        
        return item
    
    def clear_cache(self):
        """清空缓存"""
        if self.sequence_cache:
            self.sequence_cache.clear()


def collate_nucleotide_features(batch: List[Dict]) -> Dict[str, Any]:
    """Nucleotide Transformer v3特征的collate函数"""
    if not batch:
        return {}
    
    batch_size = len(batch)
    
    # 收集所有序列
    sequences = []
    labels = []
    segment_ids = []
    
    semantic_sequences = []
    confusion_sequences = []
    random_mutation_sequences = []
    
    for item in batch:
        sequences.append(item['sequence'])
        labels.append(item['labels'])
        segment_ids.append(item.get('segment_id', 'unknown'))
        
        # 变体序列
        semantic_sequences.append(item.get('semantic_sequence', None))
        confusion_sequences.append(item.get('confusion_sequence', None))
        random_mutation_sequences.append(item.get('random_mutation_sequence', None))
    
    # 堆叠标签
    labels_tensor = torch.stack(labels).squeeze()
    
    # 构建结果
    result = {
        'sequences': sequences,  # List[str] - 原始序列
        'labels': labels_tensor,
        'segment_ids': segment_ids,
    }
    
    # 处理变体序列 - 确保列表长度一致
    def process_variant_sequences(variant_list):
        # 找出有变体的索引
        valid_indices = [i for i, seq in enumerate(variant_list) if seq is not None]
        
        if valid_indices:
            # 创建完整列表，None用空字符串替代
            processed = []
            for seq in variant_list:
                if seq is not None:
                    processed.append(seq)
                else:
                    processed.append("")  # 空字符串表示无变体
            return processed
        return None
    
    result['semantic_sequences'] = process_variant_sequences(semantic_sequences)
    result['confusion_sequences'] = process_variant_sequences(confusion_sequences)
    result['random_mutation_sequences'] = process_variant_sequences(random_mutation_sequences)
    
    return result


def create_nucleotide_dataloader(
    segments: List[Dict],
    batch_size: int = 8,  # 减小批次大小以适应NTv3
    max_seq_len: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    use_cache: bool = True
) -> DataLoader:
    """创建Nucleotide Transformer v3数据加载器"""
    
    # 创建数据集
    dataset = RealtimeSequenceDataset(
        segments=segments,
        max_seq_len=max_seq_len,
        use_cache=use_cache
    )
    
    print(f"🔧 创建Nucleotide Transformer v3数据加载器:")
    print(f"  批次大小: {batch_size} ⚠️ (NTv3内存需求大)")
    print(f"  数据集大小: {len(dataset):,}")
    print(f"  是否打乱: {shuffle}")
    print(f"  Worker数量: {num_workers}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_nucleotide_features,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )