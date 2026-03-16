"""
数据预处理模块 - 处理FASTA文件，替换简并碱基，生成序列片段，并添加下采样
"""

import torch
import os
import random
import numpy as np
from Bio import SeqIO
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import pickle

# 简并碱基映射表（IUPAC标准）
degenerate_base_map = {
    'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'],
    'M': ['A', 'C'], 'R': ['A', 'G'], 'W': ['A', 'T'],
    'S': ['C', 'G'], 'Y': ['C', 'T'], 'K': ['G', 'T'],
    'V': ['A', 'C', 'G'], 'H': ['A', 'C', 'T'],
    'D': ['A', 'G', 'T'], 'B': ['C', 'G', 'T'],
    'N': ['A', 'T', 'C', 'G'],
}


def replace_degenerate_bases(seq: str) -> str:
    """将序列中的简并碱基替换为对应的标准碱基（随机选择）"""
    cleaned = []
    for base in seq.upper():
        possible_bases = degenerate_base_map.get(base)
        if possible_bases:
            cleaned.append(random.choice(possible_bases))
        else:
            cleaned.append(random.choice(degenerate_base_map['N']))
    return ''.join(cleaned)


def load_fasta_sequences(fasta_path: str, label: int) -> List[Dict]:
    """加载FASTA序列，替换简并碱基，并过滤"""
    records = []
    try:
        for rec in SeqIO.parse(fasta_path, "fasta"):
            raw_seq = str(rec.seq)
            processed_seq = replace_degenerate_bases(raw_seq)
            
            # 过滤长度过短的序列
            if len(processed_seq) >= 3:  # 最小长度，确保至少一个密码子
                records.append({
                    "id": rec.id,
                    "seq_str": processed_seq,
                    "original_seq": processed_seq,
                    "label": label,
                    "original_label": label
                })
        return records
    except Exception as e:
        print(f"❌ 加载FASTA文件失败：{fasta_path}, 错误：{str(e)}")
        return []


def generate_sequence_segments(full_seqs: List[Dict], max_length: int, min_length: int) -> List[Dict]:
    """将长序列切割成固定长度的片段"""
    segments = []
    
    for full_seq in full_seqs:
        orig_id = full_seq["id"]
        orig_seq = full_seq["seq_str"]
        orig_label = full_seq["label"]
        orig_len = len(orig_seq)
        
        start = 0
        seg_idx = 0
        
        while start < orig_len:
            # 计算当前片段的结束位置
            end = min(start + max_length, orig_len)
            
            # 确保长度是3的倍数（密码子要求）
            remainder = (end - start) % 3
            if remainder != 0:
                end -= remainder
            
            # 如果调整后的结束位置小于等于起始位置，说明无法形成有效片段
            if end <= start:
                break
                
            seg_len = end - start
            
            # 如果片段长度满足最小要求，添加到结果中
            if seg_len >= min_length:
                segments.append({
                    "segment_id": f"{orig_id}_seg{seg_idx}",
                    "original_id": orig_id,
                    "original_seq": orig_seq[start:end],
                    "label": orig_label,
                    "original_label": orig_label,
                    "length": seg_len
                })
                seg_idx += 1
            
            # 更新起始位置，准备处理下一段
            start = end
            
            # 如果剩余序列长度小于最小长度，跳出循环
            if orig_len - start < min_length:
                break
    
    return segments


def remove_duplicate_segments(segments: List[Dict]) -> Tuple[List[Dict], int]:
    """移除重复的序列片段"""
    seen_seqs = set()
    unique_segs = []
    dup_count = 0
    
    for seg in segments:
        seq_str = seg["original_seq"]
        if seq_str not in seen_seqs:
            seen_seqs.add(seq_str)
            unique_segs.append(seg)
        else:
            dup_count += 1
    
    return unique_segs, dup_count


def downsample_negative_segments(segments: List[Dict], seed: int = 42, 
                                downsample_ratio: float = 3.0) -> List[Dict]:
    """
    对非致病片段进行下采样，使其数量约为致病片段的指定倍数
    
    Args:
        segments: 所有片段列表
        seed: 随机种子
        downsample_ratio: 下采样比例，非致病片段数量/致病片段数量
    
    Returns:
        下采样后的片段列表
    """
    # 按标签分离片段
    positive_segments = [seg for seg in segments if seg['label'] == 1]
    negative_segments = [seg for seg in segments if seg['label'] == 0]
    
    positive_count = len(positive_segments)
    negative_count = len(negative_segments)
    
    print(f"📊 下采样前统计:")
    print(f"  - 致病片段: {positive_count}")
    print(f"  - 非致病片段: {negative_count}")
    print(f"  - 非致病:致病比例: {negative_count/positive_count:.2f}:1")
    
    # 如果非致病片段数量已经少于或等于目标数量，则全部保留
    target_negative_count = int(positive_count * downsample_ratio)
    
    if negative_count <= target_negative_count:
        print(f"⚠️  非致病片段数量({negative_count})已经小于或等于目标数量({target_negative_count})，跳过下采样")
        return segments
    
    # 设置随机种子以确保可重复性
    random.seed(seed)
    np.random.seed(seed)
    
    # 随机采样目标数量的非致病片段
    sampled_negative_segments = random.sample(negative_segments, target_negative_count)
    
    # 合并致病片段和采样后的非致病片段
    downsampled_segments = positive_segments + sampled_negative_segments
    
    # 可选：打乱顺序（保持随机性但可重复）
    random.shuffle(downsampled_segments)
    
    print(f"✅ 下采样完成:")
    print(f"  - 采样后致病片段: {positive_count}")
    print(f"  - 采样后非致病片段: {len(sampled_negative_segments)}")
    print(f"  - 采样后非致病:致病比例: {len(sampled_negative_segments)/positive_count:.2f}:1")
    print(f"  - 移除的非致病片段: {negative_count - target_negative_count}")
    
    return downsampled_segments


def preprocess_data(config) -> List[Dict]:
    """
    主预处理函数：加载FASTA文件，替换简并碱基，生成片段，去重，下采样
    
    Args:
        config: 配置对象（需包含downsample_ratio属性）
        
    Returns:
        预处理后的片段列表
    """
    print(f"\n🔄 数据预处理开始")
    
    # 检查文件是否存在
    if not os.path.exists(config.positive_fasta):
        print(f"⚠️  正样本文件不存在：{config.positive_fasta}")
        return []
    
    if not os.path.exists(config.negative_fasta):
        print(f"⚠️  负样本文件不存在：{config.negative_fasta}")
        return []
    
    # 1. 加载并处理原始序列
    print("📥 加载正样本序列...")
    positive_seqs = load_fasta_sequences(config.positive_fasta, label=1)
    
    print("📥 加载负样本序列...")
    negative_seqs = load_fasta_sequences(config.negative_fasta, label=0)
    
    all_seqs = positive_seqs + negative_seqs
    print(f"✅ 加载完成：正样本 {len(positive_seqs)} 条，负样本 {len(negative_seqs)} 条")
    
    # 2. 生成片段
    print("✂️  生成序列片段...")
    all_segments = generate_sequence_segments(
        all_seqs, 
        max_length=config.max_seq_length,
        min_length=config.min_segment_len
    )
    print(f"✅ 片段生成：共 {len(all_segments)} 个片段")
    
    # 3. 去重
    print("🔄 移除重复片段...")
    unique_segments, dup_count = remove_duplicate_segments(all_segments)
    print(f"✅ 去重完成：去重前 {len(all_segments)}，去重后 {len(unique_segments)}，移除 {dup_count}")
    
    # 4. 下采样非致病片段
    if hasattr(config, 'downsample_ratio') and config.downsample_ratio > 0:
        print(f"\n📉 对非致病片段进行下采样（目标比例 非致病:致病 = {config.downsample_ratio}:1）...")
        downsampled_segments = downsample_negative_segments(
            unique_segments, 
            seed=getattr(config, 'seed', 42),  # 使用配置中的随机种子，默认为42
            downsample_ratio=config.downsample_ratio
        )
        unique_segments = downsampled_segments
    
    # 5. 统计信息
    positive_count = sum(1 for seg in unique_segments if seg['label'] == 1)
    negative_count = len(unique_segments) - positive_count
    avg_length = sum(len(seg['original_seq']) for seg in unique_segments) / len(unique_segments) if unique_segments else 0
    
    print(f"\n📊 最终预处理统计：")
    print(f"  - 总片段数：{len(unique_segments)}")
    print(f"  - 致病性片段：{positive_count} ({positive_count/len(unique_segments)*100:.1f}%)")
    print(f"  - 非致病性片段：{negative_count} ({negative_count/len(unique_segments)*100:.1f}%)")
    print(f"  - 非致病:致病比例：{negative_count/positive_count:.2f}:1" if positive_count > 0 else "  - 无致病片段")
    print(f"  - 平均序列长度：{avg_length:.1f} bp")
    
    return unique_segments


def save_preprocessed_data(segments: List[Dict], cache_path: str):
    """保存预处理数据到缓存文件"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # 确保数据结构正确
    clean_segments = []
    for seg in segments:
        clean_seg = {
            'original_seq': seg.get('original_seq', ''),
            'label': seg.get('label', 0),
            'segment_id': seg.get('segment_id', f"seq_{len(clean_segments)}")
        }
        clean_segments.append(clean_seg)
    
    # 保存为pickle文件
    with open(cache_path, 'wb') as f:
        pickle.dump(clean_segments, f)
    
    print(f"💾 预处理数据已保存到：{cache_path}")
    return cache_path


def load_preprocessed_data(cache_path: str) -> List[Dict]:
    """从缓存文件加载预处理数据"""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                segments = pickle.load(f)
            print(f"✅ 从缓存加载预处理数据：{cache_path} ({len(segments)} 个片段)")
            return segments
        except Exception as e:
            print(f"❌ 缓存加载失败：{str(e)}")
    return None


# 测试函数
if __name__ == "__main__":
    print("🧪 测试数据预处理功能...")
    
    # 创建测试配置
    class TestConfig:
        positive_fasta = "test_positive.fasta"
        negative_fasta = "test_negative.fasta"
        max_seq_length = 300
        min_segment_len = 30
        downsample_ratio = 3.0  # 非致病片段是致病片段的3倍
        seed = 42  # 随机种子
    
    config = TestConfig()
    
    # 测试简并碱基替换
    test_seq = "ATNGCATMRWSN"
    cleaned_seq = replace_degenerate_bases(test_seq)
    print(f"简并碱基替换测试: {test_seq} -> {cleaned_seq}")
    
    # 测试下采样函数
    test_segments = [
        {"segment_id": f"pos_{i}", "original_seq": "ATCG"*10, "label": 1} for i in range(10)
    ] + [
        {"segment_id": f"neg_{i}", "original_seq": "GCTA"*10, "label": 0} for i in range(100)
    ]
    
    print(f"\n测试下采样：10个致病片段，100个非致病片段")
    downsampled = downsample_negative_segments(test_segments, seed=42, downsample_ratio=3.0)
    
    pos_count = sum(1 for seg in downsampled if seg['label'] == 1)
    neg_count = sum(1 for seg in downsampled if seg['label'] == 0)
    print(f"下采样后：致病片段={pos_count}, 非致病片段={neg_count}, 比例={neg_count/pos_count}:1")
    
    # 测试预处理流程
    try:
        segments = preprocess_data(config)
        if segments:
            print(f"预处理测试成功：生成 {len(segments)} 个片段")
    except Exception as e:
        print(f"预处理测试失败：{e}")
    
    print("✅ 数据预处理测试完成")