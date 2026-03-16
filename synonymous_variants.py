import random
import math
import warnings
import multiprocessing
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict

# --------------------------
# 常量定义：氨基酸替换规则与密码子映射
# --------------------------
conservative_aa_replacements = {
    'A': ['V', 'L', 'I', 'G'], 'V': ['A', 'L', 'I', 'M'], 'L': ['A', 'V', 'I', 'M', 'F'], 
    'I': ['A', 'V', 'L', 'M'], 'M': ['V', 'L', 'I', 'F'], 'F': ['Y', 'W', 'L', 'M'], 
    'Y': ['F', 'W', 'S', 'T'], 'W': ['F', 'Y', 'M'], 'S': ['T', 'N', 'Q', 'C'], 
    'T': ['S', 'N', 'Q', 'C'], 'N': ['Q', 'S', 'T', 'D'], 'Q': ['N', 'S', 'T', 'E'], 
    'C': ['S', 'T', 'A'], 'K': ['R', 'H', 'Q'], 'R': ['K', 'H', 'Q'], 
    'H': ['K', 'R', 'Y'], 'D': ['E', 'N', 'S'], 'E': ['D', 'Q', 'T'], 
    'G': ['A', 'P', 'S'], 'P': ['G', 'A', 'S'], '*': []
}

codon_to_aa = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L', 'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*', 'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L', 'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M', 'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V', 'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

# 氨基酸-密码子反向映射
aa_to_codons = defaultdict(list)
for codon, aa in codon_to_aa.items():
    if aa != '*':
        aa_to_codons[aa].append(codon)

all_valid_aas = [aa for aa in aa_to_codons.keys() if aa != '*']

# --------------------------
# 二核苷酸自由能数据 (kcal/mol)
# --------------------------
dimer_free_energy = {
    'AA': -1.00, 'AT': -0.88, 'AC': -1.45, 'AG': -1.30,
    'TA': -0.58, 'TT': -1.00, 'TC': -1.30, 'TG': -1.45,
    'CA': -1.45, 'CT': -1.30, 'CC': -2.17, 'CG': -2.24,
    'GA': -1.30, 'GT': -1.45, 'GC': -2.24, 'GG': -2.17,
    # RNA二核苷酸自由能
    'UU': -1.00, 'UA': -0.88, 'UC': -1.45, 'UG': -1.30,
    'AU': -0.58, 'AA': -1.00, 'AC': -1.30, 'AG': -1.45,
    'CU': -1.45, 'CA': -1.30, 'CC': -2.17, 'CG': -2.24,
    'GU': -1.30, 'GA': -1.45, 'GC': -2.24, 'GG': -2.17
}

# --------------------------
# 辅助函数
# --------------------------
def calculate_sequence_differences(orig_seq: str, var_seq: str) -> Dict[str, float]:
    """计算序列差异统计"""
    min_base_len = min(len(orig_seq), len(var_seq))
    base_diff_count = sum(1 for o, v in zip(orig_seq[:min_base_len], var_seq[:min_base_len]) if o != v)
    base_diff_ratio = base_diff_count / min_base_len if min_base_len > 0 else 0.0

    def dna_to_aa(dna: str) -> str:
        rna = dna.replace('T', 'U')
        codons = [rna[i:i+3] for i in range(0, len(rna), 3) if len(rna[i:i+3]) == 3]
        return ''.join([codon_to_aa.get(c, '?') for c in codons])
    
    orig_aa = dna_to_aa(orig_seq)
    var_aa = dna_to_aa(var_seq)
    min_aa_len = min(len(orig_aa), len(var_aa))
    aa_diff_count = sum(1 for o, v in zip(orig_aa[:min_aa_len], var_aa[:min_aa_len]) if o != v)
    aa_diff_ratio = aa_diff_count / min_aa_len if min_aa_len > 0 else 0.0

    return {
        "base_diff_count": base_diff_count, "base_diff_ratio": round(base_diff_ratio, 4),
        "aa_diff_count": aa_diff_count, "aa_diff_ratio": round(aa_diff_ratio, 4),
        "orig_base_len": len(orig_seq), "var_base_len": len(var_seq)
    }


def calculate_dimer_free_energy(sequence: str) -> float:
    """基于二核苷酸自由能的精确计算"""
    rna_seq = sequence.replace('T', 'U').upper()
    
    if len(rna_seq) < 2:
        return 0.0
    
    total_energy = 0.0
    dimer_count = 0
    
    for i in range(len(rna_seq) - 1):
        dimer = rna_seq[i:i+2]
        if dimer in dimer_free_energy:
            total_energy += dimer_free_energy[dimer]
            dimer_count += 1
    
    return total_energy / dimer_count if dimer_count > 0 else 0.0


def calculate_conservative_energy_change(original_codon: str, new_codon: str) -> float:
    """计算保守替换导致的自由能变化"""
    if len(original_codon) != 3 or len(new_codon) != 3:
        return 0.0
    
    # 计算原始密码子的自由能
    original_energy = calculate_dimer_free_energy(original_codon)
    
    # 计算新密码子的自由能
    new_energy = calculate_dimer_free_energy(new_codon)
    
    # 返回自由能变化
    return abs(new_energy - original_energy)


def balanced_codon_sampling(
    target_codons: List[str],
    used_codons: List[str] = None
) -> str:
    """平衡密码子采样"""
    used_codons = used_codons or []
    available_codons = [c for c in target_codons if c not in used_codons]
    if not available_codons:
        available_codons = target_codons
    
    # 均匀采样
    return random.choice(available_codons)


def get_codon_base_diff(codon1: str, codon2: str) -> int:
    """计算两个密码子的碱基差异数"""
    if len(codon1) != 3 or len(codon2) != 3:
        return 3
    return sum(c1 != c2 for c1, c2 in zip(codon1, codon2))


def reverse_complement(dna_sequence: str) -> str:
    """生成DNA序列的反向互补序列"""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(dna_sequence.upper()))


# --------------------------
# 核心：语义相似变体生成（第一个正样本）
# --------------------------

def generate_positive_variant(
    seq_str: str,
    conservative_energy_tolerance: float = 1.0,  # 放宽能量容忍度
    min_pos_base_diff: float = 0.02,  # 降低最小差异
    max_pos_base_diff: float = 0.25,  # 提高最大差异
    pos_conservative_ratio: float = 0.05,  # 降低保守替换比例
    pos_syn_ratio: float = 0.8,  # 提高同义替换比例
    max_replace_ratio: float = 0.4,  # 提高最大替换比例
    max_attempts: int = 500,  # 增加尝试次数
    min_conservative_changes: int = 1  # 最小保守替换数
) -> str:
    """为一条序列生成一个语义相似正样本变体（优化版本，提高成功率）"""
    
    def get_replaceable_positions(codons):
        """预计算可以进行替换的位置"""
        syn_positions = []
        conservative_positions = []
        
        for i, codon in enumerate(codons):
            original_aa = codon_to_aa.get(codon)
            if not original_aa or original_aa == '*':
                continue
                
            # 检查是否可以同义替换
            if len(aa_to_codons.get(original_aa, [])) > 1:
                syn_positions.append(i)
                
            # 检查是否可以保守替换
            if original_aa in conservative_aa_replacements:
                conservative_positions.append(i)
        
        return syn_positions, conservative_positions
    
    def fallback_strategy(codons, used_codons):
        """备用策略：仅使用同义替换"""
        variant_codons = codons.copy()
        codon_count = len(codons)
        
        # 计算需要的替换数量
        min_changes = max(1, int(codon_count * 0.05))  # 至少5%的密码子
        max_changes = int(codon_count * 0.25)  # 最多25%的密码子
        replace_count = random.randint(min_changes, max_changes)
        
        # 选择可替换的位置
        candidate_positions = []
        for i, codon in enumerate(codons):
            original_aa = codon_to_aa.get(codon)
            if original_aa and original_aa != '*' and len(aa_to_codons.get(original_aa, [])) > 1:
                candidate_positions.append(i)
        
        if len(candidate_positions) < min_changes:
            return None
        
        random.shuffle(candidate_positions)
        selected_positions = candidate_positions[:replace_count]
        changes_made = 0
        
        for pos in selected_positions:
            current_codon = variant_codons[pos]
            original_aa = codon_to_aa.get(current_codon)
            target_codons = [c for c in aa_to_codons.get(original_aa, []) 
                            if c != current_codon and codon_to_aa[c] != '*']
            
            if target_codons:
                selected_codon = balanced_codon_sampling(target_codons, used_codons[pos])
                variant_codons[pos] = selected_codon
                changes_made += 1
        
        return ''.join(variant_codons).replace('U', 'T') if changes_made > 0 else None

    # 主函数开始
    seq_rna = seq_str.replace('T', 'U')
    if len(seq_rna) % 3 != 0:
        raise ValueError(f"序列长度必须为3的倍数（当前：{len(seq_rna)}）")
    
    codon_count = len(seq_rna) // 3
    if codon_count < 3:
        raise ValueError(f"序列过短（{codon_count}个密码子），无法生成有效变体")

    seq_codons = [seq_rna[i:i+3] for i in range(0, len(seq_rna), 3)]
    
    # 预计算可替换位置
    all_syn_positions, all_conservative_positions = get_replaceable_positions(seq_codons)
    
    # 如果没有足够的可替换位置，直接使用备用策略
    if len(all_syn_positions) < 2 and len(all_conservative_positions) < 1:
        fallback_variant = fallback_strategy(seq_codons, defaultdict(list))
        if fallback_variant:
            diff_stats = calculate_sequence_differences(seq_str, fallback_variant)
            print(f"✅ 直接使用备用策略: 序列差异={diff_stats['base_diff_ratio']:.3f}")
            return fallback_variant
        else:
            warnings.warn(f"序列 {seq_str[:20]}... 没有足够的可替换位置")
            return ""

    for attempt in range(max_attempts):
        # 计算替换位置数
        max_replace_count = max(1, int(codon_count * max_replace_ratio))
        min_replace_count = max(1, int(codon_count * min_pos_base_diff / 3))
        replace_count = random.randint(min_replace_count, max_replace_count)
        
        # 分配替换类型
        conservative_count = max(0, int(round(replace_count * pos_conservative_ratio)))
        syn_count = max(0, replace_count - conservative_count)
        
        # 确保有足够的可替换位置
        available_conservative = min(conservative_count, len(all_conservative_positions))
        available_syn = min(syn_count, len(all_syn_positions))
        
        # 如果某种替换类型位置不足，调整另一种类型
        if available_conservative < conservative_count:
            available_syn += (conservative_count - available_conservative)
            available_conservative = 0
        if available_syn < syn_count:
            available_conservative = min(available_conservative + (syn_count - available_syn), len(all_conservative_positions))
            available_syn = len(all_syn_positions)
        
        # 选择替换位置
        conservative_positions = random.sample(all_conservative_positions, available_conservative) if available_conservative > 0 else []
        syn_positions = random.sample(all_syn_positions, available_syn) if available_syn > 0 else []

        variant_codons = seq_codons.copy()
        position_used_codons = defaultdict(list)
        valid_replace = False
        total_conservative_energy_change = 0.0
        conservative_changes_made = 0

        # 第一阶段：同义替换（成功率更高）
        for pos in syn_positions:
            current_codon = variant_codons[pos]
            original_aa = codon_to_aa.get(current_codon)
            if not original_aa or original_aa == '*':
                continue
            
            target_codons = [c for c in aa_to_codons.get(original_aa, []) 
                            if c != current_codon and codon_to_aa[c] != '*']
            
            if target_codons:
                selected_codon = balanced_codon_sampling(target_codons, position_used_codons[pos])
                variant_codons[pos] = selected_codon
                position_used_codons[pos].append(selected_codon)
                valid_replace = True

        # 第二阶段：保守替换（更宽松的条件）
        for pos in conservative_positions:
            # 如果已经达到最小要求且能量变化较大，停止保守替换
            if (conservative_changes_made >= min_conservative_changes and 
                total_conservative_energy_change > conservative_energy_tolerance * 0.7):
                break
                
            current_codon = variant_codons[pos]
            original_aa = codon_to_aa.get(current_codon)
            if not original_aa or original_aa == '*':
                continue
                
            # 扩展可接受的替换选项
            candidate_codons = []
            if original_aa in conservative_aa_replacements:
                target_aas = conservative_aa_replacements[original_aa]
                for aa in target_aas:
                    candidate_codons.extend([c for c in aa_to_codons.get(aa, []) 
                                          if codon_to_aa[c] != '*'])
            
            # 如果没有保守替换选项，跳过
            if not candidate_codons:
                continue
                
            candidate_codons = [c for c in candidate_codons if c != current_codon]
            
            if candidate_codons:
                # 优先选择能量变化小的替换
                energy_sorted_candidates = sorted(
                    candidate_codons,
                    key=lambda c: calculate_conservative_energy_change(current_codon, c)
                )
                
                # 尝试前3个能量最小的候选
                for selected_codon in energy_sorted_candidates[:3]:
                    energy_change = calculate_conservative_energy_change(current_codon, selected_codon)
                    
                    # 检查累计能量变化是否在容忍范围内
                    if total_conservative_energy_change + energy_change <= conservative_energy_tolerance:
                        variant_codons[pos] = selected_codon
                        position_used_codons[pos].append(selected_codon)
                        total_conservative_energy_change += energy_change
                        conservative_changes_made += 1
                        valid_replace = True
                        break

        if not valid_replace:
            continue

        # 检查生成的变体
        variant_rna_str = ''.join(variant_codons)
        variant_dna_str = variant_rna_str.replace('U', 'T')
        
        diff_stats = calculate_sequence_differences(seq_str, variant_dna_str)
        base_diff_ratio = diff_stats['base_diff_ratio']

        # 放宽检查条件
        if (total_conservative_energy_change <= conservative_energy_tolerance
            and min_pos_base_diff <= base_diff_ratio <= max_pos_base_diff
            and variant_dna_str != seq_str
            and conservative_changes_made >= min_conservative_changes):
            print(f"✅ 语义变体生成成功: 保守替换={conservative_changes_made}, "
                  f"自由能变化={total_conservative_energy_change:.3f}, "
                  f"序列差异={base_diff_ratio:.3f}")
            return variant_dna_str

    # 主要策略失败，尝试备用策略
    print(f"⚠️ 主要策略失败，尝试备用策略...")
    fallback_variant = fallback_strategy(seq_codons, defaultdict(list))
    if fallback_variant:
        diff_stats = calculate_sequence_differences(seq_str, fallback_variant)
        print(f"✅ 备用策略生成成功: 序列差异={diff_stats['base_diff_ratio']:.3f}")
        return fallback_variant

    # 如果仍然失败，返回警告
    warnings.warn(f"无法为序列 {seq_str[:20]}... 生成语义相似变体（尝试{max_attempts}次+备用策略）")
    return ""

# --------------------------
# 核心：DNA混淆变体生成（第二个正样本）- 不考虑自由能
# --------------------------
def generate_dna_confusion_variant(
    original_seq: str,
    min_fragment_len: int = 30,
    max_fragment_len: int = 100,
    min_fragments: int = 3,
    max_fragments: int = 8,
    flip_ratio: float = 0.3,
    max_attempts: int = 50
) -> str:
    """
    通过DNA混淆生成正样本变体
    策略：片段化 + 随机反向互补
    注意：不考虑自由能变化
    """
    seq_len = len(original_seq)
    
    # 根据序列长度动态调整参数
    if seq_len < 100:
        min_fragment_len = max(15, seq_len // 4)
        max_fragment_len = min(60, seq_len // 2)
        min_fragments = 2
        max_fragments = min(4, seq_len // min_fragment_len)
        flip_ratio = 0.4
    elif seq_len > 1000:
        min_fragment_len = 80
        max_fragment_len = min(300, seq_len // 4)
        min_fragments = 4
        max_fragments = min(15, seq_len // min_fragment_len)
        flip_ratio = 0.25
    
    # 检查序列是否仍然太短
    if seq_len < min_fragment_len * min_fragments:
        warnings.warn(f"序列过短（{seq_len}），无法进行有效的DNA混淆（需要至少{min_fragment_len * min_fragments}bp）")
        return ""
    
    for attempt in range(max_attempts):
        try:
            # 1. 确定片段数量
            num_fragments = random.randint(min_fragments, max_fragments)
            
            # 2. 生成片段边界
            fragment_boundaries = []
            remaining_len = seq_len
            current_pos = 0
            
            for i in range(num_fragments - 1):
                max_possible_len = min(
                    max_fragment_len,
                    remaining_len - (num_fragments - i - 1) * min_fragment_len
                )
                
                if max_possible_len < min_fragment_len:
                    num_fragments = i + 1
                    break
                
                fragment_len = random.randint(min_fragment_len, max_possible_len)
                fragment_boundaries.append((current_pos, current_pos + fragment_len))
                current_pos += fragment_len
                remaining_len -= fragment_len
            
            # 添加最后一个片段
            if current_pos < seq_len:
                last_fragment_len = seq_len - current_pos
                if last_fragment_len >= min_fragment_len:
                    fragment_boundaries.append((current_pos, seq_len))
                else:
                    if fragment_boundaries:
                        last_start, last_end = fragment_boundaries.pop()
                        fragment_boundaries.append((last_start, seq_len))
                    else:
                        continue
            
            if not fragment_boundaries:
                continue
            
            # 3. 随机选择要反向互补的片段
            flip_indices = set()
            num_to_flip = max(1, int(len(fragment_boundaries) * flip_ratio))
            if len(fragment_boundaries) > 1:
                flip_indices = set(random.sample(range(len(fragment_boundaries)), num_to_flip))
            else:
                flip_indices = {0}
            
            # 4. 构建混淆后的序列
            result_fragments = []
            for i, (start, end) in enumerate(fragment_boundaries):
                fragment = original_seq[start:end]
                
                if i in flip_indices:
                    fragment = reverse_complement(fragment)
                
                result_fragments.append(fragment)

            # 5. 随机打乱片段顺序
            if random.random() < 0.5 and len(result_fragments) > 1:
                random.shuffle(result_fragments)
            
            confused_seq = ''.join(result_fragments)
            
            # 6. 验证生成的序列
            if len(confused_seq) != seq_len:
                continue
            
            # 计算序列差异
            diff_stats = calculate_sequence_differences(original_seq, confused_seq)
            base_diff_ratio = diff_stats['base_diff_ratio']
            
            # DNA混淆变体不检查自由能，只要生成成功就返回
            if confused_seq != original_seq:  # 确保不是原始序列
                print(f"✅ DNA混淆变体生成成功: 序列差异={base_diff_ratio:.3f}")
                return confused_seq
                
        except Exception as e:
            if attempt == 0:
                print(f"❌ DNA混淆生成失败：{str(e)}")
            continue
    
    print(f"❌ 无法为序列生成DNA混淆变体（尝试{max_attempts}次）")
    return ""


# --------------------------
# 核心：负样本变体生成（随机碱基替换）- 不考虑自由能
# --------------------------
def generate_negative_variant(
    original_seq: str,
    mutation_rate: float = 0.5,
    max_attempts: int = 20
) -> str:
    """
    通过随机碱基替换生成负样本变体
    注意：不考虑自由能变化
    """
    seq_len = len(original_seq)
    bases = ['A', 'T', 'C', 'G']
    
    for attempt in range(max_attempts):
        try:
            # 计算需要替换的碱基数量
            num_mutations = max(1, int(seq_len * mutation_rate))
            
            # 随机选择替换位置
            mutation_positions = random.sample(range(seq_len), num_mutations)
            
            # 构建负样本序列
            negative_seq_list = list(original_seq)
            
            for pos in mutation_positions:
                original_base = negative_seq_list[pos]
                possible_bases = [b for b in bases if b != original_base]
                if possible_bases:
                    new_base = random.choice(possible_bases)
                    negative_seq_list[pos] = new_base
            
            negative_seq = ''.join(negative_seq_list)
            
            # 验证生成的序列
            if len(negative_seq) != seq_len:
                continue
                
            # 计算与原始序列的差异
            diff_count = sum(1 for a, b in zip(original_seq, negative_seq) if a != b)
            diff_ratio = diff_count / seq_len
            
            # 负样本只检查序列差异，不考虑自由能
            if 0.3 <= diff_ratio <= 0.7 and negative_seq != original_seq:
                print(f"✅ 负样本生成成功: 差异率={diff_ratio:.3f}")
                return negative_seq
            else:
                if attempt == max_attempts - 1:
                    print(f"⚠️ 负样本差异率超出范围: {diff_ratio:.3f}")
                
        except Exception as e:
            if attempt == 0:
                print(f"❌ 负样本生成失败：{str(e)}")
            continue
    
    print(f"❌ 无法为序列生成负样本变体（尝试{max_attempts}次）")
    return ""


# --------------------------
# 顶层：三元组视图生成（原始序列 + 两个正样本 + 一个负样本）
# --------------------------
def generate_triplet_views(
    current_seg: dict,
    # 语义变体参数
    conservative_energy_tolerance: float = 0.5,
    min_pos_base_diff: float = 0.05,
    max_pos_base_diff: float = 0.2,
    pos_conservative_ratio: float = 0.3,
    pos_syn_ratio: float = 0.7,
    max_replace_ratio: float = 0.3,
    max_semantic_attempts: int = 200,
    # DNA混淆参数
    min_fragment_len: int = 50,
    max_fragment_len: int = 200,
    min_fragments: int = 3,
    max_fragments: int = 10,
    flip_ratio: float = 0.3,
    max_confusion_attempts: int = 50,
    # 负样本参数
    negative_mutation_rate: float = 0.5,
    max_negative_attempts: int = 20
) -> Dict[str, Any]:
    """
    为一条序列生成三元组视图：原始序列 + 两个正样本 + 一个负样本
    语义变体只计算保守替换的自由能变化，其他变体不考虑自由能
    """
    current_seq = current_seg['original_seq']
    current_label = current_seg['label']
    seq_len = len(current_seq)
    
    positive_views = []
    negative_views = []
    view_types = []
    
    print(f"🔧 为序列生成三元组视图 (长度: {seq_len}bp)")

    # 生成语义相似变体（第一个正样本）- 只计算保守替换的自由能变化
    semantic_variant = generate_positive_variant(
        seq_str=current_seq,
        conservative_energy_tolerance=conservative_energy_tolerance,
        min_pos_base_diff=min_pos_base_diff,
        max_pos_base_diff=max_pos_base_diff,
        pos_conservative_ratio=pos_conservative_ratio,
        pos_syn_ratio=pos_syn_ratio,
        max_replace_ratio=max_replace_ratio,
        max_attempts=max_semantic_attempts
    )
    
    if semantic_variant:
        positive_views.append(semantic_variant)
        view_types.append("semantic")
        print(f"✅ 语义变体生成成功")
    else:
        print(f"❌ 语义变体生成失败")
    
    # 生成DNA混淆变体（第二个正样本）- 不考虑自由能
    min_required_length = min_fragment_len * min_fragments
    if seq_len >= min_required_length:
        confusion_variant = generate_dna_confusion_variant(
            original_seq=current_seq,
            min_fragment_len=min_fragment_len,
            max_fragment_len=max_fragment_len,
            min_fragments=min_fragments,
            max_fragments=max_fragments,
            flip_ratio=flip_ratio,
            max_attempts=max_confusion_attempts
        )
        
        if confusion_variant:
            positive_views.append(confusion_variant)
            view_types.append("confusion")
            print(f"✅ DNA混淆变体生成成功")
        else:
            print(f"❌ DNA混淆变体生成失败")
    else:
        print(f"⚠️ 序列过短 ({seq_len}bp < {min_required_length}bp)，跳过DNA混淆变体生成")
    
    # 生成负样本变体（随机碱基替换）- 不考虑自由能
    negative_variant = generate_negative_variant(
        original_seq=current_seq,
        mutation_rate=negative_mutation_rate,
        max_attempts=max_negative_attempts
    )
    
    if negative_variant:
        negative_views.append(negative_variant)
        print(f"✅ 负样本变体生成成功")
    else:
        print(f"❌ 负样本变体生成失败")
    
    # 构建结果 - 确保使用正确的键名
    result = {
        'original_seq': current_seq,
        'original_label': current_label,
        'positive_views': positive_views,  # 语义和混淆变体
        'contrastive_negative_views': negative_views,  # 随机突变变体
        'view_types': view_types,
        'segment_id': current_seg.get('segment_id', 'unknown'),
        'num_positive_views': len(positive_views),
        'num_contrastive_negative_views': len(negative_views),  # 使用新键名
        'has_variants': current_label == 1 and len(positive_views) > 0
    }
    
    # 添加统计信息
    if positive_views or negative_views:
        view_diff_details = []
        
        # 正样本差异
        for i, view in enumerate(positive_views):
            diff_stats = calculate_sequence_differences(current_seq, view)
            view_detail = {
                'type': view_types[i] if i < len(view_types) else 'unknown',
                'base_diff_ratio': diff_stats['base_diff_ratio'],
                'aa_diff_ratio': diff_stats['aa_diff_ratio'],
                'base_diff_count': diff_stats['base_diff_count']
            }
            view_diff_details.append(view_detail)
        
        # 负样本差异
        for view in negative_views:
            diff_stats = calculate_sequence_differences(current_seq, view)
            view_detail = {
                'type': 'negative',
                'base_diff_ratio': diff_stats['base_diff_ratio'],
                'aa_diff_ratio': diff_stats['aa_diff_ratio'],
                'base_diff_count': diff_stats['base_diff_count']
            }
            view_diff_details.append(view_detail)
        
        result['view_diff_details'] = view_diff_details
    
    print(f"📊 三元组生成结果: {len(positive_views)}个正样本, {len(negative_views)}个负样本")
    
    return result


# --------------------------
# 并行生成函数
# --------------------------
def _triplet_views_wrapper(args: Tuple) -> Dict[str, Any]:
    """三元组视图生成的并行包装器"""
    (current_seg, conservative_energy_tolerance, min_pos_base_diff,
     max_pos_base_diff, pos_conservative_ratio, pos_syn_ratio, max_replace_ratio,
     max_semantic_attempts, min_fragment_len, max_fragment_len,
     min_fragments, max_fragments, flip_ratio, max_confusion_attempts,
     negative_mutation_rate, max_negative_attempts) = args
     
    try:
        return generate_triplet_views(
            current_seg=current_seg,
            conservative_energy_tolerance=conservative_energy_tolerance,
            min_pos_base_diff=min_pos_base_diff,
            max_pos_base_diff=max_pos_base_diff,
            pos_conservative_ratio=pos_conservative_ratio,
            pos_syn_ratio=pos_syn_ratio,
            max_replace_ratio=max_replace_ratio,
            max_semantic_attempts=max_semantic_attempts,
            min_fragment_len=min_fragment_len,
            max_fragment_len=max_fragment_len,
            min_fragments=min_fragments,
            max_fragments=max_fragments,
            flip_ratio=flip_ratio,
            max_confusion_attempts=max_confusion_attempts,
            negative_mutation_rate=negative_mutation_rate,
            max_negative_attempts=max_negative_attempts
        )
    except Exception as e:
        warnings.warn(f"处理序列 {current_seg.get('segment_id', 'unknown')} 失败：{str(e)}")
        return {
            'original_seq': current_seg['original_seq'],
            'original_label': current_seg['label'],
            'positive_views': [],
            'contrastive_negative_views': [],  # 使用新键名
            'view_types': [],
            'segment_id': current_seg.get('segment_id', 'unknown'),
            'num_positive_views': 0,
            'num_contrastive_negative_views': 0,  # 使用新键名
            'has_variants': False,
            'error': str(e)
        }

def generate_triplet_views_parallel(
    all_segments: List[dict],
    # 语义变体参数
    conservative_energy_tolerance: float = 0.5,
    min_pos_base_diff: float = 0.05,
    max_pos_base_diff: float = 0.2,
    pos_conservative_ratio: float = 0.3,
    pos_syn_ratio: float = 0.7,
    max_replace_ratio: float = 0.3,
    max_semantic_attempts: int = 200,
    # DNA混淆参数
    min_fragment_len: int = 50,
    max_fragment_len: int = 200,
    min_fragments: int = 3,
    max_fragments: int = 10,
    flip_ratio: float = 0.3,
    max_confusion_attempts: int = 50,
    # 负样本参数
    negative_mutation_rate: float = 0.5,
    max_negative_attempts: int = 20,
    # 并行参数
    num_workers: int = None
) -> List[Dict[str, Any]]:
    """
    并行为所有序列生成三元组视图
    只对正样本生成变体，负样本保持不变
    """
    args_list = [
        (seg, conservative_energy_tolerance, min_pos_base_diff,
         max_pos_base_diff, pos_conservative_ratio, pos_syn_ratio, max_replace_ratio,
         max_semantic_attempts, min_fragment_len, max_fragment_len,
         min_fragments, max_fragments, flip_ratio, max_confusion_attempts,
         negative_mutation_rate, max_negative_attempts)
        for seg in all_segments
    ]

    num_workers = num_workers or max(1, int(multiprocessing.cpu_count() * 0.8))
    print(f"🚀 启动三元组视图生成（进程数：{num_workers}，总序列数：{len(all_segments)}）")
    print(f"📋 目标：每条正样本序列生成2个正样本变体 + 1个随机突变变体（对比学习负样本）")
    print(f"📋 负样本序列不生成任何变体")
    print(f"🔋 保守替换自由能容忍度: ±{conservative_energy_tolerance} kcal/mol")
    print(f"🔍 语义变体差异范围：[{min_pos_base_diff:.2f}, {max_pos_base_diff:.2f}]")
    print(f"🧬 DNA混淆：片段{min_fragment_len}-{max_fragment_len}bp，{min_fragments}-{max_fragments}片段")
    print(f"❌ 随机突变变体：随机突变{negative_mutation_rate:.1%}碱基（仅用于对比学习负样本）")

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(_triplet_views_wrapper, args_list)

    # 统计生成结果 - 使用新的键名
    successful_positives = sum(r['num_positive_views'] for r in results)
    successful_contrastive_negatives = sum(r['num_contrastive_negative_views'] for r in results)  # 修改
    
    semantic_success = sum(1 for r in results if any(t == "semantic" for t in r.get('view_types', [])))
    confusion_success = sum(1 for r in results if any(t == "confusion" for t in r.get('view_types', [])))
    
    # 使用新键名
    contrastive_negative_success = sum(1 for r in results if r['num_contrastive_negative_views'] > 0)  # 修改
    
    # 完整三元组：正样本且有语义+混淆+随机突变
    full_triplet_success = sum(1 for r in results if r['num_positive_views'] >= 2 and r['num_contrastive_negative_views'] >= 1)  # 修改
    
    # 统计正样本和负样本数量
    positive_segments_count = sum(1 for seg in all_segments if seg.get('label', seg.get('original_label', 0)) == 1)
    negative_segments_count = len(all_segments) - positive_segments_count
    
    print(f"\n📊 三元组生成完成统计：")
    print(f"   总处理序列数：{len(all_segments)}")
    print(f"   正样本序列数：{positive_segments_count}")
    print(f"   负样本序列数：{negative_segments_count}")
    print(f"   总正样本变体数：{successful_positives}")
    print(f"   总对比学习负样本数：{successful_contrastive_negatives}")
    print(f"   语义变体成功：{semantic_success}/{positive_segments_count} ({semantic_success/positive_segments_count*100:.1f}%)")
    print(f"   DNA混淆成功：{confusion_success}/{positive_segments_count} ({confusion_success/positive_segments_count*100:.1f}%)")
    print(f"   随机突变变体（对比学习负样本）成功：{contrastive_negative_success}/{positive_segments_count} ({contrastive_negative_success/positive_segments_count*100:.1f}%)")
    print(f"   完整三元组成功：{full_triplet_success}/{positive_segments_count} ({full_triplet_success/positive_segments_count*100:.1f}%)")

    return results
# --------------------------
# 测试函数
# --------------------------
def test_variant_generation():
    """测试变体生成功能"""
    # 测试序列
    test_seq = "ATGGCCATTGAATGGGCCGCTGCTTCTGGTGCTGCCGGTAGCGCAGTCCGTGGCGGTGCTGGTGCTGGTGCTGGCCAGCGTGGTGCTGCCG"
    
    print("🧪 测试变体生成功能（只计算保守替换自由能）...")
    
    # 测试语义相似变体
    print("\n1. 测试语义相似变体生成:")
    semantic_variant = generate_positive_variant(test_seq, max_attempts=10)
    if semantic_variant:
        diff_stats = calculate_sequence_differences(test_seq, semantic_variant)
        print(f"   ✅ 生成成功！序列差异: {diff_stats['base_diff_ratio']:.3f}")
        print(f"   原始序列: {test_seq[:30]}...")
        print(f"   语义变体: {semantic_variant[:30]}...")
    else:
        print("   ❌ 生成失败")
    
    # 测试DNA混淆变体
    print("\n2. 测试DNA混淆变体生成:")
    confusion_variant = generate_dna_confusion_variant(test_seq, max_attempts=10)
    if confusion_variant:
        diff_stats = calculate_sequence_differences(test_seq, confusion_variant)
        print(f"   ✅ 生成成功！序列差异: {diff_stats['base_diff_ratio']:.3f}")
        print(f"   原始序列: {test_seq[:30]}...")
        print(f"   DNA混淆: {confusion_variant[:30]}...")
    else:
        print("   ❌ 生成失败")
    
    # 测试负样本变体
    print("\n3. 测试负样本变体生成:")
    negative_variant = generate_negative_variant(test_seq, max_attempts=10)
    if negative_variant:
        diff_stats = calculate_sequence_differences(test_seq, negative_variant)
        print(f"   ✅ 生成成功！序列差异: {diff_stats['base_diff_ratio']:.3f}")
        print(f"   原始序列: {test_seq[:30]}...")
        print(f"   负样本: {negative_variant[:30]}...")
    else:
        print("   ❌ 生成失败")
    
    # 测试三元组视图生成
    print("\n4. 测试三元组视图生成:")
    test_segment = {
        'original_seq': test_seq,
        'label': 1,
        'segment_id': 'test_seq'
    }
    triplet_result = generate_triplet_views(test_segment, 
                                          max_semantic_attempts=10, 
                                          max_confusion_attempts=10,
                                          max_negative_attempts=10)
    print(f"   生成结果: {triplet_result['num_positive_views']}个正样本, {triplet_result['num_negative_views']}个负样本")
    print(f"   视图类型: {triplet_result.get('view_types', [])}")


if __name__ == "__main__":
    test_variant_generation()