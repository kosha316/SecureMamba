"""
完全可训练的 Nucleotide Transformer v3 (NTv3_8M_pre) 嵌入层
支持冻结NTv3参数，内部配置只解冻deconv_tower_blocks模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Optional, Dict, Any, Tuple
import os
import hashlib
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class NTv3TrainableEmbedding(nn.Module):
    """NTv3_8M_pre 可训练嵌入层，内部配置只解冻deconv_tower_blocks"""
    
    def __init__(
        self,
        model_repo: str = "InstaDeepAI/NTv3_8M_pre",
        output_dim: int = 256,  # 输出到Mamba的维度
        max_seq_len: int = 2048,  # DNA碱基数量
        use_cache: bool = False,  # 强制禁用缓存以确保维度一致
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        freeze_transformer: bool = False  # 是否冻结NTv3参数
    ):
        super().__init__()
        
        self.model_repo = model_repo
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache
        self.trust_remote_code = trust_remote_code
        self.freeze_transformer = freeze_transformer  # 存储冻结状态
        
        # 内部配置：只解冻deconv_tower_blocks模块
        # 可以在这里修改要解冻的模块
        self._unfreeze_target_modules = ["deconv_tower_blocks"]
        
        # 自动选择设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🔍 加载Nucleotide Transformer v3: {model_repo}")
        print(f"  设备: {self.device}")
        print(f"  远程代码信任: {trust_remote_code}")
        print(f"  最大序列长度: {max_seq_len}")
        print(f"  冻结NTv3: {freeze_transformer}")
        
        try:
            # 1. 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_repo,
                trust_remote_code=trust_remote_code
            )
            
            # 打印tokenizer信息
            print(f"  Tokenizer词汇表大小: {len(self.tokenizer)}")
            print(f"  填充token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
            
            # 2. 加载完整模型
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_repo,
                trust_remote_code=trust_remote_code
            ).to(self.device)
            
            print(f"✅ 模型加载成功")
            
            # NTv3_8M_pre的隐藏层维度是256
            self.hidden_size = 256
            print(f"  NTv3隐藏层大小: {self.hidden_size} (固定)")

            # 打印简单结构
            self.print_basic_info()

            # 根据freeze_transformer参数设置模型参数是否可训练
            if self.freeze_transformer:
                # 冻结参数
                for param in self.model.parameters():
                    param.requires_grad = False
                
                # 然后解冻指定的模块
                if self._unfreeze_target_modules:
                    self._unfreeze_specific_modules()
                    print(f"  🔥 已解冻指定模块: {self._unfreeze_target_modules}")
                else:
                    print(f"  ❄️  NTv3模型参数已冻结")
            else:
                print(f"  🔥 NTv3模型参数可训练（完全微调）")
            
            # 模型架构信息
            print(f"📋 模型架构: U-Net style conv tower → Transformer stack → deconv tower → LM head")
            print(f"  输入要求: 序列长度必须是128的倍数")
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            raise
        
        # 缓存系统（禁用，以确保特征维度一致）
        self.cache = None  # 强制禁用缓存
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 自适应池化层：将模型输出统一到固定长度
        self.adaptive_pool = nn.AdaptiveAvgPool1d(max_seq_len)
        
        # 投影层：将模型输出投影到我们需要的维度
        if self.hidden_size != output_dim:
            self.feature_projection = nn.Sequential(
                nn.Linear(self.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ).to(self.device)
            print(f"  启用投影层: {self.hidden_size} -> {output_dim}")
        else:
            self.feature_projection = nn.Identity()
            print(f"  无投影层 (隐藏层={output_dim})")
        
        print(f"📊 NTv3嵌入配置:")
        print(f"  模型仓库: {model_repo}")
        print(f"  原始隐藏大小: {self.hidden_size}")
        print(f"  输出维度: {output_dim}")
        print(f"  最大序列长度(碱基): {max_seq_len}")
        print(f"  使用缓存: {use_cache}")
        print(f"  冻结参数: {freeze_transformer}")
        if self.freeze_transformer and self._unfreeze_target_modules:
            print(f"  解冻模块: {self._unfreeze_target_modules}")
        
        # 打印参数统计
        self.print_param_stats()
    
    def _unfreeze_specific_modules(self):
        """解冻特定模块（内部方法）"""
        if not self._unfreeze_target_modules:
            return
        
        total_unfrozen = 0
        
        # 遍历所有模块
        for module_name in self._unfreeze_target_modules:
            module_found = False
            
            # 递归搜索模块
            for name, module in self.model.named_modules():
                if module_name in name:
                    # 找到模块，解冻其所有参数
                    for param in module.parameters():
                        param.requires_grad = True
                        total_unfrozen += param.numel()
                    module_found = True
                    print(f"    找到并解冻模块: {name}")
            
            if not module_found:
                print(f"  ⚠️  警告: 未找到模块 '{module_name}'")
        
        print(f"  ✅ 总共解冻了 {total_unfrozen:,} 个参数")
    
    def print_param_stats(self):
        """打印参数统计信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 按模块统计参数
        print("\n📈 NTv3模块参数统计:")
        print("-" * 40)
        
        # 统计主要模块的参数
        module_stats = {}
        for name, module in self.model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                module_stats[name] = module_params
        
        # 打印模块参数
        for module_name, params in module_stats.items():
            trainable = sum(p.numel() for p in getattr(self.model, module_name).parameters() if p.requires_grad)
            status = "可训练" if trainable > 0 else "冻结"
            print(f"  {module_name:20s}: {params:12,} 参数 ({status})")
        
        print("-" * 40)
        
        # 投影层参数
        proj_params = sum(p.numel() for p in self.feature_projection.parameters())
        
        print(f"\n总计:")
        print(f"  NTv3总参数: {total_params:,}")
        print(f"  NTv3可训练: {trainable_params:,}")
        print(f"  投影层参数: {proj_params:,}")
        
        if total_params > 0:
            trainable_ratio = trainable_params / total_params * 100
            print(f"  NTv3可训练比例: {trainable_ratio:.1f}%")
    
    def _preprocess_sequence(self, sequence: str) -> str:
        """预处理DNA序列 - 确保序列只包含有效字符"""
        # 确保序列只包含有效字符
        valid_chars = set('ACGTNacgtn')
        sequence = ''.join([c for c in sequence if c in valid_chars])
        
        if not sequence:
            return 'N' * 100  # 返回最小长度序列
        
        # 转换为大写
        return sequence.upper()
    
    def _tokenize_sequences(self, sequences: List[str]) -> torch.Tensor:
        """tokenize序列 - 只返回input_ids"""
        # 预处理所有序列（仅清理字符）
        processed_seqs = [self._preprocess_sequence(seq) for seq in sequences]
        
        # 使用tokenizer进行tokenization和padding
        batch = self.tokenizer(
            processed_seqs,
            add_special_tokens=False,  # 不添加特殊token
            padding=True,  # 启用padding
            pad_to_multiple_of=128,  # 填充到128的倍数
            max_length=self.max_seq_len,  # 最大长度
            truncation=True,  # 超过长度则截断
            return_tensors="pt"
        )
        
        # 只返回input_ids
        input_ids = batch['input_ids']
        
        return input_ids
    
    def _process_model_output(self, features: torch.Tensor) -> torch.Tensor:
        """处理模型输出，使用自适应池化统一特征长度"""
        batch_size, seq_len, hidden_dim = features.shape
        
        # NTv3输出特征长度可能不是max_seq_len，需要进行统一
        if seq_len != self.max_seq_len:
            # 使用自适应平均池化统一到max_seq_len长度
            # 首先转置维度: [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim, seq_len]
            features_t = features.transpose(1, 2)
            
            # 自适应池化到max_seq_len长度
            features_pooled = self.adaptive_pool(features_t)
            
            # 转置回来: [batch_size, hidden_dim, max_seq_len] -> [batch_size, max_seq_len, hidden_dim]
            features = features_pooled.transpose(1, 2)
            
            # 验证现在长度是否正确
            new_seq_len = features.shape[1]
            if new_seq_len != self.max_seq_len:
                print(f"⚠️  自适应池化后长度仍不匹配: {new_seq_len} != {self.max_seq_len}")
        
        # 投影到目标维度
        features = self.feature_projection(features)
        
        return features
    
    def extract_features(self, sequences: List[str], training_mode: bool = True) -> torch.Tensor:
        """
        提取特征 - 不使用缓存，确保维度一致
        
        Args:
            sequences: DNA序列列表
            training_mode: 是否为训练模式
        """
        if not sequences:
            return torch.empty(0, self.max_seq_len, self.output_dim, device=self.device)
        
        # Tokenize所有序列
        input_ids = self._tokenize_sequences(sequences)
        input_ids = input_ids.to(self.device)
        
        # 根据冻结状态和训练模式决定是否计算梯度
        # 检查是否有任何参数需要梯度
        has_trainable_params = any(p.requires_grad for p in self.model.parameters())
        
        if training_mode and has_trainable_params:
            # 训练模式且NTv3有可训练参数：计算梯度
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        else:
            # 评估模式或NTv3已冻结：不计算梯度
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        
        # 提取特征：使用最后一层隐藏状态
        hidden_states = outputs.hidden_states
        features = hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # 处理特征，使用自适应池化确保维度一致
        features = self._process_model_output(features)
        
        return features
    
    def forward(self, sequences: List[str], training_mode: bool = True) -> torch.Tensor:
        """前向传播"""
        return self.extract_features(sequences, training_mode=training_mode)
    
    def clear_cache(self):
        """清空缓存"""
        print("ℹ️  缓存已禁用")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'cache_enabled': False,
            'cache_size': 0,
            'hit_rate': 0.0
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        vocab_size = len(self.tokenizer) if hasattr(self, 'tokenizer') else 'N/A'
        
        return {
            'model_repo': self.model_repo,
            'vocab_size': vocab_size,
            'hidden_size': self.hidden_size,
            'output_dim': self.output_dim,
            'max_seq_len': self.max_seq_len,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'freeze_transformer': self.freeze_transformer,
            'unfreeze_modules': self._unfreeze_target_modules if self.freeze_transformer else [],
            'cache_enabled': False,
            'pad_token': self.tokenizer.pad_token,
            'pad_token_id': self.tokenizer.pad_token_id,
            'adaptive_pool': True,
            'pool_target_length': self.max_seq_len
        }
    
    def print_model_summary(self):
        """打印模型摘要"""
        info = self.get_model_info()
        print(f"🧬 NTv3嵌入模型摘要")
        print(f"  =========================================")
        print(f"  模型仓库: {info['model_repo']}")
        print(f"  词汇表大小: {info['vocab_size']}")
        print(f"  隐藏层大小: {info['hidden_size']} (NTv3_8M_pre)")
        print(f"  输出维度: {info['output_dim']}")
        print(f"  最大序列长度: {info['max_seq_len']}")
        print(f"  冻结NTv3: {info['freeze_transformer']}")
        if info['freeze_transformer'] and info['unfreeze_modules']:
            print(f"  解冻模块: {info['unfreeze_modules']}")
        print(f"  总参数量: {info['total_params']:,}")
        print(f"  可训练参数量: {info['trainable_params']:,}")
        if info['total_params'] > 0:
            trainable_ratio = info['trainable_params'] / info['total_params'] * 100
            print(f"  可训练比例: {trainable_ratio:.1f}%")
        print(f"  =========================================")
    
    def set_freeze_state(self, freeze: bool):
        """动态设置冻结状态"""
        if freeze != self.freeze_transformer:
            self.freeze_transformer = freeze
            
            if freeze:
                # 冻结所有参数
                for param in self.model.parameters():
                    param.requires_grad = False
                
                # 然后解冻指定模块
                if self._unfreeze_target_modules:
                    self._unfreeze_specific_modules()
            else:
                # 解冻所有参数
                for param in self.model.parameters():
                    param.requires_grad = True
            
            state = "冻结" if freeze else "解冻"
            print(f"🔄 NTv3参数已{state}")
            if freeze and self._unfreeze_target_modules:
                print(f"  解冻模块: {self._unfreeze_target_modules}")
            
            self.print_param_stats()
    
    def set_unfreeze_modules(self, modules: List[str]):
        """动态设置要解冻的模块（内部使用）"""
        self._unfreeze_target_modules = modules
        if self.freeze_transformer:
            print(f"🔄 更新解冻模块: {modules}")
            # 重新应用冻结/解冻状态
            self.set_freeze_state(self.freeze_transformer)

    def print_basic_info(self):
        """打印简化的NTv3网络层结构"""
        print("🧬 NTv3网络层结构 (简化版)")
        print("=" * 50)
        
        print(f"模型类型: {type(self.model).__name__}")
        print(f"隐藏层大小: {self.hidden_size}")
        print(f"输出维度: {self.output_dim}")
        
        print("\n主要模块:")
        print("-" * 30)
        
        # 只打印主要模块
        for name, module in self.model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            num_params_str = f"{num_params:,}" if num_params > 0 else "0"
            
            # 检查是否有子模块
            children = list(module.children())
            if children:
                print(f"├─ {name} ({type(module).__name__}): {num_params_str} 参数")
                # 打印第一层子模块
                for child_name, child_module in module.named_children():
                    child_params = sum(p.numel() for p in child_module.parameters())
                    if child_params > 0:
                        print(f"│  └─ {child_name}: {child_params:,} 参数")
            else:
                print(f"└─ {name}: {num_params_str} 参数")
        
        print("\n参数统计:")
        print("-" * 30)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"总参数: {total_params:,}")
        print(f"可训练: {trainable_params:,}")
        print(f"冻结: {total_params - trainable_params:,}")
        
        print("=" * 50)
        
        # 显示内部配置
        if self.freeze_transformer and self._unfreeze_target_modules:
            print(f"内部配置: 冻结NTv3，只解冻 {self._unfreeze_target_modules} 模块")
        elif self.freeze_transformer:
            print(f"内部配置: 冻结NTv3所有参数")
        else:
            print(f"内部配置: 完全解冻NTv3")


# 工厂函数
def create_ntv3_trainable_embedding(config: dict) -> NTv3TrainableEmbedding:
    """创建NTv3完全可训练嵌入层"""
    return NTv3TrainableEmbedding(
        model_repo=config.get('transformer_model_repo', 'InstaDeepAI/NTv3_8M_pre'),
        output_dim=config.get('embedding_dim', 256),
        max_seq_len=config.get('max_seq_len', 2048),
        use_cache=config.get('use_caching', False),  # 强制禁用缓存
        device=config.get('device', None),
        trust_remote_code=config.get('trust_remote_code', True),
        freeze_transformer=config.get('freeze_transformer', False)
    )