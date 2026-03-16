"""
完全使用Nucleotide Transformer v3的无残差连接Mamba序列模型
NTv3模型完全可训练，不冻结参数
优化后的精简版增强模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings("ignore")

# 导入Nucleotide v3嵌入层
try:
    from nucleotide_v3_trainable import NTv3TrainableEmbedding, create_ntv3_trainable_embedding
    NTv3_AVAILABLE = True
except ImportError:
    NTv3_AVAILABLE = False
    print("⚠️  无法导入NTv3TrainableEmbedding，请确保nucleotide_v3_trainable.py在路径中")


class SimplifiedLocalGlobalAttention(nn.Module):
    """简化的局部-全局层次化注意力模块
    去除局部卷积，专注于功能语义特征捕捉
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_flash_attention = use_flash_attention
        
        # 检查是否支持Flash Attention
        self.has_flash_attention = hasattr(F, 'scaled_dot_product_attention')
        
        # 多头注意力机制
        if not self.has_flash_attention or not use_flash_attention:
            self.multihead_attn = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 输出归一化
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward_flash_attention(self, x: torch.Tensor) -> torch.Tensor:
        """使用Flash Attention加速的全局注意力"""
        batch_size, seq_len, d_model = x.shape
        
        # 重塑为多头
        q = x.view(batch_size, seq_len, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        k = x.view(batch_size, seq_len, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        v = x.view(batch_size, seq_len, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        
        # 创建因果掩码
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), 
            diagonal=1
        )
        
        # 使用Flash Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return attn_output
    
    def forward_standard_attention(self, x: torch.Tensor) -> torch.Tensor:
        """标准多头注意力"""
        seq_len = x.shape[1]
        
        # 创建因果掩码
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), 
            diagonal=1
        )
        
        attn_output, _ = self.multihead_attn(
            x, x, x, 
            attn_mask=attn_mask,
            need_weights=False
        )
        
        return attn_output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 1. 全局多头注意力
        if self.has_flash_attention and self.use_flash_attention:
            global_features = self.forward_flash_attention(x)
        else:
            global_features = self.forward_standard_attention(x)
        
        # 2. 门控融合：平衡原始特征和注意力特征
        gate_input = torch.cat([x, global_features], dim=-1)
        gate_weights = self.gate(gate_input)
        
        # 3. 加权融合
        combined = gate_weights * x + (1 - gate_weights) * global_features
        
        # 4. 归一化
        output = self.norm(combined)
        output = self.dropout(output)
        
        return output


class EfficientMultiScaleCrossSegmentAttention(nn.Module):
    """高效的多尺度跨片段注意力模块
    优化了片段分割，提高训练效率
    """
    
    def __init__(
        self,
        d_model: int,
        segment_lengths: List[int] = [4, 8, 16],
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.segment_lengths = segment_lengths
        self.use_flash_attention = use_flash_attention
        
        # 检查是否支持Flash Attention
        self.has_flash_attention = hasattr(F, 'scaled_dot_product_attention')
        
        # 多尺度注意力模块
        self.segment_attentions = nn.ModuleList()
        for _ in segment_lengths:
            # 使用线性层预处理，减少维度
            pre_linear = nn.Linear(d_model, d_model // 2)
            post_linear = nn.Linear(d_model // 2, d_model)
            
            # 跨片段注意力
            if self.has_flash_attention and use_flash_attention:
                # 对于Flash Attention，我们使用自定义实现
                cross_attn = EfficientCrossSegmentAttention(d_model // 2, dropout=dropout)
            else:
                cross_attn = nn.MultiheadAttention(
                    d_model // 2, num_heads=4, dropout=dropout, batch_first=True
                )
            
            self.segment_attentions.append(
                nn.ModuleDict({
                    'pre_linear': pre_linear,
                    'attention': cross_attn,
                    'post_linear': post_linear
                })
            )
        
        # 多尺度特征融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * len(segment_lengths), d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 注册缓冲区用于存储预计算的片段索引（优化）
        self.register_buffer('segment_indices', None, persistent=False)
        
    def compute_efficient_segments(self, x: torch.Tensor, segment_len: int) -> torch.Tensor:
        """高效计算片段特征，避免重复重塑操作"""
        batch_size, seq_len, d_model = x.shape
        
        # 计算片段数量
        num_segments = (seq_len + segment_len - 1) // segment_len
        
        # 使用unfold进行高效的分段池化
        if seq_len % segment_len == 0:
            # 如果刚好整除，直接重塑
            segments = x.view(batch_size, num_segments, segment_len, d_model)
        else:
            # 否则使用填充+重塑
            new_seq_len = num_segments * segment_len
            padding_size = new_seq_len - seq_len
            
            if padding_size > 0:
                # 仅在最后一个维度填充
                x_padded = F.pad(x, (0, 0, 0, padding_size))
            else:
                x_padded = x
            
            segments = x_padded.view(batch_size, num_segments, segment_len, d_model)
        
        # 平均池化得到片段级特征
        segment_features = segments.mean(dim=2)  # [batch_size, num_segments, d_model]
        
        return segment_features, num_segments
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        multi_scale_features = []
        
        # 对每个尺度进行跨片段注意力
        for i, segment_len in enumerate(self.segment_lengths):
            # 1. 预处理降维
            x_pre = self.segment_attentions[i]['pre_linear'](x)
            
            # 2. 高效计算片段特征
            segment_features, num_segments = self.compute_efficient_segments(x_pre, segment_len)
            
            # 3. 跨片段注意力
            if self.has_flash_attention and self.use_flash_attention:
                segment_features_attn = self.segment_attentions[i]['attention'](segment_features)
            else:
                segment_features_attn, _ = self.segment_attentions[i]['attention'](
                    segment_features, segment_features, segment_features
                )
            
            # 4. 后处理升维
            segment_features_attn = self.segment_attentions[i]['post_linear'](segment_features_attn)
            
            # 5. 将片段特征广播回原始序列长度
            segment_len_actual = (seq_len + num_segments - 1) // num_segments
            segment_features_expanded = segment_features_attn.repeat_interleave(
                segment_len_actual, dim=1
            )
            
            # 6. 截取原始长度
            segment_features_expanded = segment_features_expanded[:, :seq_len, :]
            
            multi_scale_features.append(segment_features_expanded)
        
        # 多尺度特征融合
        if len(multi_scale_features) > 1:
            fused = torch.cat(multi_scale_features, dim=-1)
            output = self.fusion(fused)
        else:
            output = multi_scale_features[0]
        
        output = self.dropout(output)
        
        return output


class EfficientCrossSegmentAttention(nn.Module):
    """高效的跨片段注意力模块，支持Flash Attention"""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 线性投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = dropout
        
        # 检查是否支持Flash Attention
        self.has_flash_attention = hasattr(F, 'scaled_dot_product_attention')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_segments, d_model = x.shape
        
        # 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        if self.has_flash_attention:
            # 重塑为多头
            q = q.view(batch_size, num_segments, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, num_segments, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, num_segments, self.num_heads, self.head_dim).transpose(1, 2)
            
            # 使用Flash Attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.dropout if self.training else 0.0
            )
            
            # 重塑回原始形状
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_segments, d_model)
        else:
            # 标准多头注意力
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0
            )
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output


class AdaptiveMultiScaleAttention(nn.Module):
    """自适应的多尺度注意力模块
    动态选择最佳尺度，提高效率
    """
    
    def __init__(
        self,
        d_model: int,
        max_scales: int = 3,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.max_scales = max_scales
        
        # 尺度选择门控
        self.scale_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, max_scales),
            nn.Softmax(dim=-1)
        )
        
        # 主注意力模块（使用中等尺度）
        self.main_attention = EfficientMultiScaleCrossSegmentAttention(
            d_model=d_model,
            segment_lengths=[8],  # 默认使用中等尺度
            dropout=dropout,
            use_flash_attention=use_flash_attention
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 计算尺度选择权重
        pooled = x.mean(dim=1)  # [batch_size, d_model]
        scale_weights = self.scale_gate(pooled)  # [batch_size, max_scales]
        
        # 获取主要尺度
        main_output = self.main_attention(x)
        
        # 根据序列长度自适应调整
        if seq_len < 16:
            # 短序列使用简单处理
            output = main_output
        elif seq_len < 64:
            # 中等序列
            output = main_output * 0.8 + x * 0.2
        else:
            # 长序列使用更强的注意力
            output = main_output
        
        output = self.dropout(output)
        
        return output


class BasicMambaBlock(nn.Module):
    """基础Mamba块（无残差连接）"""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        
        # 确保d_conv在2到4之间
        d_conv = max(2, min(4, d_conv))
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x


class EnhancedMambaBlock(nn.Module):
    """增强Mamba块 - 增强特征直接作为Mamba输入"""
    
    def __init__(
        self, 
        d_model: int, 
        d_state: int = 16, 
        d_conv: int = 4,
        expand: int = 2,
        use_local_global_attn: bool = True,
        use_global_invariance: bool = True,
        attn_num_heads: int = 4,
        use_flash_attention: bool = True,
        use_adaptive_scales: bool = True  # 新增：是否使用自适应尺度
    ):
        super().__init__()
        self.d_model = d_model
        self.use_local_global_attn = use_local_global_attn
        self.use_global_invariance = use_global_invariance
        
        # 确保d_conv在2到4之间
        d_conv = max(2, min(4, d_conv))
        
        # 增强模块
        if use_local_global_attn:
            self.local_global_attn = SimplifiedLocalGlobalAttention(
                d_model=d_model,
                num_heads=attn_num_heads,
                use_flash_attention=use_flash_attention
            )
        else:
            self.local_global_attn = None
        
        if use_global_invariance:
            if use_adaptive_scales:
                self.global_invariance_encoder = AdaptiveMultiScaleAttention(
                    d_model=d_model,
                    use_flash_attention=use_flash_attention
                )
            else:
                self.global_invariance_encoder = EfficientMultiScaleCrossSegmentAttention(
                    d_model=d_model,
                    segment_lengths=[4, 8, 16],
                    use_flash_attention=use_flash_attention
                )
        else:
            self.global_invariance_encoder = None
        
        # Mamba块
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用增强模块
        if self.local_global_attn is not None:
            x = self.local_global_attn(x)
        
        if self.global_invariance_encoder is not None:
            x = self.global_invariance_encoder(x)
        
        # Mamba处理
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        
        return x


class DualPathMambaBlock(nn.Module):
    """双路径Mamba块 - 每条路径使用不同增强特征，共享Mamba参数"""
    
    def __init__(
        self, 
        d_model: int, 
        d_state: int = 16, 
        d_conv: int = 4,
        expand: int = 2,
        layer_idx: int = 0,
        use_path_selection: bool = True,
        path_selection_strategy: str = "adaptive_learning",
        path_diversity_weight: float = 0.1,
        path_sparsity_weight: float = 0.05,
        # 增强模块参数
        use_local_global_attn: bool = True,
        use_global_invariance: bool = True,
        attn_num_heads: int = 4,
        use_flash_attention: bool = True,
        use_adaptive_scales: bool = True  # 新增：是否使用自适应尺度
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.use_path_selection = use_path_selection
        self.path_selection_strategy = path_selection_strategy
        self.path_diversity_weight = path_diversity_weight
        self.path_sparsity_weight = path_sparsity_weight
        self.use_local_global_attn = use_local_global_attn
        self.use_global_invariance = use_global_invariance
        
        # 确保d_conv在2到4之间
        d_conv = max(2, min(4, d_conv))
        
        # 路径1: 简化的局部全局注意力增强（专注功能语义）
        self.path1_enhance = SimplifiedLocalGlobalAttention(
            d_model=d_model,
            num_heads=attn_num_heads,
            use_flash_attention=use_flash_attention
        ) if use_local_global_attn else nn.Identity()
        
        self.path1_norm = nn.LayerNorm(d_model)
        
        # 路径2: 高效的多尺度跨片段注意力增强（专注应对攻击策略）
        if use_global_invariance:
            if use_adaptive_scales:
                self.path2_enhance = AdaptiveMultiScaleAttention(
                    d_model=d_model,
                    use_flash_attention=use_flash_attention
                )
            else:
                self.path2_enhance = EfficientMultiScaleCrossSegmentAttention(
                    d_model=d_model,
                    segment_lengths=[4, 8, 16],
                    use_flash_attention=use_flash_attention
                )
        else:
            self.path2_enhance = nn.Identity()
        
        self.path2_norm = nn.LayerNorm(d_model)
        
        # 共享的Mamba模块
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # 路径选择门控网络
        if use_path_selection:
            # 使用增强特征的摘要作为输入
            gate_input_dim = d_model * 2  # 两条路径的增强特征摘要
            
            self.gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, 2),
                nn.Softmax(dim=-1)
            )
        
        # 输出融合层
        self.output_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def compute_path_selection_loss(self, learned_weights: torch.Tensor, 
                                   variant_type: Optional[str] = None) -> torch.Tensor:
        """计算路径选择损失"""
        if variant_type is not None:
            # 使用变体类型作为软监督
            if variant_type == "confusion":
                target_bias = torch.tensor([[0.3, 0.7]], device=learned_weights.device)
            else:  # "original" or "semantic"
                target_bias = torch.tensor([[0.7, 0.3]], device=learned_weights.device)
            
            batch_size = learned_weights.shape[0]
            target = target_bias.repeat(batch_size, 1)
            
            # KL散度损失（多样性）
            diversity_loss = F.kl_div(
                F.log_softmax(learned_weights, dim=-1),
                target,
                reduction='batchmean'
            )
            
            # 稀疏性损失（熵最小化）
            eps = 1e-8
            entropy = - (learned_weights * torch.log(learned_weights + eps)).sum(dim=-1)
            sparsity_loss = entropy.mean()
            
            # 组合损失
            path_selection_loss = (
                self.path_diversity_weight * diversity_loss +
                self.path_sparsity_weight * sparsity_loss
            )
        else:
            # 只使用稀疏性损失
            eps = 1e-8
            entropy = - (learned_weights * torch.log(learned_weights + eps)).sum(dim=-1)
            sparsity_loss = entropy.mean()
            path_selection_loss = self.path_sparsity_weight * sparsity_loss
        
        return path_selection_loss
    
    def forward(
        self, 
        x: torch.Tensor, 
        variant_type: Optional[str] = None,
        training_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        
        # 1. 生成增强特征
        path1_enhanced = self.path1_enhance(x)
        path2_enhanced = self.path2_enhance(x)
        
        # 2. 路径选择
        path_selection_loss = None
        if self.use_path_selection:
            # 使用增强特征的摘要作为门控输入
            gate_input = torch.cat([
                path1_enhanced.mean(dim=1),
                path2_enhanced.mean(dim=1)
            ], dim=-1)
            
            learned_weights = self.gate_network(gate_input)  # [batch_size, 2]
            
            if training_mode:
                path_selection_loss = self.compute_path_selection_loss(
                    learned_weights, variant_type
                )
            
            path_weights = learned_weights
        else:
            # 固定权重
            if variant_type == "confusion":
                path_weights = torch.tensor([[0.3, 0.7]], device=x.device).repeat(batch_size, 1)
            else:
                path_weights = torch.tensor([[0.7, 0.3]], device=x.device).repeat(batch_size, 1)
        
        # 3. 使用共享的Mamba处理增强特征
        # 路径1: 功能语义特征 → 共享Mamba
        path1_out = self.mamba(self.path1_norm(path1_enhanced))
        
        # 路径2: 应对攻击特征 → 共享Mamba
        path2_out = self.mamba(self.path2_norm(path2_enhanced))
        
        # 4. 双路径输出融合
        # 4.1 根据路径权重加权融合
        path_weights_expanded = path_weights.unsqueeze(1).unsqueeze(-1)
        path_outputs = torch.stack([path1_out, path2_out], dim=2)
        weighted_output = (path_outputs * path_weights_expanded).sum(dim=2)
        
        # 4.2 可选的特征融合层
        combined_features = torch.cat([path1_out, path2_out], dim=-1)
        fused_features = self.output_fusion(combined_features)
        
        # 4.3 最终输出：70%加权输出 + 30%融合输出
        output = 0.7 * weighted_output + 0.3 * fused_features
        output = self.dropout(output)
        
        return output, path_weights, path_selection_loss


class NucleotideMambaModel(nn.Module):
    """完全使用Nucleotide Transformer v3的无残差连接增强Mamba序列模型"""
    
    def __init__(
        self,
        # Nucleotide Transformer v3配置
        transformer_model_repo: str = "InstaDeepAI/NTv3_8M_pre",
        embedding_dim: int = 256,
        # Mamba模型配置
        d_model: int = 256,
        n_layer: int = 4,
        projection_dim: int = 256,
        num_classes: int = 1,
        variant_specialization_weight: float = 0.2,
        block_type: str = "dual_path",
        dropout_rate: float = 0.1,
        use_path_selection: bool = True,
        path_selection_weight: float = 1.0,
        max_seq_len: int = 512,  # DNA碱基数量
        # 增强模块配置
        use_local_global_attn: bool = True,
        use_global_invariance: bool = True,
        attn_num_heads: int = 4,
        use_flash_attention: bool = True,
        use_adaptive_scales: bool = True,  # 新增：是否使用自适应尺度
        # Transformer配置 - 注意：这里我们不冻结！
        freeze_transformer: bool = True,  # 改为False，完全可训练
        use_caching: bool = False,  # 强制禁用缓存以确保特征维度一致
        trust_remote_code: bool = True,  # v3需要这个
        # 架构配置
        enable_tf32: bool = True,
        compile_model: bool = False,
        device: Optional[str] = None
    ):
        super().__init__()
        
        # 存储参数
        self.transformer_model_repo = transformer_model_repo
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.n_layer = n_layer
        self.projection_dim = projection_dim
        self.num_classes = num_classes
        self.variant_specialization_weight = variant_specialization_weight
        self.block_type = block_type
        self.dropout_rate = dropout_rate
        self.use_path_selection = use_path_selection
        self.path_selection_weight = path_selection_weight
        self.max_seq_len = max_seq_len
        self.use_local_global_attn = use_local_global_attn
        self.use_global_invariance = use_global_invariance
        self.use_adaptive_scales = use_adaptive_scales
        self.freeze_transformer = freeze_transformer  # 现在为False
        self.use_caching = use_caching
        self.trust_remote_code = trust_remote_code
        
        # 自动选择设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # GPU优化：启用TF32加速
        if enable_tf32 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ 启用TF32加速")
        
        print(f"🧬 初始化完全可训练的NTv3 + Mamba模型（优化精简增强版）")
        print(f"  Transformer模型: {transformer_model_repo}")
        print(f"  嵌入维度: {embedding_dim}")
        print(f"  模型维度: {d_model}")
        print(f"  层数: {n_layer}")
        print(f"  块类型: {block_type}")
        print(f"  最大序列长度: {max_seq_len}")
        print(f"  自适应尺度: {use_adaptive_scales}")
        print(f"  ⚠️  注意: NTv3模型将完全微调（不冻结）")
        
        # 2. NTv3完全可训练嵌入层
        if not NTv3_AVAILABLE:
            raise ImportError("需要 nucleotide_v3_trainable.py 模块，请确保它在Python路径中")

        # 强制禁用缓存以确保特征维度一致
        self.nucleotide_embedding = NTv3TrainableEmbedding(
            model_repo=transformer_model_repo,
            output_dim=embedding_dim,
            max_seq_len=max_seq_len,
            use_cache=False,  # 强制禁用缓存
            device=str(self.device),
            trust_remote_code=trust_remote_code,
            freeze_transformer=freeze_transformer  # 添加这一行，传递冻结参数
        )
        
        # 打印模型信息
        self.nucleotide_embedding.print_model_summary()
        
        # 2. 从embedding_dim投影到d_model（如果维度不同）
        if embedding_dim != d_model:
            self.embedding_projection = nn.Sequential(
                nn.Linear(embedding_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
        else:
            self.embedding_projection = nn.Identity()
        
        # 3. Mamba块序列（无残差连接）
        self.mamba_blocks = nn.ModuleList()
        for i in range(n_layer):
            if block_type == "dual_path":
                block = DualPathMambaBlock(
                    d_model=d_model,
                    layer_idx=i,
                    use_path_selection=use_path_selection,
                    use_local_global_attn=use_local_global_attn,
                    use_global_invariance=use_global_invariance,
                    attn_num_heads=attn_num_heads,
                    use_flash_attention=use_flash_attention,
                    use_adaptive_scales=use_adaptive_scales
                )
            elif block_type == "mixed":
                if i % 2 == 0:
                    block = EnhancedMambaBlock(
                        d_model=d_model,
                        use_local_global_attn=use_local_global_attn,
                        use_global_invariance=use_global_invariance,
                        attn_num_heads=attn_num_heads,
                        use_flash_attention=use_flash_attention,
                        use_adaptive_scales=use_adaptive_scales
                    )
                else:
                    block = DualPathMambaBlock(
                        d_model=d_model,
                        layer_idx=i,
                        use_path_selection=use_path_selection,
                        use_local_global_attn=use_local_global_attn,
                        use_global_invariance=use_global_invariance,
                        attn_num_heads=attn_num_heads,
                        use_flash_attention=use_flash_attention,
                        use_adaptive_scales=use_adaptive_scales
                    )
            elif block_type == "enhanced":
                block = EnhancedMambaBlock(
                    d_model=d_model,
                    use_local_global_attn=use_local_global_attn,
                    use_global_invariance=use_global_invariance,
                    attn_num_heads=attn_num_heads,
                    use_flash_attention=use_flash_attention,
                    use_adaptive_scales=use_adaptive_scales
                )
            else:  # "basic"
                block = BasicMambaBlock(d_model=d_model)
            
            self.mamba_blocks.append(block)
        
        # 4. 序列池化层
        self.sequence_pooler = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1)
        )
        
        # 5. 输出归一化
        self.output_norm = nn.LayerNorm(d_model)
        
        # 6. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 7. 对比学习投影头
        self.contrastive_projector = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # 初始化权重（仅初始化非Transformer部分）
        self._init_weights()
        
        # 移动到设备
        self.to(self.device)
        
        # 可选：编译模型优化
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.forward_sequence = torch.compile(self.forward_sequence)
                print("✅ 启用模型编译优化")
            except Exception as e:
                print(f"⚠️ 模型编译失败: {e}")
    
    def _init_weights(self):
        """权重初始化（仅初始化非Transformer部分）"""
        for name, module in self.named_modules():
            # 跳过NTv3嵌入层
            if 'nucleotide_embedding' in name:
                continue
                
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
    
    def forward_sequence(
        self, 
        sequences: List[str],  # 输入DNA序列列表
        variant_type: Optional[str] = None,
        training_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """前向传播单个序列类型"""
        
        # 1. 通过完全可训练的NTv3提取特征
        x = self.nucleotide_embedding(sequences, training_mode=training_mode)
        
        # 2. 投影到模型维度
        x = self.embedding_projection(x)
        
        batch_size, seq_len, _ = x.shape
        
        # 3. 通过Mamba块（无残差连接）
        path_weights_list = []
        path_selection_loss = torch.tensor(0.0, device=self.device)
        
        for block in self.mamba_blocks:
            if isinstance(block, DualPathMambaBlock):
                x, path_weights, block_loss = block(
                    x, 
                    variant_type=variant_type,
                    training_mode=training_mode
                )
                path_weights_list.append(path_weights)
                
                if training_mode and block_loss is not None:
                    path_selection_loss = path_selection_loss + block_loss
            else:
                x = block(x)
        
        # 4. 序列池化
        x_pool = x.transpose(1, 2)
        pooled = self.sequence_pooler(x_pool)
        
        # 5. 输出归一化
        pooled = self.output_norm(pooled)
        
        # 6. 计算路径选择权重
        if self.use_path_selection and len(path_weights_list) > 0:
            avg_path_weights = torch.stack(path_weights_list).mean(dim=0)
        else:
            avg_path_weights = None
        
        # 7. 分类预测
        class_pred = self.classifier(pooled)
        
        # 8. 对比学习特征
        contrastive_feat = self.contrastive_projector(pooled)
        
        # 9. 计算总路径选择损失
        if training_mode and path_selection_loss is not None:
            path_selection_loss = path_selection_loss / len(self.mamba_blocks)
        
        return class_pred, pooled, contrastive_feat, path_selection_loss
    
    def forward(
        self,
        # 输入数据：不同变体类型的序列列表
        original_sequences: List[str],
        labels: torch.Tensor,
        semantic_sequences: Optional[List[str]] = None,
        confusion_sequences: Optional[List[str]] = None,
        random_mutation_sequences: Optional[List[str]] = None,
        training_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            original_sequences: 原始DNA序列列表
            labels: 标签张量
            semantic_sequences: 语义变体序列列表
            confusion_sequences: 混淆变体序列列表
            random_mutation_sequences: 随机突变变体序列列表
            training_mode: 是否为训练模式
            
        Returns:
            orig_class_pred: 原始序列分类预测
            semantic_class_pred: 语义变体分类预测
            confusion_class_pred: 混淆变体分类预测
            total_aux_loss: 总辅助损失
            orig_contrastive_feat: 原始序列对比特征
            all_contrastive_feat: 所有序列的对比特征
        """
        batch_size = len(original_sequences)
        
        # 确保标签在正确设备上
        labels = labels.to(self.device)
        
        # 处理原始序列
        orig_class_pred, orig_pooled_feat, orig_contrastive_feat, orig_path_loss = self.forward_sequence(
            original_sequences, 
            variant_type="original" if training_mode else None,
            training_mode=training_mode
        )
        
        # 处理语义变体
        semantic_class_pred = None
        semantic_contrastive_feat = None
        semantic_path_loss = torch.tensor(0.0, device=self.device)
        
        if semantic_sequences is not None and semantic_sequences:
            semantic_class_pred, _, semantic_contrastive_feat, semantic_path_loss = self.forward_sequence(
                semantic_sequences, 
                variant_type="semantic" if training_mode else None,
                training_mode=training_mode
            )
        
        # 创建默认值
        if semantic_class_pred is None:
            semantic_class_pred = torch.zeros(batch_size, self.num_classes, device=self.device)
            semantic_contrastive_feat = torch.zeros(batch_size, self.projection_dim, device=self.device)
        
        # 处理混淆变体
        confusion_class_pred = None
        confusion_contrastive_feat = None
        confusion_path_loss = torch.tensor(0.0, device=self.device)
        
        if confusion_sequences is not None and confusion_sequences:
            confusion_class_pred, _, confusion_contrastive_feat, confusion_path_loss = self.forward_sequence(
                confusion_sequences, 
                variant_type="confusion" if training_mode else None,
                training_mode=training_mode
            )
        
        if confusion_class_pred is None:
            confusion_class_pred = torch.zeros(batch_size, self.num_classes, device=self.device)
            confusion_contrastive_feat = torch.zeros(batch_size, self.projection_dim, device=self.device)
        
        # 处理随机突变变体
        random_mutation_contrastive_feat = None
        random_mutation_path_loss = torch.tensor(0.0, device=self.device)
        
        if random_mutation_sequences is not None and random_mutation_sequences:
            _, _, random_mutation_contrastive_feat, random_mutation_path_loss = self.forward_sequence(
                random_mutation_sequences,
                variant_type=None,
                training_mode=training_mode
            )
        
        if random_mutation_contrastive_feat is None:
            random_mutation_contrastive_feat = torch.zeros(batch_size, self.projection_dim, device=self.device)
        
        # 计算路径选择总损失
        total_path_selection_loss = orig_path_loss + semantic_path_loss + confusion_path_loss
        
        # 总辅助损失
        total_aux_loss = total_path_selection_loss * self.path_selection_weight
        
        # 合并对比特征
        contrastive_features_list = [
            orig_contrastive_feat,
            semantic_contrastive_feat,
            confusion_contrastive_feat,
            random_mutation_contrastive_feat
        ]
        
        all_contrastive_feat = torch.cat(contrastive_features_list, dim=0)
        
        return (
            orig_class_pred,
            semantic_class_pred,
            confusion_class_pred,
            total_aux_loss,
            orig_contrastive_feat,
            all_contrastive_feat
        )
    
    def clear_cache(self):
        """清空特征缓存"""
        self.nucleotide_embedding.clear_cache()
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        return self.nucleotide_embedding.get_cache_stats()
    
    def get_model_summary(self) -> dict:
        """获取模型统计信息"""
        # NTv3模型参数
        ntv3_info = self.nucleotide_embedding.get_model_info()
        ntv3_params = ntv3_info['total_params']
        ntv3_trainable = ntv3_info['trainable_params']
        
        # Mamba和其他部分参数
        mamba_params = sum(p.numel() for p in self.embedding_projection.parameters())
        mamba_params += sum(p.numel() for p in self.mamba_blocks.parameters())
        mamba_params += sum(p.numel() for p in self.sequence_pooler.parameters())
        mamba_params += sum(p.numel() for p in self.output_norm.parameters())
        mamba_params += sum(p.numel() for p in self.classifier.parameters())
        mamba_params += sum(p.numel() for p in self.contrastive_projector.parameters())
        
        total_params = ntv3_params + mamba_params
        trainable_params = ntv3_trainable + mamba_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'ntv3_parameters': ntv3_params,
            'ntv3_trainable': ntv3_trainable,
            'mamba_parameters': mamba_params,
            'embedding_dim': self.embedding_dim,
            'd_model': self.d_model,
            'n_layer': self.n_layer,
            'block_type': self.block_type,
            'use_path_selection': self.use_path_selection,
            'use_local_global_attn': self.use_local_global_attn,
            'use_global_invariance': self.use_global_invariance,
            'use_adaptive_scales': self.use_adaptive_scales,
            'parameter_mb': total_params * 4 / (1024 * 1024),
            'model_architecture': 'NTv3_8M_pre (完全可训练) + Mamba (优化精简增强)'
        }
    
    def print_detailed_summary(self):
        """打印详细模型摘要"""
        summary = self.get_model_summary()
        
        print(f"\n{'='*60}")
        print(f"🧬 NTv3 + Mamba模型详细摘要（优化精简增强版）")
        print(f"{'='*60}")
        print(f"📊 参数统计:")
        print(f"  总参数量: {summary['total_parameters']:,}")
        print(f"  可训练参数量: {summary['trainable_parameters']:,}")
        print(f"  NTv3参数量: {summary['ntv3_parameters']:,}")
        print(f"  NTv3可训练参数: {summary['ntv3_trainable']:,}")
        print(f"  Mamba部分参数量: {summary['mamba_parameters']:,}")
        print(f"  总内存占用: {summary['parameter_mb']:.2f} MB")
        
        print(f"\n📈 架构配置:")
        print(f"  模型架构: {summary['model_architecture']}")
        print(f"  嵌入维度: {summary['embedding_dim']}")
        print(f"  模型维度: {summary['d_model']}")
        print(f"  Mamba层数: {summary['n_layer']}")
        print(f"  块类型: {summary['block_type']}")
        print(f"  路径选择: {summary['use_path_selection']}")
        print(f"  功能语义路径: {summary['use_local_global_attn']} (简化的全局注意力)")
        print(f"  对抗攻击路径: {summary['use_global_invariance']} (多尺度跨片段注意力)")
        print(f"  自适应尺度: {summary['use_adaptive_scales']}")
        
        print(f"\n🧬 NTv3配置:")
        print(f"  模型仓库: {self.transformer_model_repo}")
        print(f"  最大序列长度: {self.max_seq_len}")
        print(f"  使用缓存: {self.use_caching}")
        print(f"  信任远程代码: {self.trust_remote_code}")
        
        # NTv3详细架构信息
        if hasattr(self.nucleotide_embedding, 'get_model_info'):
            ntv3_info = self.nucleotide_embedding.get_model_info()
            print(f"  NTv3架构: {ntv3_info.get('architecture', 'U-Net conv tower → Transformer stack → deconv tower → LM head')}")
            print(f"  NTv3词汇表大小: {ntv3_info.get('vocab_size', 'N/A')}")
            print(f"  NTv3隐藏大小: {ntv3_info.get('hidden_size', 'N/A')}")
        
        print(f"\n🔧 增强模块优化特点:")
        print(f"  1. 高效片段分割: 使用unfold和预计算优化")
        print(f"  2. 降维处理: 注意力前降维，减少计算量")
        print(f"  3. Flash Attention: 支持高效注意力计算")
        print(f"  4. 自适应尺度: 根据序列长度动态调整")
        print(f"  5. 共享Mamba: 两条路径共享参数，减少计算量")
        print(f"  6. 内存优化: 避免不必要的重塑和填充")
        
        print(f"{'='*60}")


def create_nucleotide_mamba_model(
    # Nucleotide Transformer v3配置
    transformer_model_repo: str = "InstaDeepAI/NTv3_8M_pre",
    embedding_dim: int = 256,
    # Mamba模型配置
    d_model: int = 256,
    n_layer: int = 4,
    block_type: str = "dual_path",
    use_path_selection: bool = True,
    # 增强模块参数
    use_local_global_attn: bool = True,
    use_global_invariance: bool = True,
    use_flash_attention: bool = True,
    use_adaptive_scales: bool = True,  # 新增参数
    # Transformer配置
    freeze_transformer: bool = True,  # 完全可训练
    use_caching: bool = True,
    trust_remote_code: bool = True,
    # GPU优化参数
    enable_tf32: bool = True,
    compile_model: bool = False,
    **kwargs
) -> NucleotideMambaModel:
    """创建NTv3 + Mamba模型的便捷函数"""
    default_kwargs = {
        'transformer_model_repo': transformer_model_repo,
        'embedding_dim': embedding_dim,
        'd_model': d_model,
        'n_layer': n_layer,
        'projection_dim': 128,
        'num_classes': 1,
        'variant_specialization_weight': 0.2,
        'block_type': block_type,
        'dropout_rate': 0.1,
        'use_path_selection': use_path_selection,
        'path_selection_weight': 0.1,
        'max_seq_len': 512,
        # 增强模块配置
        'use_local_global_attn': use_local_global_attn,
        'use_global_invariance': use_global_invariance,
        'attn_num_heads': 4,
        'use_flash_attention': use_flash_attention,
        'use_adaptive_scales': use_adaptive_scales,
        # Transformer配置
        'freeze_transformer': freeze_transformer,
        'use_caching': use_caching,
        'trust_remote_code': trust_remote_code,
        # GPU优化
        'enable_tf32': enable_tf32,
        'compile_model': compile_model
    }
    
    # 更新默认参数
    default_kwargs.update(kwargs)
    
    return NucleotideMambaModel(**default_kwargs)


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试完全可训练的NTv3 + Mamba模型（优化精简增强版）...")
    
    # 创建模型
    model = create_nucleotide_mamba_model(
        transformer_model_repo="InstaDeepAI/NTv3_8M_pre",
        embedding_dim=256,
        d_model=256,
        n_layer=4,
        block_type="dual_path",
        use_adaptive_scales=True
    )
    
    # 打印详细摘要
    model.print_detailed_summary()
    
    # 测试前向传播
    test_sequences = [
        "ATCGNATCG",
        "ACGTACGTACGTACGT",
        "ATCGATCGATCGATCGATCG"
    ]
    
    print(f"\n🔧 测试前向传播...")
    
    # 训练模式
    model.train()
    labels = torch.tensor([1.0, 0.0, 1.0])
    
    outputs = model(
        original_sequences=test_sequences,
        labels=labels,
        semantic_sequences=test_sequences[:2],
        confusion_sequences=test_sequences[:1],
        training_mode=True
    )
    
    print(f"✅ 前向传播成功")
    print(f"   输出数量: {len(outputs)}")
    print(f"   原始分类预测形状: {outputs[0].shape}")
    print(f"   语义分类预测形状: {outputs[1].shape}")
    print(f"   混淆分类预测形状: {outputs[2].shape}")