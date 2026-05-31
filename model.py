"""
Fully trainable Nucleotide Transformer v3 Mamba sequence model without residual connections
NTv3 model is fully trainable, no parameter freezing
Optimized simplified enhancement modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings("ignore")

# Import Nucleotide v3 embedding layer
try:
    from nucleotide_v3_trainable import NTv3TrainableEmbedding, create_ntv3_trainable_embedding
    NTv3_AVAILABLE = True
except ImportError:
    NTv3_AVAILABLE = False
    print("⚠️  Unable to import NTv3TrainableEmbedding, please ensure nucleotide_v3_trainable.py is in the path")


class SimplifiedLocalGlobalAttention(nn.Module):
    """Simplified local-global hierarchical attention module
    Removes local convolution, focuses on functional semantic feature capture
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
        
        # Check if Flash Attention is supported
        self.has_flash_attention = hasattr(F, 'scaled_dot_product_attention')
        
        # Multi-head attention mechanism
        if not self.has_flash_attention or not use_flash_attention:
            self.multihead_attn = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Output normalization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward_flash_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Global attention using Flash Attention acceleration"""
        batch_size, seq_len, d_model = x.shape
        
        # Reshape for multi-head
        q = x.view(batch_size, seq_len, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        k = x.view(batch_size, seq_len, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        v = x.view(batch_size, seq_len, self.num_heads, d_model // self.num_heads).transpose(1, 2)
        
        # Create causal mask
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), 
            diagonal=1
        )
        
        # Use Flash Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        # Reshape back to original shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return attn_output
    
    def forward_standard_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Standard multi-head attention"""
        seq_len = x.shape[1]
        
        # Create causal mask
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
        
        # 1. Global multi-head attention
        if self.has_flash_attention and self.use_flash_attention:
            global_features = self.forward_flash_attention(x)
        else:
            global_features = self.forward_standard_attention(x)
        
        # 2. Gated fusion: balance original features and attention features
        gate_input = torch.cat([x, global_features], dim=-1)
        gate_weights = self.gate(gate_input)
        
        # 3. Weighted fusion
        combined = gate_weights * x + (1 - gate_weights) * global_features
        
        # 4. Normalization
        output = self.norm(combined)
        output = self.dropout(output)
        
        return output


class EfficientMultiScaleCrossSegmentAttention(nn.Module):
    """Efficient multi-scale cross-segment attention module
    Optimized segment splitting for improved training efficiency
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
        
        # Check if Flash Attention is supported
        self.has_flash_attention = hasattr(F, 'scaled_dot_product_attention')
        
        # Multi-scale attention modules
        self.segment_attentions = nn.ModuleList()
        for _ in segment_lengths:
            # Use linear layers for preprocessing to reduce dimension
            pre_linear = nn.Linear(d_model, d_model // 2)
            post_linear = nn.Linear(d_model // 2, d_model)
            
            # Cross-segment attention
            if self.has_flash_attention and use_flash_attention:
                # For Flash Attention, use custom implementation
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
        
        # Multi-scale feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * len(segment_lengths), d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Register buffer for precomputed segment indices (optimization)
        self.register_buffer('segment_indices', None, persistent=False)
        
    def compute_efficient_segments(self, x: torch.Tensor, segment_len: int) -> torch.Tensor:
        """Efficiently compute segment features, avoiding repeated reshape operations"""
        batch_size, seq_len, d_model = x.shape
        
        # Calculate number of segments
        num_segments = (seq_len + segment_len - 1) // segment_len
        
        # Use unfold for efficient segmented pooling
        if seq_len % segment_len == 0:
            # If perfectly divisible, reshape directly
            segments = x.view(batch_size, num_segments, segment_len, d_model)
        else:
            # Otherwise use padding + reshape
            new_seq_len = num_segments * segment_len
            padding_size = new_seq_len - seq_len
            
            if padding_size > 0:
                # Pad only the last dimension
                x_padded = F.pad(x, (0, 0, 0, padding_size))
            else:
                x_padded = x
            
            segments = x_padded.view(batch_size, num_segments, segment_len, d_model)
        
        # Average pooling to get segment-level features
        segment_features = segments.mean(dim=2)  # [batch_size, num_segments, d_model]
        
        return segment_features, num_segments
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        multi_scale_features = []
        
        # Apply cross-segment attention for each scale
        for i, segment_len in enumerate(self.segment_lengths):
            # 1. Preprocessing dimensionality reduction
            x_pre = self.segment_attentions[i]['pre_linear'](x)
            
            # 2. Efficiently compute segment features
            segment_features, num_segments = self.compute_efficient_segments(x_pre, segment_len)
            
            # 3. Cross-segment attention
            if self.has_flash_attention and self.use_flash_attention:
                segment_features_attn = self.segment_attentions[i]['attention'](segment_features)
            else:
                segment_features_attn, _ = self.segment_attentions[i]['attention'](
                    segment_features, segment_features, segment_features
                )
            
            # 4. Post-processing dimensionality increase
            segment_features_attn = self.segment_attentions[i]['post_linear'](segment_features_attn)
            
            # 5. Broadcast segment features back to original sequence length
            segment_len_actual = (seq_len + num_segments - 1) // num_segments
            segment_features_expanded = segment_features_attn.repeat_interleave(
                segment_len_actual, dim=1
            )
            
            # 6. Truncate to original length
            segment_features_expanded = segment_features_expanded[:, :seq_len, :]
            
            multi_scale_features.append(segment_features_expanded)
        
        # Multi-scale feature fusion
        if len(multi_scale_features) > 1:
            fused = torch.cat(multi_scale_features, dim=-1)
            output = self.fusion(fused)
        else:
            output = multi_scale_features[0]
        
        output = self.dropout(output)
        
        return output


class EfficientCrossSegmentAttention(nn.Module):
    """Efficient cross-segment attention module with Flash Attention support"""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = dropout
        
        # Check if Flash Attention is supported
        self.has_flash_attention = hasattr(F, 'scaled_dot_product_attention')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_segments, d_model = x.shape
        
        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        if self.has_flash_attention:
            # Reshape for multi-head
            q = q.view(batch_size, num_segments, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, num_segments, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, num_segments, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Use Flash Attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.dropout if self.training else 0.0
            )
            
            # Reshape back to original shape
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_segments, d_model)
        else:
            # Standard multi-head attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0
            )
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class AdaptiveMultiScaleAttention(nn.Module):
    """Adaptive multi-scale attention module
    Dynamically selects the optimal scale for improved efficiency
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
        
        # Scale selection gating
        self.scale_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, max_scales),
            nn.Softmax(dim=-1)
        )
        
        # Main attention module (using medium scale)
        self.main_attention = EfficientMultiScaleCrossSegmentAttention(
            d_model=d_model,
            segment_lengths=[8],  # Default to medium scale
            dropout=dropout,
            use_flash_attention=use_flash_attention
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Calculate scale selection weights
        pooled = x.mean(dim=1)  # [batch_size, d_model]
        scale_weights = self.scale_gate(pooled)  # [batch_size, max_scales]
        
        # Get main scale output
        main_output = self.main_attention(x)
        
        # Adaptive adjustment based on sequence length
        if seq_len < 16:
            # Short sequences use simple processing
            output = main_output
        elif seq_len < 64:
            # Medium sequences
            output = main_output * 0.8 + x * 0.2
        else:
            # Long sequences use stronger attention
            output = main_output
        
        output = self.dropout(output)
        
        return output


class BasicMambaBlock(nn.Module):
    """Basic Mamba block (without residual connection)"""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        
        # Ensure d_conv is between 2 and 4
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
    """Enhanced Mamba block - enhanced features directly as Mamba input"""
    
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
        use_adaptive_scales: bool = True  # New: whether to use adaptive scales
    ):
        super().__init__()
        self.d_model = d_model
        self.use_local_global_attn = use_local_global_attn
        self.use_global_invariance = use_global_invariance
        
        # Ensure d_conv is between 2 and 4
        d_conv = max(2, min(4, d_conv))
        
        # Enhancement modules
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
        
        # Mamba block
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply enhancement modules
        if self.local_global_attn is not None:
            x = self.local_global_attn(x)
        
        if self.global_invariance_encoder is not None:
            x = self.global_invariance_encoder(x)
        
        # Mamba processing
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        
        return x


class DualPathMambaBlock(nn.Module):
    """Dual-path Mamba block - each path uses different enhanced features, shares Mamba parameters"""
    
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
        # Enhancement module parameters
        use_local_global_attn: bool = True,
        use_global_invariance: bool = True,
        attn_num_heads: int = 4,
        use_flash_attention: bool = True,
        use_adaptive_scales: bool = True  # New: whether to use adaptive scales
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
        
        # Ensure d_conv is between 2 and 4
        d_conv = max(2, min(4, d_conv))
        
        # Path 1: Simplified local-global attention enhancement (focusing on functional semantics)
        self.path1_enhance = SimplifiedLocalGlobalAttention(
            d_model=d_model,
            num_heads=attn_num_heads,
            use_flash_attention=use_flash_attention
        ) if use_local_global_attn else nn.Identity()
        
        self.path1_norm = nn.LayerNorm(d_model)
        
        # Path 2: Efficient multi-scale cross-segment attention enhancement (focusing on attack strategies)
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
        
        # Shared Mamba module
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Path selection gating network
        if use_path_selection:
            # Use summary of enhanced features as input
            gate_input_dim = d_model * 2  # Summary of enhanced features from both paths
            
            self.gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, 2),
                nn.Softmax(dim=-1)
            )
        
        # Output fusion layer
        self.output_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def compute_path_selection_loss(self, learned_weights: torch.Tensor, 
                                   variant_type: Optional[str] = None) -> torch.Tensor:
        """Calculate path selection loss"""
        if variant_type is not None:
            # Use variant type as soft supervision
            if variant_type == "confusion":
                target_bias = torch.tensor([[0.3, 0.7]], device=learned_weights.device)
            else:  # "original" or "semantic"
                target_bias = torch.tensor([[0.7, 0.3]], device=learned_weights.device)
            
            batch_size = learned_weights.shape[0]
            target = target_bias.repeat(batch_size, 1)
            
            # KL divergence loss (diversity)
            diversity_loss = F.kl_div(
                F.log_softmax(learned_weights, dim=-1),
                target,
                reduction='batchmean'
            )
            
            # Sparsity loss (entropy minimization)
            eps = 1e-8
            entropy = - (learned_weights * torch.log(learned_weights + eps)).sum(dim=-1)
            sparsity_loss = entropy.mean()
            
            # Combined loss
            path_selection_loss = (
                self.path_diversity_weight * diversity_loss +
                self.path_sparsity_weight * sparsity_loss
            )
        else:
            # Use only sparsity loss
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
        
        # 1. Generate enhanced features
        path1_enhanced = self.path1_enhance(x)
        path2_enhanced = self.path2_enhance(x)
        
        # 2. Path selection
        path_selection_loss = None
        if self.use_path_selection:
            # Use summary of enhanced features as gate input
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
            # Fixed weights
            if variant_type == "confusion":
                path_weights = torch.tensor([[0.3, 0.7]], device=x.device).repeat(batch_size, 1)
            else:
                path_weights = torch.tensor([[0.7, 0.3]], device=x.device).repeat(batch_size, 1)
        
        # 3. Process enhanced features with shared Mamba
        # Path 1: Functional semantic features -> Shared Mamba
        path1_out = self.mamba(self.path1_norm(path1_enhanced))
        
        # Path 2: Attack response features -> Shared Mamba
        path2_out = self.mamba(self.path2_norm(path2_enhanced))
        
        # 4. Dual-path output fusion
        # 4.1 Weighted fusion based on path weights
        path_weights_expanded = path_weights.unsqueeze(1).unsqueeze(-1)
        path_outputs = torch.stack([path1_out, path2_out], dim=2)
        weighted_output = (path_outputs * path_weights_expanded).sum(dim=2)
        
        # 4.2 Optional feature fusion layer
        combined_features = torch.cat([path1_out, path2_out], dim=-1)
        fused_features = self.output_fusion(combined_features)
        
        # 4.3 Final output: 70% weighted output + 30% fused output
        output = 0.7 * weighted_output + 0.3 * fused_features
        output = self.dropout(output)
        
        return output, path_weights, path_selection_loss


class NucleotideMambaModel(nn.Module):
    """Fully trainable Nucleotide Transformer v3 enhanced Mamba sequence model without residual connections"""
    
    def __init__(
        self,
        # Nucleotide Transformer v3 configuration
        transformer_model_repo: str = "InstaDeepAI/NTv3_8M_pre",
        embedding_dim: int = 256,
        # Mamba model configuration
        d_model: int = 256,
        n_layer: int = 4,
        projection_dim: int = 256,
        num_classes: int = 1,
        variant_specialization_weight: float = 0.2,
        block_type: str = "dual_path",
        dropout_rate: float = 0.1,
        use_path_selection: bool = True,
        path_selection_weight: float = 1.0,
        max_seq_len: int = 512,  # Number of DNA bases
        # Enhancement module configuration
        use_local_global_attn: bool = True,
        use_global_invariance: bool = True,
        attn_num_heads: int = 4,
        use_flash_attention: bool = True,
        use_adaptive_scales: bool = True,  # New: whether to use adaptive scales
        # Transformer configuration - Note: We do NOT freeze here!
        freeze_transformer: bool = True,  # Changed to False for full trainability
        use_caching: bool = False,  # Force disable caching to ensure feature dimension consistency
        trust_remote_code: bool = True,  # v3 requires this
        # Architecture configuration
        enable_tf32: bool = True,
        compile_model: bool = False,
        device: Optional[str] = None
    ):
        super().__init__()
        
        # Store parameters
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
        self.freeze_transformer = freeze_transformer  # Now set to False
        self.use_caching = use_caching
        self.trust_remote_code = trust_remote_code
        
        # Auto-select device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # GPU optimization: Enable TF32 acceleration
        if enable_tf32 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ TF32 acceleration enabled")
        
        print(f"🧬 Initializing fully trainable NTv3 + Mamba model (optimized simplified enhancement)")
        print(f"  Transformer model: {transformer_model_repo}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Model dimension: {d_model}")
        print(f"  Number of layers: {n_layer}")
        print(f"  Block type: {block_type}")
        print(f"  Maximum sequence length: {max_seq_len}")
        print(f"  Adaptive scales: {use_adaptive_scales}")
        print(f"  ⚠️  Note: NTv3 model will be fully fine-tuned (not frozen)")
        
        # 2. NTv3 fully trainable embedding layer
        if not NTv3_AVAILABLE:
            raise ImportError("Requires nucleotide_v3_trainable.py module, please ensure it is in Python path")

        # Force disable caching to ensure feature dimension consistency
        self.nucleotide_embedding = NTv3TrainableEmbedding(
            model_repo=transformer_model_repo,
            output_dim=embedding_dim,
            max_seq_len=max_seq_len,
            use_cache=False,  # Force disable caching
            device=str(self.device),
            trust_remote_code=trust_remote_code,
            freeze_transformer=freeze_transformer  # Add this line, pass freeze parameter
        )
        
        # Print model information
        self.nucleotide_embedding.print_model_summary()
        
        # 2. Project from embedding_dim to d_model (if dimensions differ)
        if embedding_dim != d_model:
            self.embedding_projection = nn.Sequential(
                nn.Linear(embedding_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
        else:
            self.embedding_projection = nn.Identity()
        
        # 3. Mamba block sequence (without residual connections)
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
        
        # 4. Sequence pooling layer
        self.sequence_pooler = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1)
        )
        
        # 5. Output normalization
        self.output_norm = nn.LayerNorm(d_model)
        
        # 6. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 7. Contrastive learning projection head
        self.contrastive_projector = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Initialize weights (only initialize non-Transformer parts)
        self._init_weights()
        
        # Move to device
        self.to(self.device)
        
        # Optional: Compile model for optimization
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.forward_sequence = torch.compile(self.forward_sequence)
                print("✅ Model compilation optimization enabled")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}")
    
    def _init_weights(self):
        """Weight initialization (only initialize non-Transformer parts)"""
        for name, module in self.named_modules():
            # Skip NTv3 embedding layer
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
        sequences: List[str],  # Input DNA sequence list
        variant_type: Optional[str] = None,
        training_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for a single sequence type"""
        
        # 1. Extract features through fully trainable NTv3
        x = self.nucleotide_embedding(sequences, training_mode=training_mode)
        
        # 2. Project to model dimension
        x = self.embedding_projection(x)
        
        batch_size, seq_len, _ = x.shape
        
        # 3. Pass through Mamba blocks (without residual connections)
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
        
        # 4. Sequence pooling
        x_pool = x.transpose(1, 2)
        pooled = self.sequence_pooler(x_pool)
        
        # 5. Output normalization
        pooled = self.output_norm(pooled)
        
        # 6. Calculate path selection weights
        if self.use_path_selection and len(path_weights_list) > 0:
            avg_path_weights = torch.stack(path_weights_list).mean(dim=0)
        else:
            avg_path_weights = None
        
        # 7. Classification prediction
        class_pred = self.classifier(pooled)
        
        # 8. Contrastive learning features
        contrastive_feat = self.contrastive_projector(pooled)
        
        # 9. Calculate total path selection loss
        if training_mode and path_selection_loss is not None:
            path_selection_loss = path_selection_loss / len(self.mamba_blocks)
        
        return class_pred, pooled, contrastive_feat, path_selection_loss
    
    def forward(
        self,
        # Input data: sequence lists for different variant types
        original_sequences: List[str],
        labels: torch.Tensor,
        semantic_sequences: Optional[List[str]] = None,
        confusion_sequences: Optional[List[str]] = None,
        random_mutation_sequences: Optional[List[str]] = None,
        training_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            original_sequences: List of original DNA sequences
            labels: Label tensor
            semantic_sequences: List of semantic variant sequences
            confusion_sequences: List of confusion variant sequences
            random_mutation_sequences: List of random mutation variant sequences
            training_mode: Whether in training mode
            
        Returns:
            orig_class_pred: Original sequence classification prediction
            semantic_class_pred: Semantic variant classification prediction
            confusion_class_pred: Confusion variant classification prediction
            total_aux_loss: Total auxiliary loss
            orig_contrastive_feat: Original sequence contrastive features
            all_contrastive_feat: Contrastive features for all sequences
        """
        batch_size = len(original_sequences)
        
        # Ensure labels are on correct device
        labels = labels.to(self.device)
        
        # Process original sequences
        orig_class_pred, orig_pooled_feat, orig_contrastive_feat, orig_path_loss = self.forward_sequence(
            original_sequences, 
            variant_type="original" if training_mode else None,
            training_mode=training_mode
        )
        
        # Process semantic variants
        semantic_class_pred = None
        semantic_contrastive_feat = None
        semantic_path_loss = torch.tensor(0.0, device=self.device)
        
        if semantic_sequences is not None and semantic_sequences:
            semantic_class_pred, _, semantic_contrastive_feat, semantic_path_loss = self.forward_sequence(
                semantic_sequences, 
                variant_type="semantic" if training_mode else None,
                training_mode=training_mode
            )
        
        # Create default values
        if semantic_class_pred is None:
            semantic_class_pred = torch.zeros(batch_size, self.num_classes, device=self.device)
            semantic_contrastive_feat = torch.zeros(batch_size, self.projection_dim, device=self.device)
        
        # Process confusion variants
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
        
        # Process random mutation variants
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
        
        # Calculate total path selection loss
        total_path_selection_loss = orig_path_loss + semantic_path_loss + confusion_path_loss
        
        # Total auxiliary loss
        total_aux_loss = total_path_selection_loss * self.path_selection_weight
        
        # Merge contrastive features
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
        """Clear feature cache"""
        self.nucleotide_embedding.clear_cache()
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return self.nucleotide_embedding.get_cache_stats()
    
    def get_model_summary(self) -> dict:
        """Get model statistics"""
        # NTv3 model parameters
        ntv3_info = self.nucleotide_embedding.get_model_info()
        ntv3_params = ntv3_info['total_params']
        ntv3_trainable = ntv3_info['trainable_params']
        
        # Mamba and other parts parameters
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
            'model_architecture': 'NTv3_8M_pre (fully trainable) + Mamba (optimized simplified enhancement)'
        }
    
    def print_detailed_summary(self):
        """Print detailed model summary"""
        summary = self.get_model_summary()
        
        print(f"\n{'='*60}")
        print(f"🧬 NTv3 + Mamba Model Detailed Summary (Optimized Simplified Enhancement)")
        print(f"{'='*60}")
        print(f"📊 Parameter Statistics:")
        print(f"  Total parameters: {summary['total_parameters']:,}")
        print(f"  Trainable parameters: {summary['trainable_parameters']:,}")
        print(f"  NTv3 parameters: {summary['ntv3_parameters']:,}")
        print(f"  NTv3 trainable parameters: {summary['ntv3_trainable']:,}")
        print(f"  Mamba part parameters: {summary['mamba_parameters']:,}")
        print(f"  Total memory usage: {summary['parameter_mb']:.2f} MB")
        
        print(f"\n📈 Architecture Configuration:")
        print(f"  Model architecture: {summary['model_architecture']}")
        print(f"  Embedding dimension: {summary['embedding_dim']}")
        print(f"  Model dimension: {summary['d_model']}")
        print(f"  Mamba layers: {summary['n_layer']}")
        print(f"  Block type: {summary['block_type']}")
        print(f"  Path selection: {summary['use_path_selection']}")
        print(f"  Functional semantics path: {summary['use_local_global_attn']} (simplified global attention)")
        print(f"  Attack response path: {summary['use_global_invariance']} (multi-scale cross-segment attention)")
        print(f"  Adaptive scales: {summary['use_adaptive_scales']}")
        
        print(f"\n🧬 NTv3 Configuration:")
        print(f"  Model repository: {self.transformer_model_repo}")
        print(f"  Maximum sequence length: {self.max_seq_len}")
        print(f"  Use caching: {self.use_caching}")
        print(f"  Trust remote code: {self.trust_remote_code}")
        
        # NTv3 detailed architecture information
        if hasattr(self.nucleotide_embedding, 'get_model_info'):
            ntv3_info = self.nucleotide_embedding.get_model_info()
            print(f"  NTv3 architecture: {ntv3_info.get('architecture', 'U-Net conv tower → Transformer stack → deconv tower → LM head')}")
            print(f"  NTv3 vocabulary size: {ntv3_info.get('vocab_size', 'N/A')}")
            print(f"  NTv3 hidden size: {ntv3_info.get('hidden_size', 'N/A')}")
        
        print(f"\n🔧 Enhancement Module Optimization Features:")
        print(f"  1. Efficient segment splitting: Using unfold and precomputation optimization")
        print(f"  2. Dimensionality reduction: Reduce dimension before attention, reducing computation")
        print(f"  3. Flash Attention: Support for efficient attention computation")
        print(f"  4. Adaptive scales: Dynamic adjustment based on sequence length")
        print(f"  5. Shared Mamba: Both paths share parameters, reducing computation")
        print(f"  6. Memory optimization: Avoid unnecessary reshaping and padding")
        
        print(f"{'='*60}")


def create_nucleotide_mamba_model(
    # Nucleotide Transformer v3 configuration
    transformer_model_repo: str = "InstaDeepAI/NTv3_8M_pre",
    embedding_dim: int = 256,
    # Mamba model configuration
    d_model: int = 256,
    n_layer: int = 4,
    block_type: str = "dual_path",
    use_path_selection: bool = True,
    # Enhancement module parameters
    use_local_global_attn: bool = True,
    use_global_invariance: bool = True,
    use_flash_attention: bool = True,
    use_adaptive_scales: bool = True,  # New parameter
    # Transformer configuration
    freeze_transformer: bool = True,  # Fully trainable
    use_caching: bool = True,
    trust_remote_code: bool = True,
    # GPU optimization parameters
    enable_tf32: bool = True,
    compile_model: bool = False,
    **kwargs
) -> NucleotideMambaModel:
    """Convenience function to create NTv3 + Mamba model"""
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
        # Enhancement module configuration
        'use_local_global_attn': use_local_global_attn,
        'use_global_invariance': use_global_invariance,
        'attn_num_heads': 4,
        'use_flash_attention': use_flash_attention,
        'use_adaptive_scales': use_adaptive_scales,
        # Transformer configuration
        'freeze_transformer': freeze_transformer,
        'use_caching': use_caching,
        'trust_remote_code': trust_remote_code,
        # GPU optimization
        'enable_tf32': enable_tf32,
        'compile_model': compile_model
    }
    
    # Update default parameters
    default_kwargs.update(kwargs)
    
    return NucleotideMambaModel(**default_kwargs)


if __name__ == "__main__":
    # Test code
    print("🧪 Testing fully trainable NTv3 + Mamba model (optimized simplified enhancement)...")
    
    # Create model
    model = create_nucleotide_mamba_model(
        transformer_model_repo="InstaDeepAI/NTv3_8M_pre",
        embedding_dim=256,
        d_model=256,
        n_layer=4,
        block_type="dual_path",
        use_adaptive_scales=True
    )
    
    # Print detailed summary
    model.print_detailed_summary()
    
    # Test forward pass
    test_sequences = [
        "ATCGNATCG",
        "ACGTACGTACGTACGT",
        "ATCGATCGATCGATCGATCG"
    ]
    
    print(f"\n🔧 Testing forward pass...")
    
    # Training mode
    model.train()
    labels = torch.tensor([1.0, 0.0, 1.0])
    
    outputs = model(
        original_sequences=test_sequences,
        labels=labels,
        semantic_sequences=test_sequences[:2],
        confusion_sequences=test_sequences[:1],
        training_mode=True
    )
    
    print(f"✅ Forward pass successful")
    print(f"   Number of outputs: {len(outputs)}")
    print(f"   Original classification prediction shape: {outputs[0].shape}")
    print(f"   Semantic classification prediction shape: {outputs[1].shape}")
    print(f"   Confusion classification prediction shape: {outputs[2].shape}")
