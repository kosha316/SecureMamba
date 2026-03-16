import torch
import os
from typing import List

# 基础配置
torch.manual_seed(42)

class Config:
    # ========== 基础配置 ==========
    model_type = "nucleotide_v3_mamba"  # 当前使用的模型类型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 直接指定设备
    
    # ========== 运行模式配置 ==========
    run_mode = "train"  # 运行模式: "preprocess", "generate", "train", "all", "train-only"
    
    # ========== 文件配置 ==========
    use_existing_variants = True  # 是否使用现有变体文件
    variants_file_path = "results_nucleotide_v3_mamba_ds/variants_data.pkl"  # 变体文件路径
    output_dir = "results_nucleotide_v3_mamba_ds"  # 输出目录
    
    # ========== 数据配置 ==========
    max_seq_length = 2048  # DNA序列最大长度 ⚠️ 修改为2048
    min_segment_len = 30  # 最小片段长度
    positive_fasta = "train_data/VFDB_setB_nt.fasta"  # 正样本文件
    negative_fasta = "train_data/non_pathogenic_regions.fasta"  # 负样本文件
    downsample_ratio = 3.0  # 非致病片段是致病片段的3倍
    seed = 42  # 随机种子，确保可重复性
    
    # ========== Nucleotide Transformer v3核心配置 ==========
    transformer_model_repo = "InstaDeepAI/NTv3_8M_pre"  # NTv3模型仓库
    embedding_dim = 256  # 输出嵌入维度
    max_seq_len = 2048  # NTv3最大输入长度
    freeze_transformer = True  # 冻结NTv3
    unfreeze_modules = ["deconv_tower_blocks"]  # 新增：只解冻这个模块
    trust_remote_code = True  # NTv3需要信任远程代码
    
    # ========== Mamba模型配置 ==========
    d_model = 256  # 模型内部维度，与embedding_dim相同
    n_layer = 2  # Mamba层数
    projection_dim = 128  # 对比学习投影维度
    num_classes = 1  # 分类类别数
    dropout_rate = 0.2  # dropout率
    
    # Mamba块配置
    block_type = "dual_path"  # 块类型: "dual_path", "mixed", "enhanced", "basic"
    use_path_selection = True  # 是否使用路径选择网络
    path_selection_weight = 0.1  # 路径选择损失权重
    
    # 增强模块配置
    use_local_global_attn = True  # 使用局部全局注意力
    use_global_invariance = True  # 使用全局不变性编码
    attn_num_heads = 4  # 注意力头数
    # num_segments = 8  # 片段数量
    use_flash_attention = True  # 使用Flash Attention
    
    # ========== 训练优化配置 ==========
    batch_size = 16  # 批次大小（NTv3内存需求大，2048序列长度需要更小的批次）⚠️ 修改为4
    learning_rate = 2e-6  # 基础学习率
    ntv3_learning_rate_multiplier = 0.1  # NTv3参数学习率乘子（1.0的学习率）
    num_epochs = 20  # 训练轮数
    weight_decay = 0.01  # 权重衰减
    grad_clip = 1.0  # 梯度裁剪
    
    # 学习率调度器
    scheduler_type = "onecycle"
    onecycle_max_lr = 2e-6  # 降低最大学习率
    onecycle_pct_start = 0.1  # 上升阶段比例
    steps_per_epoch = 8000  # 每个epoch的步骤数
    
    # 早停配置
    early_stop_patience = 5  # 早停耐心值
    
    # 混合精度训练
    use_amp = True  # 使用混合精度训练
    
    # ========== 变体生成配置 ==========
    # 语义变体参数
    conservative_energy_tolerance = 1.0
    min_pos_base_diff = 0.2
    max_pos_base_diff = 0.3
    pos_conservative_ratio = 0.2
    pos_syn_ratio = 0.8
    max_replace_ratio = 0.6
    max_semantic_attempts = 1000
    
    # DNA混淆变体参数
    dna_confusion_min_fragment_len = 20
    dna_confusion_max_fragment_len = 200
    dna_confusion_min_fragments = 5
    dna_confusion_max_fragments = 10
    dna_confusion_flip_ratio = 0.3
    dna_confusion_max_attempts = 100
    
    # 负样本配置
    negative_mutation_rate = 0.5
    negative_max_attempts = 20
    
    # 正样本视图数量
    num_positive_views = 2  # 语义相似 + DNA混淆
    
    # ========== 损失配置 ==========
    contrastive_weight = 0.3  # 对比损失权重
    variant_specialization_weight = 0.2  # 对齐损失权重
    triplet_margin = 1.0  # 三元组损失边界
    triplet_temperature = 0.1  # 温度参数
    use_hard_triplet = False  # 是否使用困难三元组
    
    # ========== 系统配置 ==========
    num_workers = min(os.cpu_count(), 16)  # 数据加载worker数
    pin_memory = True  # 内存锁定
    prefetch_factor = 2  # 预取因子
    persistent_workers = True  # 持久化worker
    
    # GPU优化
    enable_tf32 = True  # 启用TF32加速
    compile_model = False  # 编译模型（可选）
    
    # 缓存配置
    use_caching = False  # 启用缓存以提高性能
    
    def __init__(self):
        """初始化配置"""
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置随机种子
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # GPU优化设置
        if self.enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # 打印配置摘要
        self.print_config_summary()

    def print_config_summary(self):
        """打印配置摘要"""
        print("=" * 70)
        print("🧬 Nucleotide Transformer v3 Mamba 配置摘要 (2048长度)")
        print("=" * 70)
        
        print(f"\n📊 核心配置:")
        print(f"  模型类型: {self.model_type}")
        print(f"  设备: {self.device}")
        print(f"  运行模式: {self.run_mode}")
        print(f"  输出目录: {self.output_dir}")
        
        print(f"\n📈 数据配置:")
        print(f"  最大序列长度: {self.max_seq_length} ⚡")
        print(f"  正样本: {self.positive_fasta}")
        print(f"  负样本: {self.negative_fasta}")
        
        print(f"\n🧬 Nucleotide Transformer v3:")
        print(f"  模型仓库: {self.transformer_model_repo}")
        print(f"  嵌入维度: {self.embedding_dim} (NTv3隐藏层: 256)")
        print(f"  最大输入长度: {self.max_seq_len} ⚡")
        print(f"  完全可训练: {not self.freeze_transformer} ⭐")
        print(f"  NTv3学习率: {self.learning_rate * self.ntv3_learning_rate_multiplier:.1e}")
        
        print(f"\n🤖 Mamba模型:")
        print(f"  模型维度: {self.d_model}")
        print(f"  层数: {self.n_layer}")
        print(f"  块类型: {self.block_type}")
        print(f"  路径选择: {self.use_path_selection}")
        print(f"  局部全局注意力: {self.use_local_global_attn}")
        print(f"  全局不变性编码: {self.use_global_invariance}")
        
        print(f"\n⚡ 训练配置:")
        print(f"  批次大小: {self.batch_size} ⚠️ (2048长度需要小批次)")
        print(f"  学习率: {self.learning_rate}")
        print(f"  训练轮数: {self.num_epochs}")
        print(f"  混合精度: {self.use_amp}")
        print(f"  早停耐心: {self.early_stop_patience}")
        
        print(f"\n🎯 损失配置:")
        print(f"  对比损失权重: {self.contrastive_weight}")
        print(f"  对齐损失权重: {self.variant_specialization_weight}")
        print(f"  路径选择损失权重: {self.path_selection_weight}")
        
        print(f"\n🔄 变体生成:")
        print(f"  语义变体: {self.max_semantic_attempts}次尝试")
        print(f"  DNA混淆: {self.dna_confusion_max_fragments}个片段")
        print(f"  正样本视图: {self.num_positive_views}个")
        
        print(f"\n⚠️  重要提示:")
        print(f"  - NTv3隐藏层维度: 256")
        print(f"  - 最大序列长度: 2048")
        print(f"  - 批次大小: {self.batch_size} (需小心内存)")
        print(f"  - 禁用缓存: 确保特征维度一致")
        
        print("\n" + "=" * 70)

    def get_model_kwargs(self):
        """获取模型初始化参数 - 精简版"""
        return {
            # NTv3核心参数
            'transformer_model_repo': self.transformer_model_repo,
            'embedding_dim': self.embedding_dim,
            'max_seq_len': self.max_seq_len,
            'freeze_transformer': self.freeze_transformer,
            'trust_remote_code': self.trust_remote_code,
            'unfreeze_modules': self.unfreeze_modules,  # 传递解冻模块
            
            # Mamba模型参数
            'd_model': self.d_model,
            'n_layer': self.n_layer,
            'projection_dim': self.projection_dim,
            'num_classes': self.num_classes,
            'variant_specialization_weight': self.variant_specialization_weight,
            'block_type': self.block_type,
            'dropout_rate': self.dropout_rate,
            'use_path_selection': self.use_path_selection,
            'path_selection_weight': self.path_selection_weight,
            
            # 增强模块参数
            'use_local_global_attn': self.use_local_global_attn,
            'use_global_invariance': self.use_global_invariance,
            'attn_num_heads': self.attn_num_heads,
            # 'num_segments': self.num_segments,
            'use_flash_attention': self.use_flash_attention,
            
            # 系统参数
            'use_caching': self.use_caching,
            'enable_tf32': self.enable_tf32,
            'compile_model': self.compile_model,
            'device': str(self.device)
        }

    def get_training_config(self):
        """获取训练配置"""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'ntv3_learning_rate_multiplier': self.ntv3_learning_rate_multiplier,
            'num_epochs': self.num_epochs,
            'weight_decay': self.weight_decay,
            'grad_clip': self.grad_clip,
            
            'contrastive_weight': self.contrastive_weight,
            'variant_specialization_weight': self.variant_specialization_weight,
            
            'use_amp': self.use_amp,
            'scheduler_type': self.scheduler_type,
            'onecycle_max_lr': self.onecycle_max_lr,
            'onecycle_pct_start': self.onecycle_pct_start,
            'steps_per_epoch': self.steps_per_epoch,
            'early_stop_patience': self.early_stop_patience,
        }

    def get_variant_generation_config(self):
        """获取变体生成配置 - 精简版"""
        return {
            # 语义变体参数
            'conservative_energy_tolerance': self.conservative_energy_tolerance,
            'min_pos_base_diff': self.min_pos_base_diff,
            'max_pos_base_diff': self.max_pos_base_diff,
            'pos_conservative_ratio': self.pos_conservative_ratio,
            'pos_syn_ratio': self.pos_syn_ratio,
            'max_replace_ratio': self.max_replace_ratio,
            'max_semantic_attempts': self.max_semantic_attempts,
            
            # DNA混淆参数
            'min_fragment_len': self.dna_confusion_min_fragment_len,
            'max_fragment_len': self.dna_confusion_max_fragment_len,
            'min_fragments': self.dna_confusion_min_fragments,
            'max_fragments': self.dna_confusion_max_fragments,
            'flip_ratio': self.dna_confusion_flip_ratio,
            'max_confusion_attempts': self.dna_confusion_max_attempts,
            
            # 负样本参数
            'negative_mutation_rate': self.negative_mutation_rate,
            'max_negative_attempts': self.negative_max_attempts,
        }

    def get_loss_config(self):
        """获取损失配置"""
        return {
            'contrastive_weight': self.contrastive_weight,
            'variant_specialization_weight': self.variant_specialization_weight,
            'triplet_margin': self.triplet_margin,
            'triplet_temperature': self.triplet_temperature,
            'use_hard_triplet': self.use_hard_triplet
        }

# 全局配置实例
config = Config()