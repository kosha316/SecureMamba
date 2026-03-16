"""
主脚本 - Nucleotide Transformer v3版本
完全通过配置文件运行，无需命令行参数
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from datetime import datetime
import pickle
import json
import glob

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入自定义模块
from config import config
from data_preprocessing import preprocess_data
from synonymous_variants import generate_triplet_views_parallel
from realtime_dataset import create_nucleotide_dataloader, RealtimeSequenceDataset
from realtime_trainer import NucleotideMambaTrainer, train_nucleotide


def setup_environment():
    """设置环境"""
    print("=" * 80)
    print("🧬 Nucleotide Transformer v3 Mamba模型 (完全可训练)")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"🔧 环境设置完成")
    print(f"📊 模型类型: {config.model_type}")
    print(f"📁 输出目录: {config.output_dir}")
    print(f"⚡ NTv3训练模式: 完全可训练 (不冻结参数)")
    
    # 打印关键配置
    print(f"\n⚙️  配置参数:")
    print(f"   批次大小: {config.batch_size} ⚠️ (NTv3内存需求大)")
    print(f"   训练轮数: {config.num_epochs}")
    print(f"   最大序列长度: {config.max_seq_length}")
    print(f"   设备: {config.device}")
    print(f"   Transformer模型: {config.transformer_model_repo}")
    print(f"   NTv3学习率: {config.learning_rate * config.ntv3_learning_rate_multiplier} (其他参数: {config.learning_rate})")


def run_preprocessing():
    """运行预处理阶段"""
    print(f"\n{'='*60}")
    print(f"🚀 阶段1: 数据预处理")
    print(f"{'='*60}")
    
    try:
        print("📥 加载原始FASTA数据并预处理...")
        segments = preprocess_data(config)
        
        if not segments:
            print("❌ 预处理未生成任何数据")
            return None
        
        print(f"✅ 预处理完成，生成 {len(segments):,} 个片段")
        
        # 保存预处理结果
        preprocess_file = os.path.join(config.output_dir, "preprocessed_segments.pkl")
        with open(preprocess_file, 'wb') as f:
            pickle.dump(segments, f)
        print(f"📁 预处理数据保存到: {preprocess_file}")
        
        return segments
        
    except Exception as e:
        print(f"❌ 预处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_variants(segments, save_output=True):
    """运行变体生成阶段"""
    print(f"\n{'='*60}")
    print(f"🚀 阶段2: 生成变体数据")
    print(f"{'='*60}")
    
    if not segments:
        print("❌ 没有预处理数据，无法生成变体")
        return None
    
    # 分离正负样本
    positive_segments = [seg for seg in segments if seg['label'] == 1]
    negative_segments = [seg for seg in segments if seg['label'] == 0]
    
    print(f"📊 样本统计:")
    print(f"   总片段数: {len(segments)}")
    print(f"   致病性（正样本）: {len(positive_segments)}")
    print(f"   非致病性（负样本）: {len(negative_segments)}")
    
    if len(positive_segments) == 0:
        print("⚠️  没有致病性序列（正样本），跳过变体生成")
        
        # 为负样本创建标准格式
        all_results = []
        for seg in negative_segments:
            all_results.append({
                'original_seq': seg['original_seq'],
                'original_label': seg['label'],
                'positive_views': [],
                'contrastive_negative_views': [],
                'view_types': [],
                'segment_id': seg.get('segment_id', f"neg_{len(all_results)}"),
                'num_positive_views': 0,
                'num_contrastive_negative_views': 0,
                'has_variants': False
            })
        
        if save_output:
            variants_file = os.path.join(config.output_dir, "variants_data.pkl")
            with open(variants_file, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"📁 变体数据保存到: {variants_file}")
        
        return all_results
    
    # 变体生成配置
    variant_config = config.get_variant_generation_config()
    
    try:
        print(f"🔄 生成变体数据...")
        positive_results = generate_triplet_views_parallel(
            positive_segments,
            **variant_config
        )
        
        # 合并所有数据
        all_results = []
        
        # 添加带变体的正样本结果
        for result in positive_results:
            all_results.append(result)
        
        # 添加不带变体的负样本原始数据
        for seg in negative_segments:
            all_results.append({
                'original_seq': seg['original_seq'],
                'original_label': seg['label'],
                'positive_views': [],
                'contrastive_negative_views': [],
                'view_types': [],
                'segment_id': seg.get('segment_id', f"neg_{len(all_results)}"),
                'num_positive_views': 0,
                'num_contrastive_negative_views': 0,
                'has_variants': False
            })
        
        # 统计信息
        total_with_variants = sum(1 for r in all_results if r.get('has_variants', False))
        total_no_variants = sum(1 for r in all_results if not r.get('has_variants', False))
        
        print(f"📊 变体生成完成统计:")
        print(f"   带变体序列（正样本）: {total_with_variants}")
        print(f"   无变体序列（负样本）: {total_no_variants}")
        
        # 详细统计
        semantic_count = sum(len(r['positive_views']) for r in positive_results if len(r['positive_views']) > 0)
        confusion_count = sum(1 for r in positive_results if 'confusion' in r.get('view_types', []))
        random_mutation_count = sum(len(r['contrastive_negative_views']) for r in positive_results)
        
        print(f"   语义变体总数: {semantic_count}")
        print(f"   混淆变体总数: {confusion_count}")
        print(f"   随机突变变体总数: {random_mutation_count}")
        
        if save_output:
            # 保存变体数据
            variants_file = os.path.join(config.output_dir, "variants_data.pkl")
            with open(variants_file, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"📁 变体数据保存到: {variants_file}")
            
            # 保存统计信息
            stats = {
                'total_samples': len(all_results),
                'positive_with_variants': total_with_variants,
                'negative_without_variants': total_no_variants,
                'semantic_variants': semantic_count,
                'confusion_variants': confusion_count,
                'random_mutation_variants': random_mutation_count
            }
            
            stats_file = os.path.join(config.output_dir, "variants_stats.json")
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"📊 统计信息保存到: {stats_file}")
        
        return all_results
        
    except Exception as e:
        print(f"❌ 变体生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def load_variants_file(variants_file_path):
    """从文件加载变体数据"""
    print(f"\n📥 从文件加载变体数据: {variants_file_path}")
    
    try:
        with open(variants_file_path, 'rb') as f:
            variants_data = pickle.load(f)
        
        print(f"✅ 加载成功，共 {len(variants_data):,} 个样本")
        
        # 统计信息
        total_with_variants = sum(1 for r in variants_data if r.get('has_variants', False))
        total_no_variants = sum(1 for r in variants_data if not r.get('has_variants', False))
        
        print(f"📊 数据统计:")
        print(f"   带变体序列（正样本）: {total_with_variants}")
        print(f"   无变体序列（负样本）: {total_no_variants}")
        
        # 详细统计（如果可能）
        try:
            semantic_count = sum(len(r.get('positive_views', [])) for r in variants_data if len(r.get('positive_views', [])) > 0)
            confusion_count = sum(1 for r in variants_data if 'confusion' in r.get('view_types', []))
            random_mutation_count = sum(len(r.get('contrastive_negative_views', [])) for r in variants_data)
            
            print(f"   语义变体总数: {semantic_count}")
            print(f"   混淆变体总数: {confusion_count}")
            print(f"   随机突变变体总数: {random_mutation_count}")
        except:
            print("⚠️  无法获取详细变体统计信息")
        
        return variants_data
        
    except Exception as e:
        print(f"❌ 加载变体文件失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def find_latest_variants_file(directory=None):
    """查找最新的变体文件"""
    if directory is None:
        directory = config.output_dir
    
    # 查找可能的变体文件
    patterns = [
        "variants_data.pkl",
        "segments_with_variants.pkl",
        "*variants*.pkl",
        "*.pkl"
    ]
    
    for pattern in patterns:
        files = glob.glob(os.path.join(directory, pattern))
        if files:
            # 返回最新的文件
            latest_file = max(files, key=os.path.getmtime)
            return latest_file
    
    return None


def run_nucleotide_training(variants_data):
    """运行Nucleotide Transformer v3训练阶段"""
    print(f"\n{'='*60}")
    print(f"🚀 阶段3: Nucleotide Transformer v3训练 (完全可训练)")
    print(f"{'='*60}")
    
    try:
        # 创建训练器
        trainer = NucleotideMambaTrainer(config)
        
        # 直接从变体数据开始训练
        print("🔧 开始完全可训练的Nucleotide Transformer v3训练...")
        model, train_history, val_history = trainer.train(variants_data)
        
        if model is not None:
            print(f"✅ 训练完成")
            
            # 打印最终结果
            if val_history:
                final_val = val_history[-1]
                print(f"\n📈 最终验证集指标:")
                print(f"  损失: {final_val['loss']:.4f}")
                print(f"  综合ACC: {final_val['combined_acc']:.4f}")
                print(f"  总体ACC: {final_val['overall']['acc']:.4f}")
                print(f"  总体AUC: {final_val['overall']['auc']:.4f}")
                print(f"  总体F1: {final_val['overall']['f1']:.4f}")
                print(f"  总体MCC: {final_val['overall']['mcc']:.4f}")
                print(f"  原始ACC: {final_val['original']['acc']:.4f}")
                print(f"  语义ACC: {final_val['semantic']['acc']:.4f}")
                print(f"  混淆ACC: {final_val['confusion']['acc']:.4f}")
            
            # 模型保存路径
            best_model_path = os.path.join(config.output_dir, "best_nucleotide_v3_mamba_model.pth")
            final_model_path = os.path.join(config.output_dir, "final_nucleotide_v3_mamba_model.pth")
            log_path = os.path.join(config.output_dir, "nucleotide_v3_training_log.json")
            
            print(f"\n📁 输出文件:")
            print(f"  最佳模型: {best_model_path}")
            print(f"  最终模型: {final_model_path}")
            print(f"  训练日志: {log_path}")
            
            return True
        else:
            print(f"❌ 训练失败")
            return False
    except Exception as e:
        print(f"❌ 训练过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数 - 完全通过配置文件运行"""
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 设置环境
    setup_environment()

    
    # 从config中读取运行模式
    run_mode = getattr(config, 'run_mode', 'all')  # 默认为全流程
    
    # 从config中读取是否使用现有变体文件
    use_existing_variants = getattr(config, 'use_existing_variants', False)
    variants_file_path = getattr(config, 'variants_file_path', None)
    auto_find_variants = getattr(config, 'auto_find_variants', False)
    
    print(f"📋 运行模式: {run_mode}")
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"\n⏰ 开始时间: {start_time.strftime('%Y-%m-d %H:%M:%S')}")
    
    success = True
    segments = None
    variants_data = None
    
    # ==================== 根据模式执行 ====================
    
    if run_mode == "train-only" or (run_mode == "train" and (variants_file_path or auto_find_variants or use_existing_variants)):
        # 模式：仅训练，从文件加载变体数据
        print(f"\n🎯 模式：仅训练（从文件加载）")
        
        # 确定变体文件路径
        file_path_to_load = None
        
        if variants_file_path:
            file_path_to_load = variants_file_path
            if not os.path.exists(file_path_to_load):
                print(f"❌ 变体文件不存在: {file_path_to_load}")
                success = False
        elif auto_find_variants:
            file_path_to_load = find_latest_variants_file()
            if file_path_to_load:
                print(f"🔍 自动查找到变体文件: {file_path_to_load}")
            else:
                print("❌ 未找到变体文件")
                success = False
        elif use_existing_variants:
            # 在输出目录中查找变体文件
            default_variants_file = os.path.join(config.output_dir, "variants_data.pkl")
            if os.path.exists(default_variants_file):
                file_path_to_load = default_variants_file
                print(f"🔍 使用默认变体文件: {file_path_to_load}")
            else:
                print(f"❌ 默认变体文件不存在: {default_variants_file}")
                success = False
        
        if success and file_path_to_load:
            variants_data = load_variants_file(file_path_to_load)
            if variants_data is None:
                success = False
        
        if success and variants_data:
            success = run_nucleotide_training(variants_data)
    
    elif run_mode == "preprocess":
        # 模式：仅预处理
        print(f"\n🎯 模式：仅预处理")
        segments = run_preprocessing()
        success = (segments is not None)
    
    elif run_mode == "generate":
        # 模式：仅生成变体
        print(f"\n🎯 模式：仅生成变体")
        
        # 加载预处理数据
        preprocessed_file = getattr(config, 'preprocessed_file', None)
        if preprocessed_file and os.path.exists(preprocessed_file):
            print(f"📥 从文件加载预处理数据: {preprocessed_file}")
            with open(preprocessed_file, 'rb') as f:
                segments = pickle.load(f)
            print(f"✅ 加载成功，共 {len(segments):,} 个片段")
        else:
            print("🔄 重新运行预处理...")
            segments = run_preprocessing()
        
        if segments:
            save_variants = getattr(config, 'save_variants', True)
            variants_data = generate_variants(segments, save_output=save_variants)
            success = (variants_data is not None)
        else:
            success = False
    
    elif run_mode == "train":
        # 模式：训练（但不指定变体文件，需要先运行预处理和变体生成）
        print(f"\n🎯 模式：训练（需要先运行预处理和变体生成）")
        
        # 尝试查找现有的变体文件
        variants_file_path = find_latest_variants_file()
        if variants_file_path and use_existing_variants:
            print(f"🔍 找到现有变体文件: {variants_file_path}")
            variants_data = load_variants_file(variants_file_path)
            if variants_data:
                success = run_nucleotide_training(variants_data)
            else:
                success = False
        else:
            # 重新运行全流程
            run_mode = "all"
            print("🔄 未找到或未启用现有变体文件，运行全流程...")
    
    if run_mode == "all" and success:
        # 模式：全流程
        print(f"\n🎯 模式：全流程")
        
        # 1. 预处理
        segments = run_preprocessing()
        if segments is None:
            success = False
        
        # 2. 生成变体
        if success:
            save_variants = getattr(config, 'save_variants', True)
            variants_data = generate_variants(segments, save_output=save_variants)
            if variants_data is None:
                success = False
        
        # 3. 训练
        if success:
            success = run_nucleotide_training(variants_data)
    
    # ==================== 流程总结 ====================
    
    # 记录结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("📊 流程总结")
    print(f"{'='*80}")
    print(f"  开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  总耗时: {duration}")
    
    if run_mode == "preprocess" and success:
        preprocess_file = os.path.join(config.output_dir, "preprocessed_segments.pkl")
        print(f"\n📁 预处理数据文件: {preprocess_file}")
    
    if run_mode == "generate" and success:
        variants_file = os.path.join(config.output_dir, "variants_data.pkl")
        print(f"\n📁 变体数据文件: {variants_file}")
    
    if success:
        print(f"\n✅ 流程执行成功!")
    else:
        print(f"\n❌ 某些阶段失败，请检查日志")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)