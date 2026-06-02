"""
Main Script - Nucleotide Transformer v3 Version
Runs entirely through configuration file, no command-line arguments needed
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from datetime import datetime
import pickle
import json
import glob

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import custom modules
from config import config
from data_preprocessing import preprocess_data
from synonymous_variants import generate_triplet_views_parallel
from realtime_dataset import create_nucleotide_dataloader, RealtimeSequenceDataset
from realtime_trainer import NucleotideMambaTrainer, train_nucleotide


def setup_environment():
    """Setup environment"""
    print("=" * 80)
    print("🧬 Nucleotide Transformer v3 Mamba Model (Fully Trainable)")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"🔧 Environment setup completed")
    print(f"📊 Model type: {config.model_type}")
    print(f"📁 Output directory: {config.output_dir}")
    print(f"⚡ NTv3 training mode: Fully trainable (no parameter freezing)")
    
    # Print key configurations
    print(f"\n⚙️  Configuration parameters:")
    print(f"   Batch size: {config.batch_size} ⚠️ (NTv3 requires large memory)")
    print(f"   Number of epochs: {config.num_epochs}")
    print(f"   Maximum sequence length: {config.max_seq_length}")
    print(f"   Device: {config.device}")
    print(f"   Transformer model: {config.transformer_model_repo}")
    print(f"   NTv3 learning rate: {config.learning_rate * config.ntv3_learning_rate_multiplier} (Other parameters: {config.learning_rate})")


def run_preprocessing():
    """Run preprocessing stage"""
    print(f"\n{'='*60}")
    print(f"🚀 Stage 1: Data Preprocessing")
    print(f"{'='*60}")
    
    try:
        print("📥 Loading raw FASTA data and preprocessing...")
        segments = preprocess_data(config)
        
        if not segments:
            print("❌ Preprocessing did not generate any data")
            return None
        
        print(f"✅ Preprocessing completed, generated {len(segments):,} fragments")
        
        # Save preprocessing results
        preprocess_file = os.path.join(config.output_dir, "preprocessed_segments.pkl")
        with open(preprocess_file, 'wb') as f:
            pickle.dump(segments, f)
        print(f"📁 Preprocessed data saved to: {preprocess_file}")
        
        return segments
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_variants(segments, save_output=True):
    """Run variant generation stage"""
    print(f"\n{'='*60}")
    print(f"🚀 Stage 2: Generating Variant Data")
    print(f"{'='*60}")
    
    if not segments:
        print("❌ No preprocessed data available, cannot generate variants")
        return None
    
    # Separate positive and negative samples
    positive_segments = [seg for seg in segments if seg['label'] == 1]
    negative_segments = [seg for seg in segments if seg['label'] == 0]
    
    print(f"📊 Sample statistics:")
    print(f"   Total fragments: {len(segments)}")
    print(f"   Pathogenic (positive samples): {len(positive_segments)}")
    print(f"   Non-pathogenic (negative samples): {len(negative_segments)}")
    
    if len(positive_segments) == 0:
        print("⚠️  No pathogenic sequences (positive samples), skipping variant generation")
        
        # Create standard format for negative samples
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
            print(f"📁 Variant data saved to: {variants_file}")
        
        return all_results
    
    # Variant generation configuration
    variant_config = config.get_variant_generation_config()
    
    try:
        print(f"🔄 Generating variant data...")
        positive_results = generate_triplet_views_parallel(
            positive_segments,
            **variant_config
        )
        
        # Merge all data
        all_results = []
        
        # Add positive results with variants
        for result in positive_results:
            all_results.append(result)
        
        # Add negative sample original data without variants
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
        
        # Statistics
        total_with_variants = sum(1 for r in all_results if r.get('has_variants', False))
        total_no_variants = sum(1 for r in all_results if not r.get('has_variants', False))
        
        print(f"📊 Variant generation completed statistics:")
        print(f"   Sequences with variants (positive samples): {total_with_variants}")
        print(f"   Sequences without variants (negative samples): {total_no_variants}")
        
        # Detailed statistics
        semantic_count = sum(len(r['positive_views']) for r in positive_results if len(r['positive_views']) > 0)
        confusion_count = sum(1 for r in positive_results if 'confusion' in r.get('view_types', []))
        random_mutation_count = sum(len(r['contrastive_negative_views']) for r in positive_results)
        
        print(f"   Total semantic variants: {semantic_count}")
        print(f"   Total confusion variants: {confusion_count}")
        print(f"   Total random mutation variants: {random_mutation_count}")
        
        if save_output:
            # Save variant data
            variants_file = os.path.join(config.output_dir, "variants_data.pkl")
            with open(variants_file, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"📁 Variant data saved to: {variants_file}")
            
            # Save statistics
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
            print(f"📊 Statistics saved to: {stats_file}")
        
        return all_results
        
    except Exception as e:
        print(f"❌ Variant generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def load_variants_file(variants_file_path):
    """Load variant data from file"""
    print(f"\n📥 Loading variant data from file: {variants_file_path}")
    
    try:
        with open(variants_file_path, 'rb') as f:
            variants_data = pickle.load(f)
        
        print(f"✅ Loading successful, total {len(variants_data):,} samples")
        
        # Statistics
        total_with_variants = sum(1 for r in variants_data if r.get('has_variants', False))
        total_no_variants = sum(1 for r in variants_data if not r.get('has_variants', False))
        
        print(f"📊 Data statistics:")
        print(f"   Sequences with variants (positive samples): {total_with_variants}")
        print(f"   Sequences without variants (negative samples): {total_no_variants}")
        
        # Detailed statistics (if possible)
        try:
            semantic_count = sum(len(r.get('positive_views', [])) for r in variants_data if len(r.get('positive_views', [])) > 0)
            confusion_count = sum(1 for r in variants_data if 'confusion' in r.get('view_types', []))
            random_mutation_count = sum(len(r.get('contrastive_negative_views', [])) for r in variants_data)
            
            print(f"   Total semantic variants: {semantic_count}")
            print(f"   Total confusion variants: {confusion_count}")
            print(f"   Total random mutation variants: {random_mutation_count}")
        except:
            print("⚠️  Unable to retrieve detailed variant statistics")
        
        return variants_data
        
    except Exception as e:
        print(f"❌ Failed to load variant file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def find_latest_variants_file(directory=None):
    """Find the latest variant file"""
    if directory is None:
        directory = config.output_dir
    
    # Search for possible variant files
    patterns = [
        "variants_data.pkl",
        "segments_with_variants.pkl",
        "*variants*.pkl",
        "*.pkl"
    ]
    
    for pattern in patterns:
        files = glob.glob(os.path.join(directory, pattern))
        if files:
            # Return the latest file
            latest_file = max(files, key=os.path.getmtime)
            return latest_file
    
    return None


def run_nucleotide_training(variants_data):
    """Run Nucleotide Transformer v3 training stage"""
    print(f"\n{'='*60}")
    print(f"🚀 Stage 3: Nucleotide Transformer v3 Training (Fully Trainable)")
    print(f"{'='*60}")
    
    try:
        # Create trainer
        trainer = NucleotideMambaTrainer(config)
        
        # Start training directly from variant data
        print("🔧 Starting fully trainable Nucleotide Transformer v3 training...")
        model, train_history, val_history = trainer.train(variants_data)
        
        if model is not None:
            print(f"✅ Training completed")
            
            # Print final results
            if val_history:
                final_val = val_history[-1]
                print(f"\n📈 Final validation set metrics:")
                print(f"  Loss: {final_val['loss']:.4f}")
                print(f"  Combined ACC: {final_val['combined_acc']:.4f}")
                print(f"  Overall ACC: {final_val['overall']['acc']:.4f}")
                print(f"  Overall AUC: {final_val['overall']['auc']:.4f}")
                print(f"  Overall F1: {final_val['overall']['f1']:.4f}")
                print(f"  Overall MCC: {final_val['overall']['mcc']:.4f}")
                print(f"  Original ACC: {final_val['original']['acc']:.4f}")
                print(f"  Semantic ACC: {final_val['semantic']['acc']:.4f}")
                print(f"  Confusion ACC: {final_val['confusion']['acc']:.4f}")
            
            # Model save paths
            best_model_path = os.path.join(config.output_dir, "best_nucleotide_v3_mamba_model.pth")
            final_model_path = os.path.join(config.output_dir, "final_nucleotide_v3_mamba_model.pth")
            log_path = os.path.join(config.output_dir, "nucleotide_v3_training_log.json")
            
            print(f"\n📁 Output files:")
            print(f"  Best model: {best_model_path}")
            print(f"  Final model: {final_model_path}")
            print(f"  Training log: {log_path}")
            
            return True
        else:
            print(f"❌ Training failed")
            return False
    except Exception as e:
        print(f"❌ Training process error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function - runs entirely through configuration file"""
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # Chinese users uncomment this line
    # Setup environment
    setup_environment()

    
    # Read run mode from config
    run_mode = getattr(config, 'run_mode', 'all')  # Default to full pipeline
    
    # Read whether to use existing variant file from config
    use_existing_variants = getattr(config, 'use_existing_variants', False)
    variants_file_path = getattr(config, 'variants_file_path', None)
    auto_find_variants = getattr(config, 'auto_find_variants', False)
    
    print(f"📋 Run mode: {run_mode}")
    
    # Record start time
    start_time = datetime.now()
    print(f"\n⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = True
    segments = None
    variants_data = None
    
    # ==================== Execute based on mode ====================
    
    if run_mode == "train-only" or (run_mode == "train" and (variants_file_path or auto_find_variants or use_existing_variants)):
        # Mode: Training only, load variant data from file
        print(f"\n🎯 Mode: Training only (load from file)")
        
        # Determine variant file path
        file_path_to_load = None
        
        if variants_file_path:
            file_path_to_load = variants_file_path
            if not os.path.exists(file_path_to_load):
                print(f"❌ Variant file does not exist: {file_path_to_load}")
                success = False
        elif auto_find_variants:
            file_path_to_load = find_latest_variants_file()
            if file_path_to_load:
                print(f"🔍 Auto-found variant file: {file_path_to_load}")
            else:
                print("❌ No variant file found")
                success = False
        elif use_existing_variants:
            # Look for variant file in output directory
            default_variants_file = os.path.join(config.output_dir, "variants_data.pkl")
            if os.path.exists(default_variants_file):
                file_path_to_load = default_variants_file
                print(f"🔍 Using default variant file: {file_path_to_load}")
            else:
                print(f"❌ Default variant file does not exist: {default_variants_file}")
                success = False
        
        if success and file_path_to_load:
            variants_data = load_variants_file(file_path_to_load)
            if variants_data is None:
                success = False
        
        if success and variants_data:
            success = run_nucleotide_training(variants_data)
    
    elif run_mode == "preprocess":
        # Mode: Preprocessing only
        print(f"\n🎯 Mode: Preprocessing only")
        segments = run_preprocessing()
        success = (segments is not None)
    
    elif run_mode == "generate":
        # Mode: Generate variants only
        print(f"\n🎯 Mode: Generate variants only")
        
        # Load preprocessed data
        preprocessed_file = getattr(config, 'preprocessed_file', None)
        if preprocessed_file and os.path.exists(preprocessed_file):
            print(f"📥 Loading preprocessed data from file: {preprocessed_file}")
            with open(preprocessed_file, 'rb') as f:
                segments = pickle.load(f)
            print(f"✅ Loading successful, total {len(segments):,} fragments")
        else:
            print("🔄 Running preprocessing again...")
            segments = run_preprocessing()
        
        if segments:
            save_variants = getattr(config, 'save_variants', True)
            variants_data = generate_variants(segments, save_output=save_variants)
            success = (variants_data is not None)
        else:
            success = False
    
    elif run_mode == "train":
        # Mode: Training (but no variant file specified, need to run preprocessing and variant generation first)
        print(f"\n🎯 Mode: Training (requires preprocessing and variant generation first)")
        
        # Try to find existing variant file
        variants_file_path = find_latest_variants_file()
        if variants_file_path and use_existing_variants:
            print(f"🔍 Found existing variant file: {variants_file_path}")
            variants_data = load_variants_file(variants_file_path)
            if variants_data:
                success = run_nucleotide_training(variants_data)
            else:
                success = False
        else:
            # Run full pipeline
            run_mode = "all"
            print("🔄 No existing variant file found or not enabled, running full pipeline...")
    
    if run_mode == "all" and success:
        # Mode: Full pipeline
        print(f"\n🎯 Mode: Full pipeline")
        
        # 1. Preprocessing
        segments = run_preprocessing()
        if segments is None:
            success = False
        
        # 2. Generate variants
        if success:
            save_variants = getattr(config, 'save_variants', True)
            variants_data = generate_variants(segments, save_output=save_variants)
            if variants_data is None:
                success = False
        
        # 3. Training
        if success:
            success = run_nucleotide_training(variants_data)
    
    # ==================== Pipeline Summary ====================
    
    # Record end time
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("📊 Pipeline Summary")
    print(f"{'='*80}")
    print(f"  Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total duration: {duration}")
    
    if run_mode == "preprocess" and success:
        preprocess_file = os.path.join(config.output_dir, "preprocessed_segments.pkl")
        print(f"\n📁 Preprocessed data file: {preprocess_file}")
    
    if run_mode == "generate" and success:
        variants_file = os.path.join(config.output_dir, "variants_data.pkl")
        print(f"\n📁 Variant data file: {variants_file}")
    
    if success:
        print(f"\n✅ Pipeline execution successful!")
    else:
        print(f"\n❌ Some stages failed, please check the logs")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
