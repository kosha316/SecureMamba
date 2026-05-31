import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.manifold import TSNE
from umap import UMAP
import os
from config import config
from data_preprocessing import aa_to_codons
from typing import Tuple, Optional, List, Dict
from data_preprocessing import load_and_cache_preprocessed_data
from model import ProkBERTMambaModel, create_prokbert_mamba_model  # Updated import
from data_preprocessing import codon_to_aa, codon_to_idx, ContrastiveSequenceDataset
from datetime import datetime
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import json


def extract_validation_features(
    model: torch.nn.Module,
    val_segments: List[dict]  # 🔥 Changed to directly receive segments list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fixed feature extraction: directly process segments list
    """
    model.eval()
    all_features = []
    all_labels = []
    
    os.makedirs(config.preprocess_cache, exist_ok=True)
    
    print(f"🔍 Starting feature extraction, number of samples: {len(val_segments)}")
    
    with torch.no_grad():
        for idx, seg in enumerate(val_segments):
            try:
                # Directly get sequence data from segment
                seq_str = seg.get('original_seq') or seg.get('seq_str')
                if not seq_str:
                    continue
                    
                # Convert sequence to model input
                codons = [seq_str[i:i+3] for i in range(0, len(seq_str), 3)]
                valid_codons = [c for c in codons if c in codon_to_idx]
                if not valid_codons:
                    continue
                    
                sequence = torch.tensor([codon_to_idx[c] for c in valid_codons], 
                                      dtype=torch.long).unsqueeze(0).to(config.device)
                mask = torch.ones(1, len(valid_codons), dtype=torch.float32).to(config.device)
                
                # 🔥 Use universal encoding method
                if hasattr(model, 'encode_any_sequence'):
                    features = model.encode_any_sequence(sequence, mask)
                else:
                    # Fallback method
                    _, global_feat, _ = model._extract_variant_specialized_features(
                        sequence, mask, variant_type="original"
                    )
                    features = global_feat
                
                # Validate feature dimension
                if len(features.shape) != 2:
                    features = features.unsqueeze(0)
                
                all_features.extend(features.cpu().numpy())
                all_labels.append(seg.get('original_label', seg.get('label', 0)))
                
                if (idx + 1) % 100 == 0:
                    print(f"🔍 Feature extraction progress: {idx + 1}/{len(val_segments)}")
            
            except Exception as e:
                print(f"⚠️ Feature extraction failed for sample {idx + 1}: {str(e)}")
                continue
    
    # Convert to numpy arrays
    if not all_features:
        raise ValueError("No features extracted!")
        
    features = np.array(all_features)
    labels = np.array(all_labels)
    
    print(f"\n📊 Feature extraction completed:")
    print(f"   - Total samples: {features.shape[0]} | Feature dimension: {features.shape[1]}")
    print(f"   - Label distribution: {int(labels.sum())} positive | {len(labels)-int(labels.sum())} negative")
    
    return features, labels


def plot_feature_embedding(features, labels, method="tsne"):
    """Visualize feature embedding"""
    assert method in ["tsne", "umap"], "Unsupported dimensionality reduction method"
    plt.figure(figsize=(10, 8))
    
    print(f"🔄 Starting {method.upper()} dimensionality reduction...")
    if method == "tsne":
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embedding = tsne.fit_transform(features)
    else:
        umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = umap.fit_transform(features)
    
    # Visualization
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    plt.scatter(
        embedding[neg_mask, 0], embedding[neg_mask, 1],
        c='#1f77b4', s=30, alpha=0.7, label="Non-pathogenic"
    )
    plt.scatter(
        embedding[pos_mask, 0], embedding[pos_mask, 1],
        c='#d62728', s=30, alpha=0.7, label="Pathogenic"
    )
    
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.title(f"Sequence Feature Embedding ({method.upper()})")
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(config.output_dir, f"{method}_embedding.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 {method.upper()} visualization saved to: {save_path}")

def plot_confusion_matrix(all_labels, all_preds):
    """Plot confusion matrix"""
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["Non-pathogenic", "Pathogenic"],
        yticklabels=["Non-pathogenic", "Pathogenic"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    
    save_path = os.path.join(config.output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Confusion matrix saved to: {save_path}")


def plot_roc_curve(all_labels, all_probs):
    """Plot ROC curve"""
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = roc_auc_score(all_labels, all_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    save_path = os.path.join(config.output_dir, "roc_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 ROC curve saved to: {save_path}")



def load_pretrained_model(
    codon_to_aa: dict,
    codon_to_idx: dict,
    model_path: Optional[str] = None
) -> Tuple[ProkBERTMambaModel, dict]:
    """
    Encapsulate pretrained model loading logic, adapted for ProkBERTMambaModel
    """
    # 1. Handle default model path
    if model_path is None:
        model_path = os.path.join(config.output_dir, 'best_model.pth')
    print(f"📥 Loading pretrained model: {model_path}")
    
    # 2. Initialize ProkBERT model structure
    print(f"📌 Initializing ProkBERTMambaModel model...")
    model = create_prokbert_mamba_model(
        codon_to_idx=codon_to_idx,
        codon_to_aa=codon_to_aa,
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layer=config.n_layer,
        use_enhanced_blocks=getattr(config, 'use_enhanced_blocks', True),
        variant_specialization_weight=getattr(config, 'variant_specialization_weight', 0.2),
        freeze_prokbert=getattr(config, 'freeze_prokbert', True),
        num_classes=getattr(config, 'num_classes', 1)
    ).to(config.device)
    print(f"✅ Model structure initialization completed (device: {config.device} | total parameters: {sum(p.numel() for p in model.parameters()):,})")
    
    # 3. Safely load model weights
    try:
        # Safely load checkpoint
        checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
        
        # Extract model parameters and handle module. prefix from multi-GPU training
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if not state_dict:
            raise KeyError("model_state_dict key not found in checkpoint")
        
        # Compatible with multi-GPU training (remove module. prefix)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load parameters
        model.load_state_dict(state_dict, strict=False)
        
        # Switch to evaluation mode
        model.eval()
        
        # Print success information
        best_val_loss = checkpoint.get('best_val_loss', 'unknown')
        epoch = checkpoint.get('epoch', 'unknown')
        
        loss_str = f"{best_val_loss:.6f}" if isinstance(best_val_loss, (int, float)) else best_val_loss
        epoch_str = f"{epoch}" if isinstance(epoch, (int, float)) else epoch
        
        print(f"🎉 Pretrained model loaded successfully!")
        print(f"   - Training epoch: {epoch_str}")
        print(f"   - Best validation loss: {loss_str}")
        print(f"   - Model configuration: {checkpoint.get('config', 'none')}")
        print(f"   - Device: {config.device}")
        
        return model, checkpoint
    
    # 4. Exception handling
    except FileNotFoundError:
        raise FileNotFoundError(
            f"❌ Pretrained model file not found!\n"
            f"   Check path: {model_path}\n"
            f"   Ensure model file exists and filename is correct"
        )
    except KeyError as e:
        raise KeyError(
            f"❌ Checkpoint format error!\n"
            f"   Missing key: {e}\n"
            f"   Possible reasons: 1. Model file not trained with this code; 2. Checkpoint save format abnormal"
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"❌ Model weight and structure mismatch!\n"
            f"   Error details: {e}\n"
            f"   Solutions: 1. Ensure model parameters in config.py are consistent with training; 2. Check if codon mapping is the same"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(
            f"❌ Model loading failed!\n"
            f"   Error details: {str(e)}\n"
            f"   Check: 1. Model file integrity; 2. PyTorch version compatibility; 3. Sufficient hardware resources"
        )


def analyze_dual_positive_views_diff(segments: List[dict], name: str) -> bool:
    """
    Analyze the differences of dual positive sample views
    """
    semantic_diffs = []  # Semantic variant differences
    confusion_diffs = []  # DNA confusion differences
    
    analyzed_count = 0
    skipped_count = 0

    print(f"\n🔍 Starting {name} dual positive sample view difference analysis...")
    
    for i, seg in enumerate(segments):
        if 'original_seq' not in seg:
            skipped_count += 1
            continue
            
        if 'view_diff_details' not in seg:
            skipped_count += 1
            continue
        
        analyzed_count += 1

        # Analyze different types of view differences
        for detail in seg.get('view_diff_details', []):
            if detail['type'] == 'semantic':
                semantic_diffs.append(detail['base_diff_ratio'])
            elif detail['type'] == 'confusion':
                confusion_diffs.append(detail['base_diff_ratio'])

    # Statistical results
    print(f"\n📊 {name} dual positive sample view difference analysis:")
    print(f"  Analyzed samples: {analyzed_count}, Skipped samples: {skipped_count}")
    print(f"  Semantic variants: {len(semantic_diffs)}, DNA confusion: {len(confusion_diffs)}")
    
    if not semantic_diffs and not confusion_diffs:
        print(f"  ⚠️  Cannot calculate difference rate")
        return False

    if semantic_diffs:
        avg_semantic_diff = np.mean(semantic_diffs)
        std_semantic_diff = np.std(semantic_diffs)
        print(f"🔍 Semantic variant difference rate: mean {avg_semantic_diff:.3f} | std {std_semantic_diff:.3f} (target: 0.05~0.2)")
    
    if confusion_diffs:
        avg_confusion_diff = np.mean(confusion_diffs)
        std_confusion_diff = np.std(confusion_diffs)
        print(f"🧬 DNA confusion difference rate: mean {avg_confusion_diff:.3f} | std {std_confusion_diff:.3f} (target: 0.1~0.6)")

    # Validate data validity
    valid = True
    
    if semantic_diffs:
        avg_semantic = np.mean(semantic_diffs)
        if avg_semantic > 0.3:
            print("⚠️  Warning: Semantic variant difference too large!")
            valid = False
        elif avg_semantic < 0.02:
            print("⚠️  Warning: Semantic variant difference too small!")
            valid = False
    
    if confusion_diffs:
        avg_confusion = np.mean(confusion_diffs)
        if avg_confusion > 0.8:
            print("⚠️  Warning: DNA confusion difference too large!")
            valid = False
        elif avg_confusion < 0.05:
            print("⚠️  Warning: DNA confusion difference too small!")
            valid = False
    
    if valid:
        print("✅ Dual positive sample view difference analysis passed")
    
    return valid


def stats_dataset(dataset, name: str, pad_token: str) -> None:
    """Calculate dataset sequence lengths, valid variant proportions, etc. (adapted for dual positive sample data structure)"""
    seq_lengths = []
    valid_semantic_count = 0
    valid_confusion_count = 0
    valid_both_count = 0

    for seg in dataset.segments:
        # 🔥 Fix: Use original_seq instead of seq_str
        if 'original_seq' in seg:
            seq_str = seg['original_seq']
            valid_len = len(seq_str)
            seq_lengths.append(valid_len)
        else:
            continue
        
        # Count valid variants
        views = seg.get('positive_views', [])
        view_types = seg.get('view_types', [])
        
        has_semantic = any(t == 'semantic' for t in view_types)
        has_confusion = any(t == 'confusion' for t in view_types)
        
        if has_semantic:
            valid_semantic_count += 1
        if has_confusion:
            valid_confusion_count += 1
        if has_semantic and has_confusion:
            valid_both_count += 1

    # Output statistical results
    print(f"\n📈 {name} dataset statistics:")
    print(f"  Total samples: {len(dataset)}")
    if seq_lengths:
        print(f"  Sequence lengths: mean {np.mean(seq_lengths):.1f} | median {np.median(seq_lengths):.1f} | max {np.max(seq_lengths)} | min {np.min(seq_lengths)}")
        print(f"  Valid semantic variants: {valid_semantic_count} ({valid_semantic_count/len(dataset)*100:.1f}%)")
        print(f"  Valid DNA confusion variants: {valid_confusion_count} ({valid_confusion_count/len(dataset)*100:.1f}%)")
        print(f"  With both variants: {valid_both_count} ({valid_both_count/len(dataset)*100:.1f}%)")
    else:
        print(f"  ⚠️  Cannot retrieve sequence length information")


def analyze_feature_similarity(model, val_loader):
    """
    Analyze feature similarity between original sequences and dual positive sample variants
    """
    model.eval()
    semantic_similarities = []
    confusion_similarities = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {
                'sequences': batch['sequences'].to(config.device),
                'labels': batch['labels'].to(config.device),
                'mask': batch['mask'].to(config.device),
                'semantic_variants': batch.get('semantic_variants', None),
                'confusion_variants': batch.get('confusion_variants', None),
                'semantic_masks': batch.get('semantic_masks', None),
                'confusion_masks': batch.get('confusion_masks', None)
            }
            
            # Model forward pass - adapted for ProkBERTMambaModel
            orig_class_pred, semantic_class_pred, confusion_class_pred, aux_loss, orig_contrastive_feat, all_contrastive_feat = model(
                input_ids=batch_gpu['sequences'],
                labels=batch_gpu['labels'],
                mask=batch_gpu['mask'],
                semantic_variants=batch_gpu['semantic_variants'],
                confusion_variants=batch_gpu['confusion_variants'],
                semantic_masks=batch_gpu['semantic_masks'],
                confusion_masks=batch_gpu['confusion_masks']
            )
            
            if semantic_class_pred is not None and confusion_class_pred is not None:
                # Calculate similarity
                orig_feat_norm = F.normalize(orig_contrastive_feat, dim=1)
                
                # Semantic variant similarity
                semantic_contrastive_feat = model.encode_sequences(batch_gpu['semantic_variants'], batch_gpu['semantic_masks'])
                semantic_feat_norm = F.normalize(semantic_contrastive_feat, dim=1)
                semantic_sim = F.cosine_similarity(orig_feat_norm, semantic_feat_norm, dim=1).mean().item()
                semantic_similarities.append(semantic_sim)
                
                # DNA confusion variant similarity
                confusion_contrastive_feat = model.encode_sequences(batch_gpu['confusion_variants'], batch_gpu['confusion_masks'])
                confusion_feat_norm = F.normalize(confusion_contrastive_feat, dim=1)
                confusion_sim = F.cosine_similarity(orig_feat_norm, confusion_feat_norm, dim=1).mean().item()
                confusion_similarities.append(confusion_sim)
    
    if semantic_similarities and confusion_similarities:
        avg_semantic_sim = np.mean(semantic_similarities)
        avg_confusion_sim = np.mean(confusion_similarities)
        sim_gap = avg_semantic_sim - avg_confusion_sim
        
        print(f"\n🔍 Dual positive sample feature similarity analysis:")
        print(f"  Original-semantic variant similarity: {avg_semantic_sim:.4f}")
        print(f"  Original-DNA confusion variant similarity: {avg_confusion_sim:.4f}")
        print(f"  Similarity GAP: {sim_gap:.4f}")
        
        return avg_semantic_sim, avg_confusion_sim, sim_gap
    else:
        print("⚠️  Cannot calculate feature similarity")
        return 0, 0, 0


def plot_training_comparison(train_metrics, val_metrics, save_dir):
    """
    Plot training process key metric comparison charts (adapted for dual positive sample learning)
    """
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    
    epochs = range(1, len(train_metrics) + 1)
    
    plt.figure(figsize=(20, 12))
    
    # 1. Loss curves
    plt.subplot(2, 3, 1)
    plt.plot(epochs, [m['loss'] for m in train_metrics], label='Training Loss', linewidth=2)
    plt.plot(epochs, [m['loss'] for m in val_metrics], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. Accuracy curves
    plt.subplot(2, 3, 2)
    plt.plot(epochs, [m['acc'] for m in train_metrics], label='Training ACC', linewidth=2)
    plt.plot(epochs, [m['acc'] for m in val_metrics], label='Validation ACC', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 3. Variant classification accuracy
    plt.subplot(2, 3, 3)
    if 'semantic_acc' in train_metrics[0]:
        plt.plot(epochs, [m['semantic_acc'] for m in train_metrics], label='Training Semantic Variant ACC', linewidth=2)
        plt.plot(epochs, [m['confusion_acc'] for m in train_metrics], label='Training DNA Confusion ACC', linewidth=2)
        plt.plot(epochs, [m['semantic_acc'] for m in val_metrics], '--', label='Validation Semantic Variant ACC', linewidth=2)
        plt.plot(epochs, [m['confusion_acc'] for m in val_metrics], '--', label='Validation DNA Confusion ACC', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Variant Classification Accuracy')
    plt.title('Variant Classification Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 4. Feature similarity
    plt.subplot(2, 3, 4)
    if 'semantic_sim' in train_metrics[0]:
        plt.plot(epochs, [m['semantic_sim'] for m in train_metrics], label='Training Semantic Similarity', linewidth=2)
        plt.plot(epochs, [m['confusion_sim'] for m in train_metrics], label='Training DNA Confusion Similarity', linewidth=2)
        plt.plot(epochs, [m['semantic_sim'] for m in val_metrics], '--', label='Validation Semantic Similarity', linewidth=2)
        plt.plot(epochs, [m['confusion_sim'] for m in val_metrics], '--', label='Validation DNA Confusion Similarity', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Feature Similarity')
    plt.title('Original Sequence and Variant Feature Similarity')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 5. AUC curve
    plt.subplot(2, 3, 5)
    plt.plot(epochs, [m['auc'] for m in train_metrics], label='Training AUC', linewidth=2)
    plt.plot(epochs, [m['auc'] for m in val_metrics], label='Validation AUC', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 6. Contrastive loss
    plt.subplot(2, 3, 6)
    plt.plot(epochs, [m['contrastive_loss'] for m in train_metrics], label='Training Contrastive Loss', linewidth=2)
    plt.plot(epochs, [m['contrastive_loss'] for m in val_metrics], label='Validation Contrastive Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.title('Contrastive Learning Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'dual_positive_training_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Dual positive sample training comparison chart saved to: {save_path}")


def ensure_data_consistency(segments: List[dict]) -> List[dict]:
    """Ensure data field consistency (adapted for dual positive samples)"""
    consistent_segments = []
    
    for seg in segments:
        # Create copy to avoid modifying original data
        seg_copy = seg.copy()
        
        # 🔥 Fix: Ensure necessary fields exist
        if 'seq_str' not in seg_copy:
            if 'original_seq' in seg_copy:
                seg_copy['seq_str'] = seg_copy['original_seq']
            else:
                print(f"⚠️ Skipping invalid fragment: missing sequence field, available fields: {list(seg_copy.keys())}")
                continue
        
        if 'label' not in seg_copy:
            if 'original_label' in seg_copy:
                seg_copy['label'] = seg_copy['original_label']
            else:
                seg_copy['label'] = 0  # Default value
        
        # Ensure dual positive sample view field consistency
        if 'positive_views' in seg_copy and 'view_types' not in seg_copy:
            # If view type information is missing, try to infer
            views = seg_copy['positive_views']
            if len(views) >= 2:
                seg_copy['view_types'] = ['semantic', 'confusion']
            elif len(views) == 1:
                seg_copy['view_types'] = ['semantic']  # Default assumption as semantic variant
        
        consistent_segments.append(seg_copy)
    
    return consistent_segments


def save_variant_cache(train_segments, val_segments, cache_path):
    """Save variant data cache, ensuring field consistency"""
    # 🔥 Fix: Ensure field consistency before saving
    train_consistent = ensure_data_consistency(train_segments)
    val_consistent = ensure_data_consistency(val_segments)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((train_consistent, val_consistent), f, protocol=pickle.HIGHEST_PROTOCOL)
    
    cache_size = os.path.getsize(cache_path) / 1024 / 1024
    print(f"✅ Dual positive sample fragment cache saved: {cache_path} ({cache_size:.2f} MB)")
    return train_consistent, val_consistent


def analyze_model_predictions(model, val_loader):
    """
    Analyze model prediction performance on validation set (adapted for ProkBERTMambaModel)
    """
    model.eval()
    
    all_orig_preds = []
    all_semantic_preds = []
    all_confusion_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {
                'sequences': batch['sequences'].to(config.device),
                'labels': batch['labels'].to(config.device),
                'mask': batch['mask'].to(config.device),
                'semantic_variants': batch.get('semantic_variants', None),
                'confusion_variants': batch.get('confusion_variants', None),
                'semantic_masks': batch.get('semantic_masks', None),
                'confusion_masks': batch.get('confusion_masks', None)
            }
            
            # Use universal prediction method
            orig_pred = model.predict_any_sequence(batch_gpu['sequences'], batch_gpu['mask'])
            
            # Collect prediction results
            orig_probs = torch.sigmoid(orig_pred).cpu().numpy()
            orig_preds = (orig_probs > 0.5).astype(int)
            all_orig_preds.extend(orig_preds)
            
            # Semantic variant predictions
            if batch_gpu['semantic_variants'] is not None and batch_gpu['semantic_masks'] is not None:
                semantic_pred = model.predict_any_sequence(batch_gpu['semantic_variants'], batch_gpu['semantic_masks'])
                semantic_probs = torch.sigmoid(semantic_pred).cpu().numpy()
                semantic_preds = (semantic_probs > 0.5).astype(int)
                all_semantic_preds.extend(semantic_preds)
            
            # DNA confusion variant predictions
            if batch_gpu['confusion_variants'] is not None and batch_gpu['confusion_masks'] is not None:
                confusion_pred = model.predict_any_sequence(batch_gpu['confusion_variants'], batch_gpu['confusion_masks'])
                confusion_probs = torch.sigmoid(confusion_pred).cpu().numpy()
                confusion_preds = (confusion_probs > 0.5).astype(int)
                all_confusion_preds.extend(confusion_preds)
            
            all_labels.extend(batch_gpu['labels'].cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score
    
    print(f"\n📊 Model prediction analysis:")
    print(f"  Total samples: {len(all_labels)}")
    
    orig_acc = accuracy_score(all_labels, all_orig_preds) if all_orig_preds else 0.0
    orig_f1 = f1_score(all_labels, all_orig_preds, average='binary') if all_orig_preds else 0.0
    print(f"  Original sequence prediction: ACC {orig_acc:.4f} | F1 {orig_f1:.4f}")
    
    if all_semantic_preds:
        semantic_acc = accuracy_score(all_labels, all_semantic_preds)
        semantic_f1 = f1_score(all_labels, all_semantic_preds, average='binary')
        print(f"  Semantic variant prediction: ACC {semantic_acc:.4f} | F1 {semantic_f1:.4f}")
    
    if all_confusion_preds:
        confusion_acc = accuracy_score(all_labels, all_confusion_preds)
        confusion_f1 = f1_score(all_labels, all_confusion_preds, average='binary')
        print(f"  DNA confusion prediction: ACC {confusion_acc:.4f} | F1 {confusion_f1:.4f}")
    
    return {
        'orig_acc': orig_acc,
        'orig_f1': orig_f1,
        'semantic_acc': semantic_acc if all_semantic_preds else 0.0,
        'semantic_f1': semantic_f1 if all_semantic_preds else 0.0,
        'confusion_acc': confusion_acc if all_confusion_preds else 0.0,
        'confusion_f1': confusion_f1 if all_confusion_preds else 0.0
    }


def create_visualization_report(model, val_loader):
    """
    Create comprehensive visualization report (adapted for ProkBERTMambaModel)
    """
    print(f"\n📋 Starting dual positive sample learning visualization report creation...")
    
    # 1. Feature extraction and visualization
    print(f"🔍 Extracting validation set features...")
    features, labels = extract_validation_features(model, val_loader)
    
    # 2. Feature embedding visualization
    print(f"📊 Generating feature embedding plots...")
    plot_feature_embedding(features, labels, method="tsne")
    plot_feature_embedding(features, labels, method="umap")
    
    # 3. Analyze model predictions
    print(f"🎯 Analyzing model prediction performance...")
    pred_metrics = analyze_model_predictions(model, val_loader)
    
    # 4. Analyze feature similarity
    print(f"🔍 Analyzing feature similarity...")
    semantic_sim, confusion_sim, sim_gap = analyze_feature_similarity(model, val_loader)
    
    # 5. Save report summary
    report_data = {
        'feature_extraction': {
            'sample_count': len(features),
            'feature_dim': features.shape[1],
            'label_distribution': {
                'positive': int(labels.sum()),
                'negative': len(labels) - int(labels.sum()),
                'positive_ratio': float(labels.sum() / len(labels))
            }
        },
        'prediction_metrics': pred_metrics,
        'feature_similarity': {
            'semantic_similarity': float(semantic_sim),
            'confusion_similarity': float(confusion_sim),
            'similarity_gap': float(sim_gap)
        },
        'visualization_files': [
            'tsne_embedding.png',
            'umap_embedding.png',
            'confusion_matrix.png',
            'roc_curve.png'
        ],
        'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save report
    report_path = os.path.join(config.output_dir, 'visualization_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Visualization report saved to: {report_path}")
    
    # Print report summary
    print(f"\n📈 Visualization report summary:")
    print(f"  Feature extraction: {report_data['feature_extraction']['sample_count']} samples")
    print(f"  Label distribution: {report_data['feature_extraction']['label_distribution']['positive']} positive, {report_data['feature_extraction']['label_distribution']['negative']} negative")
    print(f"  Prediction accuracy: original {report_data['prediction_metrics']['orig_acc']:.4f}, semantic {report_data['prediction_metrics']['semantic_acc']:.4f}, DNA confusion {report_data['prediction_metrics']['confusion_acc']:.4f}")
    print(f"  Feature similarity: semantic {report_data['feature_similarity']['semantic_similarity']:.4f}, DNA confusion {report_data['feature_similarity']['confusion_similarity']:.4f}")
    
    return report_data


def setup_training_environment():
    """
    Set up training environment, including directory creation and configuration validation
    """
    print(f"🔧 Setting up training environment...")
    
    # Create necessary directories
    directories = [
        config.output_dir,
        config.preprocess_cache,
        config.cache_dir,
        os.path.join(config.output_dir, 'checkpoints'),
        os.path.join(config.output_dir, 'logs')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ Created directory: {directory}")
    
    # Validate configuration
    required_files = [config.positive_fasta, config.negative_fasta]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"  ⚠️  Warning: File does not exist: {file_path}")
    
    # Save current configuration
    config_path = os.path.join(config.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2, ensure_ascii=False)
    print(f"  ✅ Configuration saved: {config_path}")
    
    # Set random seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print(f"✅ Training environment setup completed")


# Test function
if __name__ == "__main__":
    print("🧪 Testing utility functions...")
    
    # Test data consistency check
    test_segments = [
        {'original_seq': 'ATGCGT', 'original_label': 1, 'positive_views': ['ATGAAA', 'TTTCCC'], 'view_types': ['semantic', 'confusion']},
        {'original_seq': 'GGGAAA', 'original_label': 0, 'positive_views': ['GGGTTT']},
        {'seq_str': 'CCCTTT', 'label': 1}  # Test field repair
    ]
    
    consistent_segments = ensure_data_consistency(test_segments)
    print(f"Data consistency test: {len(consistent_segments)} fragments after repair")
    
    # Test dataset statistics
    stats_dataset(ContrastiveSequenceDataset(consistent_segments), "Test Dataset", config.pad_token)
    
    # Test dual positive sample view difference analysis
    analyze_dual_positive_views_diff(consistent_segments, "Test Data")
    
    print("✅ Utility function testing completed")
