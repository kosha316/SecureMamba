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
from model import ProkBERTMambaModel, create_prokbert_mamba_model  # 更新导入
from data_preprocessing import codon_to_aa, codon_to_idx, ContrastiveSequenceDataset
from datetime import datetime
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import json


def extract_validation_features(
    model: torch.nn.Module,
    val_segments: List[dict]  # 🔥 改为直接接收 segments 列表
) -> Tuple[np.ndarray, np.ndarray]:
    """
    修复特征提取：直接处理 segments 列表
    """
    model.eval()
    all_features = []
    all_labels = []
    
    os.makedirs(config.preprocess_cache, exist_ok=True)
    
    print(f"🔍 开始特征提取，样本数: {len(val_segments)}")
    
    with torch.no_grad():
        for idx, seg in enumerate(val_segments):
            try:
                # 直接从segment获取序列数据
                seq_str = seg.get('original_seq') or seg.get('seq_str')
                if not seq_str:
                    continue
                    
                # 将序列转换为模型输入
                codons = [seq_str[i:i+3] for i in range(0, len(seq_str), 3)]
                valid_codons = [c for c in codons if c in codon_to_idx]
                if not valid_codons:
                    continue
                    
                sequence = torch.tensor([codon_to_idx[c] for c in valid_codons], 
                                      dtype=torch.long).unsqueeze(0).to(config.device)
                mask = torch.ones(1, len(valid_codons), dtype=torch.float32).to(config.device)
                
                # 🔥 使用通用编码方法
                if hasattr(model, 'encode_any_sequence'):
                    features = model.encode_any_sequence(sequence, mask)
                else:
                    # 回退方法
                    _, global_feat, _ = model._extract_variant_specialized_features(
                        sequence, mask, variant_type="original"
                    )
                    features = global_feat
                
                # 验证特征维度
                if len(features.shape) != 2:
                    features = features.unsqueeze(0)
                
                all_features.extend(features.cpu().numpy())
                all_labels.append(seg.get('original_label', seg.get('label', 0)))
                
                if (idx + 1) % 100 == 0:
                    print(f"🔍 特征提取进度：{idx + 1}/{len(val_segments)}")
            
            except Exception as e:
                print(f"⚠️ 第 {idx + 1} 个样本特征提取失败：{str(e)}")
                continue
    
    # 转换为numpy数组
    if not all_features:
        raise ValueError("未提取到任何特征！")
        
    features = np.array(all_features)
    labels = np.array(all_labels)
    
    print(f"\n📊 特征提取完成：")
    print(f"   - 总样本数：{features.shape[0]} | 特征维度：{features.shape[1]}")
    print(f"   - 标签分布：正例 {int(labels.sum())} 个 | 负例 {len(labels)-int(labels.sum())} 个")
    
    return features, labels


def plot_feature_embedding(features, labels, method="tsne"):
    """可视化特征嵌入"""
    assert method in ["tsne", "umap"], "不支持的降维方法"
    plt.figure(figsize=(10, 8))
    
    print(f"🔄 开始{method.upper()}降维...")
    if method == "tsne":
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embedding = tsne.fit_transform(features)
    else:
        umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = umap.fit_transform(features)
    
    # 可视化
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    plt.scatter(
        embedding[neg_mask, 0], embedding[neg_mask, 1],
        c='#1f77b4', s=30, alpha=0.7, label="非致病性"
    )
    plt.scatter(
        embedding[pos_mask, 0], embedding[pos_mask, 1],
        c='#d62728', s=30, alpha=0.7, label="致病性"
    )
    
    plt.xlabel(f"{method.upper()} 维度1")
    plt.ylabel(f"{method.upper()} 维度2")
    plt.title(f"序列特征嵌入 ({method.upper()})")
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(config.output_dir, f"{method}_embedding.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 {method.upper()}可视化已保存至：{save_path}")

def plot_confusion_matrix(all_labels, all_preds):
    """绘制混淆矩阵"""
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["非致病性", "致病性"],
        yticklabels=["非致病性", "致病性"]
    )
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("混淆矩阵")
    
    save_path = os.path.join(config.output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 混淆矩阵已保存至：{save_path}")


def plot_roc_curve(all_labels, all_probs):
    """绘制ROC曲线"""
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = roc_auc_score(all_labels, all_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend()
    
    save_path = os.path.join(config.output_dir, "roc_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 ROC曲线已保存至：{save_path}")



def load_pretrained_model(
    codon_to_aa: dict,
    codon_to_idx: dict,
    model_path: Optional[str] = None
) -> Tuple[ProkBERTMambaModel, dict]:
    """
    封装预训练模型加载逻辑，适配ProkBERTMambaModel
    """
    # 1. 处理默认模型路径
    if model_path is None:
        model_path = os.path.join(config.output_dir, 'best_model.pth')
    print(f"📥 开始加载预训练模型：{model_path}")
    
    # 2. 初始化ProkBERT模型结构
    print(f"📌 初始化ProkBERTMambaModel模型...")
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
    print(f"✅ 模型结构初始化完成（设备：{config.device} | 总参数：{sum(p.numel() for p in model.parameters()):,}）")
    
    # 3. 安全加载模型权重
    try:
        # 安全加载checkpoint
        checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
        
        # 提取模型参数并处理多GPU训练的module.前缀
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if not state_dict:
            raise KeyError("checkpoint中未找到model_state_dict键")
        
        # 兼容多GPU训练（移除module.前缀）
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 加载参数
        model.load_state_dict(state_dict, strict=False)
        
        # 切换到评估模式
        model.eval()
        
        # 打印加载成功信息
        best_val_loss = checkpoint.get('best_val_loss', '未知')
        epoch = checkpoint.get('epoch', '未知')
        
        loss_str = f"{best_val_loss:.6f}" if isinstance(best_val_loss, (int, float)) else best_val_loss
        epoch_str = f"{epoch}" if isinstance(epoch, (int, float)) else epoch
        
        print(f"🎉 预训练模型加载成功！")
        print(f"   - 训练epoch：{epoch_str}")
        print(f"   - 最佳验证损失：{loss_str}")
        print(f"   - 模型配置：{checkpoint.get('config', '无')}")
        print(f"   - 设备：{config.device}")
        
        return model, checkpoint
    
    # 4. 异常处理
    except FileNotFoundError:
        raise FileNotFoundError(
            f"❌ 未找到预训练模型文件！\n"
            f"   请检查路径：{model_path}\n"
            f"   确保模型文件存在且文件名正确"
        )
    except KeyError as e:
        raise KeyError(
            f"❌ checkpoint格式错误！\n"
            f"   缺少关键键：{e}\n"
            f"   可能原因：1. 模型文件不是本代码训练的；2. checkpoint保存时格式异常"
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"❌ 模型权重与结构不匹配！\n"
            f"   错误详情：{e}\n"
            f"   解决方案：1. 确保config.py中模型参数与训练时一致；2. 检查密码子映射是否相同"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(
            f"❌ 模型加载失败！\n"
            f"   错误详情：{str(e)}\n"
            f"   请检查：1. 模型文件完整性；2. PyTorch版本兼容性；3. 硬件资源是否充足"
        )


def analyze_dual_positive_views_diff(segments: List[dict], name: str) -> bool:
    """
    分析双正样本视图的差异
    """
    semantic_diffs = []  # 语义变体差异
    confusion_diffs = []  # DNA混淆差异
    
    analyzed_count = 0
    skipped_count = 0

    print(f"\n🔍 开始{name}双正样本视图差异分析...")
    
    for i, seg in enumerate(segments):
        if 'original_seq' not in seg:
            skipped_count += 1
            continue
            
        if 'view_diff_details' not in seg:
            skipped_count += 1
            continue
        
        analyzed_count += 1

        # 分析不同类型的视图差异
        for detail in seg.get('view_diff_details', []):
            if detail['type'] == 'semantic':
                semantic_diffs.append(detail['base_diff_ratio'])
            elif detail['type'] == 'confusion':
                confusion_diffs.append(detail['base_diff_ratio'])

    # 统计结果
    print(f"\n📊 {name}双正样本视图差异分析：")
    print(f"  分析样本数：{analyzed_count}，跳过样本数：{skipped_count}")
    print(f"  语义变体数：{len(semantic_diffs)}，DNA混淆数：{len(confusion_diffs)}")
    
    if not semantic_diffs and not confusion_diffs:
        print(f"  ⚠️  无法计算差异率")
        return False

    if semantic_diffs:
        avg_semantic_diff = np.mean(semantic_diffs)
        std_semantic_diff = np.std(semantic_diffs)
        print(f"🔍 语义变体差异率：均值 {avg_semantic_diff:.3f} | 标准差 {std_semantic_diff:.3f}（目标：0.05~0.2）")
    
    if confusion_diffs:
        avg_confusion_diff = np.mean(confusion_diffs)
        std_confusion_diff = np.std(confusion_diffs)
        print(f"🧬 DNA混淆差异率：均值 {avg_confusion_diff:.3f} | 标准差 {std_confusion_diff:.3f}（目标：0.1~0.6）")

    # 校验数据有效性
    valid = True
    
    if semantic_diffs:
        avg_semantic = np.mean(semantic_diffs)
        if avg_semantic > 0.3:
            print("⚠️  警告：语义变体差异过大！")
            valid = False
        elif avg_semantic < 0.02:
            print("⚠️  警告：语义变体差异过小！")
            valid = False
    
    if confusion_diffs:
        avg_confusion = np.mean(confusion_diffs)
        if avg_confusion > 0.8:
            print("⚠️  警告：DNA混淆差异过大！")
            valid = False
        elif avg_confusion < 0.05:
            print("⚠️  警告：DNA混淆差异过小！")
            valid = False
    
    if valid:
        print("✅ 双正样本视图差异分析通过")
    
    return valid


def stats_dataset(dataset, name: str, pad_token: str) -> None:
    """统计数据集序列长度、有效变体占比等关键信息（适配双正样本数据结构）"""
    seq_lengths = []
    valid_semantic_count = 0
    valid_confusion_count = 0
    valid_both_count = 0

    for seg in dataset.segments:
        # 🔥 修复：使用 original_seq 而不是 seq_str
        if 'original_seq' in seg:
            seq_str = seg['original_seq']
            valid_len = len(seq_str)
            seq_lengths.append(valid_len)
        else:
            continue
        
        # 统计有效变体
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

    # 输出统计结果
    print(f"\n📈 {name}数据集统计：")
    print(f"  样本总数：{len(dataset)}")
    if seq_lengths:
        print(f"  序列长度：均值 {np.mean(seq_lengths):.1f} | 中位数 {np.median(seq_lengths):.1f} | 最大 {np.max(seq_lengths)} | 最小 {np.min(seq_lengths)}")
        print(f"  有效语义变体：{valid_semantic_count}个 ({valid_semantic_count/len(dataset)*100:.1f}%)")
        print(f"  有效DNA混淆变体：{valid_confusion_count}个 ({valid_confusion_count/len(dataset)*100:.1f}%)")
        print(f"  同时有双变体：{valid_both_count}个 ({valid_both_count/len(dataset)*100:.1f}%)")
    else:
        print(f"  ⚠️  无法获取序列长度信息")


def analyze_feature_similarity(model, val_loader):
    """
    分析原始序列与双正样本变体的特征相似度
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
            
            # 模型前向传播 - 适配ProkBERTMambaModel
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
                # 计算相似度
                orig_feat_norm = F.normalize(orig_contrastive_feat, dim=1)
                
                # 语义变体相似度
                semantic_contrastive_feat = model.encode_sequences(batch_gpu['semantic_variants'], batch_gpu['semantic_masks'])
                semantic_feat_norm = F.normalize(semantic_contrastive_feat, dim=1)
                semantic_sim = F.cosine_similarity(orig_feat_norm, semantic_feat_norm, dim=1).mean().item()
                semantic_similarities.append(semantic_sim)
                
                # DNA混淆变体相似度
                confusion_contrastive_feat = model.encode_sequences(batch_gpu['confusion_variants'], batch_gpu['confusion_masks'])
                confusion_feat_norm = F.normalize(confusion_contrastive_feat, dim=1)
                confusion_sim = F.cosine_similarity(orig_feat_norm, confusion_feat_norm, dim=1).mean().item()
                confusion_similarities.append(confusion_sim)
    
    if semantic_similarities and confusion_similarities:
        avg_semantic_sim = np.mean(semantic_similarities)
        avg_confusion_sim = np.mean(confusion_similarities)
        sim_gap = avg_semantic_sim - avg_confusion_sim
        
        print(f"\n🔍 双正样本特征相似度分析：")
        print(f"  原始-语义变体相似度：{avg_semantic_sim:.4f}")
        print(f"  原始-DNA混淆变体相似度：{avg_confusion_sim:.4f}")
        print(f"  相似度GAP：{sim_gap:.4f}")
        
        return avg_semantic_sim, avg_confusion_sim, sim_gap
    else:
        print("⚠️  无法计算特征相似度")
        return 0, 0, 0


def plot_training_comparison(train_metrics, val_metrics, save_dir):
    """
    绘制训练过程关键指标对比图（适配双正样本学习）
    """
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    
    epochs = range(1, len(train_metrics) + 1)
    
    plt.figure(figsize=(20, 12))
    
    # 1. 损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(epochs, [m['loss'] for m in train_metrics], label='训练损失', linewidth=2)
    plt.plot(epochs, [m['loss'] for m in val_metrics], label='验证损失', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练与验证损失')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. 准确率曲线
    plt.subplot(2, 3, 2)
    plt.plot(epochs, [m['acc'] for m in train_metrics], label='训练ACC', linewidth=2)
    plt.plot(epochs, [m['acc'] for m in val_metrics], label='验证ACC', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练与验证准确率')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 3. 变体分类准确率
    plt.subplot(2, 3, 3)
    if 'semantic_acc' in train_metrics[0]:
        plt.plot(epochs, [m['semantic_acc'] for m in train_metrics], label='训练语义变体ACC', linewidth=2)
        plt.plot(epochs, [m['confusion_acc'] for m in train_metrics], label='训练DNA混淆ACC', linewidth=2)
        plt.plot(epochs, [m['semantic_acc'] for m in val_metrics], '--', label='验证语义变体ACC', linewidth=2)
        plt.plot(epochs, [m['confusion_acc'] for m in val_metrics], '--', label='验证DNA混淆ACC', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('变体分类准确率')
    plt.title('变体分类准确率')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 4. 特征相似度
    plt.subplot(2, 3, 4)
    if 'semantic_sim' in train_metrics[0]:
        plt.plot(epochs, [m['semantic_sim'] for m in train_metrics], label='训练语义相似度', linewidth=2)
        plt.plot(epochs, [m['confusion_sim'] for m in train_metrics], label='训练DNA混淆相似度', linewidth=2)
        plt.plot(epochs, [m['semantic_sim'] for m in val_metrics], '--', label='验证语义相似度', linewidth=2)
        plt.plot(epochs, [m['confusion_sim'] for m in val_metrics], '--', label='验证DNA混淆相似度', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('特征相似度')
    plt.title('原始序列与变体特征相似度')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 5. AUC曲线
    plt.subplot(2, 3, 5)
    plt.plot(epochs, [m['auc'] for m in train_metrics], label='训练AUC', linewidth=2)
    plt.plot(epochs, [m['auc'] for m in val_metrics], label='验证AUC', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('训练与验证AUC')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 6. 对比损失
    plt.subplot(2, 3, 6)
    plt.plot(epochs, [m['contrastive_loss'] for m in train_metrics], label='训练对比损失', linewidth=2)
    plt.plot(epochs, [m['contrastive_loss'] for m in val_metrics], label='验证对比损失', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('对比损失')
    plt.title('对比学习损失')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'dual_positive_training_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 双正样本训练对比图已保存至：{save_path}")


def ensure_data_consistency(segments: List[dict]) -> List[dict]:
    """确保数据字段一致性（适配双正样本）"""
    consistent_segments = []
    
    for seg in segments:
        # 创建副本以避免修改原始数据
        seg_copy = seg.copy()
        
        # 🔥 修复：确保有必要的字段
        if 'seq_str' not in seg_copy:
            if 'original_seq' in seg_copy:
                seg_copy['seq_str'] = seg_copy['original_seq']
            else:
                print(f"⚠️ 跳过无效片段：缺少序列字段，可用字段：{list(seg_copy.keys())}")
                continue
        
        if 'label' not in seg_copy:
            if 'original_label' in seg_copy:
                seg_copy['label'] = seg_copy['original_label']
            else:
                seg_copy['label'] = 0  # 默认值
        
        # 确保双正样本视图字段一致性
        if 'positive_views' in seg_copy and 'view_types' not in seg_copy:
            # 如果没有视图类型信息，尝试推断
            views = seg_copy['positive_views']
            if len(views) >= 2:
                seg_copy['view_types'] = ['semantic', 'confusion']
            elif len(views) == 1:
                seg_copy['view_types'] = ['semantic']  # 默认假设为语义变体
        
        consistent_segments.append(seg_copy)
    
    return consistent_segments


def save_variant_cache(train_segments, val_segments, cache_path):
    """保存变体数据缓存，确保字段一致性"""
    # 🔥 修复：在保存前确保字段一致性
    train_consistent = ensure_data_consistency(train_segments)
    val_consistent = ensure_data_consistency(val_segments)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((train_consistent, val_consistent), f, protocol=pickle.HIGHEST_PROTOCOL)
    
    cache_size = os.path.getsize(cache_path) / 1024 / 1024
    print(f"✅ 双正样本片段缓存已保存：{cache_path}（{cache_size:.2f} MB）")
    return train_consistent, val_consistent


def analyze_model_predictions(model, val_loader):
    """
    分析模型在验证集上的预测表现（适配ProkBERTMambaModel）
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
            
            # 使用通用预测方法
            orig_pred = model.predict_any_sequence(batch_gpu['sequences'], batch_gpu['mask'])
            
            # 收集预测结果
            orig_probs = torch.sigmoid(orig_pred).cpu().numpy()
            orig_preds = (orig_probs > 0.5).astype(int)
            all_orig_preds.extend(orig_preds)
            
            # 语义变体预测
            if batch_gpu['semantic_variants'] is not None and batch_gpu['semantic_masks'] is not None:
                semantic_pred = model.predict_any_sequence(batch_gpu['semantic_variants'], batch_gpu['semantic_masks'])
                semantic_probs = torch.sigmoid(semantic_pred).cpu().numpy()
                semantic_preds = (semantic_probs > 0.5).astype(int)
                all_semantic_preds.extend(semantic_preds)
            
            # DNA混淆变体预测
            if batch_gpu['confusion_variants'] is not None and batch_gpu['confusion_masks'] is not None:
                confusion_pred = model.predict_any_sequence(batch_gpu['confusion_variants'], batch_gpu['confusion_masks'])
                confusion_probs = torch.sigmoid(confusion_pred).cpu().numpy()
                confusion_preds = (confusion_probs > 0.5).astype(int)
                all_confusion_preds.extend(confusion_preds)
            
            all_labels.extend(batch_gpu['labels'].cpu().numpy())
    
    # 计算指标
    from sklearn.metrics import accuracy_score, f1_score
    
    print(f"\n📊 模型预测分析：")
    print(f"  样本总数：{len(all_labels)}")
    
    orig_acc = accuracy_score(all_labels, all_orig_preds) if all_orig_preds else 0.0
    orig_f1 = f1_score(all_labels, all_orig_preds, average='binary') if all_orig_preds else 0.0
    print(f"  原始序列预测：ACC {orig_acc:.4f} | F1 {orig_f1:.4f}")
    
    if all_semantic_preds:
        semantic_acc = accuracy_score(all_labels, all_semantic_preds)
        semantic_f1 = f1_score(all_labels, all_semantic_preds, average='binary')
        print(f"  语义变体预测：ACC {semantic_acc:.4f} | F1 {semantic_f1:.4f}")
    
    if all_confusion_preds:
        confusion_acc = accuracy_score(all_labels, all_confusion_preds)
        confusion_f1 = f1_score(all_labels, all_confusion_preds, average='binary')
        print(f"  DNA混淆预测：ACC {confusion_acc:.4f} | F1 {confusion_f1:.4f}")
    
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
    创建完整的可视化报告（适配ProkBERTMambaModel）
    """
    print(f"\n📋 开始创建双正样本学习可视化报告...")
    
    # 1. 特征提取和可视化
    print(f"🔍 提取验证集特征...")
    features, labels = extract_validation_features(model, val_loader)
    
    # 2. 特征嵌入可视化
    print(f"📊 生成特征嵌入图...")
    plot_feature_embedding(features, labels, method="tsne")
    plot_feature_embedding(features, labels, method="umap")
    
    # 3. 分析模型预测
    print(f"🎯 分析模型预测表现...")
    pred_metrics = analyze_model_predictions(model, val_loader)
    
    # 4. 分析特征相似度
    print(f"🔍 分析特征相似度...")
    semantic_sim, confusion_sim, sim_gap = analyze_feature_similarity(model, val_loader)
    
    # 5. 保存报告摘要
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
    
    # 保存报告
    report_path = os.path.join(config.output_dir, 'visualization_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 可视化报告已保存至：{report_path}")
    
    # 打印报告摘要
    print(f"\n📈 可视化报告摘要：")
    print(f"  特征提取：{report_data['feature_extraction']['sample_count']}个样本")
    print(f"  标签分布：正例{report_data['feature_extraction']['label_distribution']['positive']}，负例{report_data['feature_extraction']['label_distribution']['negative']}")
    print(f"  预测准确率：原始{report_data['prediction_metrics']['orig_acc']:.4f}，语义{report_data['prediction_metrics']['semantic_acc']:.4f}，DNA混淆{report_data['prediction_metrics']['confusion_acc']:.4f}")
    print(f"  特征相似度：语义{report_data['feature_similarity']['semantic_similarity']:.4f}，DNA混淆{report_data['feature_similarity']['confusion_similarity']:.4f}")
    
    return report_data


def setup_training_environment():
    """
    设置训练环境，包括目录创建和配置验证
    """
    print(f"🔧 设置训练环境...")
    
    # 创建必要目录
    directories = [
        config.output_dir,
        config.preprocess_cache,
        config.cache_dir,
        os.path.join(config.output_dir, 'checkpoints'),
        os.path.join(config.output_dir, 'logs')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ 创建目录：{directory}")
    
    # 验证配置
    required_files = [config.positive_fasta, config.negative_fasta]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"  ⚠️  警告：文件不存在：{file_path}")
    
    # 保存当前配置
    config_path = os.path.join(config.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2, ensure_ascii=False)
    print(f"  ✅ 配置已保存：{config_path}")
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print(f"✅ 训练环境设置完成")


# 测试函数
if __name__ == "__main__":
    print("🧪 测试工具函数...")
    
    # 测试数据一致性检查
    test_segments = [
        {'original_seq': 'ATGCGT', 'original_label': 1, 'positive_views': ['ATGAAA', 'TTTCCC'], 'view_types': ['semantic', 'confusion']},
        {'original_seq': 'GGGAAA', 'original_label': 0, 'positive_views': ['GGGTTT']},
        {'seq_str': 'CCCTTT', 'label': 1}  # 测试字段修复
    ]
    
    consistent_segments = ensure_data_consistency(test_segments)
    print(f"数据一致性测试：修复后 {len(consistent_segments)} 个片段")
    
    # 测试数据集统计
    stats_dataset(ContrastiveSequenceDataset(consistent_segments), "测试数据集", config.pad_token)
    
    # 测试双正样本视图差异分析
    analyze_dual_positive_views_diff(consistent_segments, "测试数据")
    
    print("✅ 工具函数测试完成")