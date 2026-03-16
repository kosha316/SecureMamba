"""
Nucleotide Transformer v3 + Mamba训练器
完全可训练的NTv3模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import time
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 导入自定义模块
from config import config
from model import create_nucleotide_mamba_model
from loss import TripletContrastiveLoss, HardTripletLoss
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from torch.amp import autocast, GradScaler
from realtime_dataset import create_nucleotide_dataloader


class NucleotideMambaTrainer:
    """Nucleotide Transformer v3 + Mamba训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        print(f"🚀 初始化完全可训练的Nucleotide Transformer v3 + Mamba训练器")
        print(f"  设备: {self.device}")
        print(f"  输出目录: {config.output_dir}")
        print(f"  批次大小: {config.batch_size} ⚠️ (NTv3内存需求大)")
        print(f"  使用Nucleotide Transformer v3: {getattr(config, 'use_nucleotide_transformer', False)}")
    
    def prepare_data(self, variants_data):
        """准备数据 - 使用Nucleotide Transformer v3"""
        print(f"\n📊 准备Nucleotide Transformer v3训练数据...")
        
        # 计算最大序列长度（DNA碱基数量）
        max_seq_len = getattr(self.config, 'max_seq_len', 512)
        
        # 检查数据
        if not variants_data:
            raise ValueError("没有提供变体数据")
        
        print(f"  总样本数: {len(variants_data):,}")
        
        # 分离正负样本用于分层抽样
        labels = []
        for item in variants_data:
            labels.append(item.get('original_label', 0.0))
        
        # 划分训练/验证集
        train_indices, val_indices = train_test_split(
            list(range(len(variants_data))),
            test_size=0.2,  # 20%验证集
            random_state=42,
            stratify=labels
        )
        
        # 创建子集
        train_data = [variants_data[i] for i in train_indices]
        val_data = [variants_data[i] for i in val_indices]
        
        print(f"📊 数据划分:")
        print(f"  训练集: {len(train_data):,} 个样本")
        print(f"  验证集: {len(val_data):,} 个样本")
        
        # 创建数据加载器
        train_loader = create_nucleotide_dataloader(
            segments=train_data,
            batch_size=self.config.batch_size,
            max_seq_len=max_seq_len,
            shuffle=True,
            num_workers=min(4, os.cpu_count()),
            use_cache=config.use_caching  # 实时处理
        )
        
        val_loader = create_nucleotide_dataloader(
            segments=val_data,
            batch_size=self.config.batch_size,
            max_seq_len=max_seq_len,
            shuffle=False,
            num_workers=min(4, os.cpu_count()),
            use_cache=config.use_caching  # 实时处理
        )
        
        return train_loader, val_loader
    
    def create_model(self):
        """创建完全可训练的Nucleotide Transformer v3 + Mamba模型"""
        print(f"\n🧠 创建完全可训练的Nucleotide Transformer v3 + Mamba模型...")
        
        # 从config获取模型参数
        model_kwargs = {
            'transformer_model_repo': getattr(self.config, 'transformer_model_repo', 'InstaDeepAI/NTv3_8M_pre'),
            'embedding_dim': getattr(self.config, 'embedding_dim', 256),
            'd_model': getattr(self.config, 'd_model', 256),
            'n_layer': getattr(self.config, 'n_layer', 4),
            'projection_dim': getattr(self.config, 'projection_dim', 128),
            'num_classes': getattr(self.config, 'num_classes', 1),
            'variant_specialization_weight': getattr(self.config, 'variant_specialization_weight', 0.2),
            'block_type': getattr(self.config, 'block_type', 'dual_path'),
            'dropout_rate': getattr(self.config, 'dropout_rate', 0.1),
            'use_path_selection': getattr(self.config, 'use_path_selection', True),
            'path_selection_weight': getattr(self.config, 'path_selection_weight', 0.1),
            'max_seq_len': getattr(self.config, 'max_seq_len', 512),
            'freeze_transformer': getattr(self.config, 'freeze_transformer', True),  # 完全可训练
            'use_caching': getattr(self.config, 'use_caching', True),
            'trust_remote_code': getattr(self.config, 'trust_remote_code', True),
            'use_local_global_attn': getattr(self.config, 'use_local_global_attn', True),
            'use_global_invariance': getattr(self.config, 'use_global_invariance', True),
            'attn_num_heads': getattr(self.config, 'attn_num_heads', 4),
            # 'num_segments': getattr(self.config, 'num_segments', 8),
            'use_flash_attention': getattr(self.config, 'use_flash_attention', True),
            'enable_tf32': getattr(self.config, 'enable_tf32', True),
            'compile_model': getattr(self.config, 'compile_model', False),
            'device': str(self.device)
        }
        
        model = create_nucleotide_mamba_model(**model_kwargs)
        
        # 打印模型统计
        model_summary = model.get_model_summary()
        print(f"✅ 模型创建完成")
        print(f"  总参数: {model_summary['total_parameters']:,}")
        print(f"  可训练参数: {model_summary['trainable_parameters']:,} ⚠️ (完全可训练)")
        print(f"  NTv3参数: {model_summary['ntv3_parameters']:,}")
        print(f"  NTv3可训练: {model_summary['ntv3_trainable']:,}")
        print(f"  Mamba参数: {model_summary['mamba_parameters']:,}")
        print(f"  参数大小: {model_summary['parameter_mb']:.2f} MB")
        print(f"  模型维度: {model_summary['d_model']}")
        print(f"  模型架构: {model_summary['model_architecture']}")
        
        return model
    
    def create_loss_functions(self):
        """创建损失函数"""
        # 分类损失
        class_criterion = nn.BCEWithLogitsLoss()
        
        # 对比损失
        if getattr(self.config, 'use_hard_triplet', False):
            contrastive_criterion = HardTripletLoss(margin=getattr(self.config, 'triplet_margin', 1.0))
        else:
            contrastive_criterion = TripletContrastiveLoss(
                margin=getattr(self.config, 'triplet_margin', 1.0),
                temperature=getattr(self.config, 'triplet_temperature', 0.1)
            )
        
        return class_criterion, contrastive_criterion
    
    def create_optimizer(self, model):
        """创建优化器 - NTv3完全可训练，需要优化所有参数"""
        # 分离NTv3参数和其他参数，可以设置不同的学习率
        ntv3_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'nucleotide_embedding' in name and 'model' in name:
                    # NTv3模型参数 - 使用更小的学习率
                    ntv3_params.append(param)
                else:
                    # 其他参数（Mamba、分类头等）
                    other_params.append(param)
        
        if not ntv3_params and not other_params:
            raise ValueError("没有可训练的参数！请检查模型配置。")
        
        # 为不同参数组设置不同的学习率
        optimizer_grouped_parameters = []
        
        # NTv3参数：较小的学习率
        if ntv3_params:
            optimizer_grouped_parameters.append({
                "params": ntv3_params,
                "lr": self.config.learning_rate * self.config.ntv3_learning_rate_multiplier,  # 10%的基础学习率
                "weight_decay": self.config.weight_decay,
            })
        
        # 其他参数：正常学习率
        if other_params:
            optimizer_grouped_parameters.append({
                "params": other_params,
                "lr": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
            })
        
        print(f"  优化器参数组:")
        print(f"    NTv3参数: {len(ntv3_params)} 组，学习率: {self.config.learning_rate * config.ntv3_learning_rate_multiplier}")
        print(f"    其他参数: {len(other_params)} 组，学习率: {self.config.learning_rate}")
        
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.onecycle_max_lr,
            epochs=self.config.num_epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            pct_start=self.config.onecycle_pct_start,
            anneal_strategy='cos'
        )
        
        return optimizer, scheduler
    
    # 以下方法保持不变...
    def calculate_metrics(self, labels, predictions, probabilities):
        """计算评估指标"""
        if len(labels) == 0:
            return {
                'acc': 0.0,
                'f1': 0.0,
                'auc': 0.0,
                'mcc': 0.0,
                'samples': 0
            }
        
        try:
            labels_np = np.array(labels).flatten()
            predictions_np = np.array(predictions).flatten()
            probabilities_np = np.array(probabilities).flatten()
            
            if len(labels_np) != len(predictions_np) or len(labels_np) != len(probabilities_np):
                min_len = min(len(labels_np), len(predictions_np), len(probabilities_np))
                labels_np = labels_np[:min_len]
                predictions_np = predictions_np[:min_len]
                probabilities_np = probabilities_np[:min_len]
            
            acc = accuracy_score(labels_np, predictions_np)
            f1 = f1_score(labels_np, predictions_np, average="binary", zero_division=0)
            
            unique_labels = np.unique(labels_np)
            if len(unique_labels) > 1:
                try:
                    auc = roc_auc_score(labels_np, probabilities_np)
                except Exception:
                    auc = 0.0
            else:
                auc = 0.0
                
            if len(unique_labels) > 1:
                try:
                    mcc = matthews_corrcoef(labels_np, predictions_np)
                except Exception:
                    mcc = 0.0
            else:
                mcc = 0.0
            
            return {
                'acc': float(acc),
                'f1': float(f1),
                'auc': float(auc),
                'mcc': float(mcc),
                'samples': len(labels_np)
            }
            
        except Exception as e:
            print(f"❌ 计算指标失败: {str(e)}")
            return {
                'acc': 0.0,
                'f1': 0.0,
                'auc': 0.0,
                'mcc': 0.0,
                'samples': 0
            }

    def train_epoch(self, model, train_loader, class_criterion, contrastive_criterion, 
                    optimizer, scheduler, scaler, epoch):
        """训练一个epoch"""
        model.train()
        
        epoch_loss = 0.0
        epoch_class_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_path_loss = 0.0
        
        # 指标收集
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        original_labels = []
        original_predictions = []
        original_probabilities = []
        
        semantic_labels = []
        semantic_predictions = []
        semantic_probabilities = []
        
        confusion_labels = []
        confusion_predictions = []
        confusion_probabilities = []
        
        # 训练前清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 使用tqdm显示进度
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # 移动到设备
                batch = self._move_batch_to_device(batch)
                
                batch_labels = batch['labels']
                batch_size = batch_labels.size(0)
                positive_mask = (batch_labels == 1)
                positive_indices = torch.where(positive_mask)[0]
                
                # 准备模型输入
                original_sequences = batch['sequences']
                semantic_sequences = batch.get('semantic_sequences', None)
                confusion_sequences = batch.get('confusion_sequences', None)
                random_mutation_sequences = batch.get('random_mutation_sequences', None)
                
                # 混合精度训练
                with autocast(device_type='cuda', enabled=True):
                    # 前向传播
                    outputs = model(
                        original_sequences=original_sequences,
                        labels=batch['labels'],
                        semantic_sequences=semantic_sequences,
                        confusion_sequences=confusion_sequences,
                        random_mutation_sequences=random_mutation_sequences,
                        training_mode=True
                    )
                    
                    # 解包输出
                    orig_class_pred, semantic_class_pred, confusion_class_pred, total_aux_loss, orig_contrastive_feat, all_contrastive_feat = outputs
                    
                    # ===== 计算分类损失 =====
                    if orig_class_pred.dim() == 2 and orig_class_pred.size(1) == 1:
                        orig_class_pred_flat = orig_class_pred.squeeze(1)
                    else:
                        orig_class_pred_flat = orig_class_pred
                    
                    orig_class_loss = class_criterion(orig_class_pred_flat, batch['labels'])
                    total_class_loss = orig_class_loss * 0.8
                    
                    # 变体分类损失：只对正样本计算
                    if semantic_class_pred is not None and len(positive_indices) > 0:
                        semantic_class_pred_positive = semantic_class_pred[positive_indices]
                        semantic_labels_positive = torch.ones_like(batch['labels'][positive_indices])
                        
                        if semantic_class_pred_positive.dim() == 2 and semantic_class_pred_positive.size(1) == 1:
                            semantic_class_pred_positive = semantic_class_pred_positive.squeeze(1)
                        
                        if semantic_class_pred_positive.size(0) > 0:
                            semantic_class_loss = class_criterion(
                                semantic_class_pred_positive, 
                                semantic_labels_positive
                            )
                            total_class_loss += semantic_class_loss * 0.1
                    
                    if confusion_class_pred is not None and len(positive_indices) > 0:
                        confusion_class_pred_positive = confusion_class_pred[positive_indices]
                        confusion_labels_positive = torch.ones_like(batch['labels'][positive_indices])
                        
                        if confusion_class_pred_positive.dim() == 2 and confusion_class_pred_positive.size(1) == 1:
                            confusion_class_pred_positive = confusion_class_pred_positive.squeeze(1)
                        
                        if confusion_class_pred_positive.size(0) > 0:
                            confusion_class_loss = class_criterion(
                                confusion_class_pred_positive, 
                                confusion_labels_positive
                            )
                            total_class_loss += confusion_class_loss * 0.1
                    
                    # ===== 计算对比损失 =====
                    contrastive_loss = torch.tensor(0.0, device=self.device)
                    
                    if (orig_contrastive_feat is not None and 
                        all_contrastive_feat is not None and 
                        len(positive_indices) > 0):
                        
                        has_semantic = semantic_sequences is not None and any(len(s) > 0 for s in semantic_sequences)
                        has_confusion = confusion_sequences is not None and any(len(s) > 0 for s in confusion_sequences)
                        has_random_mutation = random_mutation_sequences is not None and any(len(s) > 0 for s in random_mutation_sequences)
                        
                        if has_semantic and has_confusion and has_random_mutation:
                            orig_contrastive_feat_positive = orig_contrastive_feat[positive_indices]
                            
                            if all_contrastive_feat is not None:
                                split_size = batch_size
                                split_features = torch.split(all_contrastive_feat, split_size, dim=0)
                                
                                if len(split_features) >= 4:
                                    semantic_contrastive_feat_all = split_features[1]
                                    confusion_contrastive_feat_all = split_features[2]
                                    random_mutation_contrastive_feat_all = split_features[3]
                                    
                                    semantic_contrastive_feat_positive = semantic_contrastive_feat_all[positive_indices]
                                    confusion_contrastive_feat_positive = confusion_contrastive_feat_all[positive_indices]
                                    random_mutation_contrastive_feat_positive = random_mutation_contrastive_feat_all[positive_indices]
                                    
                                    if (orig_contrastive_feat_positive.size(0) > 0 and 
                                        semantic_contrastive_feat_positive.size(0) > 0 and
                                        confusion_contrastive_feat_positive.size(0) > 0 and
                                        random_mutation_contrastive_feat_positive.size(0) > 0):
                                        
                                        contrastive_loss = contrastive_criterion(
                                            anchor_feat=orig_contrastive_feat_positive.clone(),
                                            positive_feats=[
                                                semantic_contrastive_feat_positive.clone(),
                                                confusion_contrastive_feat_positive.clone()
                                            ],
                                            negative_feat=random_mutation_contrastive_feat_positive.clone()
                                        )
                    
                    # ===== 总损失 =====
                    total_loss = (total_class_loss + contrastive_loss * 0.3 + total_aux_loss * 0.1) / 1.4
                
                # 反向传播
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                
                # 梯度裁剪
                if self.config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # 累计损失
                batch_size_actual = batch['labels'].size(0)
                epoch_loss += total_loss.item() * batch_size_actual
                epoch_class_loss += total_class_loss.item() * batch_size_actual
                epoch_contrastive_loss += contrastive_loss.item() * batch_size_actual
                
                if isinstance(total_aux_loss, torch.Tensor):
                    epoch_path_loss += total_aux_loss.item() * batch_size_actual
                else:
                    epoch_path_loss += total_aux_loss * batch_size_actual
                
                # 指标收集
                batch_labels_cpu = batch['labels'].cpu().detach().numpy()
                pos_indices_np = positive_indices.cpu().numpy() if len(positive_indices) > 0 else np.array([])
                
                # 1. 原始序列预测
                orig_probs = torch.sigmoid(orig_class_pred).cpu().detach().numpy()
                if orig_probs.ndim == 2 and orig_probs.shape[1] == 1:
                    orig_probs = orig_probs.flatten()
                
                orig_preds = (orig_probs > 0.5).astype(int)
                
                # 收集原始序列指标
                original_labels.extend(batch_labels_cpu.tolist())
                original_predictions.extend(orig_preds.tolist())
                original_probabilities.extend(orig_probs.tolist())
                
                # 2. 总体指标
                all_labels.extend(batch_labels_cpu.tolist())
                all_predictions.extend(orig_preds.tolist())
                all_probabilities.extend(orig_probs.tolist())
                
                # 3. 变体指标（只对正样本）
                if len(positive_indices) > 0:
                    if semantic_class_pred is not None:
                        semantic_probs = torch.sigmoid(semantic_class_pred[positive_indices]).cpu().detach().numpy()
                        if semantic_probs.ndim == 2 and semantic_probs.shape[1] == 1:
                            semantic_probs = semantic_probs.flatten()
                        
                        semantic_preds = (semantic_probs > 0.5).astype(int)
                        semantic_labels_batch = np.ones_like(batch_labels_cpu[pos_indices_np])
                        
                        all_labels.extend(semantic_labels_batch.tolist())
                        all_predictions.extend(semantic_preds.tolist())
                        all_probabilities.extend(semantic_probs.tolist())
                        
                        semantic_labels.extend(semantic_labels_batch.tolist())
                        semantic_predictions.extend(semantic_preds.tolist())
                        semantic_probabilities.extend(semantic_probs.tolist())
                    
                    if confusion_class_pred is not None:
                        confusion_probs = torch.sigmoid(confusion_class_pred[positive_indices]).cpu().detach().numpy()
                        if confusion_probs.ndim == 2 and confusion_probs.shape[1] == 1:
                            confusion_probs = confusion_probs.flatten()
                        
                        confusion_preds = (confusion_probs > 0.5).astype(int)
                        confusion_labels_batch = np.ones_like(batch_labels_cpu[pos_indices_np])
                        
                        all_labels.extend(confusion_labels_batch.tolist())
                        all_predictions.extend(confusion_preds.tolist())
                        all_probabilities.extend(confusion_probs.tolist())
                        
                        confusion_labels.extend(confusion_labels_batch.tolist())
                        confusion_predictions.extend(confusion_preds.tolist())
                        confusion_probabilities.extend(confusion_probs.tolist())
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': total_loss.item(),
                    'class': total_class_loss.item(),
                    'contrast': contrastive_loss.item(),
                    'aux': total_aux_loss.item() if isinstance(total_aux_loss, torch.Tensor) else total_aux_loss,
                    'pos': len(positive_indices)
                })
                
                # 定期清理缓存
                if batch_idx % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 计算epoch指标
            epoch_samples = len(train_loader.dataset)
            avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
            avg_class_loss = epoch_class_loss / epoch_samples if epoch_samples > 0 else 0
            avg_contrastive_loss = epoch_contrastive_loss / epoch_samples if epoch_samples > 0 else 0
            avg_path_loss = epoch_path_loss / epoch_samples if epoch_samples > 0 else 0
            
            # 计算各项指标
            overall_metrics = self.calculate_metrics(all_labels, all_predictions, all_probabilities)
            original_metrics = self.calculate_metrics(original_labels, original_predictions, original_probabilities)
            semantic_metrics = self.calculate_metrics(semantic_labels, semantic_predictions, semantic_probabilities)
            confusion_metrics = self.calculate_metrics(confusion_labels, confusion_predictions, confusion_probabilities)
            
            # 计算综合指标
            variant_weight = getattr(self.config, 'variant_acc_weight', 0.7)
            combined_acc = original_metrics['acc'] * (1 - variant_weight) + \
                        (semantic_metrics['acc'] + confusion_metrics['acc']) / 2 * variant_weight
            
            return {
                'loss': avg_loss,
                'class_loss': avg_class_loss,
                'contrastive_loss': avg_contrastive_loss,
                'path_loss': avg_path_loss,
                'overall': overall_metrics,
                'original': original_metrics,
                'semantic': semantic_metrics,
                'confusion': confusion_metrics,
                'combined_acc': combined_acc,
                'samples': epoch_samples
            }

    def validate(self, model, val_loader, class_criterion, contrastive_criterion):
        """验证模型"""
        model.eval()
        
        val_loss = 0.0
        val_class_loss = 0.0
        val_contrastive_loss = 0.0
        val_path_loss = 0.0
        
        # 分别存储不同的指标集合
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        original_labels = []
        original_predictions = []
        original_probabilities = []
        
        semantic_labels = []
        semantic_predictions = []
        semantic_probabilities = []
        
        confusion_labels = []
        confusion_predictions = []
        confusion_probabilities = []
        
        # 验证前清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # 修复标签格式
                if 'labels' in batch:
                    labels = batch['labels']
                    if isinstance(labels, torch.Tensor):
                        if labels.dim() == 0:
                            batch['labels'] = labels.unsqueeze(0)
                    else:
                        batch['labels'] = torch.tensor([float(labels)], dtype=torch.float32)
                
                batch = self._move_batch_to_device(batch)
                
                batch_labels = batch['labels']
                if batch_labels.dim() == 0:
                    batch_labels = batch_labels.unsqueeze(0)
                    batch['labels'] = batch_labels
                
                batch_size = batch_labels.size(0)
                positive_mask = (batch_labels == 1)
                positive_indices = torch.where(positive_mask)[0]
                
                # 准备模型输入
                original_sequences = batch['sequences']
                semantic_sequences = batch.get('semantic_sequences', None)
                confusion_sequences = batch.get('confusion_sequences', None)
                random_mutation_sequences = batch.get('random_mutation_sequences', None)
                
                # 前向传播
                outputs = model(
                    original_sequences=original_sequences,
                    labels=batch['labels'],
                    semantic_sequences=semantic_sequences,
                    confusion_sequences=confusion_sequences,
                    random_mutation_sequences=random_mutation_sequences,
                    training_mode=False
                )
                
                orig_class_pred, semantic_class_pred, confusion_class_pred, total_aux_loss, orig_contrastive_feat, all_contrastive_feat = outputs
                
                # ===== 计算损失 =====
                if orig_class_pred.dim() == 2 and orig_class_pred.size(1) == 1:
                    orig_class_pred_flat = orig_class_pred.squeeze(1)
                else:
                    orig_class_pred_flat = orig_class_pred
                
                orig_class_loss = class_criterion(orig_class_pred_flat, batch['labels'])
                total_class_loss = orig_class_loss * 0.8
                
                # 语义变体损失（只对正样本）
                if semantic_class_pred is not None and len(positive_indices) > 0:
                    semantic_class_pred_positive = semantic_class_pred[positive_indices]
                    semantic_labels_positive = batch['labels'][positive_indices]
                    
                    if semantic_class_pred_positive.dim() == 2 and semantic_class_pred_positive.size(1) == 1:
                        semantic_class_pred_positive = semantic_class_pred_positive.squeeze(1)
                    
                    semantic_class_loss = class_criterion(
                        semantic_class_pred_positive, 
                        semantic_labels_positive
                    )
                    total_class_loss += semantic_class_loss * 0.1
                
                # 混淆变体损失（只对正样本）
                if confusion_class_pred is not None and len(positive_indices) > 0:
                    confusion_class_pred_positive = confusion_class_pred[positive_indices]
                    confusion_labels_positive = batch['labels'][positive_indices]
                    
                    if confusion_class_pred_positive.dim() == 2 and confusion_class_pred_positive.size(1) == 1:
                        confusion_class_pred_positive = confusion_class_pred_positive.squeeze(1)
                    
                    confusion_class_loss = class_criterion(
                        confusion_class_pred_positive, 
                        confusion_labels_positive
                    )
                    total_class_loss += confusion_class_loss * 0.1
                
                # 对比损失
                contrastive_loss = torch.tensor(0.0, device=self.device)
                
                if len(positive_indices) > 0:
                    has_semantic = semantic_sequences is not None and any(len(s) > 0 for s in semantic_sequences)
                    has_confusion = confusion_sequences is not None and any(len(s) > 0 for s in confusion_sequences)
                    has_random_mutation = random_mutation_sequences is not None and any(len(s) > 0 for s in random_mutation_sequences)
                    
                    if has_semantic and has_confusion and has_random_mutation:
                        orig_contrastive_feat_positive = orig_contrastive_feat[positive_indices]
                        
                        batch_size_val = batch['labels'].size(0)
                        split_size = batch_size_val
                        split_features = torch.split(all_contrastive_feat, split_size, dim=0)
                        
                        if len(split_features) >= 4:
                            semantic_contrastive_feat_all = split_features[1]
                            confusion_contrastive_feat_all = split_features[2]
                            random_mutation_contrastive_feat_all = split_features[3]
                            
                            semantic_contrastive_feat_positive = semantic_contrastive_feat_all[positive_indices]
                            confusion_contrastive_feat_positive = confusion_contrastive_feat_all[positive_indices]
                            random_mutation_contrastive_feat_positive = random_mutation_contrastive_feat_all[positive_indices]
                            
                            if orig_contrastive_feat_positive.size(0) > 0:
                                contrastive_loss = contrastive_criterion(
                                    anchor_feat=orig_contrastive_feat_positive,
                                    positive_feats=[semantic_contrastive_feat_positive, confusion_contrastive_feat_positive],
                                    negative_feat=random_mutation_contrastive_feat_positive
                                )
                
                total_loss = (total_class_loss * 1.0 + contrastive_loss * 0.3 + total_aux_loss * 0.1) / 1.4
                
                # 累计损失
                batch_size_val = batch['labels'].size(0)
                val_loss += total_loss.item() * batch_size_val
                val_class_loss += total_class_loss.item() * batch_size_val
                val_contrastive_loss += contrastive_loss.item() * batch_size_val
                val_path_loss += total_aux_loss.item() * batch_size_val
                
                # ===== 指标收集 =====
                batch_labels_cpu = batch['labels'].cpu().detach().numpy()
                pos_indices_np = positive_indices.cpu().numpy() if len(positive_indices) > 0 else np.array([])
                
                # 1. 原始序列指标
                orig_probs = torch.sigmoid(orig_class_pred).cpu().detach().numpy()
                if orig_probs.ndim == 2 and orig_probs.shape[1] == 1:
                    orig_probs = orig_probs.flatten()
                
                orig_preds = (orig_probs > 0.5).astype(int)
                
                original_labels.extend(batch_labels_cpu.tolist())
                original_predictions.extend(orig_preds.tolist())
                original_probabilities.extend(orig_probs.tolist())
                
                # 2. 总体指标
                all_labels.extend(batch_labels_cpu.tolist())
                all_predictions.extend(orig_preds.tolist())
                all_probabilities.extend(orig_probs.tolist())
                
                # 3. 变体指标（只对正样本）
                if len(positive_indices) > 0:
                    if semantic_class_pred is not None:
                        semantic_probs = torch.sigmoid(semantic_class_pred[positive_indices]).cpu().detach().numpy()
                        if semantic_probs.ndim == 2 and semantic_probs.shape[1] == 1:
                            semantic_probs = semantic_probs.flatten()
                        
                        semantic_preds = (semantic_probs > 0.5).astype(int)
                        semantic_labels_batch = batch_labels_cpu[pos_indices_np]
                        
                        all_labels.extend(semantic_labels_batch.tolist())
                        all_predictions.extend(semantic_preds.tolist())
                        all_probabilities.extend(semantic_probs.tolist())
                        
                        semantic_labels.extend(semantic_labels_batch.tolist())
                        semantic_predictions.extend(semantic_preds.tolist())
                        semantic_probabilities.extend(semantic_probs.tolist())
                    
                    if confusion_class_pred is not None:
                        confusion_probs = torch.sigmoid(confusion_class_pred[positive_indices]).cpu().detach().numpy()
                        if confusion_probs.ndim == 2 and confusion_probs.shape[1] == 1:
                            confusion_probs = confusion_probs.flatten()
                        
                        confusion_preds = (confusion_probs > 0.5).astype(int)
                        confusion_labels_batch = batch_labels_cpu[pos_indices_np]
                        
                        all_labels.extend(confusion_labels_batch.tolist())
                        all_predictions.extend(confusion_preds.tolist())
                        all_probabilities.extend(confusion_probs.tolist())
                        
                        confusion_labels.extend(confusion_labels_batch.tolist())
                        confusion_predictions.extend(confusion_preds.tolist())
                        confusion_probabilities.extend(confusion_probs.tolist())
        
        # ===== 计算验证指标 =====
        val_samples = sum(1 for _ in val_loader) * self.config.batch_size if len(val_loader) > 0 else 0
        avg_loss = val_loss / val_samples if val_samples > 0 else 0
        avg_class_loss = val_class_loss / val_samples if val_samples > 0 else 0
        avg_contrastive_loss = val_contrastive_loss / val_samples if val_samples > 0 else 0
        avg_path_loss = val_path_loss / val_samples if val_samples > 0 else 0
        
        # 计算各项指标
        overall_metrics = self.calculate_metrics(all_labels, all_predictions, all_probabilities)
        original_metrics = self.calculate_metrics(original_labels, original_predictions, original_probabilities)
        semantic_metrics = self.calculate_metrics(semantic_labels, semantic_predictions, semantic_probabilities)
        confusion_metrics = self.calculate_metrics(confusion_labels, confusion_predictions, confusion_probabilities)
        
        # 打印调试信息
        print(f"\n🔍 验证集统计:")
        print(f"  原始序列样本数: {len(original_labels)}")
        print(f"  语义变体样本数: {len(semantic_labels)}")
        print(f"  混淆变体样本数: {len(confusion_labels)}")
        print(f"  总体样本数: {len(all_labels)}")
        
        # 计算综合ACC
        variant_weight = getattr(self.config, 'variant_acc_weight', 0.35)
        combined_acc = original_metrics['acc'] * (1 - variant_weight) + \
                    (semantic_metrics['acc'] + confusion_metrics['acc']) / 2 * variant_weight
        
        return {
            'loss': avg_loss,
            'class_loss': avg_class_loss,
            'contrastive_loss': avg_contrastive_loss,
            'path_loss': avg_path_loss,
            'overall': overall_metrics,
            'original': original_metrics,
            'semantic': semantic_metrics,
            'confusion': confusion_metrics,
            'combined_acc': combined_acc,
            'samples': val_samples
        }
    
    def _move_batch_to_device(self, batch):
        """将批次数据移动到设备"""
        batch_gpu = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_gpu[key] = value.to(self.device, non_blocking=True)
            else:
                batch_gpu[key] = value
        return batch_gpu
    
    def save_model(self, model, optimizer, scheduler, epoch, val_metrics, path):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config.__dict__,
            'model_config': {
                'transformer_model_repo': model.transformer_model_repo,
                'embedding_dim': model.embedding_dim,
                'd_model': model.d_model,
                'n_layer': model.n_layer,
                'block_type': model.block_type,
                'use_path_selection': model.use_path_selection,
            },
            'cache_stats': model.get_cache_stats()
        }
        
        torch.save(checkpoint, path)
        print(f"💾 模型保存到: {path}")
    
    def train(self, variants_data):
        """主训练函数"""
        print(f"\n🎯 开始完全可训练的Nucleotide Transformer v3训练流程")
        
        # 1. 准备数据
        train_loader, val_loader = self.prepare_data(variants_data)
        
        # 2. 创建模型
        model = self.create_model()
        
        # 3. 创建损失函数
        class_criterion, contrastive_criterion = self.create_loss_functions()
        
        # 4. 创建优化器
        optimizer, scheduler = self.create_optimizer(model)
        
        # 5. 混合精度训练
        scaler = GradScaler(enabled=True)
        
        # 6. 训练循环
        best_combined_acc = 0.0
        early_stop_counter = 0
        early_stopped = False
        
        train_metrics_history = []
        val_metrics_history = []
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 训练
            train_metrics = self.train_epoch(
                model, train_loader, class_criterion, contrastive_criterion,
                optimizer, scheduler, scaler, epoch
            )
            
            # 验证
            val_metrics = self.validate(model, val_loader, class_criterion, contrastive_criterion)
            
            train_metrics_history.append(train_metrics)
            val_metrics_history.append(val_metrics)
            
            epoch_time = time.time() - epoch_start_time
            current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else 0
            
            print(f"\n📊 Epoch {epoch+1}/{self.config.num_epochs} 总结:")
            print(f"  耗时: {epoch_time:.1f}s | 学习率: {current_lr:.2e}")
            print(f"  训练损失: {train_metrics['loss']:.4f} | 验证损失: {val_metrics['loss']:.4f}")
            print(f"  训练综合ACC: {train_metrics['combined_acc']:.4f} | 验证综合ACC: {val_metrics['combined_acc']:.4f}")
            
            print(f"\n  📈 总体指标:")
            print(f"    训练ACC: {train_metrics['overall']['acc']:.4f} | 验证ACC: {val_metrics['overall']['acc']:.4f}")
            print(f"    训练F1: {train_metrics['overall']['f1']:.4f} | 验证F1: {val_metrics['overall']['f1']:.4f}")
            print(f"    训练AUC: {train_metrics['overall']['auc']:.4f} | 验证AUC: {val_metrics['overall']['auc']:.4f}")
            
            print(f"\n  🧬 原始序列指标:")
            print(f"    训练ACC: {train_metrics['original']['acc']:.4f} | 验证ACC: {val_metrics['original']['acc']:.4f}")
            
            print(f"\n  🔍 特异性指标:")
            print(f"    语义变体ACC: {train_metrics['semantic']['acc']:.4f} | {val_metrics['semantic']['acc']:.4f}")
            print(f"    混淆变体ACC: {train_metrics['confusion']['acc']:.4f} | {val_metrics['confusion']['acc']:.4f}")
            
            # 打印缓存统计
            cache_stats = model.get_cache_stats()
            print(f"\n  💾 缓存统计: 命中率: {cache_stats['hit_rate']:.2%} | 缓存大小: {cache_stats['cache_size']:,}")
            
            # 保存最佳模型
            if val_metrics['combined_acc'] > best_combined_acc:
                best_combined_acc = val_metrics['combined_acc']
                early_stop_counter = 0
                
                best_model_path = os.path.join(self.config.output_dir, "best_nucleotide_v3_mamba_model.pth")
                self.save_model(model, optimizer, scheduler, epoch + 1, val_metrics, best_model_path)
                
                print(f"\n✅ 新的最佳综合ACC: {best_combined_acc:.4f}")
            else:
                early_stop_counter += 1
                print(f"\n⚠️  综合ACC未改善，早停计数器: {early_stop_counter}/{self.config.early_stop_patience}")
            
            if early_stop_counter >= self.config.early_stop_patience:
                print(f"\n🛑 触发早停，停止训练")
                early_stopped = True
                break
        
        # 保存最终模型
        final_model_path = os.path.join(self.config.output_dir, "final_nucleotide_v3_mamba_model.pth")
        self.save_model(model, optimizer, scheduler, self.config.num_epochs, 
                       val_metrics_history[-1] if val_metrics_history else {}, final_model_path)
        
        # 保存训练日志
        log_data = {
            'config': self.config.__dict__,
            'train_metrics': train_metrics_history,
            'val_metrics': val_metrics_history,
            'best_combined_acc': best_combined_acc,
            'early_stopped': early_stopped,
            'total_epochs': len(train_metrics_history),
            'realtime_processing': True,
            'cache_stats_final': model.get_cache_stats(),
            'transformer_config': {
                'model_repo': model.transformer_model_repo,
                'freeze_transformer': model.freeze_transformer,
                'trainable': 'fully_trainable'
            }
        }
        
        log_path = os.path.join(self.config.output_dir, "nucleotide_v3_training_log.json")
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        print(f"\n📄 训练日志保存到: {log_path}")
        
        # 7. 调用可视化模块（可选）
        try:
            from training_visualizer import integrate_visualization
            print(f"\n🎨 生成训练可视化图表...")
            integrate_visualization(self, train_metrics_history, val_metrics_history, self.config, early_stopped)
            print(f"✅ 可视化图表生成完成")
        except ImportError as e:
            print(f"⚠️  无法导入可视化模块: {str(e)}")
        except Exception as e:
            print(f"⚠️  可视化生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 清空缓存
        model.clear_cache()
        
        return model, train_metrics_history, val_metrics_history


def train_nucleotide(variants_data):
    """Nucleotide Transformer v3训练主函数"""
    trainer = NucleotideMambaTrainer(config)
    
    try:
        model, train_history, val_history = trainer.train(variants_data)
        print(f"\n🎉 Nucleotide Transformer v3训练完成!")
        
        if val_history:
            final_val = val_history[-1]
            print(f"\n📈 最终验证集指标:")
            print(f"  综合ACC: {final_val['combined_acc']:.4f}")
            print(f"  总体ACC: {final_val['overall']['acc']:.4f}")
            print(f"  总体F1: {final_val['overall']['f1']:.4f}")
            print(f"  总体AUC: {final_val['overall']['auc']:.4f}")
            print(f"  总体MCC: {final_val['overall']['mcc']:.4f}")
            print(f"  原始ACC: {final_val['original']['acc']:.4f}")
            print(f"  语义ACC: {final_val['semantic']['acc']:.4f}")
            print(f"  混淆ACC: {final_val['confusion']['acc']:.4f}")
        
        return model
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None