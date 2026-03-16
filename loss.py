import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

# 在 loss.py 中修复 TripletContrastiveLoss：

class TripletContrastiveLoss(nn.Module):
    """
    修复的三元组对比损失
    """
    def __init__(self, margin=1.0, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, 
                anchor_feat: torch.Tensor,      # 原始序列特征 [batch_size, projection_dim]
                positive_feats: List[torch.Tensor],  # 正样本特征列表
                negative_feat: torch.Tensor     # 负样本特征 [batch_size, projection_dim]
               ) -> torch.Tensor:
        """
        计算三元组对比损失
        """
        batch_size = anchor_feat.shape[0]
        total_loss = 0.0
        valid_pairs = 0
        
        # 🔥 修复：检查维度一致性
        if anchor_feat.dim() != 2 or negative_feat.dim() != 2:
            return torch.tensor(0.0, device=anchor_feat.device)
        
        # 归一化特征
        anchor_feat = F.normalize(anchor_feat, p=2, dim=1)
        negative_feat = F.normalize(negative_feat, p=2, dim=1)
        
        # 计算锚点与负样本的相似度
        neg_similarity = F.cosine_similarity(anchor_feat, negative_feat)
        
        for i, positive_feat in enumerate(positive_feats):
            if positive_feat is None:
                continue
                
            # 🔥 修复：检查正样本特征维度
            if positive_feat.dim() != 2 or positive_feat.shape != anchor_feat.shape:
                continue
                
            # 归一化正样本特征
            positive_feat = F.normalize(positive_feat, p=2, dim=1)
            
            # 计算锚点与正样本的相似度
            pos_similarity = F.cosine_similarity(anchor_feat, positive_feat)
            
            # 计算三元组损失
            try:
                triplet_loss = F.triplet_margin_loss(
                    anchor_feat, positive_feat, negative_feat,
                    margin=self.margin, p=2, eps=1e-7
                )
                
                # 计算InfoNCE风格的对比损失
                pos_exp = torch.exp(pos_similarity / self.temperature)
                neg_exp = torch.exp(neg_similarity / self.temperature)
                
                contrastive_loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
                
                # 组合两种损失
                combined_loss = 0.7 * triplet_loss + 0.3 * contrastive_loss
                
                total_loss += combined_loss
                valid_pairs += 1
                
            except Exception as e:
                print(f"⚠️ 三元组损失计算异常：{str(e)}")
                continue
        
        if valid_pairs == 0:
            return torch.tensor(0.0, device=anchor_feat.device)
        
        return total_loss / valid_pairs

class HardTripletLoss(nn.Module):
    """
    困难三元组损失：选择最难的正负样本对
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor_feat: torch.Tensor, 
                positive_feats: List[torch.Tensor],
                negative_feat: torch.Tensor) -> torch.Tensor:
        """
        计算困难三元组损失
        
        策略：选择最难的正样本（与锚点相似度最低）和最难的负样本（与锚点相似度最高）
        """
        batch_size = anchor_feat.shape[0]
        
        # 归一化特征
        anchor_feat = F.normalize(anchor_feat, p=2, dim=1)
        negative_feat = F.normalize(negative_feat, p=2, dim=1)
        
        # 找到最难的正样本（相似度最低）
        min_pos_similarity = float('inf')
        hardest_positive = None
        
        for positive_feat in positive_feats:
            if positive_feat is None:
                continue
                
            positive_feat = F.normalize(positive_feat, p=2, dim=1)
            pos_similarity = F.cosine_similarity(anchor_feat, positive_feat).mean()
            
            if pos_similarity < min_pos_similarity:
                min_pos_similarity = pos_similarity
                hardest_positive = positive_feat
        
        if hardest_positive is None:
            return torch.tensor(0.0, device=anchor_feat.device)
        
        # 计算与负样本的相似度
        neg_similarity = F.cosine_similarity(anchor_feat, negative_feat)
        
        # 计算困难三元组损失
        hardest_loss = F.triplet_margin_loss(
            anchor_feat, hardest_positive, negative_feat,
            margin=self.margin, p=2, eps=1e-7
        )
        
        return hardest_loss


class VariantConsistencyLoss(nn.Module):
    """
    变体一致性损失
    确保变体分类预测与原始序列预测一致
    """
    def __init__(self, consistency_type: str = "kl", temperature: float = 1.0):
        super().__init__()
        self.consistency_type = consistency_type
        self.temperature = temperature
        
        if consistency_type == "kl":
            self.criterion = nn.KLDivLoss(reduction="batchmean")
        elif consistency_type == "mse":
            self.criterion = nn.MSELoss()
        elif consistency_type == "js":  # Jensen-Shannon divergence
            self.criterion = self.js_divergence
        else:
            raise ValueError(f"不支持的consistency_type: {consistency_type}")

    def js_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """计算Jensen-Shannon散度"""
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div(F.log_softmax(p, dim=1), m, reduction='batchmean') + 
                     F.kl_div(F.log_softmax(q, dim=1), m, reduction='batchmean'))

    def forward(self, orig_pred: torch.Tensor, variant_pred: torch.Tensor) -> torch.Tensor:
        """
        计算变体一致性损失
        
        Args:
            orig_pred: 原始序列预测 [batch_size, num_classes]
            variant_pred: 变体预测 [batch_size, num_classes]
            
        Returns:
            一致性损失
        """
        if self.consistency_type == "kl":
            # 使用KL散度
            orig_probs = F.log_softmax(orig_pred / self.temperature, dim=1)
            variant_probs = F.softmax(variant_pred / self.temperature, dim=1)
            loss = self.criterion(orig_probs, variant_probs)
            
        elif self.consistency_type == "mse":
            # 使用均方误差
            orig_probs = F.softmax(orig_pred / self.temperature, dim=1)
            variant_probs = F.softmax(variant_pred / self.temperature, dim=1)
            loss = self.criterion(orig_probs, variant_probs)
            
        elif self.consistency_type == "js":
            # 使用Jensen-Shannon散度
            orig_probs = F.softmax(orig_pred / self.temperature, dim=1)
            variant_probs = F.softmax(variant_pred / self.temperature, dim=1)
            loss = self.criterion(orig_probs, variant_probs)
            
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss用于处理类别不平衡
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        Args:
            inputs: 模型预测 [batch_size, num_classes] 或 [batch_size]
            targets: 真实标签 [batch_size]
            
        Returns:
            Focal Loss
        """
        if inputs.dim() > 1:
            inputs = inputs.squeeze()
            
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 防止数值不稳定
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# 测试代码
if __name__ == "__main__":
    # 测试必要的损失函数
    print("🧪 测试必要的损失函数...")
    
    # 模拟数据
    batch_size, proj_dim = 4, 128
    orig_feat = torch.randn(batch_size, proj_dim)
    semantic_feat = torch.randn(batch_size, proj_dim)
    confusion_feat = torch.randn(batch_size, proj_dim)
    negative_feat = torch.randn(batch_size, proj_dim)
    
    # 测试三元组对比损失
    triplet_loss = TripletContrastiveLoss()
    loss1 = triplet_loss(orig_feat, [semantic_feat, confusion_feat], negative_feat)
    print(f"三元组对比损失: {loss1.item():.4f}")
    
    # 测试困难三元组损失
    hard_triplet_loss = HardTripletLoss()
    loss2 = hard_triplet_loss(orig_feat, [semantic_feat, confusion_feat], negative_feat)
    print(f"困难三元组损失: {loss2.item():.4f}")
    
    # 测试变体一致性损失
    orig_pred = torch.randn(batch_size, 1)
    variant_pred = torch.randn(batch_size, 1)
    consistency_loss = VariantConsistencyLoss()
    loss3 = consistency_loss(orig_pred, variant_pred)
    print(f"变体一致性损失: {loss3.item():.4f}")
    
    # 测试Focal Loss
    labels = torch.randint(0, 2, (batch_size,)).float()
    focal_loss = FocalLoss()
    loss4 = focal_loss(orig_pred.squeeze(), labels)
    print(f"Focal Loss: {loss4.item():.4f}")
    
    print("✅ 必要的损失函数测试通过！")