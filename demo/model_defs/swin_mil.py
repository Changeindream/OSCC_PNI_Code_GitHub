# -*- coding: utf-8 -*-
"""
# @file name  : swin_mil.py
# @author     : Based on Swin Transformer for MIL
# @date       : 2024
# @brief      : Swin Transformer-based Multiple Instance Learning Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math
import sys
import os
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 尝试导入本地Swin Transformer实现
try:
    # 添加当前目录到路径以确保能找到models包
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        
    from models.swin_transformer import swin_base_patch4_window7_224
    HAS_LOCAL_SWIN = True
except ImportError:
    print("警告: 未能导入本地Swin Transformer实现，将使用TIMM版本")
    HAS_LOCAL_SWIN = False


class SwinMILAttention(nn.Module):
    """
    Swin Transformer MIL注意力机制
    结合了Swin Transformer的内部注意力和MIL的实例级注意力
    """
    
    def __init__(self, feature_dim: int, attention_dim: int = 128, num_heads: int = 8):
        super(SwinMILAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"
        
        # 多头注意力机制
        self.query_projection = nn.Linear(feature_dim, attention_dim)
        self.key_projection = nn.Linear(feature_dim, attention_dim)
        self.value_projection = nn.Linear(feature_dim, attention_dim)
        
        # 输出投影
        self.output_projection = nn.Linear(attention_dim, feature_dim)
        
        # 注意力权重计算
        self.attention_weights = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, features: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: [batch_size, num_instances, feature_dim]
            return_attention: 是否返回注意力权重
            
        Returns:
            aggregated_features: [batch_size, feature_dim]
            attention_weights: [batch_size, num_instances] (如果return_attention=True)
        """
        batch_size, num_instances, feature_dim = features.shape
        
        # 计算多头注意力
        Q = self.query_projection(features).view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_projection(features).view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_projection(features).view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = (Q @ K.transpose(-2, -1)) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 应用注意力
        context = (attention_probs @ V).transpose(1, 2).contiguous().view(batch_size, num_instances, self.attention_dim)
        
        # 输出投影
        multi_head_output = self.output_projection(context)
        
        # 残差连接和层归一化
        multi_head_output = self.layer_norm(multi_head_output + features)
        
        # 计算实例级注意力权重
        instance_attention_weights = self.attention_weights(multi_head_output)  # [batch_size, num_instances, 1]
        instance_attention_weights = F.softmax(instance_attention_weights, dim=1)  # [batch_size, num_instances, 1]
        
        # 门控机制
        gate_weights = self.gate(multi_head_output)  # [batch_size, num_instances, attention_dim]
        gated_features = multi_head_output * gate_weights
        
        # 加权聚合
        aggregated_features = torch.sum(gated_features * instance_attention_weights, dim=1)  # [batch_size, feature_dim]
        
        if return_attention:
            return aggregated_features, instance_attention_weights.squeeze(-1)
        else:
            return aggregated_features


class SwinMILModel(nn.Module):
    """
    基于Swin Transformer的多实例学习模型
    """
    
    def __init__(self, 
                 model_name: str = 'swin_base_patch4_window7_224',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 attention_dim: int = 128,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 use_gated: bool = True):
        super(SwinMILModel, self).__init__()
        
        self.num_classes = num_classes
        self.use_gated = use_gated
        
        # 使用本地Swin Transformer实现
        if HAS_LOCAL_SWIN and model_name == 'swin_base_patch4_window7_224':
            # 创建本地Swin Transformer模型，不包含分类头
            self.backbone = swin_base_patch4_window7_224(pretrained=False, num_classes=0)
            # 获取特征维度 (Swin-Base的embed_dim=128, 最后一层的特征维度是embed_dim*8=1024)
            self.feature_dim = 1024  # Swin-Base的最终特征维度
        else:
            # 如果不是支持的本地模型，回退到TIMM
            if HAS_LOCAL_SWIN and model_name != 'swin_base_patch4_window7_224':
                print(f"警告: 模型 {model_name} 不支持本地实现，使用TIMM版本")
            
            # 加载预训练的Swin Transformer模型
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.feature_dim = self.backbone.num_features
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.feature_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # MIL注意力机制
        self.mil_attention = SwinMILAttention(
            feature_dim=attention_dim,
            attention_dim=attention_dim,
            num_heads=num_heads
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(attention_dim // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, bag: torch.Tensor, return_attention: bool = False, return_features: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            bag: 输入bag，shape为[num_instances, channels, height, width]
            return_attention: 是否返回注意力权重
            return_features: 是否返回特征
            
        Returns:
            logits: [num_classes]
            attention_weights: [num_instances] (如果return_attention=True)
            features: [num_instances, feature_dim] (如果return_features=True)
        """
        # 提取每个实例的特征
        num_instances = bag.size(0)
        instance_features = []
        
        for i in range(num_instances):
            # 通过Swin Transformer提取特征
            feature = self.backbone(bag[i:i+1])  # [1, feature_dim]
            instance_features.append(feature)
        
        # 拼接所有实例特征
        instance_features = torch.cat(instance_features, dim=0)  # [num_instances, feature_dim]
        
        # 特征变换
        transformed_features = self.feature_extractor(instance_features)  # [num_instances, attention_dim]
        
        # 添加batch维度以适应MIL注意力机制
        transformed_features = transformed_features.unsqueeze(0)  # [1, num_instances, attention_dim]
        
        # MIL注意力聚合
        if return_attention:
            aggregated_features, attention_weights = self.mil_attention(transformed_features, return_attention=True)
        else:
            aggregated_features = self.mil_attention(transformed_features, return_attention=False)
        
        # 分类
        logits = self.classifier(aggregated_features.squeeze(0))  # [num_classes]
        
        # 返回结果
        result = [logits]
        if return_attention:
            result.append(attention_weights.squeeze(0))
        if return_features:
            result.append(transformed_features.squeeze(0))
        
        return result[0] if len(result) == 1 else tuple(result)
    
    def get_internal_attention_weights(self, bag: torch.Tensor) -> List[torch.Tensor]:
        """
        获取Swin Transformer内部各层的注意力权重
        
        Args:
            bag: 输入bag
            
        Returns:
            attention_weights: 各层注意力权重的列表
        """
        attention_weights = []
        
        # 注册钩子函数来获取注意力权重
        def hook_fn(module, input, output):
            if hasattr(module, 'attn') and hasattr(module.attn, 'attention_weights'):
                attention_weights.append(module.attn.attention_weights)
        
        # 为每个Swin Transformer层注册钩子
        hooks = []
        # 注意：这里假设backbone有layers属性，这对于本地Swin实现是成立的
        # 对于TIMM模型，可能需要调整
        if hasattr(self.backbone, 'layers'):
            for layer in self.backbone.layers:
                if hasattr(layer, 'blocks'):
                    for block in layer.blocks:
                        hooks.append(block.register_forward_hook(hook_fn))
        
        # 前向传播
        with torch.no_grad():
            _ = self.forward(bag)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def analyze_patch_attention(self, bag: torch.Tensor) -> dict:
        """
        分析patch级别的注意力模式
        """
        analysis = {}
        
        # 获取实例级注意力权重
        _, instance_attention = self.forward(bag, return_attention=True)
        
        # 计算注意力分布的统计信息
        analysis['instance_attention'] = {
            'weights': instance_attention.cpu().numpy(),
            'entropy': -torch.sum(instance_attention * torch.log(instance_attention + 1e-8)).item(),
            'max_weight': torch.max(instance_attention).item(),
            'min_weight': torch.min(instance_attention).item(),
            'std': torch.std(instance_attention).item()
        }
        
        # 获取内部注意力权重
        internal_attention = self.get_internal_attention_weights(bag)
        analysis['internal_attention'] = internal_attention
        
        return analysis
    
    def get_feature_diversity(self, bag: torch.Tensor) -> dict:
        """
        计算特征多样性指标
        """
        _, _, features = self.forward(bag, return_features=True)
        
        # 计算特征之间的相似性
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        # 移除对角线元素
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, 0)
        
        diversity_metrics = {
            'mean_similarity': torch.mean(similarity_matrix).item(),
            'max_similarity': torch.max(similarity_matrix).item(),
            'min_similarity': torch.min(similarity_matrix).item(),
            'diversity_index': 1 - torch.mean(similarity_matrix).item(),  # 多样性指数
            'feature_std': torch.std(features, dim=0).mean().item()
        }
        
        return diversity_metrics


class GatedSwinMILModel(SwinMILModel):
    """
    带门控机制的Swin Transformer MIL模型
    """
    
    def __init__(self, *args, **kwargs):
        super(GatedSwinMILModel, self).__init__(*args, **kwargs)
        
        # 添加门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, bag: torch.Tensor, return_attention: bool = False, return_features: bool = False) -> torch.Tensor:
        """
        前向传播（带门控机制）
        """
        # 提取每个实例的特征
        num_instances = bag.size(0)
        instance_features = []
        gate_weights = []
        
        for i in range(num_instances):
            # 通过Swin Transformer提取特征
            feature = self.backbone(bag[i:i+1])  # [1, feature_dim]
            instance_features.append(feature)
            
            # 计算门控权重
            gate_weight = self.gate_network(feature)  # [1, 1]
            gate_weights.append(gate_weight)
        
        # 拼接所有实例特征和门控权重
        instance_features = torch.cat(instance_features, dim=0)  # [num_instances, feature_dim]
        gate_weights = torch.cat(gate_weights, dim=0)  # [num_instances, 1]
        
        # 应用门控机制
        gated_features = instance_features * gate_weights
        
        # 特征变换
        transformed_features = self.feature_extractor(gated_features)  # [num_instances, attention_dim]
        
        # 添加batch维度以适应MIL注意力机制
        transformed_features = transformed_features.unsqueeze(0)  # [1, num_instances, attention_dim]
        
        # MIL注意力聚合
        if return_attention:
            aggregated_features, attention_weights = self.mil_attention(transformed_features, return_attention=True)
        else:
            aggregated_features = self.mil_attention(transformed_features, return_attention=False)
        
        # 分类
        logits = self.classifier(aggregated_features.squeeze(0))  # [num_classes]
        
        # 返回结果
        result = [logits]
        if return_attention:
            # 结合门控权重和注意力权重
            combined_weights = attention_weights.squeeze(0) * gate_weights.squeeze(-1)
            result.append(combined_weights)
        if return_features:
            result.append(transformed_features.squeeze(0))
        
        return result[0] if len(result) == 1 else tuple(result)
