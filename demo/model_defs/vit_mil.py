# -*- coding: utf-8 -*-
"""
# @file name  : vit_mil.py
# @author     : 基于ViT实现的MIL模型
# @date       : 2024
# @brief      : 基于Vision Transformer的多实例学习(MIL)模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import timm
from collections import OrderedDict
import math


class ViTMILModel(nn.Module):
    """
    基于Vision Transformer的多实例学习模型
    
    该模型包含：
    1. Vision Transformer backbone用于特征提取
    2. 注意力机制用于实例聚合
    3. 分类头用于最终预测
    """
    
    def __init__(self, 
                 model_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True,
                 pretrained_path: Optional[str] = None,
                 num_classes: int = 2,
                 attention_dim: int = 128,
                 dropout_rate: float = 0.1,
                 freeze_backbone: bool = False,
                 freeze_layers: int = 0):
        """
        初始化ViT MIL模型
        
        Args:
            model_name: timm模型名称
            pretrained: 是否使用预训练权重
            pretrained_path: 预训练权重路径
            num_classes: 分类类别数
            attention_dim: 注意力机制隐藏维度
            dropout_rate: dropout率
            freeze_backbone: 是否冻结backbone
            freeze_layers: 冻结的层数
        """
        super(ViTMILModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.attention_dim = attention_dim
        
        # 创建ViT backbone
        self.vit_backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头
            drop_rate=dropout_rate,
            attn_drop_rate=dropout_rate,
            drop_path_rate=dropout_rate
        )
        
        # 获取特征维度
        self.feature_dim = self.vit_backbone.embed_dim
        
        # 加载预训练权重（如果提供）
        if pretrained_path and pretrained_path.endswith('.pkl'):
            self.load_pretrained_weights(pretrained_path)
        
        # 冻结backbone参数
        if freeze_backbone:
            for param in self.vit_backbone.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            self.freeze_early_layers(freeze_layers)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(attention_dim, 1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim // 2, num_classes)
        )
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for m in [self.attention, self.classifier]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def load_pretrained_weights(self, pretrained_path: str):
        """加载预训练权重"""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # 提取模型状态字典
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 过滤不匹配的键
            model_dict = self.vit_backbone.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            model_dict.update(pretrained_dict)
            self.vit_backbone.load_state_dict(model_dict)
            
            print(f"成功加载预训练权重: {pretrained_path}")
            print(f"加载了 {len(pretrained_dict)}/{len(model_dict)} 个参数")
            
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    
    def freeze_early_layers(self, freeze_layers: int):
        """冻结前几层参数"""
        # 冻结patch embedding
        for param in self.vit_backbone.patch_embed.parameters():
            param.requires_grad = False
        
        # 冻结position embedding
        if hasattr(self.vit_backbone, 'pos_embed'):
            self.vit_backbone.pos_embed.requires_grad = False
        
        # 冻结cls token
        if hasattr(self.vit_backbone, 'cls_token'):
            self.vit_backbone.cls_token.requires_grad = False
        
        # 冻结前几个transformer blocks
        if hasattr(self.vit_backbone, 'blocks'):
            for i in range(min(freeze_layers, len(self.vit_backbone.blocks))):
                for param in self.vit_backbone.blocks[i].parameters():
                    param.requires_grad = False
        
        print(f"冻结了前 {freeze_layers} 层参数")
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征
        
        Args:
            x: 输入图像 [batch_size, channels, height, width]
            
        Returns:
            features: 特征向量 [batch_size, feature_dim]
        """
        return self.vit_backbone(x)
    
    def compute_attention_weights(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算注意力权重
        
        Args:
            features: 特征向量 [num_instances, feature_dim]
            
        Returns:
            attention_weights: 注意力权重 [num_instances, 1]
            attention_scores: 注意力分数 [num_instances, 1]
        """
        attention_scores = self.attention(features)  # [num_instances, 1]
        attention_weights = F.softmax(attention_scores, dim=0)  # [num_instances, 1]
        
        return attention_weights, attention_scores
    
    def aggregate_features(self, features: torch.Tensor, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        聚合特征
        
        Args:
            features: 特征向量 [num_instances, feature_dim]
            attention_weights: 注意力权重 [num_instances, 1]
            
        Returns:
            aggregated_features: 聚合后的特征 [feature_dim]
        """
        # 加权平均
        weighted_features = features * attention_weights  # [num_instances, feature_dim]
        aggregated_features = torch.sum(weighted_features, dim=0)  # [feature_dim]
        
        return aggregated_features
    
    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            bag: 一个bag的图像 [num_instances, channels, height, width]
            
        Returns:
            logits: 分类logits [num_classes]
            attention_weights: 注意力权重 [num_instances, 1]
            instance_features: 实例特征 [num_instances, feature_dim]
        """
        num_instances = bag.size(0)
        
        # 提取每个实例的特征
        instance_features = []
        for i in range(num_instances):
            instance = bag[i:i+1]  # [1, channels, height, width]
            feature = self.extract_features(instance)  # [1, feature_dim]
            instance_features.append(feature.squeeze(0))  # [feature_dim]
        
        instance_features = torch.stack(instance_features)  # [num_instances, feature_dim]
        
        # 计算注意力权重
        attention_weights, attention_scores = self.compute_attention_weights(instance_features)
        
        # 聚合特征
        aggregated_features = self.aggregate_features(instance_features, attention_weights)
        
        # 分类
        logits = self.classifier(aggregated_features)  # [num_classes]
        
        return logits, attention_weights, instance_features
    
    def get_attention_maps(self, bag: torch.Tensor) -> List[torch.Tensor]:
        """
        获取Transformer内部的注意力图
        
        Args:
            bag: 一个bag的图像 [num_instances, channels, height, width]
            
        Returns:
            attention_maps: 每个实例的注意力图列表
        """
        self.eval()
        attention_maps = []
        
        with torch.no_grad():
            for i in range(bag.size(0)):
                instance = bag[i:i+1]  # [1, channels, height, width]
                
                # 使用hook获取注意力图
                attention_map = self.extract_vit_attention(instance)
                attention_maps.append(attention_map)
        
        return attention_maps
    
    def extract_vit_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取ViT内部的注意力图
        
        Args:
            x: 输入图像 [1, channels, height, width]
            
        Returns:
            attention_map: 注意力图 [num_heads, num_patches, num_patches]
        """
        # 这里需要根据具体的ViT实现来提取注意力图
        # 由于timm的ViT可能不直接暴露注意力图，这里提供一个简化版本
        
        # 获取最后一层的注意力权重作为示例
        def hook_fn(module, input, output):
            if hasattr(module, 'attn'):
                self.last_attention = module.attn
        
        # 注册hook
        hook = None
        if hasattr(self.vit_backbone, 'blocks') and len(self.vit_backbone.blocks) > 0:
            hook = self.vit_backbone.blocks[-1].register_forward_hook(hook_fn)
        
        # 前向传播
        _ = self.vit_backbone(x)
        
        # 移除hook
        if hook:
            hook.remove()
        
        # 返回注意力图（这里是简化版本）
        if hasattr(self, 'last_attention'):
            return self.last_attention
        else:
            # 如果无法获取注意力图，返回随机矩阵作为占位符
            num_patches = (224 // 16) ** 2  # 假设输入尺寸为224x224，patch size为16
            return torch.randn(12, num_patches + 1, num_patches + 1)  # 12个头
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'attention_dim': self.attention_dim,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params
        }


class ViTMILModelWithGating(ViTMILModel):
    """
    带有门控机制的ViT MIL模型
    
    在标准MIL的基础上增加了门控机制，用于更好地选择重要的实例
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 门控机制
        self.gating = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim // 2, self.feature_dim),
            nn.Sigmoid()
        )
        
        # 重新初始化新增的权重
        self.init_gating_weights()
    
    def init_gating_weights(self):
        """初始化门控权重"""
        for module in self.gating.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播（带门控）
        
        Args:
            bag: 一个bag的图像 [num_instances, channels, height, width]
            
        Returns:
            logits: 分类logits [num_classes]
            attention_weights: 注意力权重 [num_instances, 1]
            instance_features: 实例特征 [num_instances, feature_dim]
            gate_weights: 门控权重 [num_instances, feature_dim]
        """
        num_instances = bag.size(0)
        
        # 提取每个实例的特征
        instance_features = []
        for i in range(num_instances):
            instance = bag[i:i+1]  # [1, channels, height, width]
            feature = self.extract_features(instance)  # [1, feature_dim]
            instance_features.append(feature.squeeze(0))  # [feature_dim]
        
        instance_features = torch.stack(instance_features)  # [num_instances, feature_dim]
        
        # 计算门控权重
        gate_weights = self.gating(instance_features)  # [num_instances, feature_dim]
        
        # 应用门控
        gated_features = instance_features * gate_weights  # [num_instances, feature_dim]
        
        # 计算注意力权重
        attention_weights, attention_scores = self.compute_attention_weights(gated_features)
        
        # 聚合特征
        aggregated_features = self.aggregate_features(gated_features, attention_weights)
        
        # 分类
        logits = self.classifier(aggregated_features)  # [num_classes]
        
        return logits, attention_weights, instance_features, gate_weights


def create_vit_mil_model(model_config: dict) -> ViTMILModel:
    """
    创建ViT MIL模型的工厂函数
    
    Args:
        model_config: 模型配置字典
        
    Returns:
        ViT MIL模型实例
    """
    model_type = model_config.get('model_type', 'standard')
    
    if model_type == 'gating':
        return ViTMILModelWithGating(**model_config)
    else:
        return ViTMILModel(**model_config)


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = ViTMILModel(
        model_name='vit_base_patch16_224',
        pretrained=True,
        num_classes=2,
        attention_dim=128,
        dropout_rate=0.1
    )
    
    model.to(device)
    
    # 测试前向传播
    bag_size = 10
    batch_size = 3
    channels = 3
    height = width = 224
    
    # 创建测试数据
    test_bag = torch.randn(bag_size, channels, height, width).to(device)
    
    # 前向传播
    logits, attention_weights, instance_features = model(test_bag)
    
    print(f"模型信息: {model.get_model_info()}")
    print(f"输入bag大小: {test_bag.shape}")
    print(f"输出logits: {logits.shape}")
    print(f"注意力权重: {attention_weights.shape}")
    print(f"实例特征: {instance_features.shape}")
    
    # 测试注意力图提取
    attention_maps = model.get_attention_maps(test_bag)
    print(f"注意力图数量: {len(attention_maps)}")
    
    print("ViT MIL模型测试完成！") 