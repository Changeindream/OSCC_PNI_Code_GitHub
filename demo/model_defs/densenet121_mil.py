import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DenseNet121MILModel(nn.Module):
    def __init__(self, pretrained_path:str = None, 
                 attention_dim:int = 128, num_classes:int = 2, use_pretrained:bool = True):
        """
        基于DenseNet121和注意力池化的多实例学习模型。
        :param pretrained_path: 预训练权重文件的路径。如果提供，则会加载权重。
        :param attention_dim: 注意力隐藏层维度大小。默认为128。
        :param num_classes: 输出类别数，二分类默认为2。
        :param use_pretrained: 是否使用ImageNet预训练权重。
        """
        super(DenseNet121MILModel, self).__init__()
        
        # 1. 构建DenseNet121特征提取器
        if use_pretrained:
            base_model = models.densenet121(weights='IMAGENET1K_V1')
        else:
            base_model = models.densenet121(weights=None)
        
        if pretrained_path is not None:
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                
                # 过滤掉分类器层的权重，因为维度不匹配
                filtered_state_dict = {}
                for key, value in state_dict.items():
                    if not key.startswith('classifier.'):
                        filtered_state_dict[key] = value
                
                # 加载过滤后的权重
                missing_keys, unexpected_keys = base_model.load_state_dict(filtered_state_dict, strict=False)
                print(f"成功加载预训练权重: {pretrained_path}")
                if missing_keys:
                    print(f"缺少的键: {missing_keys}")
                if unexpected_keys:
                    print(f"未期望的键: {unexpected_keys}")
            except Exception as e:
                print(f"加载预训练权重失败: {e}")
                print("将使用随机初始化的权重继续训练")
                
        feature_dim = base_model.classifier.in_features  # DenseNet121 分类层输入特征维度
        base_model.classifier = nn.Identity()            # 去掉DenseNet的分类层，使forward返回特征向量
        self.feature_extractor = base_model

        # 2. 定义注意力池化层：将每个实例特征映射到一个注意力权重
        # 注意力网络包含两层全连接：先降维到attention_dim，再输出1个注意力分数
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

        # 3. 定义分类层：输入为聚合后的袋子特征，输出为num_classes维度
        self.classifier = nn.Linear(feature_dim, num_classes)
        # 对于二分类任务，通常直接输出logits，配合CrossEntropyLoss使用
        # 不在模型中添加激活函数，让损失函数处理

    def forward(self, images: torch.Tensor):
        """
        前向传播函数。
        :param images: 张量形式的图像集合，可以是形状[N, C, H, W]的4D张量，表示N张切片组成的一个bag。
                       或者是形如 (C, H, W) 的单张图像张量（此时视为N=1）。
        :return: 返回病人级预测的概率（对于二分类，是一个0~1的值）。
        """
        # 如果输入是一张图像(3D张量)，则增加一维变为 (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        # 1. 利用DenseNet121提取每张切片的特征
        #    输出 shape: [N, feature_dim]。对于卷积模型，此步包括卷积特征提取和全局池化。
        features = self.feature_extractor(images)           # 特征提取
        if features.dim() > 2:
            # 若输出仍有高维度（例如 [N, C, 1, 1]），则展平
            features = features.view(features.size(0), -1)  # shape -> [N, feature_dim]

        # 2. 通过注意力网络计算每个实例的注意力分数
        #    attention_scores shape: [N, 1], 每个实例对应一个分数（未归一化）
        attention_scores = self.attention_net(features)     # [N, 1]
        attention_scores = attention_scores.transpose(1, 0) # 转置为 [1, N] 方便softmax
        # 对实例维度执行softmax，将分数归一化为 [0,1] 区间并且N个实例的权重和为1
        attention_weights = F.softmax(attention_scores, dim=1)  # [1, N]

        # 3. 利用注意力权重对实例特征加权求和，得到聚合的bag特征
        #    注意：这里使用矩阵乘法，相当于对每个特征维度做加权平均 (权重attention_weights)
        bag_feature = torch.mm(attention_weights, features)     # [1, N] * [N, feature_dim] -> [1, feature_dim]

        # 4. 将融合后的bag特征输入分类层，获得输出结果
        logits = self.classifier(bag_feature)  # [1, num_classes]

        # 直接返回logits，兼容CrossEntropyLoss
        return logits.squeeze(dim=0)  # 返回 [num_classes] 形状的张量 