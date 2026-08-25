import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MILModel(nn.Module):
    def __init__(self, backbone_name:str = 'resnet152', pretrained_path:str = None, 
                 attention_dim:int = 128, num_classes:int = 2, use_pretrained:bool = True):
        """
        基于注意力池化的多实例学习模型。
        :param backbone_name: 骨干特征提取网络的名称，可选 'resnet152'、'densenet121'、'vit'、'swin'
        :param pretrained_path: 预训练权重文件的路径。如果提供，则会加载权重。
        :param attention_dim: 注意力隐藏层维度大小。默认为128。
        :param num_classes: 输出类别数，二分类默认为2。
        :param use_pretrained: 是否使用ImageNet预训练权重。
        """
        super(MILModel, self).__init__()
        # 1. 根据backbone_name选择并构建特征提取器（去除最终分类层以获得特征向量）
        if backbone_name.lower() == 'resnet152':
            if use_pretrained:
                base_model = models.resnet152(weights='IMAGENET1K_V1')
            else:
                base_model = models.resnet152(weights=None)
            
            if pretrained_path is not None:
                try:
                    state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                    if "model_state_dict" in state_dict:
                        state_dict = state_dict["model_state_dict"]
                    base_model.load_state_dict(state_dict, strict=False)
                    print(f"成功加载预训练权重: {pretrained_path}")
                except Exception as e:
                    print(f"加载预训练权重失败: {e}")
                    
            feature_dim = base_model.fc.in_features  # ResNet152 最后一层全连接层输入的特征维度
            base_model.fc = nn.Identity()            # 去掉ResNet的分类层，使forward返回特征向量
            self.feature_extractor = base_model

        elif backbone_name.lower() == 'densenet121':
            base_model = models.densenet121(pretrained=False)
            if pretrained_path is not None:
                base_model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
            feature_dim = base_model.classifier.in_features  # DenseNet121 分类层输入特征维度
            base_model.classifier = nn.Identity()            # 去掉DenseNet的分类层
            self.feature_extractor = base_model

        elif backbone_name.lower() == 'vit':
            # ViT 模型需要通过 timm 或其他库获取，这里以timm的ViT-B/16为例
            try:
                import timm
            except ImportError:
                raise ImportError("需要安装 timm 库以加载 ViT 模型")
            base_model = timm.create_model('vit_base_patch16_224', pretrained=False)
            if pretrained_path is not None:
                base_model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
            # ViT的分类头通常名为 'head'
            feature_dim = base_model.head.in_features
            base_model.head = nn.Identity()  # 去除 ViT 的分类头
            self.feature_extractor = base_model

        elif backbone_name.lower().startswith('swin'):
            # Swin Transformer 模型，同样通过 timm 获取，例如 swin_base_patch4_window7_224
            try:
                import timm
            except ImportError:
                raise ImportError("需要安装 timm 库以加载 Swin Transformer 模型")
            # 默认使用 Swin-B。如果需要其他版本，可修改模型名称参数。
            model_name = 'swin_base_patch4_window7_224' if backbone_name.lower() == 'swin' else backbone_name
            base_model = timm.create_model(model_name, pretrained=False)
            if pretrained_path is not None:
                base_model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
            feature_dim = base_model.head.in_features
            base_model.head = nn.Identity()  # 去除 Swin 的分类头
            self.feature_extractor = base_model

        else:
            raise ValueError(f"不支持的backbone模型: {backbone_name}")

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
        # 1. 利用骨干网络提取每张切片的特征
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

