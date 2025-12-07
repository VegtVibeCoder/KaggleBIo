"""
模型定义模块
使用 timm 库的预训练模型作为 backbone
"""

import torch
import torch.nn as nn
import timm


class BiomassModel(nn.Module):
    """
    牧场生物量预测模型
    
    使用预训练的 CNN backbone，输出 5 个生物量指标
    """
    
    def __init__(self, model_name='tf_efficientnet_b0', pretrained=True, num_classes=5):
        """
        Args:
            model_name: timm 模型名称
            pretrained: 是否使用预训练权重
            num_classes: 输出类别数（5 个生物量指标）
        """
        super(BiomassModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # 使用 timm 创建模型
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        print(f"Model created: {model_name}")
        print(f"Pretrained: {pretrained}")
        print(f"Output classes: {num_classes}")
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 tensor [Batch_Size, 3, H, W]
        
        Returns:
            output: 预测的生物量向量 [Batch_Size, 5]
        """
        return self.backbone(x)


def create_model(model_name='tf_efficientnet_b0', pretrained=True, device='cuda'):
    """
    创建模型的辅助函数
    
    Args:
        model_name: 模型名称
        pretrained: 是否使用预训练权重
        device: 设备
    
    Returns:
        model: 创建的模型
    """
    model = BiomassModel(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=5
    )
    
    model = model.to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # 测试模型创建
    print("Testing model creation...")
    
    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = create_model(
        model_name='tf_efficientnet_b0',
        pretrained=True,
        device=device
    )
    
    # 测试前向传播
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample:\n{output}")
