# 在本地运行此脚本验证模型文件
import torch
import os

model_path = '/root/autodl-tmp/CSIRO/output/best_model.pth'

# 检查文件是否存在
if not os.path.exists(model_path):
    print(f"❌ 文件不存在: {model_path}")
else:
    # 检查文件大小
    file_size = os.path.getsize(model_path)
    print(f"✓ 文件存在，大小: {file_size / (1024*1024):.2f} MB")
    
    # 尝试加载
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✓ 模型加载成功！")
        print(f"  Keys: {list(checkpoint.keys())}")
        if 'val_rmse' in checkpoint:
            print(f"  验证 RMSE: {checkpoint['val_rmse']:.4f}")
        if 'epoch' in checkpoint:
            print(f"  训练轮次: {checkpoint['epoch']}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("\n文件可能已损坏，需要重新训练")