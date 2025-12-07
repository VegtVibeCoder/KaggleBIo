"""
在 Kaggle Notebook 中测试模型加载
复制此代码到 Kaggle Notebook Cell 运行
"""

import torch
import sys

print("PyTorch 版本:", torch.__version__)
print()

# 模型路径（根据你的 Kaggle Dataset 名称修改）
MODEL_PATH = '/kaggle/input/best-model/best_model.pth'

print("正在加载模型...")
try:
    # 重要：添加 weights_only=False 参数
    checkpoint = torch.load(MODEL_PATH, 
                           map_location='cpu', 
                           weights_only=False)
    
    print("✓ 模型加载成功！")
    print()
    print("Checkpoint Keys:", list(checkpoint.keys()))
    print()
    
    if 'val_rmse' in checkpoint:
        print(f"验证集 RMSE: {checkpoint['val_rmse']:.4f}")
    
    if 'epoch' in checkpoint:
        print(f"训练轮次: {checkpoint['epoch']}")
    
    if 'model_state_dict' in checkpoint:
        print(f"模型参数数量: {len(checkpoint['model_state_dict'])}")
    
    print()
    print("✓ 模型文件完整，可以用于推理！")
    
except Exception as e:
    print(f"✗ 加载失败: {e}")
    sys.exit(1)
