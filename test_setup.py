"""
测试脚本：验证环境和代码设置
"""

import sys
import os

def test_imports():
    """测试关键库是否可以导入"""
    print("=" * 60)
    print("测试 1: 检查库导入")
    print("-" * 60)
    
    required_packages = [
        'torch',
        'torchvision',
        'timm',
        'pandas',
        'numpy',
        'PIL',
        'sklearn',
        'tqdm',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package:15s} - OK")
        except ImportError:
            print(f"✗ {package:15s} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n警告: 缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ 所有必需的包都已安装")
        return True


def test_torch_cuda():
    """测试 PyTorch 和 CUDA"""
    print("\n" + "=" * 60)
    print("测试 2: 检查 PyTorch 和 CUDA")
    print("-" * 60)
    
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA 不可用，将使用 CPU 训练（速度较慢）")
        return True
    except Exception as e:
        print(f"✗ PyTorch 测试失败: {e}")
        return False


def test_data_files():
    """测试数据文件是否存在"""
    print("\n" + "=" * 60)
    print("测试 3: 检查数据文件")
    print("-" * 60)
    
    data_files = [
        'csiro-biomass/train.csv',
        'csiro-biomass/test.csv',
        'csiro-biomass/sample_submission.csv'
    ]
    
    all_exist = True
    for file in data_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file:35s} - {size:,} bytes")
        else:
            print(f"✗ {file:35s} - Not found")
            all_exist = False
    
    if not all_exist:
        print("\n⚠ 部分数据文件缺失")
    
    return all_exist


def test_source_code():
    """测试源代码文件"""
    print("\n" + "=" * 60)
    print("测试 4: 检查源代码文件")
    print("-" * 60)
    
    source_files = [
        'src/__init__.py',
        'src/dataset.py',
        'src/model.py',
        'src/train.py',
        'src/inference.py'
    ]
    
    all_exist = True
    for file in source_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file:25s} - {size:,} bytes")
        else:
            print(f"✗ {file:25s} - Not found")
            all_exist = False
    
    return all_exist


def test_model_creation():
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("测试 5: 测试模型创建")
    print("-" * 60)
    
    try:
        sys.path.insert(0, 'src')
        from model import BiomassModel
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = BiomassModel(model_name='tf_efficientnet_b0', pretrained=False, num_classes=5)
        model = model.to(device)
        
        # 测试前向传播
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ 模型创建成功")
        print(f"  输入形状: {dummy_input.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  设备: {device}")
        
        return True
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 60)
    print("测试 6: 测试数据加载")
    print("-" * 60)
    
    try:
        import pandas as pd
        
        train_csv = 'csiro-biomass/train.csv'
        if not os.path.exists(train_csv):
            print(f"✗ 训练数据文件不存在: {train_csv}")
            return False
        
        df = pd.read_csv(train_csv)
        print(f"✓ 数据加载成功")
        print(f"  数据形状: {df.shape}")
        print(f"  唯一图片数: {df['image_path'].nunique()}")
        print(f"  列名: {df.columns.tolist()}")
        
        return True
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("CSIRO Pasture Biomass Prediction - 环境测试")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("库导入", test_imports()))
    results.append(("PyTorch/CUDA", test_torch_cuda()))
    results.append(("数据文件", test_data_files()))
    results.append(("源代码", test_source_code()))
    results.append(("模型创建", test_model_creation()))
    results.append(("数据加载", test_data_loading()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:20s} : {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("\n" + "-" * 60)
    print(f"通过: {passed}/{total} 项测试")
    
    if passed == total:
        print("\n✓ 所有测试通过！可以开始训练了。")
        print("\n运行以下命令开始训练:")
        print("  python src/train.py --epochs 10 --batch_size 32")
        print("\n或使用快速脚本:")
        print("  ./run_baseline.sh")
    else:
        print("\n⚠ 部分测试失败，请检查环境配置。")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
