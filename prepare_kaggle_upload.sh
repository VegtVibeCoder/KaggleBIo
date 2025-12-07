#!/bin/bash

# ================================================
# Kaggle 上传文件准备脚本
# 自动打包 src 和 output 文件夹用于上传到 Kaggle
# ================================================

echo "================================================"
echo "准备 Kaggle 上传文件"
echo "================================================"

# 设置变量
UPLOAD_DIR="kaggle_upload"
ZIP_NAME="csiro-model.zip"

# 1. 创建上传目录
echo ""
echo "📁 步骤 1: 创建上传目录..."
if [ -d "$UPLOAD_DIR" ]; then
    echo "⚠️  目录已存在，先删除..."
    rm -rf "$UPLOAD_DIR"
fi
mkdir -p "$UPLOAD_DIR"
echo "✅ 创建完成: $UPLOAD_DIR/"

# 2. 复制 src 文件夹
echo ""
echo "📁 步骤 2: 复制 src 文件夹..."
if [ -d "src" ]; then
    cp -r src "$UPLOAD_DIR/"
    echo "✅ src/ 复制完成"
    echo "   包含文件:"
    ls -lh "$UPLOAD_DIR/src/"
else
    echo "❌ 错误: src 文件夹不存在"
    exit 1
fi

# 3. 复制 output 文件夹
echo ""
echo "📁 步骤 3: 复制 output 文件夹..."
if [ -d "output" ]; then
    cp -r output "$UPLOAD_DIR/"
    echo "✅ output/ 复制完成"
    echo "   包含文件:"
    ls -lh "$UPLOAD_DIR/output/"
else
    echo "⚠️  警告: output 文件夹不存在"
    echo "   请先运行训练生成模型权重"
    echo "   运行: ./run_baseline.sh"
    exit 1
fi

# 4. 检查必需文件
echo ""
echo "🔍 步骤 4: 检查必需文件..."
REQUIRED_FILES=(
    "$UPLOAD_DIR/src/model.py"
    "$UPLOAD_DIR/output/best_model.pth"
)

ALL_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "✅ $file ($size)"
    else
        echo "❌ 缺失: $file"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo ""
    echo "❌ 错误: 缺少必需文件，请检查后重试"
    exit 1
fi

# 5. 复制推理脚本（可选）
echo ""
echo "📁 步骤 5: 复制推理脚本..."
if [ -f "kaggle_notebook_cell.py" ]; then
    cp kaggle_notebook_cell.py "$UPLOAD_DIR/"
    echo "✅ kaggle_notebook_cell.py 复制完成"
fi

if [ -f "KAGGLE_GUIDE.md" ]; then
    cp KAGGLE_GUIDE.md "$UPLOAD_DIR/"
    echo "✅ KAGGLE_GUIDE.md 复制完成"
fi

# 6. 创建 README
echo ""
echo "📝 步骤 6: 创建 README..."
cat > "$UPLOAD_DIR/README.md" << 'EOF'
# CSIRO Pasture Biomass Prediction - Kaggle Upload

## 📦 包含文件

- `src/` - 源代码（包含模型定义）
- `output/` - 训练好的模型权重
- `kaggle_notebook_cell.py` - 推理脚本
- `KAGGLE_GUIDE.md` - 使用指南

## 🚀 使用方法

1. 上传此文件夹到 Kaggle Dataset
2. 在 Kaggle Notebook 中添加此 Dataset
3. 复制 `kaggle_notebook_cell.py` 的内容到 Notebook
4. 修改路径配置
5. 运行即可生成提交文件

详细说明请查看 `KAGGLE_GUIDE.md`

## 📊 模型信息

- Model: EfficientNet-B0
- Input Size: 224x224
- Outputs: 5 biomass targets

---

*Good luck with your submission! 🌿*
EOF
echo "✅ README.md 创建完成"

# 7. 显示目录结构
echo ""
echo "📊 步骤 7: 最终文件结构"
echo "----------------------------------------"
tree -L 2 "$UPLOAD_DIR" 2>/dev/null || find "$UPLOAD_DIR" -type f -o -type d | head -20
echo "----------------------------------------"

# 8. 计算总大小
echo ""
echo "📏 步骤 8: 统计文件大小..."
TOTAL_SIZE=$(du -sh "$UPLOAD_DIR" | awk '{print $1}')
echo "总大小: $TOTAL_SIZE"

# 9. 创建 ZIP 文件
echo ""
echo "📦 步骤 9: 创建 ZIP 压缩包..."
if [ -f "$ZIP_NAME" ]; then
    echo "⚠️  删除旧的 ZIP 文件..."
    rm "$ZIP_NAME"
fi

cd "$UPLOAD_DIR"
zip -r "../$ZIP_NAME" . -q
cd ..

if [ -f "$ZIP_NAME" ]; then
    ZIP_SIZE=$(ls -lh "$ZIP_NAME" | awk '{print $5}')
    echo "✅ ZIP 文件创建完成: $ZIP_NAME ($ZIP_SIZE)"
else
    echo "❌ ZIP 文件创建失败"
    exit 1
fi

# 10. 完成提示
echo ""
echo "================================================"
echo "✅ 准备完成！"
echo "================================================"
echo ""
echo "📦 生成的文件:"
echo "   1. $UPLOAD_DIR/ - 上传目录"
echo "   2. $ZIP_NAME - 压缩包"
echo ""
echo "🚀 下一步操作:"
echo "   1. 访问 https://www.kaggle.com/datasets"
echo "   2. 点击 'New Dataset'"
echo "   3. 上传 $ZIP_NAME"
echo "   4. 参考 KAGGLE_GUIDE.md 进行推理"
echo ""
echo "💡 提示:"
echo "   - 推荐使用 ZIP 文件上传（更快）"
echo "   - 也可以直接上传 $UPLOAD_DIR 文件夹"
echo "   - 上传后记录 Dataset 路径"
echo ""
echo "================================================"
