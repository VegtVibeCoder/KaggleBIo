#!/bin/bash

# CSIRO Pasture Biomass Prediction - Baseline 快速运行脚本

echo "========================================"
echo "CSIRO Pasture Biomass Prediction Baseline"
echo "========================================"

# 设置参数
EPOCHS=20
BATCH_SIZE=32
MODEL="tf_efficientnet_b0"
IMAGE_SIZE=224
LR=0.001

# Step 1: 训练模型
echo ""
echo "=== Step 1: Training Model ==="
python3 src/train.py \
    --data_dir data \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --lr $LR \
    --pretrained \
    --output_dir output

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "✓ Training completed successfully"
else
    echo "✗ Training failed"
    exit 1
fi

# Step 2: 运行推理
echo ""
echo "=== Step 2: Running Inference ==="
python3 src/inference.py \
    --test_csv csiro-biomass/test.csv \
    --root_dir . \
    --model $MODEL \
    --weights output/best_model.pth \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --output submission.csv

# 检查推理是否成功
if [ $? -eq 0 ]; then
    echo "✓ Inference completed successfully"
    echo "✓ Submission file: submission.csv"
else
    echo "✗ Inference failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Baseline pipeline completed!"
echo "========================================"
