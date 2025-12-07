#!/bin/bash
# 使用 file.io 上传文件

cd /root/autodl-tmp/CSIRO

echo "正在创建压缩包..."
if [ ! -f csiro-model-output.tar.gz ]; then
    tar -czf csiro-model-output.tar.gz output/
fi

echo ""
echo "正在上传到 file.io（文件保留 14 天，下载 1 次后自动删除）..."
echo ""

RESPONSE=$(curl -F "file=@csiro-model-output.tar.gz" https://file.io/?expires=14d)

echo ""
echo "=========================================="
echo "上传响应："
echo "$RESPONSE"
echo "=========================================="

# 提取链接
URL=$(echo $RESPONSE | grep -o '"link":"[^"]*' | cut -d'"' -f4)

if [ -n "$URL" ]; then
    echo ""
    echo "✓ 上传成功！"
    echo ""
    echo "下载链接（14天有效，下载1次后失效）："
    echo "$URL"
    echo ""
    echo "$URL" > kaggle_dataset_url.txt
    echo "链接已保存到: kaggle_dataset_url.txt"
else
    echo ""
    echo "✗ 上传失败，请尝试其他方案"
fi
