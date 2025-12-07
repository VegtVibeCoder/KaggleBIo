#!/bin/bash
# 使用 0x0.st 上传文件

cd /root/autodl-tmp/CSIRO

echo "正在创建压缩包..."
if [ ! -f csiro-model-output.tar.gz ]; then
    tar -czf csiro-model-output.tar.gz output/
fi

echo ""
echo "文件大小:"
ls -lh csiro-model-output.tar.gz

echo ""
echo "正在上传到 0x0.st..."
echo ""

URL=$(curl -F "file=@csiro-model-output.tar.gz" https://0x0.st)

echo ""
echo "=========================================="
echo "✓ 上传成功！"
echo "=========================================="
echo ""
echo "下载链接："
echo "$URL"
echo ""
echo "在 Kaggle 中使用此链接导入数据集"
echo "=========================================="
echo ""

# 保存链接
echo "$URL" > kaggle_dataset_url.txt
echo "链接已保存到: kaggle_dataset_url.txt"
