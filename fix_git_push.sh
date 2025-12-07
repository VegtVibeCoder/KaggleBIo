#!/bin/bash
# 修复 Git push 网络问题

echo "=========================================="
echo "Git Push 问题诊断与修复"
echo "=========================================="
echo ""

cd /root/autodl-tmp/CSIRO

# 方案 1: 测试网络连接
echo "1. 测试 GitHub 连接..."
if curl -I https://github.com 2>&1 | grep -q "200\|301\|302"; then
    echo "✓ GitHub 可访问"
else
    echo "✗ GitHub 连接失败"
fi
echo ""

# 方案 2: 配置 Git 使用更稳定的协议
echo "2. 配置 Git HTTP 设置..."
git config --global http.postBuffer 524288000
git config --global http.version HTTP/1.1
git config --global http.sslVerify false
echo "✓ HTTP 配置完成"
echo ""

# 方案 3: 重试推送
echo "3. 尝试推送..."
echo ""

# 尝试 3 次
for i in {1..3}; do
    echo "尝试 $i/3..."
    if git push origin master; then
        echo ""
        echo "✓ 推送成功！"
        exit 0
    else
        echo "✗ 推送失败，等待 2 秒后重试..."
        sleep 2
    fi
done

echo ""
echo "=========================================="
echo "推送失败，建议使用备用方案"
echo "=========================================="
echo ""
echo "备用方案 1: 使用 SSH"
echo "  运行: ./setup_ssh.sh"
echo ""
echo "备用方案 2: 稍后重试"
echo "  可能是网络临时问题"
echo ""
echo "备用方案 3: 检查防火墙/代理设置"
echo "=========================================="
