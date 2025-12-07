#!/bin/bash
# 使用 Token 推送到 GitHub

echo "=========================================="
echo "GitHub Push 助手"
echo "=========================================="
echo ""
echo "请准备好您的 Personal Access Token"
echo "如果没有，请访问: https://github.com/settings/tokens"
echo ""
read -p "输入用户名 (VegtVibeCoder): " username
username=${username:-VegtVibeCoder}

read -sp "输入 Personal Access Token: " token
echo ""
echo ""

cd /root/autodl-tmp/CSIRO

# 临时设置远程 URL（包含认证信息）
git remote set-url origin "https://${username}:${token}@github.com/VegtVibeCoder/CSIRO.git"

echo "正在推送..."
if git push origin master; then
    echo ""
    echo "✓ 推送成功！"
    echo ""
    echo "凭证已保存，下次推送不需要再输入"
else
    echo ""
    echo "✗ 推送失败"
fi

# 清理 URL 中的 token（安全考虑）
git remote set-url origin "https://github.com/VegtVibeCoder/CSIRO.git"
