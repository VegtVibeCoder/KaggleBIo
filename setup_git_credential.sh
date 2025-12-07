#!/bin/bash
# 安全配置 Git 凭证存储

echo "=========================================="
echo "Git 凭证安全配置"
echo "=========================================="
echo ""

cd /root/autodl-tmp/CSIRO

# 1. 先改回 HTTPS URL（不包含 token）
echo "步骤 1: 重置远程 URL..."
git remote set-url origin https://github.com/VegtVibeCoder/CSIRO.git

# 2. 配置 Git 使用凭证存储
echo "步骤 2: 配置凭证存储..."
git config --global credential.helper store

echo ""
echo "✓ 配置完成！"
echo ""
echo "=========================================="
echo "下一步操作"
echo "=========================================="
echo ""
echo "1. 访问 https://github.com/settings/tokens"
echo "2. 删除刚才的 token（已泄露）"
echo "3. 创建新的 token"
echo "4. 运行: git push origin master"
echo "5. 输入用户名: VegtVibeCoder"
echo "6. 输入密码: 粘贴新的 token"
echo ""
echo "之后 Git 会自动保存凭证，不需要每次输入"
echo "凭证会安全存储在 ~/.git-credentials 文件中"
echo "=========================================="
