# 使用 GitHub 创建可下载链接

## 方法：手动创建 GitHub Release

### 步骤 1: 下载压缩包到本地

在您的本地电脑运行：
```bash
scp your-server:/root/autodl-tmp/CSIRO/csiro-model-output.tar.gz ./
```

或者使用 FileZilla/WinSCP 等工具下载 `/root/autodl-tmp/CSIRO/csiro-model-output.tar.gz`

### 步骤 2: 在 GitHub 创建 Release

1. 访问您的 GitHub 仓库
2. 点击右侧的 "Releases" 
3. 点击 "Create a new release"
4. 填写信息：
   - Tag: `v1.0-model`
   - Title: `CSIRO Biomass Model v1.0`
   - Description: `Trained model with validation RMSE: 9.7474`
5. 拖拽上传 `csiro-model-output.tar.gz`
6. 点击 "Publish release"

### 步骤 3: 获取下载链接

发布后，右键点击文件 -> "复制链接地址"

链接格式类似：
```
https://github.com/username/repo/releases/download/v1.0-model/csiro-model-output.tar.gz
```

### 步骤 4: 在 Kaggle 导入

在 Kaggle Dataset 上传页面：
- 选择 "Link" -> "Remote URL"
- 粘贴上面的链接
- 点击导入

## 文件信息

- 文件名: `csiro-model-output.tar.gz`
- 大小: 86 MB
- 包含: `output/best_model.pth` (RMSE: 9.7474)
