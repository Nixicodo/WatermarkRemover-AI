# 打包说明与调试指南

## 1. 打包前准备

确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

## 2. 打包步骤

使用PyInstaller进行打包：
```bash
pyinstaller watermark_remover.spec
```

## 3. 打包配置说明

### watermark_remover.spec
- 入口文件：remwmgui.py
- 隐藏导入：PyQt6, transformers, torch等
- 数据文件：ui.yml, icon.ico

### requirements.txt
- 列出了所有项目依赖

## 4. 调试机制

### 调试语句
在关键位置添加了调试语句：
- remwm.py: 模型加载、设备检测
- remwmgui.py: 文件路径、执行环境检测
- utils.py: 模型设置

### 调试信息输出
程序运行时会输出以下信息：
- 当前工作目录
- Python可执行文件路径
- 脚本路径
- 是否在PyInstaller环境中运行
- 模型加载状态
- 文件是否存在

## 5. 常见问题排查

### 模型未找到
- 程序会在首次运行时自动下载模型到models/目录
- 确保网络连接正常
- 检查磁盘空间（模型约2.5GB）

### 文件路径问题
- 检查remwm.py是否与remwmgui.py在同一目录
- 确认PyInstaller打包时包含了所有必要文件

### 依赖库问题
- 确保requirements.txt中列出的所有依赖都已安装
- 检查PyInstaller的hiddenimports配置

## 6. 迭代改进

1. 根据调试输出信息定位问题
2. 修改代码或配置文件
3. 重新打包
4. 重复以上步骤直到解决问题

## 7. 注意事项

- 模型文件不会被打包进可执行文件，程序会自动下载
- 打包后的程序会创建models/目录用于存储下载的模型
- 首次运行可能需要较长时间下载模型