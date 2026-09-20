# OPS Installation Guide

## ⚠️ Python 版本相容性問題

你的系統使用的是 **Python 3.14**，這是比較新的版本，很多機器學習套件還不支援。

## 解決方案

### 方案一：使用 pyenv-win 安裝 Python 3.11/3.12（推薦）

```powershell
# 安裝 pyenv-win
irm https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install.ps1 | iex

# 安裝 Python 3.11
pyenv install 3.11.9
pyenv global 3.11.9

# 驗證
python --version  # 應該顯示 Python 3.11.9
```

### 方案二：從 python.org 下載

1. 前往 https://www.python.org/downloads/
2. 下載 Python 3.11 或 3.12
3. 安裝時勾選 "Add Python to PATH"

### 方案三：使用 Anaconda/Miniconda

```powershell
# 安裝 Miniconda
# 從 https://docs.conda.io/en/latest/miniconda.html 下載

# 建立 Python 3.11 環境
conda create -n ops python=3.11
conda activate ops

# 安裝 OPS
pip install -e "C:/Users/jksdf/ops[all]"
```

## 驗證安裝

```powershell
# 確認 Python 版本
python --version
# 輸出: Python 3.11.x 或 Python 3.12.x

# 安裝 OPS
cd C:\Users\jksdf\ops
pip install -e ".[all]"

# 初始化
ops init

# 測試
ops chat "你好"
```

## 如果只想快速測試（不需完整訓練功能）

```powershell
pip install -e ".[dev]"
python -m pytest tests/ -v
```
