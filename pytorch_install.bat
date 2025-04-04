@echo off
setlocal enabledelayedexpansion

:: 设置颜色代码
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "ERROR=[ERROR]"
set "WARNING=[WARNING]"

echo ====================================================
echo          PyTorch 2.4.0 + CUDA 12.1 安装脚本
echo ====================================================

:: 检查conda环境
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% 未找到conda，请先安装Anaconda或Miniconda
    goto :eof
)

:: 检查是否存在disprot环境
conda env list | findstr /C:"disprot" >nul
if %ERRORLEVEL% EQU 0 (
    echo %WARNING% 已存在disprot环境，将移除并重新创建
    conda env remove -n disprot
    if %ERRORLEVEL% NEQ 0 (
        echo %ERROR% 移除现有环境失败，请手动移除后重试
        echo %INFO% 手动移除命令: conda env remove -n disprot
        goto :eof
    )
)

echo %INFO% 创建新的conda环境 (Python 3.10)...
conda create -n disprot python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% 创建conda环境失败
    goto :eof
)

:: 激活环境
echo %INFO% 激活conda环境...
call conda activate disprot
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% 激活conda环境失败
    goto :eof
)

:: 安装基础依赖
echo %INFO% 安装基础依赖...
conda install -y -n disprot setuptools wheel vs2019_runtime vc=14.3
if %ERRORLEVEL% NEQ 0 (
    echo %WARNING% 安装基础依赖失败，尝试继续安装其他包
)

:: 直接使用pip安装PyTorch
echo %INFO% 安装PyTorch 2.4.0 + CUDA 12.1...
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% 安装PyTorch失败
    goto :eof
)

:: 安装其他依赖项
echo %INFO% 安装其他依赖项...
pip install numpy==1.25.0 tqdm==4.66.1 omegaconf==2.3.0 scikit-learn==1.3.2 matplotlib==3.7.3 seaborn==0.13.0 pandas==2.2.0
pip install pytorch-lightning==2.2.0 transformers==4.42.0 swanlab==0.5.4 biopython==1.83 colorama==0.4.6
pip install python-dotenv==1.0.0 tensorboard==2.15.1 pyyaml==6.0.1 ipython

:: 测试PyTorch安装
echo %INFO% 测试PyTorch安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% PyTorch测试失败
    goto :eof
)

echo %SUCCESS% 安装完成!
echo %INFO% 请使用以下命令激活环境:
echo       conda activate disprot
echo %INFO% 使用以下命令开始训练:
echo       python baseline.py --config_path config.yaml --output_dir ./outputs

endlocal 