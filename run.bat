@echo off
setlocal enabledelayedexpansion

:: 设置颜色代码
set "INFO=INFO:"
set "SUCCESS=SUCCESS:"
set "ERROR=ERROR:"
set "WARNING=WARNING:"

echo ====================================================
echo            无序蛋白质区域预测模型启动脚本
echo ====================================================

:: 检查conda环境
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% 未找到conda，请先安装Anaconda或Miniconda
    goto :eof
)

:: 检查环境变量中是否包含用户名（判断是否初始化了conda）
echo %PATH% | findstr /C:"%USERNAME%" >nul
if %ERRORLEVEL% NEQ 0 (
    echo %WARNING% conda可能未正确初始化，尝试手动激活...
    :: 尝试寻找conda.bat文件路径
    for /f "tokens=*" %%a in ('where conda') do set CONDA_PATH=%%a
    set CONDA_DIR=!CONDA_PATH:conda.exe=!
    
    :: 执行conda初始化
    call "!CONDA_DIR!Scripts\activate.bat"
    echo %INFO% conda已手动激活
)

:: 检查是否存在disprot环境
conda env list | findstr /C:"disprot" >nul
if %ERRORLEVEL% NEQ 0 (
    echo %INFO% 未找到disprot环境，正在创建...
    conda env create -f conda_environment.yml
    if %ERRORLEVEL% NEQ 0 (
        echo %ERROR% 创建conda环境失败
        goto :eof
    )
)

echo %INFO% 使用conda环境...
call conda activate disprot
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% 激活conda环境失败
    echo %INFO% 尝试使用另一种方式激活...
    call "!CONDA_DIR!Scripts\activate.bat" disprot
    if %ERRORLEVEL% NEQ 0 (
        echo %ERROR% 无法激活conda环境，请手动激活后重试
        echo %INFO% 您可以手动运行以下命令:
        echo conda activate disprot
        echo python baseline.py --config_path config.yaml --output_dir ./outputs
        goto :eof
    )
)

:: 检查数据文件
IF NOT EXIST WSAA_data_public.pkl (
    echo %ERROR% 数据文件WSAA_data_public.pkl不存在!
    echo %INFO% 请将数据文件放在当前目录下，或修改config.yaml中的data_path路径
    goto :eof
)

:: 创建输出目录
mkdir outputs 2>nul

:: 询问是否使用SwanLab
set /p use_swanlab="是否使用SwanLab进行训练可视化? (y/n): "

:: 设置SwanLab选项
set "swanlab_option="
if /i "%use_swanlab%"=="y" (
    set "swanlab_option=--use_swanlab"
    
    :: 检查SwanLab是否已安装
    python -c "import swanlab" >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo %WARNING% SwanLab未安装，正在安装...
        pip install swanlab==0.5.4
    )
    
    :: 检查是否已登录SwanLab - 使用更安全的方法
    python -c "import os; config_path = os.path.expanduser('~/.swanlab/config.json'); print('已登录' if os.path.exists(config_path) and os.path.getsize(config_path) > 0 else '未登录')" > %temp%\swanlab_status.txt
    set /p swanlab_status=<%temp%\swanlab_status.txt
    del %temp%\swanlab_status.txt
    
    if "!swanlab_status!"=="未登录" (
        echo %INFO% 您尚未登录SwanLab. 请登录...
        swanlab login
    ) else (
        echo %SUCCESS% 已登录到SwanLab!
    )
)

:: 选择训练方式
echo.
echo 请选择训练方式:
echo 1) 基本训练 (baseline.py)
echo 2) PyTorch Lightning训练 (train_with_lightning.py)
set /p train_choice="输入选择 (1/2): "

if "%train_choice%"=="1" (
    echo %INFO% 使用基本训练方式...
    python baseline.py --config_path config.yaml --output_dir ./outputs %swanlab_option%
) else if "%train_choice%"=="2" (
    echo %INFO% 使用PyTorch Lightning训练方式...
    set /p early_stopping="是否使用早停机制? (y/n): "
    
    if /i "%early_stopping%"=="y" (
        set /p patience="设置早停耐心值 (默认为5): "
        if "!patience!"=="" set "patience=5"
        python train_with_lightning.py --config_path config.yaml --output_dir ./outputs --early_stopping --patience !patience! %swanlab_option%
    ) else (
        python train_with_lightning.py --config_path config.yaml --output_dir ./outputs %swanlab_option%
    )
) else (
    echo %ERROR% 无效的选择，退出脚本
    goto :eof
)

echo %SUCCESS% 训练完成!
endlocal 