@echo off
chcp 65001 >nul
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

:: 检查data目录是否存在，不存在则创建
IF NOT EXIST data (
    echo %INFO% 数据目录不存在，正在创建...
    mkdir data
    echo %SUCCESS% 已创建数据目录！
)

:: 检查数据文件是否存在或已放置在其他位置
IF NOT EXIST WSAA_data_public.pkl (
    IF NOT EXIST data\WSAA_data_public.pkl (
        echo %WARNING% 数据文件WSAA_data_public.pkl不存在于当前目录或data目录！
        echo %INFO% 如果您已将数据文件放置在其他位置，请确保更新config.yaml中的data_path指向正确路径
        set /p continue_execution="是否继续执行? (y/n): "
        if /i NOT "!continue_execution!"=="y" (
            echo %INFO% 退出脚本
            goto :eof
        )
    ) else (
        echo %INFO% 在data目录中找到数据文件WSAA_data_public.pkl
    )
) else (
    echo %INFO% 在当前目录找到数据文件WSAA_data_public.pkl
)

:: 创建输出目录
IF NOT EXIST outputs (
    echo %INFO% 输出目录不存在，正在创建...
    mkdir outputs
    echo %SUCCESS% 已创建输出目录！
) else (
    echo %INFO% 输出目录已存在
)

:: 询问是否添加时间戳
set /p use_timestamp="是否为输出目录添加时间戳? (y/n): "
set "timestamp_option="
if /i NOT "%use_timestamp%"=="y" (
    set "timestamp_option=--no_timestamp"
    echo %INFO% 不添加时间戳到输出目录
) else (
    echo %INFO% 输出目录将添加时间戳以区分不同运行
)

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
    python baseline.py --config_path config.yaml --output_dir ./outputs %timestamp_option% %swanlab_option%
) else if "%train_choice%"=="2" (
    echo %INFO% 使用PyTorch Lightning训练方式...
    set /p early_stopping="是否使用早停机制? (y/n): "
    set /p gpu_count="设置使用的GPU数量 (默认为1): "
    if "!gpu_count!"=="" set "gpu_count=1"
    
    if /i "%early_stopping%"=="y" (
        set /p patience="设置早停耐心值 (默认为5): "
        if "!patience!"=="" set "patience=5"
        python train_with_lightning.py --config_path config.yaml --output_dir ./outputs --early_stopping --patience !patience! --gpus !gpu_count! %timestamp_option% %swanlab_option%
    ) else (
        python train_with_lightning.py --config_path config.yaml --output_dir ./outputs --gpus !gpu_count! %timestamp_option% %swanlab_option%
    )
) else (
    echo %ERROR% 无效的选择，退出脚本
    goto :eof
)

echo %SUCCESS% 训练完成!
endlocal 