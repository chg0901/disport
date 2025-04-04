@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 设置颜色代码
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "ERROR=[ERROR]"
set "WARNING=[WARNING]"

echo ====================================================
echo            无序蛋白质区域预测脚本 (Windows版)
echo ====================================================

:: 检查数据目录是否存在
if not exist saisdata (
    echo %ERROR% 输入数据目录 saisdata 不存在！
    echo %INFO% 正在创建saisdata目录...
    mkdir saisdata
    echo %SUCCESS% 已创建saisdata目录！
    echo %INFO% 请将测试数据放入saisdata目录后再运行此脚本。
    pause
    exit /b 1
)

:: 检查输出目录
if not exist saisresult (
    echo %INFO% 输出目录 saisresult 不存在，正在创建...
    mkdir saisresult
    echo %SUCCESS% 已创建saisresult目录！
)

:: 查找输入文件
set "INPUT_FILE="
if exist saisdata\test.pkl (
    set "INPUT_FILE=saisdata\test.pkl"
) else if exist saisdata\test.csv (
    set "INPUT_FILE=saisdata\test.csv"
) else (
    :: 查找任何可能的输入文件
    for %%F in (saisdata\*.*) do (
        set "INPUT_FILE=%%F"
        goto :found_file
    )
    
    echo %ERROR% 在 saisdata 目录下未找到输入文件！
    pause
    exit /b 1
)

:found_file
echo %INFO% 使用输入文件: %INPUT_FILE%
echo %INFO% 检查模型文件...

:: 首先检查顶层 outputs 目录下的最佳模型
set "MODEL_PATH=outputs\best_model.pth"
if exist %MODEL_PATH% (
    echo %INFO% 在顶层目录找到模型文件: %MODEL_PATH%
    goto :found_model
)

:: 检查所有时间戳目录下的模型
echo %INFO% 在时间戳子目录中搜索最佳模型...
for /d %%D in (outputs\run_*) do (
    if exist "%%D\best_model.pth" (
        set "MODEL_PATH=%%D\best_model.pth"
        echo %INFO% 找到模型文件: !MODEL_PATH!
        goto :found_model
    )
    
    if exist "%%D\checkpoints\*.pth" (
        :: 找到第一个checkpoint文件
        for %%F in ("%%D\checkpoints\*.pth") do (
            set "MODEL_PATH=%%F"
            echo %INFO% 找到模型文件: !MODEL_PATH!
            goto :found_model
        )
    )
)

:: 如果还没找到，检查所有可能的.pth文件
echo %INFO% 未找到特定模型文件，搜索所有.pth文件...
for /f %%F in ('dir /b /s outputs\*.pth ^| sort') do (
    set "MODEL_PATH=%%F"
    echo %INFO% 使用找到的模型文件: !MODEL_PATH!
    goto :found_model
)

echo %ERROR% 无法找到任何模型文件！
pause
exit /b 1

:found_model
echo %INFO% 使用模型文件: %MODEL_PATH%

:: 检查conda环境
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% 未找到conda，请先安装Anaconda或Miniconda
    pause
    exit /b 1
)

:: 激活conda环境
echo %INFO% 激活conda环境...
call conda activate disprot
if %ERRORLEVEL% NEQ 0 (
    echo %WARNING% 激活conda环境失败，尝试继续执行...
)

:: 执行预测
echo %INFO% 开始执行预测...

python predict_for_submission.py ^
    --model_path "%MODEL_PATH%" ^
    --config_path "config.yaml" ^
    --input_dir "saisdata" ^
    --output_dir "saisresult" ^
    --input_file "!INPUT_FILE:saisdata\=!" ^
    --output_file "submit.csv"

:: 检查预测结果
if exist saisresult\submit.csv (
    echo %SUCCESS% 预测成功完成！结果已保存至 saisresult\submit.csv
    
    :: 显示结果文件前5行
    echo %INFO% 结果文件前5行:
    type saisresult\submit.csv | findstr /n . | findstr "^[1-5]:"
) else (
    echo %ERROR% 预测失败！未生成结果文件。
    pause
    exit /b 1
)

echo %INFO% 任务完成！
pause
exit /b 0
