@echo off
setlocal enabledelayedexpansion

:: 设置颜色代码
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "ERROR=[ERROR]"
set "WARNING=[WARNING]"

echo ====================================================
echo           PyTorch DLL 依赖修复工具
echo ====================================================

:: 确保在disprot环境中运行
echo %INFO% 检查conda环境...
conda info --envs | findstr "*" | findstr "disprot" >nul
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% 请先激活disprot环境后运行此脚本
    echo %INFO% 运行: conda activate disprot
    goto :eof
)

:: 获取conda环境路径
for /f "tokens=*" %%a in ('conda info --base') do set CONDA_BASE=%%a
set CONDA_ENV=%CONDA_BASE%\envs\disprot

echo %INFO% conda环境路径: %CONDA_ENV%

:: 下载Visual C++ Redistributable
echo %INFO% 正在下载Visual C++ Redistributable 2022...
curl -L -o %TEMP%\vc_redist.x64.exe https://aka.ms/vs/17/release/vc_redist.x64.exe
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% 下载Visual C++ Redistributable失败
    echo %INFO% 请手动下载并安装: https://aka.ms/vs/17/release/vc_redist.x64.exe
) else (
    echo %INFO% 安装Visual C++ Redistributable 2022...
    %TEMP%\vc_redist.x64.exe /install /quiet /norestart
    if %ERRORLEVEL% NEQ 0 (
        echo %WARNING% 安装可能未成功，请手动安装
    ) else (
        echo %SUCCESS% Visual C++ Redistributable安装成功
    )
)

:: 重新安装PyTorch
echo.
echo %INFO% 正在重新安装PyTorch 2.4.0...
pip uninstall -y torch torchvision torchaudio
if %ERRORLEVEL% NEQ 0 (
    echo %WARNING% 卸载PyTorch失败，继续安装
)

echo %INFO% 使用pip安装PyTorch (CPU和CUDA)...
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu121
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% 安装PyTorch失败
    echo %INFO% 尝试仅安装CPU版本...
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url https://download.pytorch.org/whl/cpu
    if %ERRORLEVEL% NEQ 0 (
        echo %ERROR% 安装PyTorch (CPU版本)失败
        goto :eof
    ) else (
        echo %SUCCESS% 安装PyTorch (CPU版本)成功
    )
) else (
    echo %SUCCESS% 安装PyTorch (CUDA版本)成功
)

:: 复制系统DLL文件到PyTorch目录
echo.
echo %INFO% 检查系统DLL文件...

:: 获取PyTorch lib目录
for /f "tokens=*" %%a in ('python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"') do (
    set TORCH_LIB=%%a
)

if not defined TORCH_LIB (
    echo %ERROR% 无法获取PyTorch lib目录路径
    goto :eof
)

echo %INFO% PyTorch lib目录: %TORCH_LIB%

:: 创建备份目录
set BACKUP_DIR=%TORCH_LIB%\backup
mkdir "%BACKUP_DIR%" 2>nul

:: 复制msvcp140.dll和vcruntime140.dll
set SYSTEM32_DIR=C:\Windows\System32
set SYS_DLLS=msvcp140.dll vcruntime140.dll vcruntime140_1.dll

for %%d in (%SYS_DLLS%) do (
    if exist "%SYSTEM32_DIR%\%%d" (
        echo %INFO% 发现系统DLL: %%d
        if exist "%TORCH_LIB%\%%d" (
            echo %INFO% 备份原有DLL: %%d
            copy "%TORCH_LIB%\%%d" "%BACKUP_DIR%\%%d.bak" >nul
        )
        echo %INFO% 复制系统DLL: %%d 到 PyTorch lib目录
        copy "%SYSTEM32_DIR%\%%d" "%TORCH_LIB%\%%d" >nul
        if %ERRORLEVEL% NEQ 0 (
            echo %ERROR% 复制 %%d 失败
        ) else (
            echo %SUCCESS% 已复制 %%d
        )
    ) else (
        echo %WARNING% 系统中没有找到 %%d
    )
)

:: 检查PATH环境变量
echo.
echo %INFO% 检查PATH环境变量...
set USER_PATH=
for /f "tokens=2*" %%a in ('reg query HKCU\Environment /v PATH') do set USER_PATH=%%b

echo %SYSTEM32_DIR% | findstr /C:"%USER_PATH%" >nul
if %ERRORLEVEL% NEQ 0 (
    echo %WARNING% System32目录不在PATH中，尝试添加...
    setx PATH "%USER_PATH%;%SYSTEM32_DIR%"
    if %ERRORLEVEL% NEQ 0 (
        echo %ERROR% 更新PATH环境变量失败，请手动添加 %SYSTEM32_DIR% 到PATH
    ) else (
        echo %SUCCESS% 已将System32目录添加到PATH
        echo %INFO% 请重新启动命令提示符以应用更改
    )
) else (
    echo %SUCCESS% System32目录已在PATH中
)

:: 测试PyTorch安装
echo.
echo %INFO% 测试PyTorch安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% PyTorch导入测试失败
    echo %INFO% 请尝试重启命令提示符并再次测试
) else (
    echo %SUCCESS% PyTorch导入测试成功!
    
    python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
    if %ERRORLEVEL% NEQ 0 (
        echo %WARNING% CUDA测试失败，但PyTorch基本功能可用
    )
)

echo.
echo %INFO% 修复过程完成
echo %INFO% 如果仍然无法导入PyTorch，请尝试以下操作:
echo       1. 重启命令提示符/PowerShell
echo       2. 重新运行 nvidia_gpu_check.bat 检查GPU状态
echo       3. 安装最新NVIDIA驱动
echo       4. 安装Visual Studio 2019或2022的Visual C++ Redistributable
echo ====================================================

endlocal 