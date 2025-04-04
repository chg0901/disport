@echo off
setlocal enabledelayedexpansion

:: 设置颜色代码
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "ERROR=[ERROR]"
set "WARNING=[WARNING]"

echo ====================================================
echo              NVIDIA GPU 检查工具
echo ====================================================

:: 检查nvidia-smi是否可用
echo %INFO% 检查NVIDIA驱动和GPU...
where nvidia-smi >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% 未找到nvidia-smi工具，可能未安装NVIDIA驱动或GPU
    echo %INFO% 请前往 https://www.nvidia.com/Download/index.aspx 下载并安装最新驱动
    goto :eof
)

:: 运行nvidia-smi检查GPU
echo.
echo %INFO% NVIDIA GPU信息:
nvidia-smi
if %ERRORLEVEL% NEQ 0 (
    echo %ERROR% 运行nvidia-smi失败，请检查驱动安装
    goto :eof
)

:: 检查当前驱动版本是否支持CUDA 12
echo.
for /f "tokens=3" %%a in ('nvidia-smi ^| findstr "CUDA Version"') do (
    set cuda_version=%%a
)

if defined cuda_version (
    echo %INFO% 检测到CUDA驱动版本: !cuda_version!
    
    for /f "tokens=1 delims=." %%a in ("!cuda_version!") do (
        set cuda_major=%%a
    )
    
    if !cuda_major! GEQ 12 (
        echo %SUCCESS% CUDA版本支持PyTorch 2.4.0和CUDA 12.1
    ) else (
        echo %WARNING% CUDA驱动版本(!cuda_version!)可能不支持CUDA 12.1
        echo %INFO% 建议更新NVIDIA驱动到最新版本
    )
) else (
    echo %WARNING% 无法检测CUDA版本
)

:: 检查Visual C++ Redistributable
echo.
echo %INFO% 检查Visual C++ Redistributable...
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo %SUCCESS% Visual C++ Redistributable已安装
) else (
    echo %WARNING% 未检测到Visual C++ Redistributable 2019/2022
    echo %INFO% 请下载并安装Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
)

echo.
echo %INFO% 硬件检查完成
echo %INFO% 如果您的GPU和驱动正常，请运行pytorch_install.bat安装PyTorch环境
echo ====================================================

endlocal 