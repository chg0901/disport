#!/bin/bash
# test-env.sh - Docker容器内环境测试脚本
# 用于验证容器内环境是否正确配置

set -e

# 定义颜色输出函数
echo_info() {
    echo -e "\033[0;34m[INFO] $1\033[0m"
}

echo_success() {
    echo -e "\033[0;32m[SUCCESS] $1\033[0m"
}

echo_error() {
    echo -e "\033[0;31m[ERROR] $1\033[0m"
}

echo_warning() {
    echo -e "\033[0;33m[WARNING] $1\033[0m"
}

# 输出分隔线
separator() {
    echo -e "\033[0;36m----------------------------------------\033[0m"
}

# 显示脚本说明
separator
echo_info "Docker环境测试脚本 - 验证容器内环境配置"
separator

# 检查Python版本
echo_info "检查Python版本..."
python --version
separator

# 检查CUDA可用性
echo_info "检查CUDA环境..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA是否可用: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo_info "CUDA设备数量: $(python -c "import torch; print(torch.cuda.device_count())")"
    echo_info "当前CUDA设备: $(python -c "import torch; print(torch.cuda.get_device_name(0))")"
    echo_success "CUDA环境正常"
else
    echo_warning "CUDA不可用，将使用CPU模式运行"
fi
separator

# 检查关键依赖包
echo_info "检查关键依赖包..."
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import omegaconf; print(f'OmegaConf版本: {omegaconf.__version__}')"
python -c "import sklearn; print(f'Scikit-learn版本: {sklearn.__version__}')"
python -c "import matplotlib; print(f'Matplotlib版本: {matplotlib.__version__}')"
separator

# 列出所有已安装的pip包
echo_info "已安装的Python包列表:"
pip list
separator

# 检查模型文件
echo_info "检查模型文件..."
find /app -name "*.pth" | while read -r model_file; do
    echo_success "发现模型文件: $model_file"
done

if ! find /app -name "*.pth" | grep -q .; then
    echo_warning "未找到模型文件 (.pth)"
fi
separator

# 检查工作目录结构
echo_info "检查工作目录结构..."
ls -la /app
separator

# 检查数据目录和输出目录
echo_info "检查数据目录和输出目录..."
if [ -d "/saisdata" ]; then
    echo_success "数据目录(/saisdata)存在"
    echo_info "数据目录内容:"
    ls -la /saisdata
else
    echo_error "数据目录(/saisdata)不存在"
fi

if [ -d "/saisresult" ]; then
    echo_success "输出目录(/saisresult)存在"
    echo_info "输出目录内容:"
    ls -la /saisresult
else
    echo_error "输出目录(/saisresult)不存在"
fi
separator

# 检查run.sh脚本
echo_info "检查run.sh脚本..."
if [ -f "/app/run.sh" ]; then
    echo_success "run.sh脚本存在"
    echo_info "run.sh内容:"
    head -n 20 /app/run.sh
    echo "..."
    tail -n 5 /app/run.sh
else
    echo_error "run.sh脚本不存在"
fi
separator

# 检查predict_for_submission.py脚本
echo_info "检查predict_for_submission.py脚本..."
if [ -f "/app/predict_for_submission.py" ]; then
    echo_success "predict_for_submission.py脚本存在"
else
    echo_error "predict_for_submission.py脚本不存在"
fi
separator

# 测试模型加载
echo_info "测试模型加载..."
MODEL_PATH=$(find /app -name "*.pth" | head -n 1)
if [ -n "$MODEL_PATH" ]; then
    echo_info "使用模型文件: $MODEL_PATH"
    python -c "
import torch
from baseline import DisProtModel
from omegaconf import OmegaConf

try:
    config = OmegaConf.load('/app/config.yaml')
    model = DisProtModel(config.model)
    model.load_state_dict(torch.load('$MODEL_PATH', map_location=torch.device('cpu')))
    print('模型加载成功!')
except Exception as e:
    print(f'模型加载失败: {str(e)}')
    exit(1)
"
    if [ $? -eq 0 ]; then
        echo_success "模型加载测试通过"
    else
        echo_error "模型加载测试失败"
    fi
else
    echo_warning "未找到模型文件，跳过模型加载测试"
fi
separator

echo_success "环境测试完成!" 