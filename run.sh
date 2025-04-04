#!/bin/bash
set -e

# 定义颜色输出函数
function echo_info() {
    echo -e "\033[34m[INFO] $1\033[0m"
}

function echo_success() {
    echo -e "\033[32m[SUCCESS] $1\033[0m"
}

function echo_error() {
    echo -e "\033[31m[ERROR] $1\033[0m"
}

# 显示环境信息
echo_info "Python版本:"
python --version

echo_info "PyTorch版本:"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}')"

echo_info "CUDA设备检测:"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}, 设备数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# 判断数据目录是否存在
if [ ! -d "/saisdata" ]; then
    echo_error "输入数据目录 /saisdata 不存在!"
    exit 1
fi

# 确保输出目录存在
mkdir -p /saisresult

# 查找输入文件
INPUT_FILE=""
if [ -f "/saisdata/test.pkl" ]; then
    INPUT_FILE="/saisdata/test.pkl"
elif [ -f "/saisdata/test.csv" ]; then
    INPUT_FILE="/saisdata/test.csv"
else
    # 查找任何可能的输入文件
    FOUND_FILES=$(find /saisdata -type f | head -n 1)
    if [ -n "$FOUND_FILES" ]; then
        INPUT_FILE=$FOUND_FILES
    else
        echo_error "在 /saisdata 目录下未找到输入文件!"
        exit 1
    fi
fi

echo_info "使用输入文件: $INPUT_FILE"
echo_info "检查模型文件..."

# 检查模型文件
MODEL_PATH="/app/outputs/best_model.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo_info "未找到主模型文件，尝试查找其他模型文件"
    MODEL_FILES=$(find /app -name "*.pth" | sort)
    if [ -n "$MODEL_FILES" ]; then
        MODEL_PATH=$(echo "$MODEL_FILES" | head -n 1)
        echo_info "使用找到的模型文件: $MODEL_PATH"
    else
        echo_error "无法找到任何模型文件!"
        exit 1
    fi
fi

# 执行预测
echo_info "开始执行预测..."
python predict_for_submission.py \
    --model_path "$MODEL_PATH" \
    --config_path "/app/config.yaml" \
    --input_dir "/saisdata" \
    --output_dir "/saisresult" \
    --input_file "$(basename "$INPUT_FILE")" \
    --output_file "submit.csv"

# 检查预测结果
if [ -f "/saisresult/submit.csv" ]; then
    echo_success "预测成功完成! 结果已保存至 /saisresult/submit.csv"
    echo_info "结果文件前5行:"
    head -n 5 /saisresult/submit.csv
else
    echo_error "预测失败! 未生成结果文件。"
    exit 1
fi

exit 0 