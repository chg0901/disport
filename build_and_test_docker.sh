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

# 显示脚本用途说明
echo_info "=== 无序蛋白质区域预测Docker镜像构建与测试脚本 ==="
echo_info "该脚本将构建Docker镜像并在本地测试其功能"
echo

# 确保当前目录是项目根目录
if [ ! -f "Dockerfile" ] || [ ! -f "predict_for_submission.py" ]; then
    echo_error "请在项目根目录运行此脚本"
    exit 1
fi

# 检查模型文件是否存在
MODEL_DIR="./outputs"
if [ ! -d "$MODEL_DIR" ]; then
    echo_info "创建模型目录: $MODEL_DIR"
    mkdir -p "$MODEL_DIR"
fi

# 检查是否有模型文件
MODEL_FILES=$(find $MODEL_DIR -name "*.pth" 2>/dev/null)
if [ -z "$MODEL_FILES" ]; then
    echo_error "错误: 在 $MODEL_DIR 目录下未找到模型权重文件 (*.pth)"
    echo_info "请先训练模型或将模型文件放入此目录"
    echo_info "例如: python baseline.py --config_path config.yaml --output_dir $MODEL_DIR"
    exit 1
else
    echo_info "发现模型文件:"
    echo "$MODEL_FILES"
fi

# 创建测试数据目录
TEST_DATA_DIR="./test_data"
TEST_OUTPUT_DIR="./test_output"

if [ ! -d "$TEST_DATA_DIR" ]; then
    echo_info "创建测试数据目录: $TEST_DATA_DIR"
    mkdir -p "$TEST_DATA_DIR"
fi

if [ ! -d "$TEST_OUTPUT_DIR" ]; then
    echo_info "创建测试输出目录: $TEST_OUTPUT_DIR"
    mkdir -p "$TEST_OUTPUT_DIR"
fi

# 检查测试数据
TEST_FILES=$(find $TEST_DATA_DIR -type f 2>/dev/null)
if [ -z "$TEST_FILES" ]; then
    echo_info "测试数据目录为空，创建示例测试数据"
    echo 'proteinID,sequence' > "$TEST_DATA_DIR/test_example.csv"
    echo 'test1,MEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPP' >> "$TEST_DATA_DIR/test_example.csv" 
    echo 'test2,MEEPQSDLWKLLPENNVLSPLPSQAVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQS' >> "$TEST_DATA_DIR/test_example.csv"
    echo_info "已创建示例测试数据: $TEST_DATA_DIR/test_example.csv"
fi

# 构建Docker镜像
echo_info "开始构建Docker镜像..."
IMAGE_NAME="disprot-prediction"
TAG="latest"

docker build -t ${IMAGE_NAME}:${TAG} .

if [ $? -ne 0 ]; then
    echo_error "Docker镜像构建失败"
    exit 1
else
    echo_success "Docker镜像构建成功: ${IMAGE_NAME}:${TAG}"
fi

# 运行Docker容器进行测试
echo_info "开始运行Docker容器进行测试..."
echo_info "挂载测试数据目录: $TEST_DATA_DIR -> /saisdata"
echo_info "挂载测试输出目录: $TEST_OUTPUT_DIR -> /saisresult"

# 检查是否支持GPU
if command -v nvidia-smi &> /dev/null && [ "$(nvidia-smi -L | grep GPU)" ]; then
    echo_info "检测到GPU，将启用GPU支持"
    DOCKER_CMD="docker run --rm --gpus all -v $(pwd)/$TEST_DATA_DIR:/saisdata -v $(pwd)/$TEST_OUTPUT_DIR:/saisresult ${IMAGE_NAME}:${TAG}"
else
    echo_info "未检测到GPU或nvidia-docker，将使用CPU模式运行"
    DOCKER_CMD="docker run --rm -v $(pwd)/$TEST_DATA_DIR:/saisdata -v $(pwd)/$TEST_OUTPUT_DIR:/saisresult ${IMAGE_NAME}:${TAG}"
fi

echo_info "执行命令: $DOCKER_CMD"
eval $DOCKER_CMD

# 检查测试结果
if [ -f "$TEST_OUTPUT_DIR/submit.csv" ]; then
    echo_success "测试成功! 输出文件已生成: $TEST_OUTPUT_DIR/submit.csv"
    
    # 显示结果预览
    echo_info "结果预览:"
    head -n 5 "$TEST_OUTPUT_DIR/submit.csv"
    
    echo_success "Docker镜像测试完成"
    echo_info "您可以使用以下命令提交Docker镜像:"
    echo "docker tag ${IMAGE_NAME}:${TAG} <registry>/<username>/${IMAGE_NAME}:${TAG}"
    echo "docker push <registry>/<username>/${IMAGE_NAME}:${TAG}"
else
    echo_error "测试失败! 未找到输出文件: $TEST_OUTPUT_DIR/submit.csv"
    echo_info "请检查Docker容器日志以获取更多信息"
fi 