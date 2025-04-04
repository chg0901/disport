FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置pip源为国内源并升级pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --no-cache-dir --upgrade pip

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p /saisdata /saisresult

# 为模型目录创建软链接
RUN mkdir -p /app/outputs && \
    ln -sf /app/outputs /app/checkpoints

# 确保run.sh脚本权限正确
RUN chmod +x /app/run.sh

# 添加环境测试脚本
COPY test-env.sh /app/
RUN chmod +x /app/test-env.sh

# 添加健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import os, torch; exit(0 if os.path.exists('/app') and torch.__version__ >= '2.4.0' else 1)"

# 设置入口点
ENTRYPOINT ["/app/run.sh"] 