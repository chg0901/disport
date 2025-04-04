# 无序蛋白质区域预测模型 (DisProt Prediction Model)

[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn)

这是一个基于Transformer架构的无序蛋白质区域（Intrinsically Disordered Regions, IDRs）预测模型。该模型使用蛋白质序列作为输入，预测每个氨基酸位置是有序还是无序。本项目作为上海科技智能大赛的参赛作品，旨在提高无序蛋白质区域预测的准确性。

## 比赛背景

无序蛋白质区域在蛋白质功能中扮演重要角色，包括信号传导、转录调控和蛋白质相互作用等。与结构稳定的有序区域不同，IDRs在生理条件下没有稳定的三维结构，这种内在的柔性使它们能够与多种目标相互作用，在细胞功能中发挥关键作用。准确预测蛋白质序列中的IDRs对理解蛋白质功能和疾病机制具有重要意义。

### 赛题描述

本次比赛任务是开发算法，通过氨基酸序列准确预测蛋白质中的无序区域。参赛者需要:
1. 分析蛋白质序列特征
2. 构建预测模型
3. 精确识别无序区域

### 比赛日程

- **报名组队阶段**: 2025.2.23~2025.5月上旬
- **初赛阶段**: 2025.3.24~2025.5月上旬
- **复赛阶段**: 2025.5月中旬~2025.6月中旬
- **决赛**: 2025年7月

### 评价指标

本次任务采用实验真实结果与预测结果的F1 score进行评测：

- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 score = (2 × Precision × Recall) / (Precision + Recall)

其中TP、FP、FN分别代表真阳性、假阳性和假阴性预测结果。

### 奖项设置

本次大赛总奖金池为100万元人民币，本赛道奖金池为20万元人民币，具体如下：

- **冠军**: 1支队伍，奖金10万元人民币，颁发获奖证书
- **亚军**: 2支队伍，奖金3万元人民币，颁发获奖证书
- **季军**: 3支队伍，奖金1万元人民币，颁发获奖证书
- **优胜奖**: 4支队伍，奖金2500元，颁发获奖证书

## 项目结构

- `baseline.py`: 主要代码文件，包含数据处理、模型定义和训练逻辑
- `config.yaml`: 配置文件，包含模型参数和训练设置
- `requirements.txt`: 项目依赖包列表
- `conda_environment.yml`: Conda环境配置文件
- `run.sh`: Linux/Mac系统的一键运行脚本
- `run.bat`: Windows系统的一键运行脚本
- `predict.py`: 使用训练好的模型进行预测的脚本
- `predict_for_submission.py`: 专门用于比赛提交的预测脚本
- `train_with_lightning.py`: 使用PyTorch Lightning框架的增强版训练脚本
- `Dockerfile`: 用于构建比赛提交的Docker镜像
- `docker-compose.yml`: 本地开发测试Docker镜像的配置文件
- `build_and_test_docker.sh`: 构建和测试Docker镜像的辅助脚本
- `test_data/`: 测试数据目录
- `image_test.py`: 测试PyTorch和CUDA环境的脚本
- `test-env.sh`: Docker容器内环境测试脚本
- `WSAA_data_public.pkl`: 数据文件（需自行准备）

## 环境配置

### 系统要求

- **操作系统**: Linux（推荐Ubuntu 20.04或更高版本）、Windows 10/11、macOS
- **Python版本**: 3.10或更高版本
- **GPU支持**: 支持CUDA 12.1的NVIDIA GPU（推荐RTX系列）
- **内存**: 至少8GB RAM（推荐16GB或更多）
- **存储**: 至少10GB可用空间

### 方法一：使用一键运行脚本（推荐）

根据您的操作系统选择对应的脚本：

```bash
# Linux/Mac
chmod +x run.sh
./run.sh

# Windows
run.bat
```

脚本会自动检查环境并安装所需依赖。

### 方法二：手动配置

1. 克隆仓库后，首先创建并激活一个虚拟环境（推荐使用conda或venv, **以下三种方法选择其一即可**）:

```bash
# 使用conda
conda create -n disprot python=3.10 -y
conda activate disprot

# 或者使用conda环境文件
conda env create -f conda_environment.yml
conda activate disprot

# 或使用venv
python -m venv disprot
source disprot/bin/activate  # Linux/Mac
# 或者
disprot\Scripts\activate  # Windows
```

2. 如果使用GPU训练，需要确保已安装CUDA（建议CUDA 12.1版本）并安装对应版本的PyTorch:

```bash
# 对于CUDA 12.1的示例（已在requirements.txt中指定）
# CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# with conda
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

```


3. 安装其他依赖包:

```bash
pip install -r requirements.txt
```


4. 验证环境配置:

```bash
# 运行环境测试脚本
python image_test.py
```

## 数据准备

将数据文件`WSAA_data_public.pkl`放在项目根目录下，或者在`config.yaml`中修改`data_path`指向正确的数据文件位置。

数据文件应是一个Python pickle文件，其中包含一个字典列表，每个字典具有以下结构:
```python
{
    'sequence': 'MVKPKRPRSAYN...', # 蛋白质序列（氨基酸单字母代码）
    'label': '00000111100...'     # 对应标签，0表示有序，1表示无序
}
```

## 模型训练

### 基本训练

通过以下命令开始基本训练:

```bash
python baseline.py --config_path config.yaml --output_dir ./outputs
```

### 使用PyTorch Lightning训练（推荐）

使用PyTorch Lightning可以获得更好的训练管理、日志记录和模型检查点功能:

```bash
python train_with_lightning.py --config_path config.yaml --early_stopping --patience 10 --output_dir ./outputs
```

PyTorch Lightning版本提供的额外功能：
- 自动模型检查点保存
- 早停机制防止过拟合
- TensorBoard集成用于可视化训练过程
- 更高效的多GPU训练支持
- 学习率调度器自动管理

命令行参数:
- `--max_epochs`: 最大训练轮数
- `--gpus`: 使用的GPU数量
- `--early_stopping`: 启用早停机制
- `--patience`: 早停耐心值（几个epoch无改善后停止）
- `--output_dir`: 输出目录

### 训练可视化 - SwanLab支持

本项目集成了SwanLab，可以轻松可视化训练过程和实验结果：

```bash
# 使用SwanLab进行训练跟踪
python baseline.py --config_path config.yaml --output_dir ./outputs --use_swanlab

# 或者使用PyTorch Lightning与SwanLab集成
python train_with_lightning.py --config_path config.yaml --output_dir ./outputs --use_swanlab
```

如果您是第一次使用SwanLab，需要先注册账号并登录：

```bash
# 登录SwanLab
swanlab login
```

您也可以尝试运行SwanLab示例脚本：

```bash
python examples/swanlab_example.py
```

更多关于SwanLab的信息，请访问[SwanLab官方文档](https://github.com/SwanHubX/SwanLab)。

## 使用模型进行预测

训练完成后，可以使用`predict.py`脚本对新的蛋白质序列进行预测:

```bash
# 直接输入序列
python predict.py --model_path ./outputs/best_model.pth --sequence "MVKPKRPRSAYNIYVSESFQEAKDDSAQGK"

# 或从文件读取序列（支持FASTA格式）
python predict.py --model_path ./outputs/best_model.pth --sequence_file ./your_protein.fasta --output ./prediction_result.txt
```

命令行参数:
- `--model_path`: 训练好的模型权重路径（必需）
- `--config_path`: 配置文件路径（默认为'./config.yaml'）
- `--sequence`: 直接输入的蛋白质序列
- `--sequence_file`: 包含蛋白质序列的文件路径
- `--output`: 保存预测结果的文件路径
- `--device`: 计算设备，'cpu'或'cuda'（默认为'cpu'）

## 比赛提交

### 提交格式要求

测试集不公开发布，选手需要将模型代码、权重等文件通过docker镜像上传至服务器进行评测。

提交的镜像要求：
1. 将执行脚本`run.sh`放在`/app`目录
2. 将输出结果放在`/saisresult`目录，结果文件名统一命名为`submit.csv`
3. 在`/saisdata`目录读取输入数据集

### 预测结果格式

提交的预测结果文件`submit.csv`需要遵循下述格式：
- 第一行为标题行：`proteinID,sequence,IDRs`
- 从第二行开始，每行输出具体蛋白质ID、序列，以及每个残基是否为IDR的分类，用逗号分隔
- IDR的预测值应以1|0的字符串表示形式提供

示例：
```
proteinID,sequence,IDRs
disprot1,XXXXXX,110011
disprot2,XXXXXX,100001
```

### 使用Docker进行测试和提交

本项目提供了完整的Docker支持，包括Dockerfile和辅助脚本，以便快速构建和测试您的提交。

#### 自动构建和测试

使用提供的`build_and_test_docker.sh`脚本可以一键构建和测试Docker镜像：

```bash
# 赋予脚本执行权限
chmod +x build_and_test_docker.sh

# 运行脚本
./build_and_test_docker.sh
```

该脚本会：
1. 检查模型文件是否存在
2. 构建Docker镜像
3. 运行Docker容器进行测试
4. 验证输出结果

#### 手动构建和测试

如果您需要更多控制，可以手动执行以下步骤：

1. 确保模型已训练完成，模型权重保存在`outputs/best_model.pth`
2. 构建Docker镜像：
```bash
docker build -t disprot-prediction:latest .
```

3. 使用docker-compose进行测试：
```bash
docker-compose up
```

4. 或者直接使用docker命令测试：
```bash
# 使用GPU
docker run --rm --gpus all -v $(pwd)/test_data:/saisdata -v $(pwd)/test_output:/saisresult disprot-prediction:latest

# 不使用GPU
docker run --rm -v $(pwd)/test_data:/saisdata -v $(pwd)/test_output:/saisresult disprot-prediction:latest
```

5. 测试环境配置（可选）：
```bash
# 使用环境测试脚本
docker run --rm -v $(pwd)/test_data:/saisdata -v $(pwd)/test_output:/saisresult disprot-prediction:latest /app/test-env.sh
```

### 提交Docker镜像

完成测试后，按照比赛要求提交Docker镜像：

1. 为镜像打标签：
```bash
docker tag disprot-prediction:latest [registry]/[username]/disprot-prediction:latest
```

2. 上传镜像：
```bash
docker push [registry]/[username]/disprot-prediction:latest
```

### Docker镜像最佳实践

根据天池比赛经验，以下做法可以提高Docker镜像的稳定性和性能：

1. **使用官方镜像作为基础**：本项目使用`pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime`作为基础镜像，确保PyTorch和CUDA环境稳定可靠
2. **精简依赖**：只安装必要的依赖包，减小镜像体积
3. **健壮的错误处理**：代码中添加了完善的异常处理机制，确保预测过程不会因单个样本错误而中断
4. **日志记录**：使用Python的logging模块记录详细日志，便于调试
5. **灵活的输入处理**：支持多种输入格式（pickle、CSV、文本文件）
6. **批处理超长序列**：对于超长蛋白质序列，实现了分批处理机制
7. **健康检查**：Dockerfile中添加了HEALTHCHECK指令，便于监控容器状态
8. **入口点脚本**：优化的run.sh脚本具有良好的错误处理和日志输出
9. **环境测试工具**：提供测试脚本验证Docker容器内环境配置

## 配置说明

在`config.yaml`中，您可以调整以下参数:

- `model`: 模型架构相关参数
  - `i_dim`: 输入维度（氨基酸one-hot编码的维度）
  - `o_dim`: 输出维度（通常为2，表示二分类任务）
  - `d_model`: 模型的隐藏层维度
  - `n_head`: Transformer注意力头数量
  - `n_layer`: Transformer层数

- `train`: 训练相关参数
  - `epochs`: 训练轮数
  - `dataloader`: 数据加载器设置
    - `batch_size`: 批大小
    - `shuffle`: 是否打乱数据
    - `num_workers`: 数据加载的并行工作进程数
    - `drop_last`: 是否丢弃不完整的最后一批数据
  - `optimizer`: 优化器设置
    - `lr`: 学习率
    - `weight_decay`: 权重衰减系数（L2正则化）

## 性能优化建议

如需进一步提升模型性能，可以尝试以下方法：

1. **增加模型复杂度**:
   - 增大 `d_model` 值（如128、256）
   - 增加 `n_head` 和 `n_layer` 参数

2. **数据增强**:
   - 对序列进行随机裁剪
   - 在序列中随机掩码某些位置
   - 引入蛋白质家族相关信息

3. **特征工程**:
   - 引入氨基酸理化性质特征
   - 添加保守性信息
   - 集成进化信息如PSSM矩阵

4. **超参数调优**:
   - 尝试不同的学习率
   - 使用学习率预热和余弦退火调度
   - 实验不同的批大小

5. **集成学习**:
   - 训练多个模型并集成他们的预测结果
   - 使用不同随机种子训练相同架构的模型

6. **利用现代硬件优势**:
   - 使用PyTorch 2.4的编译器功能提升推理性能
   - 充分利用CUDA 12.1的新特性提高GPU利用率
   - 采用半精度训练（FP16）减少显存使用并加快训练速度

## 结果评估

模型使用F1分数作为主要评估指标。训练过程中会在每个epoch后在验证集上计算并输出F1分数。

在预测时，`predict.py`脚本会提供以下输出：
- 预测的有序/无序状态（'O'表示有序，'D'表示无序）
- 无序区域的比例和数量统计
- 可选的预测结果保存到文件

### 性能基准

以下是在不同硬件和配置下的性能基准：

| 硬件配置 | PyTorch版本 | CUDA版本 | 训练时间(20轮) | 推理速度(序列/秒) |
|---------|------------|---------|--------------|-----------------|
| RTX 4090 | 2.4.0 | 12.1 | 约10分钟 | ~500 |
| RTX 3080 | 2.4.0 | 12.1 | 约15分钟 | ~300 |
| CPU (8核) | 2.4.0 | N/A | 约2小时 | ~50 |

## 常见问题解答（FAQ）

### 训练相关问题

**Q: 如何处理训练过程中的"CUDA out of memory"错误？**

A: 尝试减小批大小（batch_size）或使用梯度累积。您也可以在`config.yaml`中降低模型的`d_model`参数值。

**Q: 如何在多GPU上训练？**

A: 使用PyTorch Lightning版本的训练脚本，并设置`--gpus`参数：
```bash
python train_with_lightning.py --config_path config.yaml --gpus 2 --output_dir ./outputs
```

### Docker相关问题

**Q: Docker构建失败，显示"no matching manifest for linux/amd64 in the manifest list"**

A: 这是因为镜像不支持您的系统架构。尝试使用兼容您架构的基础镜像，或使用BuildKit构建：
```bash
DOCKER_BUILDKIT=1 docker build -t disprot-prediction:latest .
```

**Q: 在Windows上运行Docker时出现路径问题**

A: 请确保使用正确的路径格式，在Windows上使用PowerShell时，使用以下命令：
```powershell
docker run --rm -v ${PWD}/test_data:/saisdata -v ${PWD}/test_output:/saisresult disprot-prediction:latest
```

## 组织机构

- **指导单位**: 上海市科学技术委员会、上海市发展和改革委员会、上海市经济和信息化委员会、上海市教育委员会
- **主办方**: 上海科学智能研究院、复旦大学
- **协办单位**: 阿里云计算有限公司、上海飞机设计研究院、中国南方电网电力调度控制中心、上海市漕河泾新兴技术开发区发展总公司、上海复星医药（集团）股份有限公司、晶泰科技、艾昆纬企业管理咨询（上海）有限公司
- **战略媒体伙伴**: ScienceAI、集智俱乐部、InfoQ
- **办赛平台**: 上智院平台

## 引用

如果您在研究中使用了本项目，请考虑以下列格式引用：

```
@software{DisProt_Prediction_Model,
  author = {Your Name},
  title = {DisProt Prediction Model: A Transformer-based Approach for Predicting Intrinsically Disordered Regions in Proteins},
  year = {2025},
  url = {https://github.com/your-username/disprot-prediction}
}
```

## 许可证

本项目采用MIT许可证 - 详见 LICENSE 文件 