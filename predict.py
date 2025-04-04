import argparse
import torch
import pickle
import numpy as np
from omegaconf import OmegaConf
from baseline import DisProtModel, restypes

def load_model(model_path, config_path):
    """
    加载训练好的模型
    
    参数:
        model_path: 模型权重文件路径
        config_path: 配置文件路径
    
    返回:
        加载好的模型
    """
    config = OmegaConf.load(config_path)
    model = DisProtModel(config.model)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_sequence(sequence):
    """
    预处理输入的蛋白质序列
    
    参数:
        sequence: 蛋白质序列（氨基酸单字母代码字符串）
    
    返回:
        处理后的序列张量
    """
    # 构建氨基酸映射字典
    residue_mapping = {'X': 20}
    residue_mapping.update(dict(zip(restypes, range(len(restypes)))))
    
    # 创建one-hot编码
    seq_tensor = torch.zeros(len(sequence), len(residue_mapping))
    for i, c in enumerate(sequence):
        if c not in restypes:
            c = 'X'
        seq_tensor[i][residue_mapping[c]] = 1
    
    return seq_tensor.unsqueeze(0)  # 添加批维度

def predict(model, sequence, device='cpu'):
    """
    使用模型预测蛋白质序列的无序区域
    
    参数:
        model: 加载的模型
        sequence: 输入的蛋白质序列
        device: 计算设备
    
    返回:
        预测结果
    """
    model = model.to(device)
    # 预处理序列
    seq_tensor = preprocess_sequence(sequence).to(device)
    
    # 预测
    with torch.no_grad():
        pred = model(seq_tensor)
    
    # 获取每个位置的预测标签（0表示有序，1表示无序）
    pred_labels = torch.argmax(pred, dim=-1).squeeze().cpu().numpy()
    
    return pred_labels

def visualize_prediction(sequence, prediction):
    """
    可视化预测结果
    
    参数:
        sequence: 输入序列
        prediction: 预测的标签
    """
    # 打印序列
    print("序列:")
    print(sequence)
    
    # 打印结构预测（O表示有序，D表示无序）
    prediction_str = ''.join(['D' if p == 1 else 'O' for p in prediction])
    print("结构预测 (O=有序, D=无序):")
    print(prediction_str)
    
    # 打印统计信息
    disorder_ratio = np.mean(prediction)
    print(f"无序区域比例: {disorder_ratio:.2f}")
    print(f"无序残基数量: {np.sum(prediction)} / {len(prediction)}")

def save_prediction(sequence, prediction, output_file):
    """
    保存预测结果到文件
    
    参数:
        sequence: 输入序列
        prediction: 预测的标签
        output_file: 输出文件路径
    """
    prediction_str = ''.join(['1' if p == 1 else '0' for p in prediction])
    with open(output_file, 'w') as f:
        f.write(">sequence\n")
        f.write(sequence + "\n")
        f.write(">prediction (0=ordered, 1=disordered)\n")
        f.write(prediction_str + "\n")
    
    print(f"预测结果已保存到: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='蛋白质无序区域预测')
    parser.add_argument('--model_path', required=True, help='模型权重文件路径')
    parser.add_argument('--config_path', default='./config.yaml', help='配置文件路径')
    parser.add_argument('--sequence', help='输入的蛋白质序列')
    parser.add_argument('--sequence_file', help='包含蛋白质序列的文件路径')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='计算设备')
    
    args = parser.parse_args()
    
    # 检查设备可用性
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU替代")
        device = 'cpu'
    
    # 加载模型
    model = load_model(args.model_path, args.config_path)
    
    # 获取序列
    sequence = args.sequence
    if args.sequence_file:
        with open(args.sequence_file, 'r') as f:
            # 简单的FASTA格式解析
            lines = f.readlines()
            sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
    
    if not sequence:
        parser.error("请提供蛋白质序列，使用--sequence或--sequence_file")
    
    # 进行预测
    prediction = predict(model, sequence, device)
    
    # 显示预测结果
    visualize_prediction(sequence, prediction)
    
    # 保存结果到文件
    if args.output:
        save_prediction(sequence, prediction, args.output) 