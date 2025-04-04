#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试PyTorch和CUDA环境配置
用于验证Docker镜像中的PyTorch和CUDA版本是否符合要求
"""

import os
import sys
import torch
import platform
import numpy as np
from packaging import version

# 定义颜色输出函数
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_info(message):
    print(f"{Colors.BLUE}[INFO] {message}{Colors.ENDC}")

def print_success(message):
    print(f"{Colors.GREEN}[SUCCESS] {message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.YELLOW}[WARNING] {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.RED}[ERROR] {message}{Colors.ENDC}")

def print_section(title):
    print(f"\n{Colors.BOLD}{title}{Colors.ENDC}")
    print("-" * 50)

def check_torch_version():
    """检查PyTorch版本是否满足要求"""
    print_section("PyTorch版本检查")
    
    required_version = "2.4.0"
    current_version = torch.__version__
    
    print_info(f"当前PyTorch版本: {current_version}")
    print_info(f"要求PyTorch版本: >={required_version}")
    
    if version.parse(current_version) >= version.parse(required_version):
        print_success(f"PyTorch版本检查通过: {current_version} >= {required_version}")
        return True
    else:
        print_error(f"PyTorch版本不满足要求: {current_version} < {required_version}")
        return False

def check_cuda_version():
    """检查CUDA版本是否满足要求"""
    print_section("CUDA版本检查")
    
    if not torch.cuda.is_available():
        print_warning("CUDA不可用，将使用CPU模式")
        return False
    
    cuda_version = torch.version.cuda
    required_version = "12.0"
    
    print_info(f"当前CUDA版本: {cuda_version}")
    print_info(f"要求CUDA版本: >={required_version}")
    
    if version.parse(cuda_version) >= version.parse(required_version):
        print_success(f"CUDA版本检查通过: {cuda_version} >= {required_version}")
        print_info(f"CUDA设备数量: {torch.cuda.device_count()}")
        
        # 打印所有可用的GPU设备信息
        for i in range(torch.cuda.device_count()):
            print_info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        return True
    else:
        print_error(f"CUDA版本不满足要求: {cuda_version} < {required_version}")
        return False

def check_numpy_version():
    """检查NumPy版本"""
    print_section("NumPy版本检查")
    
    current_version = np.__version__
    required_version = "1.25.0"
    
    print_info(f"当前NumPy版本: {current_version}")
    print_info(f"要求NumPy版本: >={required_version}")
    
    if version.parse(current_version) >= version.parse(required_version):
        print_success(f"NumPy版本检查通过: {current_version} >= {required_version}")
        return True
    else:
        print_warning(f"NumPy版本较低: {current_version} < {required_version}")
        return False

def check_system_info():
    """检查系统信息"""
    print_section("系统信息")
    
    print_info(f"Python版本: {platform.python_version()}")
    print_info(f"操作系统: {platform.system()} {platform.release()}")
    
    # 检查是否在Docker容器中运行
    in_docker = os.path.exists('/.dockerenv')
    if in_docker:
        print_info("当前运行在Docker容器中")
    else:
        print_info("当前不在Docker容器中")

def test_torch_functionality():
    """测试PyTorch基本功能"""
    print_section("PyTorch功能测试")
    
    try:
        # 创建一个小型张量并执行基本操作
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        
        # 测试加法
        z = x + y
        print_info(f"张量加法: \n{x} + \n{y} = \n{z}")
        
        # 测试矩阵乘法
        w = torch.matmul(x, y.t())
        print_info(f"矩阵乘法结果: \n{w}")
        
        # 如果CUDA可用，还测试GPU操作
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = x_gpu + y_gpu
            print_info("GPU张量操作成功")
            
        print_success("PyTorch基本功能测试通过")
        return True
    except Exception as e:
        print_error(f"PyTorch功能测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    print_section("PyTorch和CUDA环境检查")
    
    check_system_info()
    
    torch_version_ok = check_torch_version()
    cuda_version_ok = check_cuda_version()
    numpy_version_ok = check_numpy_version()
    functionality_ok = test_torch_functionality()
    
    # 总结结果
    print_section("测试结果汇总")
    
    if torch_version_ok:
        print_success("✓ PyTorch版本符合要求")
    else:
        print_error("✗ PyTorch版本不符合要求")
    
    if cuda_version_ok:
        print_success("✓ CUDA版本符合要求")
    elif not torch.cuda.is_available():
        print_warning("⚠ CUDA不可用")
    else:
        print_error("✗ CUDA版本不符合要求")
    
    if numpy_version_ok:
        print_success("✓ NumPy版本符合要求")
    else:
        print_warning("⚠ NumPy版本较低")
    
    if functionality_ok:
        print_success("✓ PyTorch功能测试通过")
    else:
        print_error("✗ PyTorch功能测试失败")
    
    # 总体评估
    if all([torch_version_ok, functionality_ok]) and (cuda_version_ok or not torch.cuda.is_available()):
        print_success("✅ 环境检查通过，所有关键组件满足要求")
        return 0
    else:
        print_error("❌ 环境检查未通过，请解决上述问题")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 




########################################################################################
############################  Windows CPU 测试
########################################################################################

# PyTorch和CUDA环境检查
# --------------------------------------------------

# 系统信息
# --------------------------------------------------
# [INFO] Python版本: 3.10.16
# [INFO] 操作系统: Windows 10
# [INFO] 当前不在Docker容器中

# PyTorch版本检查
# --------------------------------------------------
# [INFO] 当前PyTorch版本: 2.4.0+cpu
# [INFO] 要求PyTorch版本: >=2.4.0
# [SUCCESS] PyTorch版本检查通过: 2.4.0+cpu >= 2.4.0

# CUDA版本检查
# --------------------------------------------------
# [WARNING] CUDA不可用，将使用CPU模式

# NumPy版本检查
# --------------------------------------------------
# [INFO] 当前NumPy版本: 1.25.0
# [INFO] 要求NumPy版本: >=1.25.0
# [SUCCESS] NumPy版本检查通过: 1.25.0 >= 1.25.0

# PyTorch功能测试
# --------------------------------------------------
# [INFO] 张量加法: 
# tensor([[1., 2.],
#         [3., 4.]]) +
# tensor([[5., 6.],
#         [7., 8.]]) =
# tensor([[ 6.,  8.],
#         [10., 12.]])
# [INFO] 矩阵乘法结果:
# tensor([[17., 23.],
#         [39., 53.]])
# [SUCCESS] PyTorch基本功能测试通过

# 测试结果汇总
# --------------------------------------------------
# [SUCCESS] ✓ PyTorch版本符合要求
# [WARNING] ⚠ CUDA不可用
# [SUCCESS] ✓ NumPy版本符合要求
# [SUCCESS] ✓ PyTorch功能测试通过
# [SUCCESS] ✅ 环境检查通过，所有关键组件满足要求









########################################################################################
############################  Linux GPU 测试
########################################################################################

# PyTorch和CUDA环境检查
# --------------------------------------------------

# 系统信息
# --------------------------------------------------
# [INFO] Python版本: 3.10.16
# [INFO] 操作系统: Linux 5.15.0-134-generic
# [INFO] 当前不在Docker容器中

# PyTorch版本检查
# --------------------------------------------------
# [INFO] 当前PyTorch版本: 2.4.0+cu121
# [INFO] 要求PyTorch版本: >=2.4.0
# [SUCCESS] PyTorch版本检查通过: 2.4.0+cu121 >= 2.4.0

# CUDA版本检查
# --------------------------------------------------
# [INFO] 当前CUDA版本: 12.1
# [INFO] 要求CUDA版本: >=12.0
# [SUCCESS] CUDA版本检查通过: 12.1 >= 12.0
# [INFO] CUDA设备数量: 4
# [INFO] GPU 0: NVIDIA RTX A6000
# [INFO] GPU 1: NVIDIA RTX A6000
# [INFO] GPU 2: NVIDIA RTX A6000
# [INFO] GPU 3: NVIDIA RTX A6000

# NumPy版本检查
# --------------------------------------------------
# [INFO] 当前NumPy版本: 1.25.0
# [INFO] 要求NumPy版本: >=1.25.0
# [SUCCESS] NumPy版本检查通过: 1.25.0 >= 1.25.0

# PyTorch功能测试
# --------------------------------------------------
# [INFO] 张量加法: 
# tensor([[1., 2.],
#         [3., 4.]]) + 
# tensor([[5., 6.],
#         [7., 8.]]) = 
# tensor([[ 6.,  8.],
#         [10., 12.]])
# [INFO] 矩阵乘法结果: 
# tensor([[17., 23.],
#         [39., 53.]])
# [INFO] GPU张量操作成功
# [SUCCESS] PyTorch基本功能测试通过

# 测试结果汇总
# --------------------------------------------------
# [SUCCESS] ✓ PyTorch版本符合要求
# [SUCCESS] ✓ CUDA版本符合要求
# [SUCCESS] ✓ NumPy版本符合要求
# [SUCCESS] ✓ PyTorch功能测试通过
# [SUCCESS] ✅ 环境检查通过，所有关键组件满足要求


