#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch安装测试脚本
用于检测PyTorch是否正确安装及CUDA是否可用
"""

import os
import sys
import platform
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

def print_divider():
    print("=" * 50)

def print_system_info():
    print_divider()
    print("系统信息:")
    print(f"- 操作系统: {platform.system()} {platform.release()}")
    print(f"- Python版本: {platform.python_version()}")
    print(f"- 解释器路径: {sys.executable}")
    print(f"- 工作目录: {os.getcwd()}")
    print_divider()

def test_torch():
    print("PyTorch测试:")
    
    if not torch_available:
        print("❌ PyTorch导入失败，请检查安装")
        return
    
    print(f"✓ PyTorch版本: {torch.__version__}")
    print(f"- PyTorch路径: {torch.__file__}")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.version.cuda}")
        print(f"- CUDA设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ CUDA不可用")
    
    # 测试基本操作
    try:
        x = torch.rand(5, 3)
        y = torch.rand(5, 3)
        z = x + y
        print("✓ 基本张量操作成功")
    except Exception as e:
        print(f"❌ 基本张量操作失败: {str(e)}")
    
    # 如果CUDA可用，测试CUDA操作
    if torch.cuda.is_available():
        try:
            x_cuda = x.cuda()
            y_cuda = y.cuda()
            z_cuda = x_cuda + y_cuda
            print("✓ CUDA张量操作成功")
        except Exception as e:
            print(f"❌ CUDA张量操作失败: {str(e)}")
    
    print_divider()

def check_dlls():
    """检查PyTorch DLL文件状态"""
    if not torch_available:
        return
    
    print("DLL文件检查:")
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    
    # 检查常见的DLL文件
    dll_files = [
        "fbgemm.dll",
        "torch_cpu.dll", 
        "c10.dll",
        "torch_cuda.dll",
        "torch_global_deps.dll"
    ]
    
    for dll in dll_files:
        dll_path = os.path.join(torch_lib_dir, dll)
        if os.path.exists(dll_path):
            print(f"✓ 已找到: {dll}")
        else:
            print(f"❌ 未找到: {dll}")
    
    print_divider()

def main():
    print_system_info()
    test_torch()
    check_dlls()
    
    # 总结
    if torch_available:
        print("总结: PyTorch已安装")
        if torch.cuda.is_available():
            print("CUDA支持: 可用")
        else:
            print("CUDA支持: 不可用")
    else:
        print("总结: PyTorch未正确安装")
    
    print_divider()

if __name__ == "__main__":
    main() 


########################################################################################
############################  Windows CPU 测试
########################################################################################



########################################################################################
############################  Linux GPU 测试
########################################################################################

# ==================================================
# 系统信息:
# - 操作系统: Linux 5.15.0-134-generic
# - Python版本: 3.10.16
# - 解释器路径: /home/cine/anaconda3/envs/disprot/bin/python
# - 工作目录: /home/cine/Documents/Github/disport
# ==================================================
# PyTorch测试:
# ✓ PyTorch版本: 2.4.0+cu121
# - PyTorch路径: /home/cine/anaconda3/envs/disprot/lib/python3.10/site-packages/torch/__init__.py
# ✓ CUDA可用: 12.1
# - CUDA设备数量: 4
#   - GPU 0: NVIDIA RTX A6000
#   - GPU 1: NVIDIA RTX A6000
#   - GPU 2: NVIDIA RTX A6000
#   - GPU 3: NVIDIA RTX A6000
# ✓ 基本张量操作成功
# ✓ CUDA张量操作成功
# ==================================================
# DLL文件检查:
# ❌ 未找到: fbgemm.dll
# ❌ 未找到: torch_cpu.dll
# ❌ 未找到: c10.dll
# ❌ 未找到: torch_cuda.dll
# ❌ 未找到: torch_global_deps.dll
# ==================================================
# 总结: PyTorch已安装
# CUDA支持: 可用
# ==================================================

