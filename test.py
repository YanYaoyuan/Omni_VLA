import torch

### 5090
# pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 1. 检查版本
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 版本: {torch.version.cuda}")

# 2. 检查支持的架构 (关键!)
print(f"支持的架构: {torch.cuda.get_arch_list()}")

# 3. 检查 GPU 是否可用
print(f"GPU 可用: {torch.cuda.is_available()}")
print(f"GPU 名称: {torch.cuda.get_device_name(0)}")

# 4. 简单计算测试
x = torch.randn(1000, 1000).cuda()
y = x @ x
print(f"计算测试通过! 结果形状: {y.shape}")
