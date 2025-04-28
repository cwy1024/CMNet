from CMNet.CMMNet.Mamba_cnn.CMNet import *
import torch
import thop
import torch.nn as nn
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = Mamba_Cnn().to(device)
# model = Unet().to(device)

# 示例输入
input1 = torch.randn(1, 4, 224, 224).to(device)  # batch_size=1, channels=4, height=224, width=224

# 使用thop计算FLOPs和参数量
flops, params = thop.profile(model, inputs=(input1,))

# 输出FLOPs和参数量
print(f"GFLOPs: {flops / 1e9:.4f}")  # 将FLOPs转换为GigaFLOPs (GFLOPs)
print(f"Params (M): {params / 1e6:.4f}")  # 将参数数量转换为百万(M)

# 计算FPS
warmup = 10  # 预热次数
repeat = 100  # 重复测量次数

# 预热
with torch.no_grad():
    for _ in range(warmup):
        _ = model(input1)

# 测量推理时间
start_time = time.time()
with torch.no_grad():
    for _ in range(repeat):
        _ = model(input1)
total_time = time.time() - start_time

# 计算平均FPS
avg_fps = repeat / total_time
print(f"FPS: {avg_fps:.2f}")

# 可选：计算每帧的平均推理时间（毫秒）
avg_time_per_frame = total_time * 1000 / repeat
print(f"Average inference time per frame: {avg_time_per_frame:.2f} ms")