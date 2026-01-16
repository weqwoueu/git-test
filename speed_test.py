import torch
import time

print("正在准备数据，请稍等...")
# 生成两个巨大的 10000x10000 矩ied
# 这相当于一亿次乘法运算
n = 10000
a = torch.randn(n, n)
b = torch.randn(n, n)

print("--- 开始 CPU 计算 ---")
start_time = time.time()
# 让 CPU 算
c = torch.matmul(a, b)
print(f"CPU 耗时: {time.time() - start_time:.4f} 秒")

print("\n--- 开始 GPU 计算 ---")
# 把数据搬运到显卡上
device = torch.device("cuda")
a_gpu = a.to(device)
b_gpu = b.to(device)

# 预热一下（显卡刚启动需要一点时间）
torch.matmul(a_gpu, b_gpu)
torch.cuda.synchronize()

start_time = time.time()
# 让 GPU 算
c_gpu = torch.matmul(a_gpu, b_gpu)
torch.cuda.synchronize() # 等待显卡算完
print(f"GPU 耗时: {time.time() - start_time:.4f} 秒")