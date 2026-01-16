import torch
import time

n=40

start_time = time.time()
def fibo(n):
    if n == 1:
        return 1
    elif n==0:
        return 0
    else:
        return fibo(n-1) + fibo(n-2)
    
print(fibo(n)  , time.time()-start_time )

device = torch.device("cuda")

# 定义基础矩阵 Q = [[1, 1], [1, 0]]
Q = torch.tensor([[1, 1], [1, 0]], dtype=torch.float64).to(device)

start_time = time.time()

# 核心：计算 Q 的 n 次方
# torch.linalg.matrix_power 是专门做矩阵快速幂的
# 这比递归快了一万倍
M_pow = torch.linalg.matrix_power(Q, n)

# 结果提取
result = M_pow[0, 1] 
# 注意：最后要 .item() 拿回 CPU 读数
print(f"GPU(矩阵) 结果: {int(result.item())}")
print(f"GPU 耗时: {time.time()-start_time:.4f}s")