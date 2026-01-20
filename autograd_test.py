import torch

x = torch.arange(4, dtype=torch.float32)
print(x)
x.requires_grad_(True)  # 设置 requires_grad 属性为 True，表示需要计算梯度
y = x ** x
g_x = x**x*(torch.log(x)+1)
sum_y = y.sum()
print(sum_y)
sum_y.backward()  # 反向传播，计算梯度
print(x.grad)  # 输出 x 的梯度
print(g_x==x.grad)  # 验证计算的梯度是否正确

for i in range(4):
    y = x*x + 2*x + 1
    sum_y = y.sum()
    sum_y.backward()  # 反向传播，计算梯度
print(x.grad)  # 输出 x 的梯度 此时梯度是之前的4倍因为x.grad没有清零，每次的梯度都会累加
#正确的做法是每次反向传播前清零梯度
for i in range(4):
    y = x*x + 2*x + 1
    sum_y = y.sum()
    x.grad.zero_()  # 清零梯度
    sum_y.backward()  # 反向传播，计算梯度
print(x.grad)  # 输出 x 的梯度 现在是正确的结果
#二次求导
x.grad.zero_()  # 清零梯度
y = x*x + 2*x + 1
dy = 2*x +2
ddy = 2
sum_y = y.sum()
sum_y.backward(create_graph=True)  # 反向传播，计算梯度，设置 create_graph=True 使得到的梯度保留计算图(dy与x的ui关系)
print(f"一阶梯度{x.grad}")  # 输出 x 的梯度

grad_y = x.grad #把x.grad当另一个y,即dy = x.grad
dy_sum = grad_y.sum()#标量才能.backward()
x.grad.zero_()  # 清零梯度
dy_sum.backward()
print(f"二阶梯度{x.grad}")










