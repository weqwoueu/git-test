import torch
import time
import matplotlib.pyplot as plt


# 生成模拟数据
x = torch.linspace(-1,1,100)#生成-1到1之间的100个点
y_true = 4*x**2 + 5*x + 6 + torch.randn(x.shape)*0.1 #真实函数加噪声

# 定义模型参数
a = torch.tensor(1,dtype=torch.float64,requires_grad = True)
b = torch.tensor(1,dtype=torch.float64,requires_grad = True)
c = torch.tensor(1,dtype=torch.float64,requires_grad = True)


lr = 0.01 #定义步长
stime = time.time()

#训练开始
for epoch in range(10000):
    y_gress = a*x**2 + b*x + c #每次训练周期重新定义abc和y_gress
    loss = ((y_true-y_gress)**2).mean() #100个y的均方值，mean（）是求平均值、
    #此时loss（a，b，c）是abc的二次函数，极值必然存在且为零

    loss.backward()
    #根据梯度调整abc值，目标是loss=0（到达极值点时abc拟合完成）

    with torch.no_grad():#此时开始不再跟踪abc计算图，否则每次更新数据都调用到最初更新前，内存爆炸
        a -= lr*a.grad
        b -= lr*b.grad
        c -= lr*c.grad
        #记得用完清零
        a.grad.zero_()
        b.grad.zero_()
        c.grad.zero_()
    if epoch % 100 == 0:
        print(f"epoch:{epoch}a:{a},b:{b},c:{c}")
print(f"训练结束,用时{time.time()-stime} 预测 a:{a},b:{b},c:{c}")

