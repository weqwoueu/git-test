import torch
import time
import numpy as np
#创建tensor
a = torch.arange(16).reshape(4,4)
a[0,0]=1
af = torch.arange(16,dtype=torch.float64).reshape(4,4)
af[0,0]=1.0
af[1,0]=5.0
af[2,0]=7.0
a8 = torch.arange(8,dtype=torch.float64).reshape(2,2,2)
a8[0,0,0]=1.0

date = torch.tensor([[1,2,3,4,5,6,7,8,9,10],
                     [1,2,3,4,5,6,6,7,8,9]])
zero = torch.zeros((4,4))
cleaned_date = date.clone() # 克隆一份新的数据
z_date = torch.zeros_like(zero)
r_date = torch.randn_like(zero)
fdate = date.reshape(-1,20)
randn = torch.randn((4,4))
one = torch.ones((4,4))
#tensor和numpy互转
A = date.numpy() #tensor转numpy
B = torch.from_numpy(A) #numpy转tensor
#增加和减少维度
a1 = a.unsqueeze(0) #给零件加个包装盒
a2 = a1.squeeze() #把多余的包装盒扔掉

#广播机制
b = torch.tensor([1,2,3,4])
e = torch.reshape(b,(4,-1)) #把b变成4行1列
c = a + b #b会自动扩展成和a一样的形状再做加法
d = a + e #e会自动扩展成和a一样的形状再做加法
f = b + e #b和e会自动扩展成一样的形状再做加法

print(zero)
print(date.shape) #矩阵维度
print(date.numel()) #元素个数
print(fdate.shape)
print(fdate)
print(randn)
print(randn.shape)
print(randn.numel())
print(zero*randn) #对应加减乘除
print(one*randn)
print(one+randn)
print(torch.cat([zero,randn,one],dim=0))#黏合0为粘a行，1为粘列
print(torch.stack([zero,randn,one],dim=0))#堆叠0为增加a维，1为增加列维
print(torch.zeros_like(randn))#生成和randn一样维度的0矩阵
print(randn.sum())#所有元素求和
print(randn.mean())#所有元素求平均
print(randn.max())#所有元素求最大值
print(date)
print(date[0:3,4:6])#切片
print(date[1:3])
date[1:3] = 7
print(date)#切片赋值
print(f"data的形状: {date.shape}") 
print(f"data的数据类型: {date.dtype}")
print(f"data在哪个设备: {date.device}") # cpu 还是 cuda
print(f"A的类型: {type(A)}")
print(f"B的类型: {type(B)}")
print(z_date>r_date)
print(a,a1,a2)

print(a)
print(b)
print(c)
print(e)
print(d)
print(f)
print(a.T) #矩阵转置
print(a.matmul(b.reshape(4,1))) #矩阵乘法
print(torch.matmul(a,b.reshape(4,1))) #矩阵乘法
print(a @ b.reshape(4,1)) #矩阵乘法
print(af)
print(af.inverse()) #矩阵求逆
print(a8)
print(a8.inverse()) #矩阵维度变换
print(a8.reshape(1,-1).squeeze())#reshape矩阵变成一行后去掉多余维度
print(a8.inverse().reshape(1,-1).squeeze())
print(torch.dot(a8.reshape(1,-1).squeeze(),a8.inverse().reshape(1,-1).squeeze())) #矩阵点积
print(torch.mm(a8.reshape(2,4),a8.inverse().reshape(2,4).T))#矩阵乘法