import torch
import torch.nn as nn ##角色： 标准件仓库 (Parts Catalog)解释： nn 是 Neural Network (神经网络) 的缩写。这里面全是现成的、封装好的精密零件。你想用全连接层？调 nn.Linear。想用卷积层？调 nn.Conv2d。想用损失函数（质检仪）？调 nn.CrossEntropyLoss。机械类比： 你不需要自己车削螺丝，直接去库房领 M5 的螺栓就行。我们定义的 class Net(nn.Module) 就是从这里继承的。
import torch.optim as optim #optim 是 Optimizer (优化器) 的缩写角色： 调机师傅 (Tuner)解释： optim 是 Optimizer (优化器) 的缩写。还记得昨天我们手动写的 w -= lr * w.grad 吗？在工业级代码里，我们不用手写更新公式，而是雇佣这个“师傅”来帮我们调参。常用的师傅有：optim.SGD (随机梯度下降，老实稳重), optim.Adam (自适应动量，聪明且快)。作用： 每次反向传播算出梯度后，你喊一声 optimizer.step()，这个师傅就会拿着扳手，把网络里几百万个参数都微调一遍。
from torchvision import datasets, transforms  #角色： 原材料供应商 & 预处理车间解释： torchvision 是 PyTorch 官方专门做计算机视觉的扩展包。datasets (供应商)： 它这就好像一个巨大的云端仓库。你想做手写数字识别？直接 datasets.MNIST。想做自然场景识别？直接 datasets.CIFAR10。它会自动帮你下载数据。transforms (预处理)： 这是粗加工车间。图片太大了？transforms.Resize。格式不对？transforms.ToTensor（把 jpg 图片转成 Tensor 矩阵）。太暗了？transforms.Normalize（归一化）。作用： 保证送上生产线的原材料是标准统一的。
from torch.utils.data import DataLoader #角色： 震动盘 / 自动上料机 (Feeder)解释： 这是数据流水线的核心组件。你肯定不想一只手拿一张图片去训练，太慢了。DataLoader 的作用：Batching (打包)： 它把 64 张或者 128 张图捆成一捆（Batch），一次性喂给 GPU。Shuffling (洗牌)： 每学完一遍，它会自动把数据打乱顺序（防止死记硬背）。Parallelism (多线程)： 它可以在 CPU 上用多个线程偷偷帮你加载数据，这样 GPU 计算的时候就不需要等数据了。

# 设置参数

catch_size = 64 #一次训练抓64张图
learn_rate = 0.01 #学习率
epochs = 5 #训练周期(次数)

device = torch.device("cuda"if torch.cuda.is_available() else "cpu")


#准备数据

#数据处理方法
transform = transforms.Compose([
    transforms.ToTensor(),#原始图片是 PIL 格式（0-255的整数，H x W x C）。这步把它转成 Tensor 格式（0-1的浮点数，C x H x W），方便 GPU 计算。
    transforms.Normalize((0.1307),(0.3081))#标准化result = (input - mean) / std  0.1307,0.3081分别为mean和std,标准化后收敛快
])
#下载训练数据
# root='./data': 货仓位置。下载的文件会存在你项目下的 data 文件夹里。train = True 用于训练，download = True自动下载，transform = transform初始处理方式
train_dataset = datasets.MNIST(root='./data',train = True,download = True,transform = transform)
#数据上货方式下面的分别是数据，上的数量，shuffle=True: 打乱上货顺序
train_loader = DataLoader(train_dataset,batch_size=catch_size,shuffle=True)

#下载测试数据
test_dataset = datasets.MNIST(root='./data',train = False,download = True,transform = transform)
test_loader = DataLoader(train_dataset,batch_size=catch_size,shuffle=False)

#定义网络结构
class net(nn.Module):
    def __init__(self):
        super().__init__()
         # 输入 784 (像素点), 输出 10 (数字 0-9 的概率
        self.fc1 = nn.Linear(784,128)# 第一层：输入784 -> 隐藏128
        self.fc2 = nn.Linear(128,10) # 第二层：隐藏128 -> 输出10
        #网络结构
    def forward(self,x):
        # 我们要把它拉直成 [64, 784]64个长条一次加工
        x= x.view(-1,784) # x 的形状一开始是 [64, 1, 28, 28] (Batch, Channel, Height, Width)
        x = self.fc1(x)
        x = torch.relu(x)#非线性激活：这是灵魂。如果没有它，两层 Linear 叠在一起 mathematically 等于一层 Linear。作用：它像个单向阀，把负数信号滤掉，保留正数。有了它，网络才能学会复杂的非线性逻辑（比如弯曲的数字 '8'）。
        x = self.fc2(x)
        return x
model = net().to(device)

#循环训练

#优化器和损失函数
optimizer = optim.SGD(model.parameters(),lr = learn_rate)# 他的职责：根据反馈回来的误差，拿着扳手去拧网络里的参数 (w, b)
# model.parameters(): 告诉师傅，“这台机器里所有能转的旋钮都在这儿了”
# lr=0.01: 每次拧多大幅度？(步长)。太大容易拧过头，太小调整太慢。
criterion = nn.CrossEntropyLoss()  #loss

#开始循环
print("\n开始训练")
for i in range(epochs):
    model.train()# 【关键】开启“训练模式”。
    for batch_idx, (data,target) in enumerate(train_loader):
        ## train_loader: 震动盘每次吐出 64 个零件 (batch)
        #train_loader里面有64张手写图, 64个对应的数字标签（一次循环）data,target代表吐出的包一包二这样写好看分布后续使用包一包二
        #enumerate (计数器)
        data , target = data.to(device) , target.to(device) # A. 搬运原材料 (Data Transfer)
        # 把数据从 CPU 内存搬到 GPU 显存。
        # 必须保证 data (输入) 和 model (机器) 都在同一个设备上。
        optimizer.zero_grad()#归零
        output = model(data)#data过神经网络net后optimizer.step()更新model参数
        loss = criterion(output , target)
        loss.backward()#求导
        optimizer.step()#类似w = w - lr * gradu更新u参数
        if batch_idx % 100 == 0:
            print (f"epochs{i}  Loss:{loss.item()}")

#测试训练结果
print("\n开始测试")
model.eval() # 【关键】开启“测试/评估模式”。

correct = 0
total = 0
with torch.no_grad():# 告诉监工：“这一段不需要记账”。因为测试不需要反向传播，不记账可以省内存、提速。
    for batch_idx, (data,target) in enumerate(test_loader):
        data , target = data.to(device) , target.to(device)#同上搬运材料到GPU

        output = model(data)#过调好的model得到[64,10]的概率表，10代表0到9的分别的概率
        predicted = output.argmax(dim=1)#每行最大值得[64,1]即预测结果
        is_correct = (predicted == target)#得到对错表[True, False, True, True ...]
        correct_in_batch = is_correct.sum().item()
        # sum(): 把 True 当作 1, False 当作 0 加起来 -> 得到 tensor(58)
        # item(): 把 tensor(58) 变成普通的 Python 整数 58
        #每循环计算正确数和总体数
        correct += correct_in_batch
        total += target.size(0) 
        if batch_idx==1:
            print(f"target.shape:{target.shape}")

print(f"correct_rate:{100*(correct/total)}%")
torch.save(model.state_dict(),"mnist_reco_num.pth")
print("模型已保存为 mnist_reco_num.pth")







