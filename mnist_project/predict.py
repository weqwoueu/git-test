import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image # 专门处理图片的库

# === 1. 重新定义一遍网络结构 ===
# (因为我们只保存了参数，没保存结构，所以必须要有图纸才能装配)
class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 784) 
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# === 2. 准备“加工工艺” (预处理) ===
# 真实世界的图片千奇百怪，必须处理成和训练时一模一样的格式
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 强转为灰度图 (防止你存成彩色的)
    transforms.Resize((28, 28)),                 # 缩放到 28x28 像素
    transforms.ToTensor(),                       # 转成 Tensor
    transforms.Normalize((0.1307,), (0.3081,))   # 归一化 (和训练时保持一致)
])

def predict_image(image_path):
    # A. 加载设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # B. 加载模型结构
    model = net().to(device)
    
    # C. 加载权重 (读档)
    # map_location: 即使你在 GPU 训练的，也可以强制在 CPU 上跑预测
    model.load_state_dict(torch.load("mnist_reco_num.pth", map_location=device))
    
    # D. 开启预测模式 (必须！)
    model.eval()
    
    # E. 处理图片
    image = Image.open(image_path)
    
    # 【关键点】处理白底黑字 vs 黑底白字
    # MNIST 训练集是【黑底白字】(数字是亮的)
    # 如果你用画图板画的是【白底黑字】，必须反转颜色！
    # 我们可以通过判断左上角像素是不是白的来自动反转
    if image.getpixel((0, 0))[0] > 128: # 如果左上角是亮的(白色背景)
        from PIL import ImageOps
        image = ImageOps.invert(image) # 反转颜色
        print("检测到白底图片，已自动反转颜色...")

    # 应用预处理工艺
    image_tensor = transform(image)

    # === 【新增】保存中间过程图，看看模型到底看到了什么 ===
    # 把 Tensor 转回 PIL 图片保存
    debug_img = transforms.ToPILImage()(image_tensor)
    debug_img = debug_img.resize((300, 300), resample=Image.NEAREST) # 放大一点方便你看
    debug_img.save("what_ai_sees.png")
    print("已保存模型看到的图: what_ai_sees.png <--- 快去打开看看它长什么样！")
    # ==================================================

    # 增加 Batch 维度
    # ... (后面的代码不变)
    
    # 增加 Batch 维度
    # 目前是 [1, 28, 28]，要变成 [1, 1, 28, 28] (假装只有一个人的Batch)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # F. 预测
    with torch.no_grad():
        output = model(image_tensor)
        # 拿到概率最大的数字
        prediction = output.argmax(dim=1).item()
        
        # 拿到具体的概率值 (可选)
        probability = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()

    print(f"---------------------------")
    print(f"图片: {image_path}")
    print(f"模型预测结果: {prediction}")
    print(f"置信度: {probability*100:.2f}%")
    print(f"---------------------------")

# === 运行预测 ===
# 确保你文件夹里有这张图，或者改成你自己的图片路径
try:
    predict_image("test_pic.png")
except FileNotFoundError:
    print("错误：找不到图片！请在代码目录下放一张 'test_pic.png'")

