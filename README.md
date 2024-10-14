# _NCUSCC 2024秋季考核Pytorch试题实验报告_
## ***目录***
### 1.实验环境搭建
* 虚拟机搭建 
* GPU驱动装置
* PyTorch安装
### 2.数据集准备及训练模型搭建
* CIFAR-10 数据集的下载、处理、与格式转换
* PyTorch实现深度学习模型
* 训练模型与性能验证
### 3.模型优化与加速
* GPU＆CPU对比
* 调整 batch size
* 混合精度训练
* 调整workers
* 其他负优化的行为(批量一体化、调整学习率……)
### 4.不同优化方式性能可视化
### 5.实验过程中遇到的问题及解决方案
* 虚拟机平台选择
* GPU驱动装置与CUDATOOLKIT下载方式
* PyTorch安装须知
* CIFAR的正确下载方法
* 负优化的原因
### 6.参考资料&特别鸣谢


* * *

* * *

## Ⅰ.实验环境搭建
### ①.虚拟机搭建
##### **~~VMware?~~**   &nbsp;     ❌
##### **WSL2！** &nbsp;             ✔️
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;首先，我们要知道此次考核的深度学习要求我们使用NVIDIA驱动，从而实现训练模型在GPU上的运行，因此我们虚拟机的GPU也应为NVIDIA，方便后面CUDA的使用以及模型的训练。
#### VMware的致命缺陷
&nbsp;      与WSL不同，VMware等传统虚拟机软件创建的是一个完全隔离的虚拟环境，每个虚拟机都需要有自己的操作系统和驱动程序。通俗来讲，也就是VMware上的Linux系统几乎完全独立于我们的主机，通常使用虚拟化的硬件，这里面也就包括虚拟GPU，这些虚拟硬件需要在虚拟机内部模拟，性能通常不如直接访问主机硬件的WSL2。想要做到VMware直通主系统，是十分复制且困难的，因此，本次试题我采用WSL搭建虚拟机环境，接下来我将讲解WSL搭建&nbsp;**Ubuntu 22.04 LTS 操作系统**&nbsp;的过程
![https://www.helloimg.com/i/2024/10/13/670bc0a497032.png](https://www.helloimg.com/i/2024/10/13/670bc0a497032.png)
(此时我VMware上的乌班图已经删掉了，故用CentOS 9 为例)
可以看到，此时的GPU是一个虚拟的
&nbsp;
#### WSL2搭建Ubuntu 22.04 LTS

##### 1.检查虚拟化设置
&nbsp;      打开"任务管理器"，找到性能选项，检查虚拟化启用状态

![c100c67cea1443b1afdd4f2fa3e5eed9.png](https://www.helloimg.com/i/2024/10/13/670bc168dc9f3.png)

确保虚拟化设置正确，方便后面搭建乌班图虚拟机
##### 2.安装wsl2
&nbsp;      以管理员身份打开Windows PowerShell 并执行以下指令
&nbsp;      &nbsp;  `  wsl --set-default-version 2`
&nbsp;     &nbsp;     `wsl --update`
&nbsp;      完成后重启系统
#### 3.搭建 **Ubuntu 22.04 LTS**
&nbsp;      搭建乌班图虚拟机有很多种方法，包括通过命令行下载安装，以及在Windows自带的应用商城中下载等，为了节约时间，这里直接使用第二种方法。

![1f988326f3d82f200eb145aba3448fe8.png](https://www.helloimg.com/i/2024/10/13/670bc1694e7b5.png)

选择需要的版本下载后直接打开,设置初始账号密码
为了方便后续的操作，我在命令行中输入 `sudo -i ` 进入root账户
&nbsp;
### ②.NVIDIA 驱动和 CUDA Toolkit的安装
#### 1.安装Nvidia驱动
&nbsp;      在命令行中输入 `nvidia-smi`  查询自己的显卡驱动版本，这里我的版本已经是新版，故跳过更新步骤，稍后我会在第五个大板块讲解更新步骤

![83c568643c6708547c1bd8fcaf84224b.png](https://www.helloimg.com/i/2024/10/13/670bc16897da5.png)

&nbsp;      
#### 2.安装CUDA Tookit
&nbsp;      安装CUDA的顺序有两种，一种是决定pytorch的版本以此来选择cuda版本，第二种是先决定cuda的版本在以此来选择pytorch的版本，在这里我先是以第二种顺序进行操作，但最终出现了巨大失误导致必须重装系统，这里我将以第一种顺序讲解安装过程，稍后我将会在第五个大板块中讲解出错原因
&nbsp;      打开      [PyTorch](https://pytorch.org/get-started/previous-versions/)   

![b6c128ff6a0b5636543f2fcf29c3a33d.png](https://www.helloimg.com/i/2024/10/13/670bc1fe1f4f3.png)

&nbsp;      打开上方网站，我们可以查询到各版本的PyTorch支持的CUDA版本
&nbsp;      这里我选择安装较早的11.8版本，以避免版本过新BUG多和版本过老功能缺失的问题
&nbsp;      由网站可知支持PyTorch11.8版本的CUDA版本为11.8，这就是我们要下载的版本

&nbsp;      决定好我们要下载的CUDA版本后 打开 [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) 

&nbsp;      找到我要下载的版本后，根据我的系统版本，如图选择

![1e25580c45b27addbe6f23bea793e3ac.png](https://www.helloimg.com/i/2024/10/13/670bc1fe4f292.png)

&nbsp;      接下来，在Ubuntu命令行输入网页提供给我们的代码，进入CUDA的安装程序

![30aa7fe1ff3dda2b395163bf01e2c4fc.png](https://www.helloimg.com/i/2024/10/13/670bc3618d987.png)

（当时忘记截图了，暂时用一下别人的图）
&nbsp;      输入accept
&nbsp;      接下来一直回车，等待安装完毕即可
&nbsp;      接下来，我们进行环境配置
&nbsp;      输入`vim ~/.bashrc`  编辑该文件
&nbsp;      将下列配置写进去
```
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CPUTI/lib64
export CUDA_HOME=/usr/local/cuda/bin
export PATH=$PATH:$LD_LIBRARY_PATH:$CUDA_HOME
```
&nbsp;      保存并提出，输入 `source ~/.bashrc` 完成配置更新
&nbsp;      shuru `nvcc -V` 检查配置是否正确

&nbsp;      ![ac1c4ddf3d7d7d4e9fb90c4fe4f49ec7.png](https://www.helloimg.com/i/2024/10/13/670bc1fe4f292.png)

&nbsp;      如图，我们可以知道我们成功下载了CUDA11.8版本，可喜可贺
&nbsp;      **！！！！ 注 ：这几部一定要在root账户下完成，否则可能出现下载下来的版本与安装包不匹配的情况**
&nbsp;      
### ③.Anaconda与PyTorch的安装
####  1.conda环境的搭建
&nbsp;      在Ubunt虚拟机自带的firefox浏览器中，进入 [Anaconda](https://www.anaconda.com/) 进行Anaconda的安装与下载

![68457393d34e118c73a61e17e58845b4.png](https://www.helloimg.com/i/2024/10/13/670bc1feb40a3.png)

&nbsp;      提供邮箱后，选择我们所需要的安装包

![e4d2e5e34e8d4cb053ea5ea6c254d4d9.png](https://www.helloimg.com/i/2024/10/13/670bc1feb40a3.png)

&nbsp;      然后，在Ubuntu命令行输入`sudo bash /root/Anaconda3-2023.03-1-Linux-x86_64.sh` 进行Anaconda的安装
&nbsp;      输入之后，一直回车直到出现要求你输入yes或no的指令时，输入yes，不出意外的话，会出现这个界面

![214ae07ed21aa90fa2caf89520ace206.png](https://www.helloimg.com/i/2024/10/13/670bc2b2307f7.png)

&nbsp;      这个地方，我建议不要修改路径，否则可能对后续的操作带来麻烦，直接回车即可
&nbsp;      **下面这一步非常重要！！**！千万不要手快了按回车，这里我们应该输入yes，再按回车去初始化anaconda，要不然需要自己去配置路径，这样非常麻烦

![b3306ec418226e7d93f57c38f64922f7.png](https://www.helloimg.com/i/2024/10/13/670bc2b26778e.png)

&nbsp;      接下来，静候Anaconda安装完成即可
&nbsp;      接下来，我们执行 `conda create -n dl python=3.8` 创建python3.8且名为dl的环境
&nbsp;      之后，出现这样的界面，输入yes即可

![c5f6af50b77207fc59f22dd767677553.png](https://www.helloimg.com/i/2024/10/13/670bc2b2a03f4.png)

&nbsp;      这样，我们的conda环境就创建好了
&nbsp;      
#### 2.PyTorch的安装
&nbsp;      重新进入 [PyTorch](https://pytorch.org/get-started/previous-versions/)，找到我们之前决定的Pytorch版本，复制网页提供的命令并安装

```
#CUDA 11.8
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia

```
&nbsp;      跟着安装提示进行即可，此时我们的PyTorch就安装完成了，接下来，我们检验一下PyTorch是否安装成功
&nbsp;      通过 `python` 命令，进入python，导入torch库 `import torch` 并回车，然后执行 `print(torch.cuda.is_available())` 并回车
![c393eb793a8b2b28b49b1d2bd1f47543.png](https://www.helloimg.com/i/2024/10/13/670bc2b2a03f4.png)
&nbsp;      输出True，说明我们成功安装了PyTorch环境，同时也说明你的pytorch是可以正常调用GPU去进行计算的
&nbsp;      以上，我们就完成了Anaconda与PyTorch的安装，接下来，我们将实现数据集准备及训练模型搭建
&nbsp;      
## Ⅱ.数据集准备及训练模型搭建
&nbsp;   
### ①.CIFAR-10 数据集的下载与处理
&nbsp;      在完成这一步之前，我们应该下载一个能够运行python程序的编辑器，以方便代码的书写，保存和运行，这里我选择了vscode。使用vscode可以选择在Windows主系统上进行远程连接，或者直接在虚拟机内下载，这里我为了节约时间，选择了后者
&nbsp;      **！！！！注意：在使用VScode时，检查一下我们的Python环境，我们选择的Python环境一定是我们刚刚创建好的dl环境内的，因为只有在这个环境中我们安装了PyTorch，否则的话torch无法正常使用，我之前就中了这个坑**
&nbsp;      CIFAR-10数据集的下载和处理我找到的有两种方式，一种是使用非官方的，他人提供的，一种是官方的，在之前我并没有找到官方的方式，故使用了第一种，但是遇到了严重的test set size = 0 问题导致运行精度无法测算，后来找到了官方的处理方法才得以解决，稍后我会在第五个大板块中讲解。下面是官方加载数据的指令
```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
```
&nbsp;      这样，我们就完成了CIFAR-10数据集的下载与处理，并将其转换为 PyTorch DataLoader 格式，确保数据集可以高效加载到 GPU 进行训练。
&nbsp;      
### ②.实现深度学习模型
&nbsp;      在考核中，试题要求我们使用 PyTorch 实现一个基础的卷积神经网络（CNN），模型结构可以包括卷积层、池化层和全连接层，不调用现成的模型库（如 torchvision.models）。因此，我使用了以下指令来实现深度学习模型的构建
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 池化层 1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 全连接层 1
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        # 全连接层 2
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # 第一个卷积层 + 激活函数 + 池化层
        x = self.pool(F.relu(self.conv1(x)))
        # 第二个卷积层 + 激活函数 + 池化层
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图
        x = x.view(-1, 64 * 8 * 8)
        # 第一个全连接层 + 激活函数
        x = F.relu(self.fc1(x))
        # 第二个全连接层
        x = self.fc2(x)
        return x
```

```python
# 实例化模型并移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
```
&nbsp;      接下来，使用以下指令，确保模型训练在GPU上进行
```python
# 实例化模型并移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
```
&nbsp;      接着，我们来设置损失函数和优化器，对于分类问题，交叉熵损失函数是一个常见选择。这里我们选择Adam 优化器
```python
import torch.optim as optim

# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
&nbsp;      然后，我们开始编写训练循环来训练模型。在每个 epoch 中，遍历训练数据集，计算损失，执行反向传播，并更新模型的权重。
```python
num_epochs = 10  # 根据需要调整 epoch 数量

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()  # 清零梯度

        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        running_loss += loss.item()
        if i % 2000 == 1999:  # 每 2000 个批次打印一次
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
```
&nbsp;      最终，我们来计算运行的精度与loss，来量化我们模型的性能
```python
correct = 0
total = 0
with torch.no_grad():  # 在评估模式下，不计算梯度
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
```
&nbsp;      最后，再加上时间计算，内存占用情况等代码，再将这些代码进行整合，我们得到了一个初始的，能够计算运算时间，内存占用情况，loss，精度，并能够调整batch size，worker和学习率的训练模型
```python
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import psutil
# 数据集放置路径
data_save_pth = "./data"

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 使用官方方式加载数据集
trainset = torchvision.datasets.CIFAR10(root=data_save_pth, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_save_pth, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# 检查数据集大小
print(f'Training set size: {len(trainset)}')
print(f'Test set size: {len(testset)}')

# 定义CNN模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化网络和优化器
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
criterion.to(device)

start_time = time.time()

# 训练循环
for epoch in range(10):  # 这里使用2个epoch作为示例
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000个小批量打印一次
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
            print(f'Memory Usage: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.2f} MB')  # 打印内存使用情况

print('Finished Training')

end_time = time.time()

# 计算运行时间
elapsed_time = end_time - start_time
print(f'Training took {elapsed_time:.2f} seconds')

# 测试模型性能
# 测试模型性能
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)  # 确保标签也在 GPU 上
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

if total > 0:
    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')
else:
    print('No test data found.')
```
以10次循环为例，我们可以得到以下结果
```
[1, 2000] loss: 2.214
Memory Usage: 984.87 MB
[1, 4000] loss: 1.896
Memory Usage: 984.88 MB
[1, 6000] loss: 1.683
Memory Usage: 984.93 MB
[1, 8000] loss: 1.584
Memory Usage: 984.93 MB
[1, 10000] loss: 1.530
Memory Usage: 984.93 MB
[1, 12000] loss: 1.460
Memory Usage: 984.93 MB
[2, 2000] loss: 1.412
Memory Usage: 985.41 MB
………………………………………………（省略中间部分）
Memory Usage: 987.01 MB
[10, 8000] loss: 0.873
Memory Usage: 987.01 MB
[10, 10000] loss: 0.876
Memory Usage: 987.01 MB
[10, 12000] loss: 0.916
Memory Usage: 987.01 MB
Finished Training
Training took 403.63 seconds
Accuracy of the network on the test images: 61.46%
```
&nbsp;      由此可知，我们已经成功实现了模型的搭建与训练，总共运行时间为403.43s，loss在逐渐减少，精度为61.16%，内存平均占用986MB，之后我又运行了几次，算出平均的时间，精度，内存占用分别为404.15s，994.33MB，61.02%，接下来，我将尝试不同的优化策略，寻找最高效的优化方法

## Ⅲ.模型优化与加速
### ①.CPU＆GPU
&nbsp;      为了对比模型在CPU和GPU上的训练情况，我写了下面这个程序，以对比两者在时间，精度，以及内存占用上的对比。
```python
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import psutil

# 数据集放置路径
data_save_pth = "./data"

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 使用官方方式加载数据集
trainset = torchvision.datasets.CIFAR10(root=data_save_pth, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_save_pth, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# 检查数据集大小
print(f'Training set size: {len(trainset)}')
print(f'Test set size: {len(testset)}')

# 定义CNN模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_and_evaluate(device):
    net = Net().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    start_time = time.time()

    for epoch in range(30):  # 使用10个epoch作为示例
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # 每2000个小批量打印一次
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                print(f'Memory Usage: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.2f} MB')  # 打印内存使用情况

    print('Finished Training')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Training took {elapsed_time:.2f} seconds on {device}')

    # 测试模型性能
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total > 0:
        print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')
    else:
        print('No test data found.')

    return elapsed_time

# 训练和评估在CPU上
cpu_time = train_and_evaluate(torch.device("cpu"))

# 训练和评估在GPU上
if torch.cuda.is_available():
    gpu_time = train_and_evaluate(torch.device("cuda"))
    print(f'Speedup on GPU compared to CPU: {cpu_time / gpu_time:.2f}x')
else:
    print("CUDA is not available. Cannot compare with GPU.")
```
&nbsp;      以2次为例，我们得到以下结果
&nbsp;      Memory Usage: 659.07 MB
&nbsp;      Finished Training
&nbsp;      Training took 57.07 seconds on cpu
&nbsp;      Accuracy of the network on the test images: 54.07%
&nbsp;      Memory Usage: 997.85 MB
&nbsp;      Finished Training
&nbsp;      Training took 84.73 seconds on cuda
&nbsp;      Accuracy of the network on the test images: 53.79%
&nbsp;      Speedup on GPU compared to CPU: 0.67x
&nbsp;      可以看到在循环数为2的情况下，CPU的各方面表现得都比GPU好
&nbsp;      那么10次呢？
&nbsp;      在10次的情况下，CPU上运行模型的评价时间，精度，占用内存分别为293.87s，62.07%，662.47MB，仍然是比GPU有锈
&nbsp;      如果我运算次数扩大到30次，或者更多呢？
&nbsp;      在30次的情况下，我们可以发现GPU的精度开始超过CPU，那我们可以合理推测，在数据量逐渐增大的情况下，GPU的精度会逐渐优于CPU
&nbsp;      关于GPU运算时间远超CPU，查询资料后，我有以下两个推论

* 数据传输时间：将数据从CPU传输到GPU需要时间，如果处理的数据量很小，这个传输时间可能会抵消GPU加速的优势。对于小规模的数据或模型，CPU可能会更快。
* 模型复杂度：如果模型过于简单，CPU可能迅速完成计算，而GPU还没来得及发挥其并行处理的优势。在这种情况下，可以尝试使用更复杂的模型或加深加宽现有的神经网络模型。

&nbsp;      可以推测，CIFAR-10是一个并不算复杂的数据集，此时CPU的优势开始凸显，但随着循环次数的增加，数据量逐渐增大，GPU的优势就会逐渐显露，从而导致我们上面展现的结果

### ②.不同的优化方式
#### 1.batch_size的调整
&nbsp;      调整 Batch Size 是深度学习中优化模型性能的重要策略之一。Batch Size直接影响模型训练的稳定性、收敛速度和最终的泛化能力。这是我目前发现的最高效的优化方法
&nbsp;      下面我将展示这种优化后的代码
```python
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import psutil

# 数据集放置路径
data_save_pth = "./data"

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 使用官方方式加载数据集
trainset = torchvision.datasets.CIFAR10(root=data_save_pth, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_save_pth, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

# 检查数据集大小
print(f'Training set size: {len(trainset)}')
print(f'Test set size: {len(testset)}')

# 定义CNN模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化网络和优化器
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
criterion.to(device)

start_time = time.time()

# 训练循环
for epoch in range(10):  # 这里使用2个epoch作为示例
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000个小批量打印一次
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
            print(f'Memory Usage: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.2f} MB')  # 打印内存使用情况

print('Finished Training')

end_time = time.time()

# 计算运行时间
elapsed_time = end_time - start_time
print(f'Training took {elapsed_time:.2f} seconds')

# 测试模型性能
# 测试模型性能
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)  # 确保标签也在 GPU 上
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

if total > 0:
    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')
else:
    print('No test data found.')
```
&nbsp;      仅仅是增长了一倍的batch_size值，我们的运行速度就提高了整整一倍，并获得了4-5个百分点的精度提升
#### 2.混合精度训练
&nbsp;      相比其他的方式，混合精度似乎提升并不高，但是将这种优化方式和其他方式结合，往往能够得到1+1>2的效果，下面我将展示只采用混合精度的代码
```python
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import psutil

# 数据集放置路径
data_save_pth = "./data"

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 使用官方方式加载数据集
trainset = torchvision.datasets.CIFAR10(root=data_save_pth, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_save_pth, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# 检查数据集大小
print(f'Training set size: {len(trainset)}')
print(f'Test set size: {len(testset)}')

# 定义CNN模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化网络和优化器
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
criterion.to(device)

start_time = time.time()
# 导入自动混合精度的库
from torch.cuda.amp import GradScaler, autocast

# 初始化梯度缩放器
scaler = GradScaler()

# 训练循环
for epoch in range(10):  # 这里使用2个epoch作为示例
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 使用自动混合精度
        with autocast():
            outputs = net(inputs)
            loss = criterion(outputs, labels)
        
        # 缩放损失，然后反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000个小批量打印一次
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
            print(f'Memory Usage: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.2f} MB')  # 打印内存使用情况

            

print('Finished Training')
end_time = time.time()

# 计算运行时间
elapsed_time = end_time - start_time
print(f'Training took {elapsed_time:.2f} seconds')

# 测试模型性能
# 测试模型性能
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)  # 确保标签也在 GPU 上
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

if total > 0:
    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')
else:
    print('No test data found.')
```

&nbsp;      通过计算，我们可以得到通过混合精度运算优化后的平均时长，精度，和内存占用分别为526.15s，62.47%，1062.43MB，可以看到精度方面有着不小的提升。
#### 3.调整workers
&nbsp;      在DataLoader中设置num_workers参数，使用多线程或多进程同时加载数据，可以有效减少数据加载瓶颈
&nbsp;      注：经过测试，发现调整workers和调整batch_size结合才是最高效的方法，下面是代码
```python
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import psutil

# 数据集放置路径
data_save_pth = "./data"

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 使用官方方式加载数据集
trainset = torchvision.datasets.CIFAR10(root=data_save_pth, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root=data_save_pth, train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)

# 检查数据集大小
print(f'Training set size: {len(trainset)}')
print(f'Test set size: {len(testset)}')

# 定义CNN模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化网络和优化器
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
criterion.to(device)

start_time = time.time()

# 训练循环
for epoch in range(10):  # 这里使用2个epoch作为示例
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000个小批量打印一次
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
            print(f'Memory Usage: {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.2f} MB')  # 打印内存使用情况

print('Finished Training')

end_time = time.time()

# 计算运行时间
elapsed_time = end_time - start_time
print(f'Training took {elapsed_time:.2f} seconds')

# 测试模型性能
# 测试模型性能
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)  # 确保标签也在 GPU 上
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

if total > 0:
    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')
else:
    print('No test data found.')
```

&nbsp;      经过多次测试，发现将batch_size设为16，workers设为4才是最优解
&nbsp;      数据大概为下
&nbsp;      Memory Usage: 988.84 MB
&nbsp;      Finished Training
&nbsp;      Training took 111.21 seconds
&nbsp;      Accuracy of the network on the test images: 64.09%
#### 4.其他负优化行为
注：由于是负优化，下列测试都只进行了寥寥几次，代码不进行完整的展示
##### &nbsp;      a.批量一体化

&nbsp;      使用批量一体化后，精度反而下降，时间变长
&nbsp;      Memory Usage: 1026.61 MB
&nbsp;      Finished Training
&nbsp;      Training took 489.16 seconds
&nbsp;      Accuracy of the network on the test images: 51.08%
&nbsp;      下面是实现批量一体化的代码部分
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)  # 添加批量归一化
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)  # 添加批量归一化
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)  # 添加批量归一化
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)  # 添加批量归一化
        self.fc3 = nn.Linear(84, 10)
```
##### &nbsp;      b.调整学习率
&nbsp;      同样的，使用调整学习率后，精度下降
```python
[10, 12000] loss: 1.039
Memory Usage: 1000.86 MB
Finished Training
Training took 409.62 seconds
Accuracy of the network on the test images: 57.38%
```

&nbsp;      我会在第五个大板块讲解这些本应优化的策略负优化的原因

## Ⅳ.不同优化方式性能可视化
&nbsp;      为了使数据可视化，我先将测算出的数据放在一个csv文件当中
```csv
Optimization Technique,Training Time (seconds),Accuracy (%),Memory Usage (MB),Epochs
cpu,293.87,62.07,662.47,10
g-initial,404.15,61.02,994.33,10
g-torch.cuda.amp,526.15,62.47,1062.43,10
g-b32-w4,66.44,60.39,977.11,10
g-b16-w4,113.21,64.91,988.84,10
g-b16w8,109.48,62.77,994.29,10
```
&nbsp;      再通过使用pandas库读取csv文件中的内容，然后再使用 matplotlib.pyplot 函数制作柱状图
&nbsp;      以下是生成柱状图的代码
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置Seaborn的样式
sns.set(style="whitegrid")

# 读取CSV文件
df = pd.read_csv('training_data.csv')

# 设置图表大小
plt.figure(figsize=(12, 10))  # 调整图表大小以适应三个子图

# 定义颜色列表
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 绘制Training Time柱状图
plt.subplot(3, 1, 1)
sns.barplot(x='Optimization Technique', y='Training Time (seconds)', data=df, palette=colors)
plt.title('Training Time by Optimization Technique', fontsize=16)
plt.xlabel('Optimization Technique', fontsize=14)
plt.ylabel('Training Time (seconds)', fontsize=14)

# 在柱状图上添加数据标签
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width() / 2., p.get_height(), 
             f'{p.get_height():.2f}', 
             ha='center', va='bottom')

# 绘制Accuracy柱状图
plt.subplot(3, 1, 2)
sns.barplot(x='Optimization Technique', y='Accuracy (%)', data=df, palette=colors)
plt.title('Accuracy by Optimization Technique', fontsize=16)
plt.xlabel('Optimization Technique', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)

# 在柱状图上添加数据标签
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width() / 2., p.get_height(), 
             f'{p.get_height()}%', 
             ha='center', va='bottom')

# 绘制Memory Usage柱状图
plt.subplot(3, 1, 3)
sns.barplot(x='Optimization Technique', y='Memory Usage (MB)', data=df, palette=colors)
plt.title('Memory Usage by Optimization Technique', fontsize=16)
plt.xlabel('Optimization Technique', fontsize=14)
plt.ylabel('Memory Usage (MB)', fontsize=14)

# 在柱状图上添加数据标签
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width() / 2., p.get_height(), 
             f'{p.get_height():.2f} MB', 
             ha='center', va='bottom')

# 调整子图间距
plt.tight_layout()

# 保存图表
plt.savefig('optimization_techniques_comparison.png', dpi=300)  # 设置高分辨率

# 显示图表
plt.show()
```
&nbsp;      这是最终呈现的图片
&nbsp;      注：这里g代表gpu，w代表worker的数量，b代表batch的数量
![8b4a307e57aff7dea25224c2de29b660.png](https://www.helloimg.com/i/2024/10/13/670bcaeab36f8.png)
## Ⅴ.实验过程中遇到的问题及解决方案
&nbsp;      
### ①.虚拟机平台选择
&nbsp;      正如我之前所说，与WSL不同，VMware等传统虚拟机软件创建的是一个完全隔离的虚拟环境。使用的是虚拟的硬件，而VMvare等平台做到直通功能又十分艰难，我第一次搭建Anaconda环境时，就因此卡在了安装nvidia驱动这一步，后来经过查询csdn等平台才选择了使用WSL2，轻而易举地做到了NVIDIA驱动与CUDATOOLKIT的安装
&nbsp;      
### ②.GPU驱动装置与CUDATOOLKIT下载方式
&nbsp;      
&nbsp;      在这两步中，我犯了同样的错误，即在windows系统上安装好适用于Linux系统的安装包后，使用cp指令将安装包复制到Linux系统上，事实上，这是不安全的，这可能会导致安装包被放在一个即存在又不存在的Downloads文件中，即使下载成功，也会遇到其他奇怪的问题，从而导致PyTorch下载失败，比如说，conda文件内可能出现错误的版本号以及莫名其妙的波浪号，我vim很多了相关文件也没找到所谓的的~在哪里，换源之后，又出现了所谓的多余的“2.7”，后面重装了一遍系统，使用Ubuntu自带的firefox浏览器进行下载后，才解决了这个问题
```
(base) olddove@localhost:~$ conda install pytorch torchvision torchaudio cudatoolkit=11.5 -c pytorch Solving environment: failed InvalidVersionSpecError: Invalid version spec: =2.7


(base) olddove@localhost:~$ conda install pytorch torchvision torchaudio -c pytorch Solving environment: failed CondaValueError: Malformed version string '~': invalid character(s).

```

&nbsp;      这两个是我因为复制安装包而遇到的两个错误

### ③.PyTorch安装须知

&nbsp;      在之前，我提到过PyTorch安装有两种顺序，一种是决定pytorch的版本以此来选择cuda版本，第二种是先决定cuda的版本在以此来选择pytorch的版本，在第一次安装PyTorch时，我选择了第二种，这是不适合新手的。在第一次安装过程中，由于使用了第二种方法，我遇到了许多版本不兼容问题，究其根本是对CUDA 的版本通常由硬件和驱动程序决定这个细节，导致选择一个与 PyTorch 兼容的 CUDA 版本这个过程变得复杂。
&nbsp;      下面是第一种方法的几个优势

 * PyTorch 官方网站提供了一个非常直观的安装指引，用户可以通过选择操作系统、Python 版本、CUDA 版本来获取相应的安装命令。这种方法简化了选择过程，使得新手可以更容易地找到适合自己环境的安装命令。
 * PyTorch 通常每隔一段时间就会发布新版本，通过先选择 PyTorch 版本，可以更容易地保持我们的的环境是最新的。
 * 如果先选择了CUDA的一个较老版本，可能会导致一些功能的缺失，从而导致CUDA和PyTorch版本之间的不兼容
 &nbsp;      在安装cuda过程中，我还遇到了11.8版本的安装包下载后变为10.1版本的问题，登录root账户后，得到了解决

### ④.CIFAR-10的正确下载方法
&nbsp;      在之前，我说过CIFAR-10有官方和非官方两种下载方法，第一次由于不知道官方的下载方法，我在csdn中寻找了非官方的下载方法，指令如下
```python
import numpy as np
import pickle
import os
from torchvision import datasets
from imageio import imwrite
 
# 数据集放置路径
data_save_pth = "./data"
train_pth = os.path.join(data_save_pth, "train")
test_pth = os.path.join(data_save_pth, "test")
 
 
# 创建必要的目录
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
 
 
create_dir(train_pth)
create_dir(test_pth)
 
# 解压路径
data_dir = os.path.join(data_save_pth, "cifar-10-batches-py")
 
 
# 数据集下载
def download_data():
    datasets.CIFAR10(root=data_save_pth, train=True, download=True)
 
 
# 解压缩数据
def unpickle(file):
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding="bytes")
 
 
# 保存图像
def save_images(data, output_dir, offset):
    for i in range(0, 10000):
        img = np.reshape(data[b'data'][i], (3, 32, 32)).transpose(1, 2, 0)
        label = str(data[b'labels'][i])
        label_dir = os.path.join(output_dir, label)
        create_dir(label_dir)
 
        img_name = f'{label}_{i + offset}.png'
        img_path = os.path.join(label_dir, img_name)
        imwrite(img_path, img)
 
 
if __name__ == '__main__':
    download_data()
    for j in range(1, 6):
        path = os.path.join(data_dir, f"data_batch_{j}")
        data = unpickle(path)
        print(f"{path} is loading...")
        save_images(data, train_pth, (j - 1) * 10000)
        print(f"{path} loaded")
 
    test_data_path = os.path.join(data_dir, "test_batch")
    test_data = unpickle(test_data_path)
    save_images(test_data, test_pth, 0)
    print("test_batch loaded")
```

&nbsp;      那么，这样做会出现什么问题呢？
&nbsp;      在使用非官方方式下载的过程中，我发现无论采取什么措施，模型训练时输出的 Test set size 永远为0，从而导致无法计算出我们模型的精度，试题也就无法继续进行下去，同时，每次运行模型时，这种加载方式都会消耗大量的时间去下载数据集，造成巨大的时间浪费。
&nbsp;      为什么会出现这种情况，我也不清楚，推测有可能是图像文件格式错误或是测试集目录路径错误。

### ⑤.负优化的原因
&nbsp;      在之前进行模型优化的过程中，我提到过某些应为优化的策略造成了负优化，这是为什么呢，接下来就是我的解释
#### 1.批量一体化
&nbsp;       a.批量大小选择：批量归一化是在小批量数据上计算均值和方差的，如果批量大小太小，可能会导致归一化效果不佳，从而影响模型的精度。也许是我设置的批量太小，导致优化效果不明显甚至倒退。

&nbsp;       b. 模型结构问题：如果模型结构不适合特定的数据集，即使使用了批量归一化，也可能导致精度下降。也许是我设定的模型结构不适合CIFAR-10数据集，导致精度下降
&nbsp;      
#### 2.调整学习率
&nbsp;       不同的优化算法对学习率的敏感度不同。一些优化算法如Adam具有自适应学习率调整的能力，而SGD等则需要手动调整学习率。如果更换了优化算法而没有重新调整学习率，可能会影响模型性能。在我的模型中，我使用了Adam，这可能是调整学习率负优化的原因
&nbsp;       

## Ⅵ.参考资料&特别鸣谢
### 参考资料
>[在Windows中安装wsl2和ubuntu22.04](https://blog.csdn.net/qianbin3200896/article/details/136937960)
>[2023年最新Ubuntu安装pytorch教程](https://blog.csdn.net/qq_35768355/article/details/131261838)
>[用PyTorch搭建卷积神经网络](https://blog.csdn.net/juhanishen/article/details/123462838)
>[\[ 数据集 \] CIFAR-10 数据集介绍](https://blog.csdn.net/weixin_45084253/article/details/123597716)

### 特别鸣谢
**[Kimi](https://kimi.moonshot.cn/)**
wwb的茶饼
10+ 的打压
炒蒜的大家


## 老鸽 2024.10.13
