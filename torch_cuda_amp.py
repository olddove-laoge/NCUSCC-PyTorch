#这是使用了精度混合的模型

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
