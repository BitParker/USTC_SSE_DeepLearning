import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层：输入通道为1（黑白图像），输出通道为32，卷积核大小为5x5，padding为2
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        # 第一个池化层：最大池化，池化核大小为2x2，步长为2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积层：输入通道为32，输出通道为64，卷积核大小为5x5，padding为2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        # 第二个池化层：最大池化，池化核大小为2x2，步长为2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层：输入大小为64*7*7，输出大小为1024
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.relu3 = nn.ReLU()
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.5)
        # 输出层：输入大小为1024，输出大小为10（类别数）
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # 第一层卷积、激活函数和池化
        x = self.pool1(self.relu1(self.conv1(x)))
        # 第二层卷积、激活函数和池化
        x = self.pool2(self.relu2(self.conv2(x)))
        # 将特征图展平
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层、激活函数和Dropout
        x = self.dropout(self.relu3(self.fc1(x)))
        # 输出层
        x = self.fc2(x)
        return x


# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 定义模型、损失函数和优化器
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 5
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, len(train_loader),
                                                                     loss.item()))

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += torch.sum(predicted == labels).item()

print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
