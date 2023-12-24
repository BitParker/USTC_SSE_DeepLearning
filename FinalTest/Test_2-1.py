# test2 - 方案1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import time
# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = datasets.MNIST(root='./', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.1307, ], [0.3081, ])
]), download=True)

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

test_data = datasets.MNIST(root='./', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.1307, ], [0.3081, ])
]))
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)
# 搭建模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1_192 = nn.Conv2d(1, 4, 5, padding=2)  # 输入1个通道，输出4个通道，5*5卷积，填充2
        self.bn1_192 = nn.BatchNorm2d(4)
        self.conv2_192 = nn.Conv2d(4, 8, 5, padding=2)
        self.bn2_192 = nn.BatchNorm2d(8)
        self.conv3_192 = nn.Conv2d(8, 8, 5, padding=2)
        self.bn3_192 = nn.BatchNorm2d(8)
        self.pool_192 = nn.MaxPool2d(4, 4)
        self.fc1_192 = nn.Linear((28 * 28) // (4 * 4) * 8, 512)
        self.fc2_192 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1_192(self.conv1_192(x)))
        x = F.relu(self.bn2_192(self.conv2_192(x)))
        x = F.relu(self.bn3_192(self.conv3_192(x)))
        x = self.pool_192(x)
        x = x.view(-1, (28 * 28) // (4 * 4) * 8)
        x = self.fc1_192(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2_192(x)
        return x

# 初始化模型
net = Model().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 动量SGD

# 训练
start_time = time.time()
for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        net.train()
        pred = net(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += loss.item()
    if epoch % 5 == 0:
        print('Epoch %d, Loss: %.3f' % (epoch, running_loss / len(train_loader)))
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time:{training_time}")

# 测试
rights = 0
length = 0
with torch.no_grad():
    for i, data in enumerate(test_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        net.eval()
        pred = net(x)
        _, pred_labels = torch.max(pred, 1)
        rights += torch.sum(pred_labels == y).item()
        length += len(y)
print (rights, length, rights/ length)