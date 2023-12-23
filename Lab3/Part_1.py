import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# 生成样本数据集
# n_samples=1000 表示生成1000个样本
# n_features=2 表示每个样本有2个特征
# centers=5 表示5个簇
# cluster_std=0.5 表示5个簇的方差
# random_state=16 表示随机种子，用于可复现性
X, y = make_blobs(n_samples=1000, n_features=2, centers=5, cluster_std=0.5, random_state=16)
# 绘制散点图
# data[:, 0] 表示样本数据集的第一个特征
# data[:, 1] 表示样本数据集的第二个特征
# c=target 表示不同簇用不同颜色表示
# marker='o' 表示指定数据点的显示形状为圆圈
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o')
plt.show()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=16)

# 转换为PyTorch张量
train_x = torch.from_numpy(X_train).float()
train_y = torch.from_numpy(y_train).long()
test_x = torch.from_numpy(X_test).float()
test_y = torch.from_numpy(y_test).long()

# 构建数据集和数据加载器
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)


# 定义神经网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 5)  # 输出类别数设置为聚类中心数

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 实例化模型、损失函数和优化器
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 在测试集上验证模型准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f'Test Accuracy: {accuracy:.2f}%')
