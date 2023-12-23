# ========================== 数据预处理 ==========================
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

data = pd.read_csv("D:\\Master\\课程\\深度学习实践\\实验\\数据\\iris.csv")
for i in range(len(data)):
    if data.loc[i, 'Species'] == 'setosa':
        data.loc[i, 'Species'] = 0
    if data.loc[i, 'Species'] == 'versicolor':
        data.loc[i, 'Species'] = 1
    if data.loc[i, 'Species'] == 'virginica':
        data.loc[i, 'Species'] = 2
data = data.drop('Unnamed: 0', axis=1)
# 打散数据
data = shuffle(data)
data.index = range(len(data))

# ========================== 数据标准化 ==========================
col_titles = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
for i in col_titles:
    mean, std = data[i].mean(), data[i].std()
    data[i] = (data[i] - mean) / std

# ========================== 数据集处理 ==========================
train_data = data[:-32]
train_x = train_data.drop(['Species'], axis=1).values  # 删除Species列
train_y = train_data['Species'].values.astype(int)
train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
train_y = torch.from_numpy(train_y).type(torch.LongTensor)

test_data = data[-32:]
test_x = test_data.drop(['Species'], axis=1).values  # 删除Species列
test_y = test_data['Species'].values.astype(int)
test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
test_y = torch.from_numpy(test_y).type(torch.LongTensor)

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

# 尝试不同的批次大小，以获得更好的泛化性能。
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)  # 训练数据加载器
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)  # 测试数据加载器


# ========================== 构建网络 ==========================
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.fc1 = nn.Linear(4, 6)  # 调整节点数
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = model()
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
opt = torch.optim.SGD(net.parameters(), lr=0.05)  # 优化器  #调整学习率lr  #在优化器中添加正则化，防止过拟合   #尝试其他优化器如 Adam、Adagrad 或 RMSprop

# ========================== 训练 ==========================
epochs = 900
for epoch in range(epochs):  # 增加训练轮数，让模型有更多机会学习数据的特征
    for i, data in enumerate(train_loader):
        x, y = data
        pred = net(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# ========================== 测试 ==========================
rights = 0
length = 0


def rightness(pred, labels):
    pred = torch.max(pred.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


for i, data in enumerate(test_loader):
    x, y = data
    pred = net(x)
    rights = rights + rightness(pred, y)[0]
    length = length + rightness(pred, y)[1]
    # print(y)
    # print(torch.max(pred.data, 1)[1], '\n')

accuracy = rights / length * 100
print(f'Test Accuracy: {accuracy:.2f}%')
