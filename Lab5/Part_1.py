import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def getSeq(start, n):
    x = [3 * x + 2 for x in range(start, start + n)]  # 生成长度为n的序列
    return x


data = []
for i in range(100):
    rnd = np.random.randint(0, 25)
    data.append(getSeq(rnd, 6))
data = np.array(data)
data = torch.from_numpy(data)
target = data[:, -1:].type(torch.FloatTensor)  # 标签
data = data[:, :-1].type(torch.FloatTensor)  # 特征
train_x = data[:90]
train_y = target[:90]
test_x = data[90:]
test_y = target[90:]
train_dataset = TensorDataset(train_x, train_y)  # 训练数据集
test_dataset = TensorDataset(test_x, test_y)  # 测试数据集
train_loader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=True)  # 加载器
test_loader = DataLoader(dataset=test_dataset, batch_size=5, shuffle=True)


# 网络
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 10, batch_first=True)  # 输入1个特征，隐层10个单元，第一个维度为batch
        self.fc = nn.Linear(10, 1)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        output = output[:, -1, :]
        output = self.fc(output)
        return output


net = model()
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=0.001)  # 优化器，学习率设置为0.001

# 训练
# 初始化h0, c0,如果batchsize不是5，要考虑最后一个batch的样本数
print("========================= 训练 =========================")
h0 = torch.zeros(1, 5, 10)
c0 = torch.zeros(1, 5, 10)
for epoch in range(1200):
    for i, data in enumerate(train_loader):
        x, y = data
        x = x.view(-1, 5, 1)
        pred = net(x, (h0, c0))
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if (epoch + 1) % 100 == 0:
        print("epoch: {}, loss: {}".format(epoch + 1, loss.item()))

# 测试
rights = 0
length = 0
print("========================= 测试 =========================")
for i, data in enumerate(test_loader):
    x, y = data
    x = x.view(-1, 5, 1)
    pred = net(x, (h0, c0))
    print(y.view(1, -1).data)
    print("预测值：", pred.view(1, -1).data, '\n')
