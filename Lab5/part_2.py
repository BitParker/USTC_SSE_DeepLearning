import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

data = []
start = 0
for i in range(200):
    x = [np.sin(x / 10) for x in range(start, start + 11)]
    data.append(x)
    start = start + 1

data = np.array(data)
data = torch.from_numpy(data)

target = data[:, -1:].type(torch.FloatTensor)  # 标签
data = data[:, :-1].type(torch.FloatTensor)  # 特征

train_x = data[:150]
train_y = target[:150]
test_x = data[150:]
test_y = target[150:]
train_dataset = TensorDataset(train_x, train_y)  # 训练数据集
test_dataset = TensorDataset(test_x, test_y)  # 测试数据集
train_loader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=True)  # 加载器
# 注意测试集不能用shuffle，否则画出来的正弦波形不对
test_loader = DataLoader(dataset=test_dataset, batch_size=5, shuffle=False)


# 网络
class model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.rnn = nn.RNN(1, 10, batch_first=True)  # 输入1个特征，隐层10个单元，第一个维度为batch
        # self.lstm=nn.LSTM(1, 10, batch_first=True) #输入1个特征，隐层10个单元，第一个维度为batch
        self.gru = nn.GRU(1, 10, batch_first=True)  # 输入1个特征，隐层10个单元，第一个维度为batch
        self.fc = nn.Linear(10, 1)

    def forward(self, x, hidden):
        # output, hidden=self.rnn (x, hidden)
        # output, hidden=self.lstm (x, hidden)
        output, hidden = self.gru(x, hidden)
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
for epoch in range(500):
    for i, data in enumerate(train_loader):
        x, y = data
        x = x.view(-1, 10, 1)
        pred = net(x, h0)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if (epoch + 1) % 50 == 0:
        print("epoch: {}, loss: {}".format(epoch + 1, loss.item()))

# 测试
print("========================= 测试 =========================")
preds = []
for i, data in enumerate(test_loader):
    x, y = data
    x = x.view(-1, 10, 1)
    pred = net(x, h0)
    preds.append(pred.data.numpy())
    print("实际值：", y.view(1, -1).data)
    print("预测值：", pred.view(1, -1).data)
    accuracy = 1 - abs(y.view(1, -1).data - pred.view(1, -1).data) / y.view(1, -1).data
    print("准确率：", accuracy, '\n')
plt.scatter(range(len(train_y)), train_y.data.numpy(), marker='o')
plt.scatter(range(150, 200), preds, marker='s')
plt.show()
