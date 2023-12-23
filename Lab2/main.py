# ===================离散数据处理===================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

data = pd.read_csv('D:\\Master\\课程\\深度学习实践\\实验\\数据\\bikes.csv')
col_titles = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for i in col_titles:
    dummies = pd.get_dummies(data[i], prefix=i)
    data = pd.concat([data, dummies], axis=1)
col_titles_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'mnth', 'workingday', 'hr']
data = data.drop(col_titles_to_drop, axis=1)

# ===================连续数据标准化处理===================
col_titles = ['cnt', 'temp', 'hum', 'windspeed']
for i in col_titles:
    mean, std = data[i].mean(), data[i].std()
    if i == 'cnt':
        mean_cnt, std_cnt = mean, std  # 保存cnt的均值和方差
    data[i] = (data[i] - mean) / std

# ===================数据集处理===================
test_data = data[-30 * 24:]
train_data = data[:-30 * 24]
X = train_data.drop(['cnt'], axis=1)
X = X.values
Y = train_data['cnt']
Y = Y.values.astype(float)
Y = np.reshape(Y, [len(Y), 1])

# ===================搭建神经网络===================
input_size = X.shape[1]  # 输入张量的特征数量
hidden_size = 10  # 隐层神经元数量
output_size = 1  # 输出层神经元数量
batch_size = 128  # 每个batch的数量
neu = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size)
)
loss_fn = torch.nn.MSELoss()  # 均方差损失函数
opt = torch.optim.SGD(neu.parameters(), lr=0.01)  # 参数设置

# ===================训练===================
losses = []
for i in range(1000):
    batch_loss = []
    for start in range(0, len(X), batch_size):
        if start + batch_size < len(X):
            end = start + batch_size
        else:
            end = len(X)

        # 此处注意将numpy数据转换为Pytorch张量
        X = X.astype(np.float32)
        x = torch.FloatTensor(X[start:end])  # 生成一个batch的训练数据
        Y = Y.astype(np.float32)
        y = torch.FloatTensor(Y[start:end])
        pred = neu(x)
        loss = loss_fn(pred, y)  # 调用损失函数计算损失
        opt.zero_grad()
        loss.backward()
        opt.step()
        batch_loss.append(loss.data.numpy())
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))
plt.plot(np.arange(len(losses)) * 100, losses)
plt.xlabel("batch")
plt.ylabel("MSE")
plt.show()

# ===================测试（验证）===================
X = test_data.drop(['cnt'], axis=1)
Y = test_data['cnt']
Y = Y.values.reshape([len(Y), 1])

# 此处注意将numpy数据转换为torch数据
X = X.astype(np.float32)
X = torch.FloatTensor(X.values)
Y = torch.FloatTensor(Y)
pred = neu(X)  # 用训练好的模型进行预测

Y = Y.data.numpy() * std_cnt + mean_cnt  # 把经过标准化处理的数据再变回去
pred = pred.data.numpy() * std_cnt + mean_cnt

plt.figure(figsize=(10, 7))
xplot, = plt.plot(np.arange(X.size(0)), Y)
yplot, = plt.plot(np.arange(X.size(0)), pred, ':')
plt.show()
