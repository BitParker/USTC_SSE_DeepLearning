import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# 设定随机种子以便结果可重现
np.random.seed(0)
torch.manual_seed(0)

# 数据生成
N_SAMPLES = 1000
x = np.linspace(-6, 6, N_SAMPLES).reshape(-1, 1)
a, b, c, d = np.random.randn(4)
y = a * np.sin(x) + b * x ** 2 + c * x + d
y += np.random.normal(0, 0.1, y.shape)

# 转换为PyTorch张量
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 划分训练集和验证集
dataset = TensorDataset(x_tensor, y_tensor)
train_dataset, val_dataset = random_split(dataset, [int(N_SAMPLES * 0.9), int(N_SAMPLES * 0.1)])

# 加载数据
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1_192 = nn.Linear(1, 5)
        self.fc2_192 = nn.Linear(5, 8)
        self.fc3_192 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1_192(x))
        x = torch.relu(self.fc2_192(x))
        return self.fc3_192(x)


# 定义LSTM模型
class LSTMNetwork(nn.Module):
    def __init__(self):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=5, num_layers=1, batch_first=True)
        self.fc_192 = nn.Linear(5, 1)

    def forward(self, x):
        x = x.view(len(x), 1, -1)
        x, _ = self.lstm(x)
        x = self.fc_192(x[:, -1, :])
        return x


# 实例化模型
neural_net = NeuralNetwork()
lstm_net = LSTMNetwork()

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer_nn = torch.optim.Adam(neural_net.parameters(), lr=0.01)
optimizer_lstm = torch.optim.Adam(lstm_net.parameters(), lr=0.01)


# 训练模型的函数，返回训练集上的平均损失
def train(model, optimizer, data_loader, criterion):
    model.train()
    total_loss = 0
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    return average_loss


# 验证模型的函数，返回验证集上的平均损失
def validate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    return average_loss


print("============================== CNN ==============================")
for epoch in range(100):  # 这里设定迭代次数为100
    train_loss = train(neural_net, optimizer_nn, train_loader, criterion)
    val_loss = validate(neural_net, DataLoader(val_dataset, batch_size=64, shuffle=False), criterion)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1},CNN Training Loss: {train_loss:.4f}, CNN Validation Loss: {val_loss:.4f}')

print("============================== LSTM ==============================")
for epoch in range(100):
    train_loss = train(lstm_net, optimizer_lstm, train_loader, criterion)
    val_loss = validate(lstm_net, DataLoader(val_dataset, batch_size=64, shuffle=False), criterion)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, LSTM Training Loss: {train_loss:.4f}, LSTM Validation Loss: {val_loss:.4f}')

# 评估模型并绘制结果
neural_net.eval()
lstm_net.eval()
with torch.no_grad():
    x_val, y_val = val_dataset.dataset.tensors
    nn_preds = neural_net(x_val).view(-1)
    lstm_preds = lstm_net(x_val.unsqueeze(1)).view(-1)

# 绘制真实数据和预测结果
plt.figure(figsize=(10, 6))
plt.plot(x_val.numpy(), y_val.numpy(), label='Real Date', color='blue')
plt.plot(x_val.numpy(), nn_preds.numpy(), label='CNN Prediction', color='green', linestyle='--')
plt.plot(x_val.numpy(), lstm_preds.numpy(), label='LSTM Prediction', color='red', linestyle=':')
plt.legend()
plt.title("y=asinx+bx^2+cx+d x∈[-6,6]")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
