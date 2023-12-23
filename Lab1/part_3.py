import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# 生成加噪数据
def generate_noisy_data_cubic(a, b, c, d, num_points, noise_std):
    x = torch.linspace(-1, 1, num_points)
    # y_true=a*x+b
    y_true = a * x ** 3 + b * x ** 2 + c * x + d
    noise = torch.normal(0, noise_std, size=(num_points,))
    y_noisy = y_true + noise
    return x, y_noisy, y_true


# 模型
class CubicModel(nn.Module):
    def __init__(self):
        super(CubicModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.pow(x, 3)
        x = self.fc1(x)
        x = torch.pow(x, 2)
        x = self.fc2(x)
        return x


# 训练模型
def train_cubic_model(x, y, model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


# 参数设置
a_true = input("input a: ")
b_true = input("input b: ")
c_true = input("input c: ")
d_true = input("input d: ")
a_true = int(a_true)
b_true = int(b_true)
c_true = int(c_true)
d_true = int(d_true)

num_points = 100
noise_std = 0.5
learning_rate = 0.01
num_epochs = 100

# 生成加噪数据
x, y_noisy, y_true = generate_noisy_data_cubic(a_true, b_true, c_true, d_true, num_points, noise_std)

# 转换数据为PyTorch张量
x = x.view(-1, 1)
y_noisy = y_noisy.view(-1, 1)

# 定义模型、损失函数和优化器
model_cubic = CubicModel()
criterion_cubic = nn.MSELoss()
optimizer_cubic = optim.SGD(model_cubic.parameters(), lr=learning_rate)

# 训练模型
train_cubic_model(x, y_noisy, model_cubic, criterion_cubic, optimizer_cubic, num_epochs)

# 绘制拟合效果
plt.scatter(x, y_noisy, label='Noisy Data')
plt.plot(x, y_true, label='True Function', linestyle='--', color='green')
plt.plot(x, model_cubic(x).detach().numpy(), label='Fitted Curve', linestyle='-', color='red')
plt.legend()
plt.title('Fitting Cubic Model with Neural Network')
plt.show()
