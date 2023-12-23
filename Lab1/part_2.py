import matplotlib.pyplot as plt
import numpy as np


# 生成加噪数据
def generate_noisy_data(a, b, c, d, num_points, noise_std):
    x = np.linspace(-1, 1, num_points)
    y_true = a * x ** 3 + b * x ** 2 + c * x + d
    noise = np.random.normal(0, noise_std, size=num_points)
    y_noisy = y_true + noise
    return x, y_noisy, y_true


# 计算模型预测值
def predict(x, params):
    a, b, c, d = params
    return a * x ** 3 + b * x ** 2 + c * x + d


# 计算损失函数
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# 使用梯度下降算法更新参数
def gradient_descent(x, y_true, params, learning_rate, num_iterations):
    for _ in range(num_iterations):
        y_pred = predict(x, params)
        loss = compute_loss(y_true, y_pred)

        # 计算梯度
        gradient = -2 * np.mean((y_true - y_pred) * x ** 3), -2 * np.mean((y_true - y_pred) * x ** 2), \
                   -2 * np.mean((y_true - y_pred) * x), -2 * np.mean(y_true - y_pred)

        # 更新参数
        params = tuple(param - learning_rate * grad for param, grad in zip(params, gradient))

        print(f"Iteration {_ + 1}/{num_iterations}, Loss: {loss}")

    return params


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
num_iterations = 100

# 生成加噪数据
x, y_noisy, y_true = generate_noisy_data(a_true, b_true, c_true, d_true, num_points, noise_std)

# 初始参数
initial_params = (1.0, -2.0, 1.0, 0.0)

# 使用梯度下降算法迭代得到参数
optimized_params = gradient_descent(x, y_noisy, initial_params, learning_rate, num_iterations)

# 绘制拟合效果
plt.scatter(x, y_noisy, label='Noisy Data')
plt.plot(x, y_true, label='True Function', linestyle='--', color='green')
plt.plot(x, predict(x, optimized_params), label='Fitted Curve', linestyle='-', color='red')
plt.legend()
plt.title('Fitting Polynomial Curve with Gradient Descent')
plt.show()
