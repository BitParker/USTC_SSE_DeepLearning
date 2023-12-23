import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

# 数据处理
data_path = "D:\\23876\\Documents\\train"
data_file_names = os.listdir(data_path)
train_image_paths = []
test_image_paths = []
train_labels = []
test_labels = []

for data_file in data_file_names:
    label = data_file.split('.')[0]
    idx = int(data_file.split('.')[1])
    if idx >= 2500:
        test_path = os.path.join(data_path, data_file)
        test_image_paths.append(test_path)
        test_labels.append(label)
    else:
        train_path = os.path.join(data_path, data_file)
        train_image_paths.append(train_path)
        train_labels.append(label)


# 定义数据集类
class CatDogDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = 0 if self.labels[idx] == 'cat' else 1

        if self.transform:
            image = self.transform(image)

        return image, label


# 图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 创建数据集
train_dataset = CatDogDataset(train_image_paths, train_labels, transform=transform)
test_dataset = CatDogDataset(test_image_paths, test_labels, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义ResNet18模型
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# 固定模型参数
for param in model.parameters():
    param.requires_grad = False
# 修改全连接层以适应二分类问题
features = model.fc.in_features
model.fc = nn.Linear(features, 2)
# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.01)

# 训练模型
print("========================= 训练 =========================")
num_epochs = 20
if torch.cuda.is_available():
    model = model.cuda()
    model.fc = model.fc.cuda()

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        x, y = data
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # if (epoch + 1) % 2 == 0:
    print("epoch: {}, loss: {}".format(epoch + 1, loss.item()))

# 测试模型
print("========================= 测试 =========================")
rights = 0
length = 0
for i, data in enumerate(test_loader):
    x, y = data
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    model.eval()
    pred = model(x)
    _, pred_labels = torch.max(pred, 1)
    rights += torch.sum(pred_labels == y).item()
    length += len(y)

accuracy = rights / length * 100
print(f'准确率: {accuracy:.2f}%')
