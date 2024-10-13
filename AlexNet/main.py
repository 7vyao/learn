import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from alexnet_model import AlexNet
from alexnet_transform import Transform
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transformer = Transform(resize=224, mean=0.5, std=0.5)
transform = transformer.get_transform()

# 下载并加载MNIST数据集
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
model = AlexNet().to(device)
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 用于记录训练和测试的损失与准确率
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# 定义训练过程
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # 模型进入训练模式
        running_loss = 0.0
        correct = 0
        total = 0


        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # 记录每个epoch的平均损失和准确率
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(100. * correct / total)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, '
              f'Accuracy: {train_accuracies[-1]:.2f}%')

        # 在每个epoch结束后对模型进行测试
        test_model(model, test_loader, criterion)

# 定义测试过程
def test_model(model, test_loader, criterion):
    model.eval()  # 模型进入评估模式
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # 记录测试集的平均损失和准确率
    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(100. * correct / total)

    print(f'Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.2f}%')

# 计算并保存指标
def evaluate_metrics(model, test_loader):
    model.eval()  # 评估模式
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # 收集预测结果和真实标签
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算各项指标
    cm = confusion_matrix(all_targets, all_predictions)
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')

    print(f'Confusion Matrix:\n{cm}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    # 将结果保存到Excel文件中
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    }

    # 创建DataFrame并写入Excel
    metrics_df = pd.DataFrame(metrics_data)
    cm_df = pd.DataFrame(cm, index=[f'Actual {i}' for i in range(10)], columns=[f'Predicted {i}' for i in range(10)])

    # 使用ExcelWriter保存多个DataFrame
    with pd.ExcelWriter('model_metrics.xlsx') as writer:
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        cm_df.to_excel(writer, sheet_name='Confusion Matrix')

    print('Metrics and Confusion Matrix saved to model_metrics.xlsx')

# 训练模型
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# 评估模型并保存指标
evaluate_metrics(model, test_loader)

# 保存模型
torch.save(model.state_dict(), 'data/MNIST/alexnet_mnist.pth')

# 加载模型
model.load_state_dict(torch.load('data/MNIST/alexnet_mnist.pth'))
model.eval()  # 设置为评估模式

# 可视化训练和测试的Loss和Accuracy
def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # 绘制Loss曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs[:len(test_losses)], test_losses, label='Test Loss')  # 避免test_losses比train_losses短
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs[:len(test_accuracies)], test_accuracies, label='Test Accuracy')  # 同样避免不匹配
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # 显示图表
    plt.tight_layout()
    plt.show()

# 绘制训练和测试的Loss与Accuracy曲线
plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)