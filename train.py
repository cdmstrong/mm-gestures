from models.model import LSTMNet
import torch 
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from utils.dataloaders import mm_DataSet
from tqdm import tqdm
def run():
    # 构建训练数据
    dataset = mm_DataSet()
    # cached_dataset = torch.utils.data.dataset.DatasetCacher(dataset)

    train_size = int(0.8 * len(dataset))  # 80% 数据用于训练
    val_size = len(dataset) - train_size  # 剩余20% 数据用于验证

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # 创建数据加载器并使用缓存
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_size = 80  # 输入特征数，即数据的维度（如温度、湿度等）
    hidden_size = 128  # LSTM的隐藏层大小，可以根据需要进行调整
    output_size = 12  # 输出特征数，这里只需要预测一个值，所以为1
    model = LSTMNet(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    loss_function = nn.MSELoss()  # 使用均方误差损失函数
    epochs = 100  # 训练轮数，可以根据需要进行调整
    print("start train")
    print(model)
    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        optimizer.zero_grad()  # 清空梯度缓存
        train_tqdm = tqdm(train_loader, total=(len(train_loader)))
        for X, y in train_tqdm:
            print(X.shape)
            X = (X.view(X.shape[0], X.shape[1], -1)).double()
            print(X.dtype)
            y_pred = model(X)  # 前向传播，得到预测值
            loss = loss_function(y_pred, y)  # 计算损失函数值
            loss.backward()  # 反向传播，计算梯度值
            optimizer.step()  # 更新权重参数值（根据梯度下降法）
        # 跑验证集合
        model.eval()  # 将模型设置为评估模式
        total_correct = 0
        for X, y in val_loader:
            with torch.no_grad():  # 关闭梯度计算
                X = X.view(X.shape[0], X.shape[1], -1).double()
                y_pred = model(X)  # 前向传播，得到预测值
                pred = torch.argmax(y_pred, dim=1)  # 对预测值进行最大值索引，得到预测类别
                correct = pred.eq(y).sum().item()  # 统计预测正确的数量
                total_correct += correct  # 累加预测正确的数量
        accuracy = total_correct / len(val_loader) * 100  # 计算准确率并乘以100转换为百分比
        print(f"epoch: {e}, loss: {loss}, val_acc: {accuracy}", epoch, loss, accuracy)

if __name__ == "__main__":
    run()