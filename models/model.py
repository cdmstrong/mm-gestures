import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        embedding_dim: 词向量的维度
        hidden_dim: LSTM神经元的个数
        layer_dim: LSTM的层数
        output_dim: 输出的维度
        """
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        
        # LSTM + FC
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)  # 全0初始化h0
        # r_out : [batch, time_step, hidden_size]
        # h_n: [n_layers, batch, hidden_size]
        # h_c: [n_layers, batch, hidden_size]
        out = self.fc1(r_out[:, -1, :])   # 选取最后一个时间点的out
        return out   