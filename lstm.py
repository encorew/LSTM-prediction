import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, series_dim, hidden_size, num_layers=1, drop_out=0.5):
        super(LSTM, self).__init__()
        # series_dim 是每时刻时间点的维度大小
        self.input_size = self.output_size = series_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(series_dim, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, self.output_size)
        self.dropout = nn.Dropout(drop_out)

    def __call__(self, X, state):
        # X = self.dropout(X)
        # print(f"X.shape {X.shape} h.shape {state[1].shape}")
        hiddens, state = self.lstm(X, state)
        outputs = self.linear(hiddens)
        return outputs, state

    def begin_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size,
                        self.hidden_size, requires_grad=True)
        c = torch.zeros(self.num_layers, batch_size,
                        self.hidden_size, requires_grad=True)
        h, c = h.to(device), c.to(device)
        return (h, c)
