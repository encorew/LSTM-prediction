import torch.optim
from d2l import torch as d2l
from torch import nn
from lstm import LSTM
from utils.data_preprocess import get_data_iter_from_dataset


def train_epoch(net, train_iter, updater, batch_size, loss_fn, device):
    metric = d2l.Accumulator(2)
    net.train()
    for X, Y in train_iter:
        # 第一维和第二维交换
        if len(X.shape) == 3:
            X, Y = X.permute(1, 0, 2), Y.permute(1, 0, 2)
        elif len(X.shape) == 2:
            X, Y = X.T, Y.T
        X, Y = X.to(device), Y.to(device)
        state = net.begin_state(batch_size, device)
        Y_hat, state = net(X, state)
        l = loss_fn(Y_hat, Y)
        updater.zero_grad()
        l.backward()
        updater.step()
        metric.add(float(l) * batch_size, batch_size)
    return metric[0] / metric[1]


def train(net, num_epochs, train_iter, updater, batch_size, loss_fn, device, validation_iter):
    net = net.to(device)
    for epoch in range(num_epochs):
        training_loss = train_epoch(net, train_iter, updater, batch_size, loss_fn, device)
        if epoch % 10 == 0:
            validation_loss = validate(net, validation_iter, batch_size, loss_fn, device)
            print(
                f"epoch:{epoch} ==> training average loss:{training_loss}\n                  validation loss:{validation_loss}\n")


def validate(net, validation_iter, batch_size, loss_fn, device):
    metric = d2l.Accumulator(2)
    net.eval()
    with torch.no_grad():
        for X, Y in validation_iter:
            # 第一维和第二维交换
            if len(X.shape) == 3:
                X, Y = X.permute(1, 0, 2), Y.permute(1, 0, 2)
            elif len(X.shape) == 2:
                X, Y = X.T, Y.T
            X, Y = X.to(device), Y.to(device)
            state = net.begin_state(batch_size, device)
            Y_hat, state = net(X, state)
            l = loss_fn(Y_hat, Y)
            metric.add(float(l) * batch_size, batch_size)
    return metric[0] / metric[1]


if __name__ == "__main__":
    seris_dim = 1
    # 时间序列维度
    dir_path = "data"
    file_name = "grouped_dm_pc_online_amt_stats.csv"
    batch_size = 5
    num_steps = 50
    hidden_size = 256
    num_epochs = 100
    lr = 0.01
    train_iter, validation_iter = get_data_iter_from_dataset(dir_path, file_name, columns_num=1, window_size=num_steps,
                                                             batch_size=batch_size)
    net = LSTM(seris_dim, hidden_size)
    loss_fn = nn.MSELoss()
    updater = torch.optim.SGD(net.parameters(), lr)
    device = d2l.try_gpu()
    train(net, num_epochs, train_iter, updater, batch_size, loss_fn, device, validation_iter)
