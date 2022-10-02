import os

import numpy as np
import pandas as pd
import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt
from torch import nn
from lstm import LSTM
from train import train
from utils.data_preprocess import get_data_iter_from_dataset, get_raw_data, inverse_normalization
from utils.plot_show import Plot


def predict(net, predict_base, device):
    # 先把前window_size个放好
    net = net.to(device)
    window_size, vector_size = predict_base[0].shape[0], predict_base[0].shape[1]
    whole_prediction = [predict_base[0][i, :] for i in range(window_size)]
    for X in predict_base:
        X = X.reshape(window_size, 1, vector_size)
        X = torch.Tensor(X).to(device)
        state = net.begin_state(batch_size=1, device=device)
        outputs, state = net(X, state)
        # 最后一个点的第一个batch,因为就一个batch
        present_predict = outputs[-1, 0].cpu().detach().numpy()
        whole_prediction.append(present_predict)
    return np.array(whole_prediction)


def save_prediction_as_file(prediction, real, time, dir, file_name):
    data = {'RATE_TIME': time, 'original_value': real.reshape(-1), 'predicted_value': prediction.reshape(-1)}
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(dir, file_name + ".csv"), index=False)


def plot_show_predition_performance(prediction_file_dir, prediction_file_name):
    raw_data, time, _, _ = get_raw_data(prediction_file_dir, prediction_file_name, columns_num=[1, 2],
                                        time_column_num=0,
                                        normalize=False)
    real, prediction = raw_data[:, 0], raw_data[:, 1]
    # myPlot = Plot(real, prediction, time=time)
    # myPlot.draw()
    plt.plot(time, real)
    plt.plot(time, prediction)
    plt.show()


if __name__ == "__main__":
    seris_dim = 1
    # 时间序列维度
    dir_path = "data"
    prediction_path = "data/predicted"
    model_paramater_path = "model_parameters"
    file_name = "grouped_dm_agent_ele_purchase_analyse_result.csv"
    file_name2 = "grouped_dm_pc_online_amt_stats.csv"
    file_name3 = "grouped_dm_pub_pile_recharge.csv"
    file_name4 = "grouped_dm_uc_user_center_business_analyse.csv"
    file = file_name3
    batch_size = 5
    num_steps = 50
    hidden_size = 250
    num_epochs = 500
    lr = 0.01
    columns_num = 3
    save_parameter_name = 'col-' + str(columns_num) + 'b' + str(batch_size) + 'w' + str(num_steps) + 'h' + str(
        hidden_size) + '.pkl'
    raw_data, time, columns_name, normalizer = get_raw_data(dir_path, file, columns_num, time_column_num=0,
                                                            normalize=True)
    train_iter, validation_iter, predict_base = get_data_iter_from_dataset(dir_path, file, columns_num=columns_num,
                                                                           window_size=num_steps,
                                                                           batch_size=batch_size,
                                                                           shuffle=True)
    net = LSTM(seris_dim, hidden_size)
    loss_fn = nn.MSELoss()
    updater = torch.optim.SGD(net.parameters(), lr)
    device = d2l.try_gpu()
    save_folder = os.path.join(model_paramater_path, file)
    train(net, num_epochs, train_iter, updater, batch_size, loss_fn, device, save_folder, validation_iter,
          save_parameter_name)
    net.load_state_dict(torch.load(os.path.join(save_folder, "best " + save_parameter_name)))
    prediction = predict(net, predict_base, device)
    raw_data = inverse_normalization(raw_data, normalizer)
    prediction = inverse_normalization(prediction, normalizer)
    plt.plot(time, raw_data)
    plt.plot(time, prediction)
    plt.show()
    save_prediction_as_file(prediction, raw_data, time, prediction_path, columns_name)
    plot_show_predition_performance(prediction_path, columns_name + ".csv")
