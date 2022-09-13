import os
import random
from copy import deepcopy

import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import torch
import torch.utils.data as Data
import pickle

from utils.plot_show import Plot


def normalization(dataset):
    normalizer = MinMaxScaler(feature_range=(0, 1))
    # sklearn中所有数据都需要2维
    if len(dataset.shape) == 1:
        dataset = dataset.reshape(-1, 1)
    return normalizer.fit_transform(dataset), normalizer


def inverse_normalization(dataset, normalizer):
    return normalizer.inverse_transform(dataset)


def read_data(data_dir, data_name, mode=None):
    if not mode:
        data_path = os.path.join(data_dir, data_name)
        extension = data_name.split('.')[-1]
        if extension == "pkl":
            fr = open(data_path, 'rb+')
            matrix_data = pickle.load(fr)
            return matrix_data
        elif extension == "csv":
            data_frame = pd.read_csv(data_path)
            return data_frame


def extract_raw_data_from_frame(csv_data_frame, columns_col_number, is_time=False):
    # iloc函数可以像numpy里下标那样对dataframe使用
    if not is_time:
        return np.array(csv_data_frame.iloc[:, columns_col_number])
    else:
        time_df = pd.to_datetime(csv_data_frame.iloc[:, columns_col_number].astype('str'))
        return np.array(time_df)


def split_data_by_window(data, window_size):
    total_num_steps = data.shape[0]
    sequence_list = [data[i:i + window_size] for i in range(total_num_steps - window_size + 1)]
    return sequence_list


def generate_data_iter(X, Y, batch_size, shuffle=False, drop_last=True):
    X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    data_set = Data.TensorDataset(X, Y)
    data_iter = Data.DataLoader(data_set, batch_size, shuffle, drop_last=drop_last)
    return data_iter


def get_data_iter_from_dataset(data_dir, file_name, columns_num, window_size, batch_size, time_column_num=0,
                                normalize=True):
    data_frame = read_data(data_dir, file_name)
    time = extract_raw_data_from_frame(data_frame, time_column_num, is_time=True)
    raw_data = extract_raw_data_from_frame(data_frame, columns_num)
    if normalize:
        raw_data, normalizer = normalization(raw_data)
    # myPlot = Plot(raw_data, time=time)
    # myPlot.draw()
    sequence_list = split_data_by_window(raw_data, window_size)
    training_fold_bound = -len(sequence_list) // 5
    train_list = sequence_list[:training_fold_bound]
    validation_list = sequence_list[training_fold_bound:]
    train_X, train_Y = np.array(train_list[:-1]), np.array(train_list[1:])
    validation_X, validation_Y = np.array(validation_list[:-1]), np.array(validation_list[1:])
    train_iter = generate_data_iter(train_X, train_Y, batch_size=batch_size)
    validation_iter = generate_data_iter(validation_X, validation_Y, batch_size=batch_size)
    return train_iter, validation_iter


if __name__ == "__main__":
    dir_path = "../data"
    file_name = "grouped_dm_pc_online_amt_stats.csv"
