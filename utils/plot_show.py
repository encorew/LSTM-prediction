import numpy as np
from matplotlib import pyplot as plt


class Plot:
    def __init__(self, raw_data, prediction=np.array([0, 0]), time=np.array([0, 0]), anomaly_indice=np.array([0, 0])):
        self.raw_data = raw_data
        self.prediction = prediction
        self.anomaly_indice = anomaly_indice
        self.time = time
        if time == np.array([0, 0]):
            self.time = range(len(raw_data))

    def draw(self, select_dim=0):
        if len(self.raw_data.shape) > 1:
            # 多维
            plt.plot(self.time, self.raw_data[:, select_dim])
            if self.prediction != np.array([0, 0]):
                plt.plot(self.time, self.prediction[:, select_dim])
        else:
            plt.plot(self.time, self.raw_data)
            if self.prediction != np.array([0, 0]):
                plt.plot(self.time, self.prediction)
        plt.xticks(rotation=45)
        # plt.show()
