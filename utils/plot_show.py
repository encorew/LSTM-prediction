from matplotlib import pyplot as plt


class Plot:
    def __init__(self, raw_data, anomaly_indice=None, time=None):
        self.raw_data = raw_data
        self.anomaly_indice = anomaly_indice
        self.time = time

    def draw(self):
        if len(self.raw_data.shape) > 1:
            # 多维
            plt.plot(self.time, self.raw_data[:, 0])
        else:
            plt.plot(self.time, self.raw_data)
        plt.xticks(rotation=45)
        plt.show()
