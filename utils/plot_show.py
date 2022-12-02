import numpy as np
from matplotlib import pyplot as plt

from utils.data_preprocess import read_data


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


def draw_anomaly(raw_data, labels):
    plt.plot(raw_data[:, 5])
    anomaly_regions = []
    anomaly_region_start = False
    region_left = 0
    region_right = 0
    for idx, label in enumerate(labels):
        if not anomaly_region_start and label == 1:
            anomaly_region_start = True
            region_left = idx - 0.2
        elif anomaly_region_start and label == 0:
            anomaly_region_start = False
            region_right = idx - 0.8
            anomaly_regions.append((region_left, region_right))
    for region in anomaly_regions:
        plt.axvspan(xmin=region[0], xmax=region[1], facecolor="r", alpha=0.3)
    plt.show()


if __name__ == '__main__':
    # data = read_data('../../data', 'MSL_test.pkl')
    # labels = read_data('../../data', 'MSL_test_label.pkl')
    # draw_anomaly(data, labels)
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(0, 10, 1)
    y = np.array([3,4,30,20,21,24,26,27,11,10])
    z1 = np.polyfit(x, y, 3)  # 用3次多项式拟合，输出系数从高到0
    p1 = np.poly1d(z1)  # 使用次数合成多项式
    x_for_show = np.arange(0,10,0.1)
    print(p1)
    yvals = p1(x_for_show)
    plt.ylim((-40,40))
    plt.plot(x, y, '*')
    plt.plot(x_for_show, yvals)
    plt.show()