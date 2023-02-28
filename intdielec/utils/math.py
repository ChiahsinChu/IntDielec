import numpy as np


def get_dev(x, y):
    x = np.array(x)
    y = np.array(y)
    delta_x = np.diff(x)
    delta_y = np.diff(y)
    out_x = (x[1:] + x[:-1]) / 2
    return out_x, delta_y / delta_x


def get_bin_ave(data, bin_size=10):
    total_size = len(data)
    out_size = bin_size * total_size // bin_size
    data = np.reshape(data[:out_size], (-1, bin_size))
    data = np.mean(data, axis=-1)
    return data


def get_int(x, y):
    x = np.array(x)
    y = np.array(y)
    ave_y = (y[1:] + y[:-1]) / 2
    delta_x = np.diff(x)
    x = (x[1:] + x[:-1]) / 2
    return x, np.cumsum(ave_y * delta_x)


def get_cum_ave(data):
    cum_sum = data.cumsum()
    cum_ave = cum_sum / (np.arange(len(data)) + 1)
    return cum_ave