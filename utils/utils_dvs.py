import numpy as np
import os
import struct


def one_hot(alphabet_size, idx):
    assert idx <= alphabet_size
    out = [0]*alphabet_size
    if idx > 0:
        out[idx - 1] = 1
    return out


def load_dvs(datafile, S_prime, min_pxl_value=48, max_pxl_value=80, window_length=25000, alphabet_size=None, polarity=True):
    # constants
    aeLen = 8  # 1 AE event takes 8 bytes
    readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
    td = 0.000001  # timestep is 1us
    xmask = 0x00fe
    ymask = 0x7f00
    pmask = 0x1

    aerdatafh = open(datafile, 'rb')
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)

    length = statinfo.st_size

    # header
    lt = aerdatafh.readline()
    while lt and lt[:1] == b'#':
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        continue

    # variables to parse
    timestamps = []
    xaddr = []
    yaddr = []
    polarities = []

    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen

    while p < length:
        addr, ts = struct.unpack(readMode, s)

        # parse event's data
        x_addr = 128 - 1 - ((xmask & addr) >> 1)
        y_addr = ((ymask & addr) >> 8)
        a_pol = 1 - 2 * (addr & pmask)

        if (x_addr >= min_pxl_value) & (x_addr <= max_pxl_value) & (y_addr >= min_pxl_value) & (y_addr <= max_pxl_value):
            timestamps.append(ts)
            xaddr.append(x_addr - min_pxl_value)
            yaddr.append(y_addr - min_pxl_value)
            polarities.append(a_pol)

        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen

    # create windows to store eventa
    windows = list(range(window_length - 1000, max(timestamps), window_length))
    window_ptr = 0
    ts_pointer = 0
    n_neurons = (max_pxl_value - min_pxl_value + 1)**2

    timestamps_grouped = [[] for _ in range(len(windows))]
    current_group = []

    while (ts_pointer < len(timestamps)) & (window_ptr < len(windows)):
        if timestamps[ts_pointer] <= windows[window_ptr]:
            current_group += [ts_pointer]
        else:
            timestamps_grouped[window_ptr] += current_group
            window_ptr += 1
            current_group = [ts_pointer]
        ts_pointer += 1

    if polarity:
        spiking_neurons_per_ts = [[(xaddr[ts] * (max_pxl_value - min_pxl_value + 1) + yaddr[ts]) * polarities[ts] for ts in group] for group in timestamps_grouped]
        if S_prime <= len(windows):
            input_signal = np.array([[[0, 1] if n in spiking_neurons_per_ts[s] else [1, 0] if -n in spiking_neurons_per_ts[s] else [0, 0] for n in range(n_neurons)] for s in range(S_prime)])
            input_signal = input_signal.transpose(1, 2, 0)[None, :]

        else:
            input_signal = np.array([[[0, 1] if n in spiking_neurons_per_ts[s] else [1, 0] if -n in spiking_neurons_per_ts[s] else [0, 0] for n in range(n_neurons)] for s in range(len(windows))])
            padding = np.zeros([S_prime - len(windows), n_neurons, 2])

            input_signal = np.vstack((input_signal, padding))
            input_signal = input_signal.transpose(1, 2, 0)[None, :]

    elif alphabet_size:
        spiking_neurons_per_ts = [[xaddr[ts] * (max_pxl_value - min_pxl_value + 1) + yaddr[ts] for ts in group] for group in timestamps_grouped]
        if S_prime <= len(windows):
            input_signal = np.array([[one_hot(alphabet_size, min(alphabet_size, spiking_neurons_per_ts[s].count(n))) for n in range(n_neurons)] for s in range(S_prime)])
            input_signal = input_signal.transpose(1, 2, 0)[None, :]

        else:
            input_signal = np.array([[one_hot(alphabet_size, min(alphabet_size, spiking_neurons_per_ts[s].count(n))) for n in range(n_neurons)] for s in range(len(windows))])
            padding = np.zeros([S_prime - len(windows), n_neurons, alphabet_size])

            input_signal = np.vstack((input_signal, padding))
            input_signal = input_signal.transpose(1, 2, 0)[None, :]

    return input_signal.astype(bool)


# load_dvs(r'C:\Users\K1804053\Desktop\PhD\Federated SNN\processed\processed_data0\scale4\mnist_0_scale04_0002.aedat',
#          80, min_pxl_value=48, max_pxl_value=73, window_length=25000, alphabet_size=5, polarity=False)
