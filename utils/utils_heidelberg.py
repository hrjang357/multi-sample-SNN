import numpy as np
import tables


def one_hot(alphabet_size, idx):
    assert idx <= alphabet_size
    out = [0]*alphabet_size
    if idx > 0:
        out[idx - 1] = 1
    return out


def load_shd_accum(datafile, S_prime, digits, alphabet_size, window_length=10000):
    hdf5_file = tables.open_file(datafile, 'r')
    samples = np.hstack([np.where(hdf5_file.root.labels[:] == i) for i in digits]).flatten()
    n_neurons = 700

    res = []
    for i in samples:
        # variables to parse
        timestamps = hdf5_file.root.spikes.times[i] * 10e6  # get times in mus
        addr = hdf5_file.root.spikes.units[i]

        # create windows to store eventa
        windows = list(range(window_length, int(max(timestamps)), window_length))
        window_ptr = 0
        ts_pointer = 0

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

        spiking_neurons_per_ts = [[addr[ts] for ts in group] for group in timestamps_grouped]

        if S_prime <= len(windows):
            input_signal = np.array([[one_hot(alphabet_size, min(alphabet_size, spiking_neurons_per_ts[s].count(n))) for n in range(n_neurons)] for s in range(S_prime)])
            input_signal = input_signal.transpose(1, 2, 0)[None, :]

        else:
            input_signal = np.array([[one_hot(alphabet_size, min(alphabet_size, spiking_neurons_per_ts[s].count(n))) for n in range(n_neurons)] for s in range(len(windows))])
            padding = np.zeros([S_prime - len(windows), n_neurons, 2])

            input_signal = np.vstack((input_signal, padding))
            input_signal = input_signal.transpose(1, 2, 0)[None, :]

        res.append(input_signal.astype(bool))
    return np.vstack(res).astype(bool)
