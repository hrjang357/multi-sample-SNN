import tables
import os
import glob
from utils_dvs import load_dvs
from utils_heidelberg import load_shd_accum
import re
import numpy as np
import math


def make_mnist_dvs(path_to_data, path_to_hdf5, digits, max_pxl_value, min_pxl_value, T_max, window_length, scale, alphabet_size, polarity=True):

    """"
    Preprocess the .aedat file and save the dataset as an .hdf5 file
    """

    dirs = [r'/' + dir_ for dir_ in os.listdir(path_to_data)]
    allFiles = {key: [] for key in dirs}

    S_prime = math.ceil(T_max/window_length)

    pattern = [1]#, 0, 0, 0, 0]   # the pattern used as output for the considered digit

    hdf5_file = tables.open_file(path_to_hdf5, 'w')

    train = hdf5_file.create_group(where=hdf5_file.root, name='train')
    train_data_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='data', atom=tables.BoolAtom(), shape=(0, (max_pxl_value - min_pxl_value + 1)**2,
                                                                                                                       alphabet_size, S_prime))
    train_labels_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='label', atom=tables.BoolAtom(), shape=(0, len(digits), alphabet_size, S_prime))

    test = hdf5_file.create_group(where=hdf5_file.root, name='test')
    test_data_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='data', atom=tables.BoolAtom(), shape=(0, (max_pxl_value - min_pxl_value + 1)**2,
                                                                                                                     alphabet_size, S_prime))
    test_labels_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='label', atom=tables.BoolAtom(), shape=(0, len(digits), alphabet_size, S_prime))

    for i, digit in enumerate(digits):
        for dir_ in dirs:
            if dir_.find(str(digit)) != -1:
                for subdir, _, _ in os.walk(path_to_data + dir_):
                    if subdir.find(scale) != -1:
                        for j, file in enumerate(glob.glob(subdir + r'/*.aedat')):
                            output_signal = np.vstack((np.array([[[0] * S_prime] * i
                                                                    + [pattern * int(S_prime / len(pattern)) + pattern[:(S_prime % len(pattern))]]
                                                                    + [[0] * S_prime] * (len(digits) - 1 - i)], dtype=bool),
                                                        np.zeros([alphabet_size - 1, len(digits), S_prime], dtype=bool))).transpose(1, 0, 2)[None, :]


                            if j < 0.9*len(glob.glob(subdir + r'/*.aedat')):
                                if j % 50 == 0:
                                    print('train', file)

                                train_data_array.append(load_dvs(file, S_prime, min_pxl_value=min_pxl_value, max_pxl_value=max_pxl_value,
                                                                    window_length=window_length, alphabet_size=alphabet_size, polarity=polarity))
                                train_labels_array.append(output_signal)
                            else:
                                if j % 20 == 0:
                                    print('test', file)
                                    
                                test_data_array.append(load_dvs(file, S_prime, min_pxl_value=min_pxl_value, max_pxl_value=max_pxl_value,
                                                                window_length=window_length, alphabet_size=alphabet_size, polarity=polarity))
                                test_labels_array.append(output_signal)

    train_shape = np.concatenate((hdf5_file.root.train.data[:], hdf5_file.root.train.label[:]), axis=1).shape
    test_shape = np.concatenate((hdf5_file.root.test.data[:], hdf5_file.root.test.label[:]), axis=1).shape

    stats = hdf5_file.create_group(where=hdf5_file.root, name='stats')
    train_data_array = hdf5_file.create_array(where=hdf5_file.root.stats, name='train', atom=tables.Atom.from_dtype(np.dtype('int')),
                                                  obj=train_shape)
    test_data_array = hdf5_file.create_array(where=hdf5_file.root.stats, name='test', atom=tables.Atom.from_dtype(np.dtype('int')),
                                                  obj=test_shape)

    hdf5_file.close()


def make_shd(path_to_train, path_to_test, path_to_hdf5, digits, alphabet_size, window_length):

    train_data = tables.open_file(path_to_train, 'r')
    test_data = tables.open_file(path_to_test, 'r')

    T_max = max([max(train_data.root.spikes.times[i]) for i in range(len(train_data.root.labels))]) * 1e6
    S_prime = math.ceil(T_max/window_length)

    pattern = [1, 0, 0, 0, 0]   # the pattern used as output for the considered digit

    hdf5_file = tables.open_file(path_to_hdf5, 'w')


    # Make train group and arrays
    train = hdf5_file.create_group(where=hdf5_file.root, name='train')
    train_data_array = hdf5_file.create_array(where=hdf5_file.root.train, name='data', atom=tables.BoolAtom(),
                                              obj=load_shd_accum(path_to_train, S_prime, digits, alphabet_size, window_length))

    n_samples_per_label = np.array([len(np.where(train_data.root.labels[:] == i)[0]) for i in digits])
    output_signal = np.zeros([np.sum(n_samples_per_label), len(digits), alphabet_size, S_prime])

    for i, _ in enumerate(digits):
        output_signal[np.sum(n_samples_per_label[:i]):np.sum(n_samples_per_label[:i+1]), i, 0, :] \
            = np.array(pattern * int(S_prime/len(pattern)) + pattern[:(S_prime % len(pattern))])
    train_labels_array = hdf5_file.create_earray(where=hdf5_file.root.train, name='label', atom=tables.BoolAtom(), obj=output_signal.astype(np.bool))


    test = hdf5_file.create_group(where=hdf5_file.root, name='test')
    test_data_array = hdf5_file.create_array(where=hdf5_file.root.test, name='data', atom=tables.BoolAtom(),
                                             obj=load_shd_accum(path_to_test, S_prime, digits, alphabet_size, window_length))

    n_samples_per_label = np.array([len(np.where(test_data.root.labels[:] == i)[0]) for i in digits])
    output_signal = np.zeros([np.sum(n_samples_per_label), len(digits), alphabet_size, S_prime])
    for i, _ in enumerate(digits):
        output_signal[np.sum(n_samples_per_label[:i]):np.sum(n_samples_per_label[:i+1]), i, 0, :] \
            = np.array(pattern * int(S_prime/len(pattern)) + pattern[:(S_prime % len(pattern))])
    test_labels_array = hdf5_file.create_earray(where=hdf5_file.root.test, name='label', atom=tables.BoolAtom(), obj=output_signal.astype(np.bool))

    train_data.close()
    test_data.close()
    hdf5_file.close()


if __name__ == "__main__":
    # path_to_data = r'path/to/mnist-dvs-processed'
    #path_to_data = r'C:\Users\K1804053\Desktop\PhD\Federated SNN\processed_polarity'
    path_to_data = r'C:\Users\k1775597\Dropbox\Research\simulations\workspace\datasets\mnist-dvs'

    # digits to consider
    digits = [i for i in range(10)]
    #digits = [1, 7]

    # Pixel values to consider
    max_pxl_value = 73
    min_pxl_value = 48

    T_max = int(2e6)  # maximum duration of an example in us
    window_length = 25000

    scale = 'scale4'
    polarity = False
    alphabet_size = 2

    # path_to_hdf5 = r'C:/Users/K1804053/PycharmProjects/datasets/mnist-dvs/mnist_dvs_%dms_%dpxl_%d_digits_polarity.hdf5' \
    #                % (int(window_length / 1000), max_pxl_value - min_pxl_value + 1, len(digits))
    #path_to_hdf5 = r'C:/Users/K1804053/PycharmProjects/datasets/mnist-dvs/mnist_dvs_%dms_%dpxl_%d_digits_C_%d.hdf5' \
    #               % (int(window_length / 1000), max_pxl_value - min_pxl_value + 1, len(digits), alphabet_size)
    path_to_hdf5 = r'C:\Users\k1775597\Dropbox\Research\simulations\workspace\datasets\mnist-dvs\mnist_dvs_%dms_%dpxl_%d_digits_C_%d_temp.hdf5' \
                   % (int(window_length / 1000), max_pxl_value - min_pxl_value + 1, len(digits), alphabet_size)                   

    #
    make_mnist_dvs(path_to_data, path_to_hdf5, digits, max_pxl_value, min_pxl_value, T_max, window_length, scale, alphabet_size, polarity)