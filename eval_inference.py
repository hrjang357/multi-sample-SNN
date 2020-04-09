import os
import numpy as np
import tables
import torch
from utils.inference_utils import get_distance, get_loss_bound

def evaluate_from_inference(path_dir, dataset_path, overwrite, query, tau):

    files_in_path = os.listdir(path_dir)
    
    for f in files_in_path:
        filename = os.path.splitext(f)  # split filename and ext
        if not filename[1] == '.pth':  # read only pth files
            continue

        if not '_t-' in f:
            continue 

        eval_filename_t = filename[0].split('_t-')
        eval_filename = eval_filename_t[0] + '_tau-' + str(tau) + '_t-' + eval_filename_t[1] +'.eval'

        if not overwrite:
            if eval_filename in files_in_path:
                continue

        if not query == None:
            if not query in filename[0]: 
                # print(filename[0])
                continue

        print('Start calculate for', filename[0], flush=True)

        # store the hyperparameters from filename to dictionary
        hp = dict()
        for v in filename[0].split('_'):
            k = v.split('-')
            if k[0] in ['digits', 'eptrain', 'eptest', 'Nk']:
                hp[k[0]] = int(k[1])
            else:
                hp[k[0]] = k[1]

        if hp['mode'] == 'mb':
            mode = 'train_ml_multi_elbo'
        elif hp['mode'] == 'iw':
            mode = 'train_ml_multi'
        elif hp['mode'] == 'iwb':
            mode = 'train_ml_multi_ls'
        elif hp['mode'] == 'gem':
            mode = 'train_ml_multi_nols'
        elif hp['mode'] == 'mb1':
            mode = 'train_ml'

        load_filename=os.path.join(path_dir, f)
        load_data = torch.load(load_filename)
        network = load_data['network']
        network.set_tau(tau)

        # Load dataset
        i_train, o_train, i_test, o_test = get_dataset(dataset_path, hp['digits'], hp['task'], hp['ex'], hp['eptrain'], hp['eptest'])

        # Calculate the results
        loss_bound_train, _, _ = get_loss_bound(network, i_train, o_train, hp['Nk'], mode)
        loss_bound_test, _, _ = get_loss_bound(network, i_test, o_test, hp['Nk'], mode)
        
        d_avg_train, d_majority_train, d_fix_train, spikenum_hid_train, _, spikenum_hid_majority_train, spikenum_hid_fix_train = get_distance(network, i_train, o_train, hp['Nk'], mode)
        d_avg_test, d_majority_test, d_fix_test, spikenum_hid_test, _, spikenum_hid_majority_test, spikenum_hid_fix_test = get_distance(network, i_test, o_test, hp['Nk'], mode)

        result_dict = dict()
        result_dict['loss_bound_train'] = loss_bound_train
        result_dict['loss_bound_test'] = loss_bound_test
        result_dict['d_avg_train'] = d_avg_train
        result_dict['d_majority_train'] = d_majority_train
        result_dict['d_fix_train'] = d_fix_train
        result_dict['spikenum_hid_train'] = spikenum_hid_train
        result_dict['spikenum_hid_majority_train'] = spikenum_hid_majority_train
        result_dict['spikenum_hid_fix_train'] = spikenum_hid_fix_train
        result_dict['d_avg_test'] = d_avg_test
        result_dict['d_majority_test'] = d_majority_test
        result_dict['d_fix_test'] = d_fix_test
        result_dict['spikenum_hid_test'] = spikenum_hid_test
        result_dict['spikenum_hid_majority_test'] = spikenum_hid_majority_test
        result_dict['spikenum_hid_fix_test'] = spikenum_hid_fix_test

        save_path = os.path.join(path_dir, eval_filename)
        torch.save(result_dict, save_path)

def get_dataset(dataset_path, digits, task, train_indices_mode, eptrain, eptest):

    # random seed 
    torch.manual_seed(0)
    np.random.seed(0)   
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if digits == 2:
        digit_list = [1, 7]
    elif digits <= 10:
        digit_list = [i for i in range(digits)]
    else:
        digit_list = [i for i in range(10)]

    data = tables.open_file(dataset_path)

    input_train_total = torch.FloatTensor(data.root.train.data[:])
    output_train_total = torch.FloatTensor(data.root.train.label[:])

    input_test_total = torch.FloatTensor(data.root.test.data[:])
    output_test_total = torch.FloatTensor(data.root.test.label[:])

    indices_train = np.hstack([np.where(np.argmax(np.sum(data.root.train.label[:], axis=(-1, -2)), axis=-1) == i)[0] for i in digit_list])
    input_train = input_train_total[indices_train]
    output_train = output_train_total[indices_train]
    output_train = output_train[:,digit_list]

    test_indices = np.hstack([np.where(np.argmax(np.sum(data.root.test.label[:], axis=(-1, -2)), axis=-1) == i)[0] for i in digit_list])
    input_test = input_test_total[test_indices]
    output_test = output_test_total[test_indices]
    output_test = output_test[:,digit_list]

    # for prediction
    if task == 'prediction':
        input_train_old = input_train
        input_test_old = input_test
        img_size = int(np.sqrt(input_train.shape[1]))

        input_idx = np.arange(input_train_old.shape[1]).reshape((img_size, img_size))[:,np.arange(int(img_size/2))].reshape(-1)
        output_idx = np.arange(input_train_old.shape[1]).reshape((img_size, img_size))[:,int(img_size/2):].reshape(-1)

        input_train = input_train_old[:, input_idx, :, :]
        output_train = input_train_old[:, output_idx, :, :]
        input_test = input_test_old[:, input_idx, :, :]
        output_test = input_test_old[:, output_idx, :, :]

    # Randomly select training/testing samples
    if train_indices_mode == 'single':
        train_indices = np.random.choice(np.arange(1), [eptrain], replace=True)
    elif train_indices_mode == 'multiple':
        n_ref = int(eptrain/len(digit_list))
        list_ref = []
        for i in range(len(digit_list)):
            list_ref = np.append( list_ref, np.random.choice(np.arange( int(input_train.shape[0]/len(digit_list))*(i), int(input_train.shape[0]/len(digit_list))*(i)+5 ), [n_ref], replace=True) )
        train_indices = np.random.permutation(list_ref)
        #indices = np.random.permutation(np.append( np.random.choice(np.arange(5), [int(epochs/2)], replace=True), np.random.choice(np.arange(input_train.shape[0]/2, input_train.shape[0]/2+5), [int(epochs/2)], replace=True)  ))
    else:
        if eptrain > input_train.shape[0]:
            train_indices = np.random.choice(np.arange(input_train.shape[0]), [eptrain], replace=True)    
        else:
            train_indices = np.random.choice(np.arange(input_train.shape[0]), [eptrain], replace=False)    

    if eptest > input_test.shape[0]:
        test_indices = np.random.choice(np.arange(input_test.shape[0]), [eptest], replace=True)
    else:
        test_indices = np.random.choice(np.arange(input_test.shape[0]), [eptest], replace=False)

    return input_train[train_indices], output_train[train_indices], input_test[test_indices], output_test[test_indices]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluation with inference.')
    parser.add_argument('-o', '--overwrite', help='Overwrite all results',
                    action="store_true")
    parser.add_argument('-q', '--query', help="Calculate the files if name includes the query")
    parser.add_argument('-p', '--path', default='results', help="Path of eval result files (pth files)")
    parser.add_argument('-d', '--dataset', default='../datasets/mnist-dvs/mnist_dvs_25ms_26pxl_10_digits_C_1.hdf5',
                    help="Path of dataset")
    parser.add_argument('-t', '--tau', default=20, type=int, help="Tau")
    args = parser.parse_args()

    evaluate_from_inference(args.path, args.dataset, args.overwrite, args.query, args.tau)
