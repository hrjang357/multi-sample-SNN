from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from SNN import SNNetwork
from utils.training_multi_utils import *
from utils.training_utils import *
from utils.inference_utils import *
from plot_snn.eval import *
import time
import numpy as np
import tables
import math
import argparse
import os
import matplotlib.pyplot as plt
import pdb
from torch.utils.tensorboard import SummaryWriter
import datetime
import scipy.misc
from scipy import ndimage

if __name__ == "__main__":
    
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    #task_digits_ex_dec_eptrain_eptest_mode_niter_Nh_Nk_lr_lrconst_kappa_Nb_nll_time.pth

    # Training arguments
    parser.add_argument('--where')
    parser.add_argument('--dataset')
    parser.add_argument('--task', default=None, type=str, help='Type of training task: classification or prediction')
    parser.add_argument('--digits', default=2, type=int, help='Number of digits for training and testing: 2 (1,7) / 5 (0-4) / 10 (0-9)')
    parser.add_argument('--ex', help='Single or Multiple training examples')
    parser.add_argument('--dec', help='Single, majority, or manxum of decision rule for classification')
    parser.add_argument('--eptrain', default=None, type=int, help='Number of samples to train on for each experiment')
    parser.add_argument('--eptest', default=None, type=int, help='Number of samples to test on')
    parser.add_argument('--mode', help='Feedforward or interactive readout')
    parser.add_argument('--niter', default=10, type=int, help='Number of times every experiment will be repeated')
    parser.add_argument('--num_hid', default=1, type=int, help='Number of hidden neurons')
    parser.add_argument('--num_sample', default=1, type=int, help='Number of samples')
    parser.add_argument('--lr', default=0.05, type=float, help='Learning rate')
    parser.add_argument('--lrconst', default=1, type=float, help='Constant for learning rate decrease')
    parser.add_argument('--kappa', default=0.05, type=float, help='Learning signal and eligibility trace decay coefficient')
    parser.add_argument('--n_basis', default=8, type=int, help='Number of synaptic filters')
    parser.add_argument('--nll', default=10, type=int, help='Number of samples to estiate log-likelihood')
    parser.add_argument('--alpha', default=3, type=float, help='Alpha softmax coefficient')
    parser.add_argument('--debug_period', default=10, type=int, help='Period for debuging in training')
    parser.add_argument('--plot_flag', default=1, type=int, help='Flag for plotting graphs')
    parser.add_argument('--tau', default=10, type=int, help='Tau (memory) of synaptic filters')
    parser.add_argument('--time', default=None, type=str, help='Execution time')
    parser.add_argument(
        "--time_range",
        nargs="*",
        type=str,
        default=None,
        help='time range of execution time',
    )
    parser.add_argument(
        "--timestep",
        nargs="*",
        type=str,
        default=None,
        help='time step lists for inference',
    )

    args = parser.parse_args()

niter = args.niter
task = args.task

# random seed (0) for training/test datasets
torch.manual_seed(0)
np.random.seed(0)   
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


dataset = r'../datasets/mnist-dvs/mnist_dvs_25ms_26pxl_10_digits_C_1.hdf5'
data = tables.open_file(dataset)

input_train_total = torch.FloatTensor(data.root.train.data[:])
output_train_total = torch.FloatTensor(data.root.train.label[:])

input_test_total = torch.FloatTensor(data.root.test.data[:])
output_test_total = torch.FloatTensor(data.root.test.label[:])

if args.digits == 2:
    digits = [1,7]
elif args.digits <= 10:
    digits = [i for i in range(args.digits)]
else:
    digits = [i for i in range(10)]

indices_train = np.hstack([np.where(np.argmax(np.sum(data.root.train.label[:], axis=(-1, -2)), axis=-1) == i)[0] for i in digits])
input_train = input_train_total[indices_train]
output_train = output_train_total[indices_train]
output_train = output_train[:,digits]

test_indices = np.hstack([np.where(np.argmax(np.sum(data.root.test.label[:], axis=(-1, -2)), axis=-1) == i)[0] for i in digits])
input_test = input_test_total[test_indices]
output_test = output_test_total[test_indices]
output_test = output_test[:,digits]
 
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


# input arguments
train_indices_mode = args.ex
dec_type = args.dec
if args.eptrain:
    epochs = args.eptrain
else:
    epochs = input_train.shape[0]

if args.eptest:
    epochs_test = args.eptest
else:
    epochs_test = input_test.shape[0]

if args.num_hid:
    n_hidden_neurons = args.num_hid
else:
    n_hidden_neurons = 1

if args.num_sample:
    num_samples = args.num_sample
else:
    num_samples = 1

if args.time:
    current_time = args.time
else: 
    current_time = args.time_range[0]

if args.time_range:
    time_range = args.time_range
else:
    time_range = [str(args.time), str(args.time+10)]

if args.timestep:
    timestep = args.timestep
else: 
    print('no timestep input arguments')

if args.mode == 'mb':
    mode = 'train_ml_multi_elbo'
elif args.mode == 'iw':
    mode = 'train_ml_multi'
elif args.mode == 'iwb':
    mode = 'train_ml_multi_ls'
elif args.mode == 'gem':
    mode = 'train_ml_multi_nols'
elif args.mode == 'mb1':
    mode = 'train_ml'
else:
    print('invalid training mode')

learning_rate = args.lr
kappa = args.kappa
lr_const = args.lrconst
alpha = args.alpha
num_ll_est = args.nll
plot_flag = args.plot_flag
n_basis = args.n_basis
tau = args.tau

load_path = os.getcwd() + r'/results'
load_file_name = '/task-'+str(task)+'_digits-'+str(args.digits)+'_ex-'+str(train_indices_mode)+'_dec-'+str(dec_type)\
    +'_eptrain-'+str(epochs)+'_eptest-'+str(epochs_test)+'_mode-'+str(args.mode)+'_niter-'+str(niter)\
        +'_Nh-'+str(n_hidden_neurons)+'_Nk-'+str(num_samples)+'_lr-'+str(learning_rate)+'_lrconst-'+str(lr_const)+\
            '_kappa-'+str(kappa)+'_Nb-'+str(n_basis)+'_nll-'+str(num_ll_est)+'_time-'+str(current_time)+'_t-'+str(timestep[0])

print(' ', flush=True)
print(load_file_name + '.pth', flush=True)


# Randomly select training/testing samples
if train_indices_mode == 'single':
    indices = np.random.choice(np.arange(1), [epochs], replace=True)
elif train_indices_mode == 'multiple':
    n_ref = int(epochs/len(digits))
    list_ref = []
    for i in range(len(digits)):
        list_ref = np.append( list_ref, np.random.choice(np.arange( int(input_train.shape[0]/len(digits))*(i), int(input_train.shape[0]/len(digits))*(i)+5 ), [n_ref], replace=True) )
    indices = np.random.permutation(list_ref)
else:
    if epochs > input_train.shape[0]:
        indices = np.random.choice(np.arange(input_train.shape[0]), [epochs], replace=True)    
    else:
        indices = np.random.choice(np.arange(input_train.shape[0]), [epochs], replace=False)    

if epochs_test > input_test.shape[0]:
    test_indices = np.random.choice(np.arange(input_test.shape[0]), [epochs_test], replace=True)
else:
    test_indices = np.random.choice(np.arange(input_test.shape[0]), [epochs_test], replace=False)


# # random seed for niter
# torch.manual_seed(niter)
# np.random.seed(niter)   
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(niter)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

print('Task %s - Nh %d, Nk %d, lr %4.3f kappa %3.2f, Nbasis %d, tau %d, time range [ %s, %s]' \
    %(task, n_hidden_neurons, num_samples, learning_rate, kappa, n_basis, tau, time_range[0], time_range[1]), flush=True)


# Inference 
hprof_file_list = get_hprofs_from_files(load_path)
print('total %d files' %(len(hprof_file_list)), flush=True)

query_hprof_t = { 
'task': str(task),
'digits': str(args.digits),
'ex': str(train_indices_mode),
'dec': str(dec_type),
'eptrain': str(epochs),
'eptest': str(epochs_test),
'mode': str(args.mode),
'Nh': str(n_hidden_neurons),
'Nk': str(num_samples),
'lr': str(learning_rate),
'lrconst': str(lr_const),
'kappa': str(kappa),
'Nb': str(n_basis),
'nll': str(num_ll_est),
'time': str(current_time)
}

# sweep mode
sweep_t = {'hyperparam': 't', 'value': timestep}

# Inference on Training dataset
inference_metrics = ['loss_bound', 'distance_avg', 'distance_ref']

print(' ', flush=True)
print('Inference performance on training dataset', flush=True)
t0 = time.time()
get_inference_eval(hprof_file_list, sweep_t, inference_metrics, query_hprof_t, input_train[indices], output_train[indices], time_range)
print('Number of samples for inference: %d, time: %f' % (epochs, time.time() - t0), flush=True)


print(' ', flush=True)
print('Inference performance on test dataset', flush=True)
t0 = time.time()
get_inference_eval(hprof_file_list, sweep_t, inference_metrics, query_hprof_t, input_test[test_indices], output_test[test_indices], time_range)
print('Number of samples for inference: %d, time: %f' % (epochs_test, time.time() - t0), flush=True)
print(' ', flush=True)

plt.show()