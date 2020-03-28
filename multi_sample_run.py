from __future__ import print_function
import torch
from SNN import SNNetwork
from utils.training_utils import train_ml_online, get_acc, get_log_likelihood, make_network_parameters
from utils.training_multi_utils import train_ml_multi_online, train_ml_multi_elbo_online, duplicate_networks, train_ml_multi_nols_online, train_ml_multi_online_ls
from utils.inference_utils import get_inference, get_distance, get_loss_bound
import utils.training_utils
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

''''
Code snippet to train a multivalued SNN.
'''

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
        "--debug_flag",
        nargs="*",
        type=int,
        default=None,
        help='Flag for debuging in training (train/test)',
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

save_path = os.getcwd() + r'/results'
save_log_path = os.getcwd() + r'/runs'

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


### Network parameters
n_input_neurons = input_train.shape[1]
n_output_neurons = output_train.shape[1]
alphabet_size = input_train.shape[-2]
#mode = args.mode
train_indices_mode = args.ex
dec_type = args.dec


### Learning parameters
if args.eptrain:
    epochs = args.eptrain
else:
    epochs = input_train.shape[0]

if args.eptest:
    epochs_test = args.eptest
else:
    epochs_test = input_test.shape[0]

if args.debug_flag:
    debug_flag = args.debug_flag
    n_debug = args.debug_period
else:
    debug_flag = [0,0]

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
    current_time = datetime.datetime.now().strftime("%m%d%H%M")

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

# save file name
#task-prediction_digits-2_ex-single_dec-majority_eptrain-30_eptest-30_mode-iw_niter-3_Nh-2_Nk-10_lr-0.05_lrconst-1.5_kappa-0.05_Nb-8_nll-10_time-03131500.pth

save_file_name = '/task-'+str(task)+'_digits-'+str(args.digits)+'_ex-'+str(train_indices_mode)+'_dec-'+str(dec_type)\
    +'_eptrain-'+str(epochs)+'_eptest-'+str(epochs_test)+'_mode-'+str(args.mode)+'_niter-'+str(niter)\
        +'_Nh-'+str(n_hidden_neurons)+'_Nk-'+str(num_samples)+'_lr-'+str(learning_rate)+'_lrconst-'+str(lr_const)+\
            '_kappa-'+str(kappa)+'_Nb-'+str(n_basis)+'_nll-'+str(num_ll_est)+'_time-'+str(current_time)
            #+'_C-'+str(alphabet_size)

writer_folder_name = '/task-'+str(task)+'_digits-'+str(args.digits)+'_ex-'+str(train_indices_mode)+'_dec-'+str(dec_type)\
    +'_eptrain-'+str(epochs)+'_eptest-'+str(epochs_test)+'_mode-'+str(args.mode)+'_niter-'+str(niter)\
        +'_Nh-'+str(n_hidden_neurons)+'_Nk-'+str(num_samples)+'_lr-'+str(learning_rate)+'_lrconst-'+str(lr_const)+\
            '_kappa-'+str(kappa)+'_Nb-'+str(n_basis)+'_nll-'+str(num_ll_est)+'_time-'+str(current_time)


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


# random seed for niter
torch.manual_seed(niter)
np.random.seed(niter)   
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(niter)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print('Task %s - Exp %d: Nx %d Ny %d Nh %d, Nk %d, lr %4.3f kappa %3.2f, Nbasis %d, tau %d' \
    %(task, niter, n_input_neurons, n_output_neurons, n_hidden_neurons, num_samples, learning_rate, kappa, n_basis, tau), flush=True)

# writer for summary
writer = SummaryWriter(save_log_path + writer_folder_name)

# Randomly select training samples
print(indices.astype(int), flush=True)

S_prime = input_train.shape[-1]
S = epochs * S_prime

### Run training
# Create the network
network = SNNetwork(**utils.training_utils.make_network_parameters(n_input_neurons, n_output_neurons, n_hidden_neurons, alphabet_size, mode, n_basis, tau, tau))

# Train it
t0 = time.time()
if mode == 'train_ml': # MB1
    print('Train mode: train-ml-online with single sample', flush=True)
    acc_train, loss_output_train, spikenum_hid_train, spikenum_output_train, acc_test, loss_output_test, timestamp = \
        train_ml_online(network, input_train[indices], output_train[indices], input_test[test_indices], output_test[test_indices], learning_rate, lr_const, kappa, kappa, debug_flag, n_debug, num_ll_est, dec_type, writer, save_path+save_file_name)
elif mode == 'train_ml_multi': # IW
    print('Train mode: train-ml-online with IWAE multiple samples', flush=True)
    acc_train, loss_output_train, spikenum_hid_train, spikenum_output_train, acc_test, loss_output_test, timestamp = \
        train_ml_multi_online(network, input_train[indices], output_train[indices], input_test[test_indices], output_test[test_indices], learning_rate, lr_const, kappa, kappa, num_samples, debug_flag, n_debug, num_ll_est, dec_type, writer, save_path+save_file_name)
elif mode == 'train_ml_multi_ls': # IW-b
    print('Train mode: train-ml-online with IWAE multiple samples and per-sample learning signal', flush=True)
    acc_train, loss_output_train, spikenum_hid_train, spikenum_output_train, acc_test, loss_output_test, timestamp = \
        train_ml_multi_online_ls(network, input_train[indices], output_train[indices], input_test[test_indices], output_test[test_indices], learning_rate, lr_const, kappa, kappa, num_samples, debug_flag, n_debug, num_ll_est, dec_type, writer, save_path+save_file_name)
elif mode == 'train_ml_multi_nols': # GEM
    print('Train mode: train-ml-online with IWAE multiple samples without learning signal', flush=True)
    acc_train, loss_output_train, spikenum_hid_train, spikenum_output_train, acc_test, loss_output_test, timestamp = \
        train_ml_multi_nols_online(network, input_train[indices], output_train[indices], input_test[test_indices], output_test[test_indices], learning_rate, lr_const, kappa, kappa, num_samples, debug_flag, n_debug, num_ll_est, dec_type, writer, save_path+save_file_name)
elif mode == 'train_ml_multi_elbo': # MB
    print('Train mode: train-ml-online with standard multiple sample averaging', flush=True)
    acc_train, loss_output_train, spikenum_hid_train, spikenum_output_train, acc_test, loss_output_test, timestamp = \
        train_ml_multi_elbo_online(network, input_train[indices], output_train[indices], input_test[test_indices], output_test[test_indices], learning_rate, lr_const, kappa, kappa, num_samples, debug_flag, n_debug, num_ll_est, dec_type, writer, save_path+save_file_name)
    

# Copy the trained network for evaluation 
temp_net = SNNetwork(**make_network_parameters(network.n_input_neurons, network.n_output_neurons, network.n_hidden_neurons, network.alphabet_size, network.mode, network.n_basis_feedforward, network.tau_ff, network.tau_fb))
temp_net_params = network.get_parameters()
temp_net.set_parameters(temp_net_params)                        

# Final performance evaluation
training_sequence = torch.cat((input_train[indices], output_train[indices]), dim=1)
test_sequence = torch.cat((input_test[test_indices], output_test[test_indices]), dim=1)

# inference
dec_type_inference = 'majority'

# ### Train accuracy
loss_train_f, loss_output_train_f = get_log_likelihood(temp_net, training_sequence, num_ll_est)
acc_train_f = get_acc(temp_net, input_train[indices], output_train[indices], dec_type, num_samples)
acc_train_inference, spikenum_hid_train_inference, spikenum_output_train_inference, _ = get_inference(temp_net, input_train[indices], output_train[indices], dec_type_inference, num_samples, writer)
print('Exp %d: Final train accuracy (%s): %f' % (niter, dec_type, acc_train_f), flush=True)
print('Exp %d: Final train loss-output: %f' % (niter, loss_output_train_f), flush=True)
print('Exp %d: Final train (inference) accuracy (%s): %f' % (niter, dec_type_inference, acc_train_inference), flush=True)
print('Exp %d: Final spike num (inference) hidden: %d, %f / output: %d, %f' % (niter, \
    spikenum_hid_train_inference, spikenum_hid_train_inference/num_samples/n_hidden_neurons/(S), \
        spikenum_output_train_inference, spikenum_output_train_inference/num_samples/n_output_neurons/(S)), flush=True)

# ### Test accuracy
loss_test_f, loss_output_test_f = get_log_likelihood(temp_net, test_sequence, num_ll_est)
acc_test_f = get_acc(temp_net, input_test[test_indices], output_test[test_indices], dec_type, num_samples)
acc_test_inference, spikenum_hid_test_inference, spikenum_output_test_inference, _ = get_inference(temp_net, input_test[test_indices], output_test[test_indices], dec_type_inference, num_samples, writer)
print('Exp %d: Final test accuracy (%s): %f' % (niter, dec_type, acc_test_f), flush=True)
print('Exp %d: Final test loss-output: %f' % (niter, loss_output_test_f), flush=True)
print('Exp %d: Final test (inference) accuracy (%s): %f' % (niter, dec_type_inference, acc_test_inference), flush=True)
print('Exp %d: Final spike num (inference) hidden: %d, %f / output: %d, %f' % (niter, \
    spikenum_hid_test_inference, spikenum_hid_test_inference/num_samples/n_hidden_neurons/(epochs_test*S_prime), \
        spikenum_output_test_inference, spikenum_output_test_inference/num_samples/n_output_neurons/(epochs_test*S_prime)), flush=True)


# Save results 
save_state = {
    'mode': mode,
    'num_ite': niter,
    'epochs': epochs,
    'epochs_test': epochs_test, 
    'num_ll_est': num_ll_est,
    'learning_rate': learning_rate, 
    'lr_const': lr_const,
    'kappa': kappa,
    'alpha': alpha,
    'debug_flag': debug_flag,
    'n_debug': n_debug,
    'timestamp': timestamp,
    'n_hidden_neurons': n_hidden_neurons,
    'num_samples': num_samples,
    'acc_train': acc_train,
    'loss_output_train': loss_output_train,
    'acc_train_f': torch.Tensor([[acc_train_f]]),
    'loss_output_train_f': torch.Tensor([[loss_output_train_f]]),
    'spikenum_hid_train': spikenum_hid_train,
    'spikenum_output_train': spikenum_output_train,
    'acc_train_inference': torch.Tensor([[acc_train_inference]]),
    'spikenum_hid_train_inference': torch.Tensor([[spikenum_hid_train_inference]]),
    'spikenum_output_train_inference': torch.Tensor([[spikenum_output_train_inference]]),
    'acc_test': acc_test,
    'loss_output_test': loss_output_test,
    'acc_test_f': torch.Tensor([[acc_test_f]]),
    'loss_output_test_f': torch.Tensor([[loss_output_test_f]]),
    'acc_test_inference': torch.Tensor([[acc_test_inference]]),
    'spikenum_hid_test_inference': torch.Tensor([[spikenum_hid_test_inference]]),
    'spikenum_output_test_inference': torch.Tensor([[spikenum_output_test_inference]]),
    'network': network,
    'alphabet_size': alphabet_size,
    'dec_type': dec_type
}
torch.save(save_state, save_path + save_file_name+'.pth')

print('Number of samples trained on: %d, time: %f' % (epochs, time.time() - t0), flush=True)
print('')

writer.close()


if plot_flag == 1:
    xpos = [[i] for i in range(n_debug+1)]
    plt.figure()

    plt.plot(xpos, loss_output_train, 'o-', ms=3)

    plt.legend(framealpha=1, frameon=True)
    plt.xlabel('learning epochs')
    plt.ylabel('log-likelihood of output neurons')
    plt.title('%s: Nh %d, Nk %d' %(args.mode, n_hidden_neurons, num_samples))
    plt.grid(True)

    plt.show(block=False)
           
    plt.show()




