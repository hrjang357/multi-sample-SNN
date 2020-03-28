import torch
import torch.nn as nn
import torch.nn.functional as F
import tables
import utils.filters as filters
import numpy as np
from SNN import SNNetwork
import pdb
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage

def custom_softmax(input_tensor, alpha):
    u = torch.max(input_tensor)

    return torch.exp(alpha*(input_tensor - u)) / torch.sum(torch.exp(alpha*(input_tensor - u)))


#def make_network_parameters(n_input_neurons, n_output_neurons, n_hidden_neurons, alphabet_size, mode, n_basis_ff=8, ff_filter=filters.raised_cosine_pillow_08, n_basis_fb=1,
#                            fb_filter=filters.raised_cosine_pillow_08, tau_ff=10, tau_fb=10, weights_magnitude=0.1, mu=1.5, task='supervised'):
def make_network_parameters(n_input_neurons, n_output_neurons, n_hidden_neurons, alphabet_size, mode, n_basis_ff, tau_ff, tau_fb, ff_filter=filters.raised_cosine_pillow_08, n_basis_fb=1,
                            fb_filter=filters.raised_cosine_pillow_08, weights_magnitude=0.1, mu=1.5, task='supervised'):
    """"
    Initializes a dictionary of network parameters with standard training values
    """
    if mode == 'train_ml':
        topology = torch.FloatTensor([[1] * n_input_neurons + [1] * n_hidden_neurons + [0] * n_output_neurons] * n_hidden_neurons +
                                     [[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_output_neurons)
    elif mode == 'train_rl':
        topology = torch.FloatTensor([[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_hidden_neurons +
                                     [[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_output_neurons)
    elif mode == 'train_ml_multi':
        topology = torch.FloatTensor([[1] * n_input_neurons + [1] * n_hidden_neurons + [0] * n_output_neurons] * n_hidden_neurons +
                                     [[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_output_neurons)
    elif mode == 'train_ml_multi_ls':
        topology = torch.FloatTensor([[1] * n_input_neurons + [1] * n_hidden_neurons + [0] * n_output_neurons] * n_hidden_neurons +
                                     [[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_output_neurons)                                     
    elif mode == 'train_ml_multi_varK':
        topology = torch.FloatTensor([[1] * n_input_neurons + [1] * n_hidden_neurons + [0] * n_output_neurons] * n_hidden_neurons +
                                     [[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_output_neurons)
    elif mode == 'train_ml_multi_nols':
        topology = torch.FloatTensor([[1] * n_input_neurons + [1] * n_hidden_neurons + [0] * n_output_neurons] * n_hidden_neurons +
                                     [[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_output_neurons)
    elif mode == 'train_ml_multi_elbo':
        topology = torch.FloatTensor([[1] * n_input_neurons + [1] * n_hidden_neurons + [0] * n_output_neurons] * n_hidden_neurons +
                                     [[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_output_neurons)
    elif mode == 'inference' or mode == 'test-ll':
        topology = torch.FloatTensor([[1] * n_input_neurons + [1] * n_hidden_neurons + [0] * n_output_neurons] * n_hidden_neurons +
                                     [[1] * n_input_neurons + [1] * n_hidden_neurons + [1] * n_output_neurons] * n_output_neurons)   
    else:
        print('At train time, mode should be one of "train_ml", "train_ml_multi", "train_ml_multi_ls", "train_ml_multi_elbo", "train_ml_multi_nols" or "train_rl"', flush=True)
        raise AttributeError

    network_parameters = {'n_input_neurons': n_input_neurons,
                          'n_output_neurons': n_output_neurons,
                          'n_hidden_neurons': n_hidden_neurons,
                          'topology': topology,
                          'alphabet_size': alphabet_size,
                          'n_basis_feedforward': n_basis_ff,
                          'feedforward_filter': ff_filter,
                          'n_basis_feedback': n_basis_fb,
                          'feedback_filter': fb_filter,
                          'tau_ff': tau_ff,
                          'tau_fb': tau_fb,
                          'weights_magnitude': weights_magnitude,
                          'mu': mu,
                          'task': task,
                          'mode': mode
                          }

    return network_parameters


def get_acc_and_loss(network, input_sequence, output_sequence):
    """"
    Compute loss and accuracy on the indices from the dataset precised as arguments
    """
    network.set_mode('test')
    network.reset_internal_state()

    S_prime = input_sequence.shape[-1]
    epochs = input_sequence.shape[0]
    S = S_prime * epochs

    loss = 0
    loss_output = 0
    outputs = torch.zeros([epochs, network.n_output_neurons])

    for s in range(S):
        if s % S_prime == 0:
            network.reset_internal_state()

        log_proba = network(input_sequence[int(s / S_prime), :, :, s % S_prime])
        loss += torch.sum(log_proba).numpy()
        loss_output += torch.sum(log_proba[network.output_neurons - network.n_non_learnable_neurons]).numpy()
        
        outputs[int(s / S_prime), :] += torch.sum(network.spiking_history[network.output_neurons, :, -1], dim=-2)

    predictions = torch.max(outputs, dim=-1).indices
    true_classes = torch.max(torch.sum(output_sequence, dim=(-1, -2)), dim=-1).indices

    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    return acc, loss, loss_output


def get_log_likelihood(network, input_sequence, num_ll_est):
    """
    Compute log-likelihood of input/output sequences precised as arguments with sufficient number of generations
    """
    network.set_mode('test-ll')
    network.reset_internal_state()

    S_prime = input_sequence.shape[-1]
    epochs = input_sequence.shape[0]
    
    S = S_prime * epochs
    loss = 0
    loss_output = 0
    
    for _ in range(num_ll_est):
        network.reset_internal_state()

        for s in range(S):
            if s % S_prime == 0:
                network.reset_internal_state()

            log_proba = network(input_sequence[int(s / S_prime), :, :, s % S_prime])
            loss += torch.sum(log_proba).numpy()
            loss_output += torch.sum(log_proba[network.output_neurons - network.n_non_learnable_neurons]).numpy()
            
    loss /= num_ll_est*epochs
    loss_output /= num_ll_est*epochs

    return loss, loss_output


def get_acc(network, input_sequence, output_sequence, dec_type, num_ll_est):
    """
    Compute accuracy of output sequence given input sequence precised as arguments with sufficient number of generations
    """
    network.set_mode('test-class')
    network.reset_internal_state()

    S_prime = input_sequence.shape[-1]
    epochs = input_sequence.shape[0]
    
    S = S_prime * epochs
    acc = 0
    
    if dec_type == 'single':
        num_ll_est = 1

    outputs = torch.zeros([num_ll_est, epochs, network.n_output_neurons])

    for nl in range(num_ll_est):
        network.reset_internal_state()
        
        for s in range(S):
            if s % S_prime == 0:
                network.reset_internal_state()

            log_proba = network(input_sequence[int(s / S_prime), :, :, s % S_prime])
            outputs[nl, int(s / S_prime), :] += torch.sum(network.spiking_history[network.output_neurons, :, -1], dim=-2)

    if (dec_type == 'majority'):
        tempout, tempcount = torch.unique(torch.max(outputs, dim=-1).indices, sorted=True, return_counts=True, dim=0)
        predictions = tempout[torch.max(tempcount, dim=-1).indices]        
    elif (dec_type == 'maxnum') | (dec_type == 'single'):
        predictions = torch.max(torch.sum(outputs, dim=0), dim=-1).indices

    true_classes = torch.max(torch.sum(output_sequence, dim=(-1,-2)), dim=-1).indices
    acc = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))

    return acc


def init_ml_vars(network, input_train, s, S_prime, learning_rate):
    """"
    Initializes the parameters for maximum likelihood training.
    """

    log_proba = network(input_train[int(s / S_prime), :, :, s % S_prime])

    reward = torch.sum(log_proba[network.output_neurons - network.n_non_learnable_neurons])

    eligibility_trace = {parameter: network.gradients[parameter] for parameter in network.gradients}

    baseline_num = {parameter: eligibility_trace[parameter].pow(2)*reward for parameter in eligibility_trace}
    baseline_den = {parameter: eligibility_trace[parameter].pow(2) for parameter in eligibility_trace}

    baseline = {parameter: (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07) for parameter in network.gradients}

    updates = {parameter: eligibility_trace[parameter] for parameter in network.gradients}

    # Compute update
    for parameter in updates:
        updates[parameter][network.hidden_neurons - network.n_non_learnable_neurons] *= (reward - baseline[parameter][network.hidden_neurons - network.n_non_learnable_neurons])
        network.get_parameters()[parameter] += updates[parameter] * learning_rate

    return baseline_num, baseline_den, baseline, updates, eligibility_trace


def train_ml_online(network, input_train, output_train, input_test, output_test, learning_rate, lr_const, kappa, beta, debug_flag, n_debug, num_ll_est, dec_type, writer, save_file_name):
    """"
    Train a network using maximum likelihood.
    """

    assert torch.sum(network.feedforward_mask[network.hidden_neurons - network.n_non_learnable_neurons, :, -2:, :, :]) == 0, 'There must be no backward connection from output' \
                                                                                                                             'to hidden neurons.'
    network.set_mode('train_ml')

    training_sequence = torch.cat((input_train, output_train), dim=1)
    test_sequence = torch.cat((input_test, output_test), dim=1)

    epochs = training_sequence.shape[0]
    S_prime = training_sequence.shape[-1]
    S = epochs * S_prime

    img_size = int(np.sqrt(input_train.shape[1]*2))

    timestamp = torch.zeros([n_debug,1])
    acc_train = torch.zeros([n_debug,1])
    loss_output_train = torch.zeros([n_debug,1])
    spikenum_hid_train = torch.zeros([n_debug,1])
    spikenum_output_train = torch.zeros([n_debug,1])
    acc_test = torch.zeros([n_debug,1])
    loss_output_test = torch.zeros([n_debug,1])

    img_train = torch.zeros([n_debug, 1, S_prime, img_size, img_size])
    img_true = torch.zeros([n_debug, 1, S_prime, img_size, img_size])

    output_spikes_train = torch.zeros([n_debug, 1, network.n_output_neurons, network.alphabet_size, S_prime])

    eligibility_reward = 0

    for s in range(S):
        # Reset network & training variables for each example
        if s % S_prime == 0:
            network.reset_internal_state()
            baseline_num, baseline_den, baseline, updates, eligibility_trace = init_ml_vars(network, training_sequence, s, S_prime, learning_rate)

        if s % min( int(S / 5), S_prime * 200 ) == 0 and s > 0:
            learning_rate /= lr_const

        log_proba = network(training_sequence[int(s / S_prime), :, :, s % S_prime])

        reward = torch.sum(log_proba[network.output_neurons - network.n_non_learnable_neurons])

        if s % S_prime == 0:
            eligibility_reward = reward
        elif s % S_prime > 0:
            eligibility_reward = kappa*eligibility_reward + reward

        for parameter in updates:
            if s % S_prime > 0:
                eligibility_trace[parameter] = kappa*eligibility_trace[parameter] + network.gradients[parameter]

                baseline_num[parameter] = beta * baseline_num[parameter] + eligibility_trace[parameter].pow(2) * eligibility_reward
                baseline_den[parameter] = beta * baseline_den[parameter] + eligibility_trace[parameter].pow(2)
            
            baseline[parameter] = (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07)

            # update 
            updates[parameter][network.hidden_neurons - network.n_non_learnable_neurons] = \
                (eligibility_reward - baseline[parameter][network.hidden_neurons - network.n_non_learnable_neurons]) * eligibility_trace[parameter][network.hidden_neurons - network.n_non_learnable_neurons]

            updates[parameter][network.output_neurons - network.n_non_learnable_neurons] = eligibility_trace[parameter][network.output_neurons - network.n_non_learnable_neurons]
            
            network.get_parameters()[parameter] += updates[parameter] * learning_rate


        # compute spike numbers for one example presentation S_prime
        p_idx = int(s / (S/n_debug))
    
        spikenum_hid_train[p_idx] += torch.nonzero(network.spiking_history[network.hidden_neurons, :, -1]).shape[0]
        spikenum_output_train[p_idx] += torch.nonzero(network.spiking_history[network.output_neurons, :, -1]).shape[0]
        output_spikes_train[p_idx, 0, :, :, s % S_prime] = network.spiking_history[network.output_neurons, :, -1]

        if s % int(S / n_debug) == int(S / n_debug)-1:
            timestamp[p_idx] = s+1

            if debug_flag[0] == 1:
                # Copy current network for in-training evaluation
                temp_net = SNNetwork(**make_network_parameters(network.n_input_neurons, network.n_output_neurons, network.n_hidden_neurons, network.alphabet_size, network.mode, network.n_basis_feedforward, network.tau_ff, network.tau_fb))
                temp_net_params = network.get_parameters()
                temp_net.set_parameters(temp_net_params)

                _, loss_output_train[p_idx] = get_log_likelihood(temp_net, training_sequence, num_ll_est)
                acc_train[p_idx] = get_acc(temp_net, input_train, output_train, dec_type, 1)
                print('Step %d out of %d: lr %6.5f, Train accuracy (%s) %f / Train loss-output %f' % (s+1, S, learning_rate, dec_type, acc_train[p_idx], loss_output_train[p_idx]), flush=True)
                
                writer.add_scalar('accuracy/train', acc_train[p_idx], s+1)
                writer.add_scalar('log-likelihood/train', loss_output_train[p_idx], s+1)

                print('Step %d out of %d: lr %6.5f, Spikenum hid %d, rate %f / Spikenum output %d, rate %f' % (s+1, S, learning_rate, spikenum_hid_train[p_idx], spikenum_hid_train[p_idx]/(S/n_debug)/network.n_hidden_neurons, \
                    spikenum_output_train[p_idx], spikenum_output_train[p_idx]/(S/n_debug)/network.n_output_neurons), flush=True)

                img_train[p_idx, 0, :, np.arange(int(img_size/2)), :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(input_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(1,1,1,1)
                img_train[p_idx, 0, :, int(img_size/2):, :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(output_spikes_train[p_idx, :, :, :, :].reshape(1, img_size, int(img_size/2), S_prime), 270, (1,2)), (0,3,1,2)))
                img_true[p_idx, 0, :, np.arange(int(img_size/2)), :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(input_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(1,1,1,1)
                img_true[p_idx, 0, :, int(img_size/2):, :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(output_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(1,1,1,1)
                
                writer.add_images('trained_images', img_train[p_idx, 0, :, :, :].reshape(S_prime, 1, img_size,img_size), s+1, dataformats = 'NCHW')
                writer.add_images('true images', img_true[p_idx, 0, :, :, :].reshape(S_prime, 1, img_size, img_size), s+1, dataformats = 'NCHW')

                # save file 
                save_state = {
                    'network': network,
                    'timestep': s+1,
                    'acc_train': acc_train[p_idx],
                    'loss_output_train': torch.Tensor([loss_output_train[p_idx]]), 
                    'spikenum_hid_train': spikenum_hid_train[p_idx],
                    'spikenum_output_train': spikenum_output_train[p_idx],
                }
                torch.save(save_state, save_file_name+'_t-'+str(s+1)+'.pth')

                if debug_flag[1] == 1:
                    _, loss_output_test[p_idx] = get_log_likelihood(temp_net, test_sequence, num_ll_est)
                    acc_test[p_idx] = get_acc(temp_net, input_test, output_test, dec_type, 1)
                    print('Step %d out of %d: Test accuracy (%s) %f / Test loss-output %f' % (s+1, S, dec_type, acc_test[p_idx], loss_output_test[p_idx]), flush=True)

                    writer.add_scalar('accuracy/test', acc_test[p_idx], s+1)
                    writer.add_scalar('log-likelihood/test', loss_output_test[p_idx], s+1)

            else:
                print('Step %d out of %d' % (s, S), flush=True)

    return acc_train, loss_output_train, spikenum_hid_train, spikenum_output_train, acc_test, loss_output_test, timestamp
                
