import torch
import torch.nn as nn
import torch.nn.functional as F
import tables
import utils.filters as filters
import numpy as np
from utils.training_utils import *
from SNN import SNNetwork
import pdb
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage


def train_ml_multi_online(network, input_train, output_train, input_test, output_test, learning_rate, lr_const, kappa, beta, num_samples, debug_flag, n_debug, num_ll_est, dec_type, writer, save_file_name):
    """"
    Train a network using maximum likelihood (K-sample importance weights)
    """    

    assert torch.sum(network.feedforward_mask[network.hidden_neurons - network.n_non_learnable_neurons, :, -2:, :, :]) == 0, 'There must be no backward connection from output to hidden neurons.'

    network.set_mode('train_ml_multi')

    # Duplicate the network
    network_list = duplicate_networks(network, num_samples)
    training_params_list = list()
    for k in range(num_samples):
        training_params_list.append(TrainingParams(k))        

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
    
    img_train = torch.zeros([n_debug, num_samples, S_prime, img_size, img_size])
    img_true = torch.zeros([n_debug, num_samples, S_prime, img_size, img_size])

    output_spikes_train = torch.zeros([n_debug, num_samples, network.n_output_neurons, network.alphabet_size, S_prime])

    unnormalized_weight = torch.zeros([num_samples,1])
    normalized_weight = torch.zeros([num_samples,1])

    for s in range(S):

        if s % S_prime == 0:
            network.reset_internal_state()
            baseline_num, baseline_den, baseline, updates, _ = init_ml_vars(network, training_sequence, s, S_prime, learning_rate)

        if s % min( int(S / 5), S_prime * 200 ) == 0 and s > 0:
            learning_rate /= lr_const

        for k in range(num_samples):

            # Reset network & training variables for each example
            if s % S_prime == 0:
                network_list[k].reset_internal_state()
                baseline_num_, baseline_den_, baseline_, updates_, eligibility_trace_ = init_ml_vars(network_list[k], training_sequence, s, S_prime, learning_rate)
                training_params_list[k].set_init_params(baseline_num_, baseline_den_, baseline_, updates_, eligibility_trace_)

            training_params_list[k].log_proba = network_list[k](training_sequence[int(s / S_prime), :, :, s % S_prime])
            training_params_list[k].reward = torch.sum(training_params_list[k].log_proba[network_list[k].output_neurons - network_list[k].n_non_learnable_neurons])

            for parameter in training_params_list[k].updates:  
                if s % S_prime > 0:
                    training_params_list[k].eligibility_trace[parameter] = kappa*training_params_list[k].eligibility_trace[parameter] + network_list[k].gradients[parameter]

            if s % S_prime == 0:
                training_params_list[k].eligibility_reward = training_params_list[k].reward    
            elif s % S_prime > 0:
                training_params_list[k].eligibility_reward = kappa*training_params_list[k].eligibility_reward + training_params_list[k].reward

            unnormalized_weight[k] = training_params_list[k].eligibility_reward

        # K normalization via softmax
        normalized_weight = F.softmax( unnormalized_weight, dim=0 )
        
        learning_sig = logsumexp(unnormalized_weight, dim=0) - torch.log(torch.ones([1]) * num_samples)

        if s % S_prime == 0:
            baseline_num = {parameter: updates[parameter].pow(2) * learning_sig for parameter in updates}
            baseline_den = {parameter: updates[parameter].pow(2) for parameter in updates}
            baseline = {parameter: (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07) for parameter in updates}

        # Update
        for parameter in updates:
            updates[parameter] -= updates[parameter]

            for k in range(num_samples):
                # Update of each sample 
                updates[parameter][network_list[k].hidden_neurons - network_list[k].n_non_learnable_neurons] += training_params_list[k].eligibility_trace[parameter][network_list[k].hidden_neurons - network_list[k].n_non_learnable_neurons]
                updates[parameter][network_list[k].output_neurons - network_list[k].n_non_learnable_neurons] += normalized_weight[k] * training_params_list[k].eligibility_trace[parameter][network_list[k].output_neurons - network_list[k].n_non_learnable_neurons]

            if s % S_prime > 0:
                baseline_num[parameter] = beta * baseline_num[parameter] + updates[parameter].pow(2) * learning_sig
                baseline_den[parameter] = beta * baseline_den[parameter] + updates[parameter].pow(2)
        
            baseline[parameter] = (baseline_num[parameter]) / (baseline_den[parameter] + 1e-07)

            updates[parameter][network.hidden_neurons - network.n_non_learnable_neurons] *= learning_sig - baseline[parameter][network.hidden_neurons - network.n_non_learnable_neurons]

            # update using learning_rate
            network.get_parameters()[parameter] += updates[parameter] * learning_rate

        # Copy params of network to duplicated nets
        net_params = network.get_parameters()
        for k in range(num_samples):
            network_list[k].set_parameters(net_params)        

        # compute spike numbers for one example presentation S_prime
        p_idx = int(s / (S/n_debug))

        for k in range(num_samples):
            spikenum_hid_train[p_idx] += torch.nonzero(network_list[k].spiking_history[network_list[k].hidden_neurons, :, -1]).shape[0]
            spikenum_output_train[p_idx] += torch.nonzero(network_list[k].spiking_history[network_list[k].output_neurons, :, -1]).shape[0]
            output_spikes_train[p_idx, k, :, :, s % S_prime] = network_list[k].spiking_history[network_list[k].output_neurons, :, -1]

        if s % int(S / n_debug) == int(S / n_debug)-1:
            timestamp[p_idx] = s+1

            if debug_flag[0] == 1:

                # Copy current network for in-training evaluation
                temp_net = SNNetwork(**make_network_parameters(network.n_input_neurons, network.n_output_neurons, network.n_hidden_neurons, network.alphabet_size, network.mode, network.n_basis_feedforward, network.tau_ff, network.tau_fb))
                temp_net_params = network.get_parameters()
                temp_net.set_parameters(temp_net_params)

                _, loss_output_train[p_idx] = get_log_likelihood(temp_net, training_sequence, num_ll_est)
                acc_train[p_idx] = get_acc(temp_net, input_train, output_train, dec_type, num_samples)
                print('Step %d out of %d: lr %6.5f,  Train accuracy (%s) %f / Train loss-output %f' % (s+1, S, learning_rate, dec_type, acc_train[p_idx], loss_output_train[p_idx]), flush=True)
                #print('accuracy single %f, majority %f, maxnum %f' %(get_acc(temp_net, input_train, output_train, 'single', num_samples), get_acc(temp_net, input_train, output_train, 'majority', num_samples), get_acc(temp_net, input_train, output_train, 'maxnum', num_samples)), flush=True)

                writer.add_scalar('accuracy/train', acc_train[p_idx], s+1)
                writer.add_scalar('log-likelihood/train', loss_output_train[p_idx], s+1)

                print('Step %d out of %d: lr %6.5f,  Spikenum hid %d, rate %f / Spikenum output %d, rate %f' % (s+1, S, learning_rate, spikenum_hid_train[p_idx], spikenum_hid_train[p_idx]/(S/n_debug)/num_samples/network.n_hidden_neurons, \
                    spikenum_output_train[p_idx], spikenum_output_train[p_idx]/(S/n_debug)/num_samples/network.n_output_neurons), flush=True)

                img_train[p_idx, :, :, np.arange(int(img_size/2)), :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(input_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_samples,1,1,1)
                img_train[p_idx, :, :, int(img_size/2):, :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(output_spikes_train[p_idx, :, :, :, :].reshape(num_samples, img_size, int(img_size/2), S_prime), 270, (1,2)), (0,3,1,2)))
                img_true[p_idx, :, :, np.arange(int(img_size/2)), :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(input_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_samples,1,1,1)
                img_true[p_idx, :, :, int(img_size/2):, :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(output_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_samples,1,1,1)

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
                    acc_test[p_idx] = get_acc(temp_net, input_test, output_test, dec_type, num_samples)
                    print('Step %d out of %d: Test accuracy (%s) %f / Test loss-output %f' % (s+1, S, dec_type, acc_test[p_idx], loss_output_test[p_idx]), flush=True)

                    writer.add_scalar('accuracy/test', acc_test[p_idx], s+1)
                    writer.add_scalar('log-likelihood/test', loss_output_test[p_idx], s+1)

            else:
                print('Step %d out of %d' % (s, S), flush=True)

    return acc_train, loss_output_train, spikenum_hid_train, spikenum_output_train, acc_test, loss_output_test, timestamp


def train_ml_multi_online_ls(network, input_train, output_train, input_test, output_test, learning_rate, lr_const, kappa, beta, num_samples, debug_flag, n_debug, num_ll_est, dec_type, writer, save_file_name):
    """"
    Train a network using maximum likelihood (K-sample importance weights) + per-sample learning signal
    """    

    assert torch.sum(network.feedforward_mask[network.hidden_neurons - network.n_non_learnable_neurons, :, -2:, :, :]) == 0, 'There must be no backward connection from output to hidden neurons.'

    network.set_mode('train_ml_multi_ls')

    # Duplicate the network
    network_list = duplicate_networks(network, num_samples)
    training_params_list = list()
    for k in range(num_samples):
        training_params_list.append(TrainingParams(k))        

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

    img_train = torch.zeros([n_debug, num_samples, S_prime, img_size, img_size])
    img_true = torch.zeros([n_debug, num_samples, S_prime, img_size, img_size])

    output_spikes_train = torch.zeros([n_debug, num_samples, network.n_output_neurons, network.alphabet_size, S_prime])

    unnormalized_weight = torch.zeros([num_samples,1])
    normalized_weight = torch.zeros([num_samples,1])
    learning_sig = torch.zeros([num_samples,1])

    for s in range(S):

        if s % S_prime == 0:
            network.reset_internal_state()
            _, _, _, updates, _ = init_ml_vars(network, training_sequence, s, S_prime, learning_rate)

        if s % min( int(S / 5), S_prime * 200 ) == 0 and s > 0:
            learning_rate /= lr_const

        for k in range(num_samples):

            # Reset network & training variables for each example
            if s % S_prime == 0:
                network_list[k].reset_internal_state()
                baseline_num_, baseline_den_, baseline_, updates_, eligibility_trace_ = init_ml_vars(network_list[k], training_sequence, s, S_prime, learning_rate)
                training_params_list[k].set_init_params(baseline_num_, baseline_den_, baseline_, updates_, eligibility_trace_)

            training_params_list[k].log_proba = network_list[k](training_sequence[int(s / S_prime), :, :, s % S_prime])
            training_params_list[k].reward = torch.sum(training_params_list[k].log_proba[network_list[k].output_neurons - network_list[k].n_non_learnable_neurons])

            for parameter in training_params_list[k].updates:  
                if s % S_prime > 0:
                    training_params_list[k].eligibility_trace[parameter] = kappa*training_params_list[k].eligibility_trace[parameter] + network_list[k].gradients[parameter]

            if s % S_prime == 0:
                training_params_list[k].eligibility_reward = training_params_list[k].reward    
            elif s % S_prime > 0:
                training_params_list[k].eligibility_reward = kappa*training_params_list[k].eligibility_reward + training_params_list[k].reward

            unnormalized_weight[k] = training_params_list[k].eligibility_reward
            
        # K normalization via softmax
        normalized_weight = F.softmax( unnormalized_weight, dim=0 )
        
        learning_sig_common = logsumexp(unnormalized_weight, dim=0) - torch.log(torch.ones([1]) * num_samples)

        # per-sample learning signal (if num_samples >= 2)
        if num_samples > 1:
            for k in range(num_samples):
                temp_unnormalized_weight = unnormalized_weight
                temp_unnormalized_weight[k] = 1 / (num_samples-1) * (torch.mean(unnormalized_weight, dim=0)*num_samples - unnormalized_weight[k])
                learning_sig[k] = learning_sig_common - logsumexp(temp_unnormalized_weight, dim=0) - torch.log(torch.ones([1]) * num_samples)

                if s % S_prime > 0:
                    training_params_list[k].baseline_num[parameter] = beta * training_params_list[k].baseline_num[parameter] + training_params_list[k].eligibility_trace[parameter].pow(2) * learning_sig[k]
                    training_params_list[k].baseline_den[parameter] = beta * training_params_list[k].baseline_den[parameter] + training_params_list[k].eligibility_trace[parameter].pow(2)
        
                training_params_list[k].baseline[parameter] = (training_params_list[k].baseline_num[parameter]) / (training_params_list[k].baseline_den[parameter] + 1e-07)

        # Update
        for parameter in updates:
            updates[parameter] -= updates[parameter]
            
            for k in range(num_samples):
                # update of each sample (with baseline)
                #updates[parameter][network_list[k].hidden_neurons - network_list[k].n_non_learnable_neurons] += \
                #    (learning_sig[k] - training_params_list[k].baseline[parameter][network_list[k].hidden_neurons - network_list[k].n_non_learnable_neurons]) \
                #        * training_params_list[k].eligibility_trace[parameter][network_list[k].hidden_neurons - network_list[k].n_non_learnable_neurons]

                # Update of each sample (without baseline)
                updates[parameter][network_list[k].hidden_neurons - network_list[k].n_non_learnable_neurons] += \
                    (learning_sig[k]) * training_params_list[k].eligibility_trace[parameter][network_list[k].hidden_neurons - network_list[k].n_non_learnable_neurons]

                updates[parameter][network_list[k].output_neurons - network_list[k].n_non_learnable_neurons] += normalized_weight[k] * training_params_list[k].eligibility_trace[parameter][network_list[k].output_neurons - network_list[k].n_non_learnable_neurons]

            # update using learning_rate
            network.get_parameters()[parameter] += updates[parameter] * learning_rate
        

        # Copy params of network to duplicated nets
        net_params = network.get_parameters()
        for k in range(num_samples):
            network_list[k].set_parameters(net_params)        
        
        # compute spike numbers for one example presentation S_prime
        p_idx = int(s / (S/n_debug))
        
        for k in range(num_samples):
            spikenum_hid_train[p_idx] += torch.nonzero(network_list[k].spiking_history[network_list[k].hidden_neurons, :, -1]).shape[0]
            spikenum_output_train[p_idx] += torch.nonzero(network_list[k].spiking_history[network_list[k].output_neurons, :, -1]).shape[0]
            output_spikes_train[p_idx, k, :, :, s % S_prime] = network_list[k].spiking_history[network_list[k].output_neurons, :, -1]

        if s % int(S / n_debug) == int(S / n_debug)-1:
            timestamp[p_idx] = s+1

            if debug_flag[0] == 1:
                # Copy current network for in-training evaluation
                temp_net = SNNetwork(**make_network_parameters(network.n_input_neurons, network.n_output_neurons, network.n_hidden_neurons, network.alphabet_size, network.mode, network.n_basis_feedforward, network.tau_ff, network.tau_fb))
                temp_net_params = network.get_parameters()
                temp_net.set_parameters(temp_net_params)

                _, loss_output_train[p_idx] = get_log_likelihood(temp_net, training_sequence, num_ll_est)
                acc_train[p_idx] = get_acc(temp_net, input_train, output_train, dec_type, num_samples)
                print('Step %d out of %d: lr %6.5f, Train accuracy (%s) %f / Train loss-output %f' % (s+1, S, learning_rate, dec_type, acc_train[p_idx], loss_output_train[p_idx]), flush=True)

                writer.add_scalar('accuracy/train', acc_train[p_idx], s+1)
                writer.add_scalar('log-likelihood/train', loss_output_train[p_idx], s+1)

                print('Step %d out of %d: lr %6.5f, Spikenum hid %d, rate %f / Spikenum output %d, rate %f' % (s+1, S, learning_rate, spikenum_hid_train[p_idx], spikenum_hid_train[p_idx]/(S/n_debug)/num_samples/network.n_hidden_neurons, \
                    spikenum_output_train[p_idx], spikenum_output_train[p_idx]/(S/n_debug)/num_samples/network.n_output_neurons), flush=True)

                img_train[p_idx, :, :, np.arange(int(img_size/2)), :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(input_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_samples,1,1,1)
                img_train[p_idx, :, :, int(img_size/2):, :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(output_spikes_train[p_idx, :, :, :, :].reshape(num_samples, img_size, int(img_size/2), S_prime), 270, (1,2)), (0,3,1,2)))
                img_true[p_idx, :, :, np.arange(int(img_size/2)), :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(input_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_samples,1,1,1)
                img_true[p_idx, :, :, int(img_size/2):, :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(output_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_samples,1,1,1)

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
                    acc_test[p_idx] = get_acc(temp_net, input_test, output_test, dec_type, num_samples)
                    print('Step %d out of %d: Test accuracy (%s) %f / Test loss-output %f' % (s+1, S, dec_type, acc_test[p_idx], loss_output_test[p_idx]), flush=True)

                    writer.add_scalar('accuracy/test', acc_test[p_idx], s+1)
                    writer.add_scalar('log-likelihood/test', loss_output_test[p_idx], s+1)

            else:
                print('Step %d out of %d' % (s, S), flush=True)

    return acc_train, loss_output_train, spikenum_hid_train, spikenum_output_train, acc_test, loss_output_test, timestamp


def train_ml_multi_elbo_online(network, input_train, output_train, input_test, output_test, learning_rate, lr_const, kappa, beta, num_samples, debug_flag, n_debug, num_ll_est, dec_type, writer, save_file_name):
    """
    Train a network using maximum likelihood with a multiple-sample ELBO (K-sample averaging)
    """

    assert torch.sum(network.feedforward_mask[network.hidden_neurons - network.n_non_learnable_neurons, :, -2:, :, :]) == 0, 'There must be no backward connection from output to hidden neurons.'

    network.set_mode('train_ml_multi_elbo')

    # Duplicate the networks
    network_list = duplicate_networks(network, num_samples)
    training_params_list = list()
    for k in range(num_samples):
        training_params_list.append(TrainingParams(k))
    
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
    
    img_train = torch.zeros([n_debug, num_samples, S_prime, img_size, img_size])
    img_true = torch.zeros([n_debug, num_samples, S_prime, img_size, img_size])

    output_spikes_train = torch.zeros([n_debug, num_samples, network.n_output_neurons, network.alphabet_size, S_prime])

    for s in range(S):

        if s % S_prime == 0:
            network.reset_internal_state()
            _, _, _, updates, _ = init_ml_vars(network, training_sequence, s, S_prime, learning_rate)

        if s % min( int(S / 5), S_prime * 200 ) == 0 and s > 0:
            learning_rate /= lr_const

        for k in range(num_samples):

            # Reset network & training variables for each example
            if s % S_prime == 0:
                network_list[k].reset_internal_state()
                baseline_num_, baseline_den_, baseline_, updates_, eligibility_trace_ = init_ml_vars(network_list[k], training_sequence, s, S_prime, learning_rate)
                training_params_list[k].set_init_params(baseline_num_, baseline_den_, baseline_, updates_, eligibility_trace_)

            training_params_list[k].log_proba = network_list[k](training_sequence[int(s / S_prime), :, :, s % S_prime])
            training_params_list[k].reward = torch.sum(training_params_list[k].log_proba[network_list[k].output_neurons - network_list[k].n_non_learnable_neurons])

            if s % S_prime == 0:
                training_params_list[k].eligibility_reward = training_params_list[k].reward    
            elif s % S_prime > 0:
                training_params_list[k].eligibility_reward = kappa*training_params_list[k].eligibility_reward + training_params_list[k].reward

            for parameter in training_params_list[k].updates:  
                if s % S_prime > 0:
                    training_params_list[k].eligibility_trace[parameter] = kappa*training_params_list[k].eligibility_trace[parameter] + network_list[k].gradients[parameter]

                    training_params_list[k].baseline_num[parameter] = beta * training_params_list[k].baseline_num[parameter] + training_params_list[k].eligibility_trace[parameter].pow(2) * training_params_list[k].eligibility_reward
                    training_params_list[k].baseline_den[parameter] = beta * training_params_list[k].baseline_den[parameter] + training_params_list[k].eligibility_trace[parameter].pow(2)
                
                training_params_list[k].baseline[parameter] = (training_params_list[k].baseline_num[parameter]) / (training_params_list[k].baseline_den[parameter] + 1e-07)

                training_params_list[k].updates[parameter][network_list[k].hidden_neurons - network_list[k].n_non_learnable_neurons] = \
                    (training_params_list[k].eligibility_reward - training_params_list[k].baseline[parameter][network_list[k].hidden_neurons - network_list[k].n_non_learnable_neurons]) \
                        * training_params_list[k].eligibility_trace[parameter][network_list[k].hidden_neurons - network_list[k].n_non_learnable_neurons]

                training_params_list[k].updates[parameter][network_list[k].output_neurons - network_list[k].n_non_learnable_neurons] = \
                    training_params_list[k].eligibility_trace[parameter][network_list[k].output_neurons - network_list[k].n_non_learnable_neurons]

        # update by K-sample averaging
        for parameter in updates:
            updates[parameter] -= updates[parameter]

            for k in range(num_samples):
                updates[parameter] += training_params_list[k].updates[parameter]
            
            updates[parameter] = updates[parameter] / num_samples 

            network.get_parameters()[parameter] += updates[parameter] * learning_rate

        # Copy params of network to duplicated nets
        net_params = network.get_parameters()
        for k in range(num_samples):
            network_list[k].set_parameters(net_params)

        # compute spike numbers for one example presentation S_prime
        p_idx = int(s / (S/n_debug))
        
        for k in range(num_samples):
            spikenum_hid_train[p_idx] += torch.nonzero(network_list[k].spiking_history[network_list[k].hidden_neurons, :, -1]).shape[0]
            spikenum_output_train[p_idx] += torch.nonzero(network_list[k].spiking_history[network_list[k].output_neurons, :, -1]).shape[0]
            output_spikes_train[p_idx, k, :, :, s % S_prime] = network_list[k].spiking_history[network_list[k].output_neurons, :, -1]

        if s % int(S / n_debug) == int(S / n_debug)-1:
            timestamp[p_idx] = s+1

            if debug_flag[0] == 1:
                # Copy current network for in-training evaluation
                temp_net = SNNetwork(**make_network_parameters(network.n_input_neurons, network.n_output_neurons, network.n_hidden_neurons, network.alphabet_size, network.mode, network.n_basis_feedforward, network.tau_ff, network.tau_fb))
                temp_net_params = network.get_parameters()
                temp_net.set_parameters(temp_net_params)

                _, loss_output_train[p_idx] = get_log_likelihood(temp_net, training_sequence, num_ll_est)
                acc_train[p_idx] = get_acc(temp_net, input_train, output_train, dec_type, num_samples)
                print('Step %d out of %d: lr %6.5f, Train accuracy (%s) %f / Train loss-output %f' % (s+1, S, learning_rate, dec_type, acc_train[p_idx], loss_output_train[p_idx]), flush=True)

                writer.add_scalar('accuracy/train', acc_train[p_idx], s+1)
                writer.add_scalar('log-likelihood/train', loss_output_train[p_idx], s+1)

                print('Step %d out of %d: lr %6.5f, Spikenum hid %d, rate %f / Spikenum output %d, rate %f' % (s+1, S, learning_rate, spikenum_hid_train[p_idx], spikenum_hid_train[p_idx]/(S/n_debug)/num_samples/network.n_hidden_neurons, \
                    spikenum_output_train[p_idx], spikenum_output_train[p_idx]/(S/n_debug)/num_samples/network.n_output_neurons), flush=True)
                
                img_train[p_idx, :, :, np.arange(int(img_size/2)), :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(input_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_samples,1,1,1)
                img_train[p_idx, :, :, int(img_size/2):, :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(output_spikes_train[p_idx, :, :, :, :].reshape(num_samples, img_size, int(img_size/2), S_prime), 270, (1,2)), (0,3,1,2)))
                img_true[p_idx, :, :, np.arange(int(img_size/2)), :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(input_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_samples,1,1,1)
                img_true[p_idx, :, :, int(img_size/2):, :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(output_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_samples,1,1,1)

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
                    acc_test[p_idx] = get_acc(temp_net, input_test, output_test, dec_type, num_samples)
                    print('Step %d out of %d: Test accuracy (%s) %f / Test loss-output %f' % (s+1, S, dec_type, acc_test[p_idx], loss_output_test[p_idx]), flush=True)

                    writer.add_scalar('accuracy/test', acc_test[p_idx], s+1)
                    writer.add_scalar('log-likelihood/test', loss_output_test[p_idx], s+1)

            else:
                print('Step %d out of %d' % (s, S), flush=True)

    return acc_train, loss_output_train, spikenum_hid_train, spikenum_output_train, acc_test, loss_output_test, timestamp


def train_ml_multi_nols_online(network, input_train, output_train, input_test, output_test, learning_rate, lr_const, kappa, beta, num_samples, debug_flag, n_debug, num_ll_est, dec_type, writer, save_file_name):
    """"
    Train a network using maximum likelihood (K-sample importance weights without learning signal)
    """    

    assert torch.sum(network.feedforward_mask[network.hidden_neurons - network.n_non_learnable_neurons, :, -2:, :, :]) == 0, 'There must be no backward connection from output to hidden neurons.'

    network.set_mode('train_ml_multi_nols')

    # Duplicate the network
    network_list = duplicate_networks(network, num_samples)
    training_params_list = list()
    for k in range(num_samples):
        training_params_list.append(TrainingParams(k))        

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
    
    img_train = torch.zeros([n_debug, num_samples, S_prime, img_size, img_size])
    img_true = torch.zeros([n_debug, num_samples, S_prime, img_size, img_size])

    output_spikes_train = torch.zeros([n_debug, num_samples, network.n_output_neurons, network.alphabet_size, S_prime])

    unnormalized_weight = torch.zeros([num_samples,1])
    normalized_weight = torch.zeros([num_samples,1])

    for s in range(S):

        if s % S_prime == 0:
            network.reset_internal_state()
            baseline_num, baseline_den, baseline, updates, eligibility_trace = init_ml_vars(network, training_sequence, s, S_prime, learning_rate)

        if s % min( int(S / 5), S_prime * 200 ) == 0 and s > 0:
            learning_rate /= lr_const

        for k in range(num_samples):

            # Reset network & training variables for each example
            if s % S_prime == 0:
                network_list[k].reset_internal_state()
                _, _, _, updates_, eligibility_trace_ = init_ml_vars(network_list[k], training_sequence, s, S_prime, learning_rate)
                training_params_list[k].set_init_params(_, _, _, updates_, eligibility_trace_)

            training_params_list[k].log_proba = network_list[k](training_sequence[int(s / S_prime), :, :, s % S_prime])
            training_params_list[k].reward = torch.sum(training_params_list[k].log_proba[network_list[k].output_neurons - network_list[k].n_non_learnable_neurons])

            if s % S_prime == 0:
                training_params_list[k].eligibility_reward = training_params_list[k].reward
            
            for parameter in training_params_list[k].updates:
                if s % S_prime > 0:  
                    training_params_list[k].eligibility_trace[parameter] = kappa*training_params_list[k].eligibility_trace[parameter] + network_list[k].gradients[parameter]

            if s % S_prime == 0:
                training_params_list[k].eligibility_reward = training_params_list[k].reward    
            elif s % S_prime > 0:
                training_params_list[k].eligibility_reward = kappa*training_params_list[k].eligibility_reward + training_params_list[k].reward
            
            unnormalized_weight[k] = training_params_list[k].eligibility_reward
  
        # K normalization via softmax
        normalized_weight = F.softmax( unnormalized_weight, dim=0 )
        
        # Update
        for parameter in updates:
            updates[parameter] -= updates[parameter]
            
            for k in range(num_samples):
                # Update of each sample 
                updates[parameter][network_list[k].hidden_neurons - network_list[k].n_non_learnable_neurons] += normalized_weight[k] * training_params_list[k].eligibility_trace[parameter][network_list[k].hidden_neurons - network_list[k].n_non_learnable_neurons]
                updates[parameter][network_list[k].output_neurons - network_list[k].n_non_learnable_neurons] += normalized_weight[k] * training_params_list[k].eligibility_trace[parameter][network_list[k].output_neurons - network_list[k].n_non_learnable_neurons]

            # update using learning_rate
            network.get_parameters()[parameter] += updates[parameter] * learning_rate
  
        # Copy params of network to duplicated nets
        net_params = network.get_parameters()
        for k in range(num_samples):
            network_list[k].set_parameters(net_params)        

        # compute spike numbers for one example presentation S_prime
        p_idx = int(s / (S/n_debug))
        
        for k in range(num_samples):
            spikenum_hid_train[p_idx] += torch.nonzero(network_list[k].spiking_history[network_list[k].hidden_neurons, :, -1]).shape[0]
            spikenum_output_train[p_idx] += torch.nonzero(network_list[k].spiking_history[network_list[k].output_neurons, :, -1]).shape[0]
            output_spikes_train[p_idx, k, :, :, s % S_prime] = network_list[k].spiking_history[network_list[k].output_neurons, :, -1]

        if s % int(S / n_debug) == int(S / n_debug)-1:
            timestamp[p_idx] = s+1

            if debug_flag[0] == 1:
                # Copy current network for in-training evaluation
                temp_net = SNNetwork(**make_network_parameters(network.n_input_neurons, network.n_output_neurons, network.n_hidden_neurons, network.alphabet_size, network.mode, network.n_basis_feedforward, network.tau_ff, network.tau_fb))
                temp_net_params = network.get_parameters()
                temp_net.set_parameters(temp_net_params)

                _, loss_output_train[p_idx] = get_log_likelihood(temp_net, training_sequence, num_ll_est)
                acc_train[p_idx] = get_acc(temp_net, input_train, output_train, dec_type, num_samples)
                print('Step %d out of %d: lr %6.5f, Train accuracy (%s) %f / Train loss-output %f' % (s+1, S, learning_rate, dec_type, acc_train[p_idx], loss_output_train[p_idx]), flush=True)

                writer.add_scalar('accuracy/train', acc_train[p_idx], s+1)
                writer.add_scalar('log-likelihood/train', loss_output_train[p_idx], s+1)

                print('Step %d out of %d: lr %6.5f, Spikenum hid %d, rate %f / Spikenum output %d, rate %f' % (s+1, S, learning_rate, spikenum_hid_train[p_idx], spikenum_hid_train[p_idx]/(S/n_debug)/num_samples/network.n_hidden_neurons, \
                    spikenum_output_train[p_idx], spikenum_output_train[p_idx]/(S/n_debug)/num_samples/network.n_output_neurons), flush=True)

                img_train[p_idx, :, :, np.arange(int(img_size/2)), :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(input_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_samples,1,1,1)
                img_train[p_idx, :, :, int(img_size/2):, :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(output_spikes_train[p_idx, :, :, :, :].reshape(num_samples, img_size, int(img_size/2), S_prime), 270, (1,2)), (0,3,1,2)))
                img_true[p_idx, :, :, np.arange(int(img_size/2)), :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(input_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_samples,1,1,1)
                img_true[p_idx, :, :, int(img_size/2):, :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(output_train[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_samples,1,1,1)

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
                    acc_test[p_idx] = get_acc(temp_net, input_test, output_test, dec_type, num_samples)
                    print('Step %d out of %d: Test accuracy (%s) %f / Test loss-output %f' % (s+1, S, dec_type, acc_test[p_idx], loss_output_test[p_idx]), flush=True)

                    writer.add_scalar('accuracy/test', acc_test[p_idx], s+1)
                    writer.add_scalar('log-likelihood/test', loss_output_test[p_idx], s+1)

            else:
                print('Step %d out of %d' % (s, S), flush=True)

    return acc_train, loss_output_train, spikenum_hid_train, spikenum_output_train, acc_test, loss_output_test, timestamp


def duplicate_networks(network, num_samples):
    n_input_neurons = network.n_input_neurons
    n_output_neurons = network.n_output_neurons
    n_hidden_neurons = network.n_hidden_neurons
    alphabet_size = network.alphabet_size
    mode = network.mode
    n_basis = network.n_basis_feedforward
    tau = network.tau_ff
    net_params = network.get_parameters()

    network_list = list()
    for _ in range(num_samples):
        copied_net = SNNetwork(**make_network_parameters(n_input_neurons, n_output_neurons, n_hidden_neurons, alphabet_size, mode, n_basis, tau, tau))
        copied_net.set_parameters(net_params)
        network_list.append(copied_net)

    return network_list


class TrainingParams(object):
    def __init__(self, id):
        self.id = id
        self.baseline_num = 0
        self.baseline_den = 0
        self.baseline = 0
        self.updates = None
        self.eligibility_trace = None
        self.log_proba = 0
        self.reward = 0
        self.gradients = None
        self.eligibility_reward = 0

    def set_init_params(self, baseline_num, baseline_den, baseline, updates, eligibility_trace):
        self.baseline_num = baseline_num
        self.baseline_den = baseline_den
        self.baseline = baseline
        self.updates = updates
        self.eligibility_trace = eligibility_trace

    def print(self):
        print(self.id, self.reward)

    def calculate_importance_weight(self):
        return self.reward

def logsumexp(inputs, dim=None, keepdim=False):
    return (inputs - F.log_softmax(inputs, dim=dim)).mean(dim=dim, keepdim=keepdim)

