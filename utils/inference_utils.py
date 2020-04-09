import torch
import utils.filters as filters
from utils.training_multi_utils import duplicate_networks, TrainingParams
from SNN import SNNetwork
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.training_utils import *
from utils.training_multi_utils import *
import pdb
import scipy.misc
from scipy import ndimage


def get_loss_bound(network, input_sequence, output_sequence, num_samples, training_mode, writer=None):
    """
    Compute training objective value (lower bound) given input_sequence, output_sequence, network) with num_samples compartments 
    """

    #training_mode = network.mode

    network.set_mode('test-ll')
    network.reset_internal_state()

    S_prime = input_sequence.shape[-1]
    epochs = input_sequence.shape[0]
    S = S_prime * epochs
    kappa = 0.05

    # compute the training objective (lower bound) for each learning scheme
    loss_bound = 0
    
    training_sequence = torch.cat((input_sequence, output_sequence), dim=1)

    # Duplicate the networks 
    network_list = duplicate_networks(network, num_samples)
    training_params_list = list()
    for k in range(num_samples):
        training_params_list.append(TrainingParams(k))        
    
    output_spikes_inference = torch.zeros([num_samples, network.n_output_neurons, network.alphabet_size, S_prime])
    spikenum_hid_inference = 0
    spikenum_output_inference = 0

    unnormalized_weight = torch.zeros([num_samples,1])
    normalized_weight = torch.zeros([num_samples,1])
    
    if training_mode == 'train_ml_multi_elbo':
        temp_weight = 0
    else:
        temp_weight = torch.zeros([num_samples,1])
    
    for s in range(S):
        for k in range(num_samples):
            # reset K networks and internal states for each example
            if s % S_prime == 0:
                network_list[k].reset_internal_state()
                
            training_params_list[k].log_proba = network_list[k](training_sequence[int(s / S_prime), :, :, s % S_prime])
            training_params_list[k].reward = torch.sum(training_params_list[k].log_proba[network_list[k].output_neurons - network_list[k].n_non_learnable_neurons])
            
            if s % S_prime == 0:
                training_params_list[k].eligibility_reward = training_params_list[k].reward
            elif s % S_prime > 0:
                training_params_list[k].eligibility_reward = kappa*training_params_list[k].eligibility_reward + training_params_list[k].reward
            
            unnormalized_weight[k] = training_params_list[k].eligibility_reward
            
            output_spikes_inference[k, :, :, s % S_prime] = network_list[k].spiking_history[network_list[k].output_neurons, :, -1]
            spikenum_hid_inference += torch.nonzero(network_list[k].spiking_history[network_list[k].hidden_neurons, :, -1]).shape[0]
            spikenum_output_inference += torch.nonzero(network_list[k].spiking_history[network_list[k].output_neurons, :, -1]).shape[0]
            
        # K normalization via softmax
        normalized_weight = F.softmax( unnormalized_weight, dim=0 )
        
        if training_mode == 'train_ml_multi_elbo':
            temp_weight += torch.sum(unnormalized_weight, dim=0)
        elif training_mode == 'train_ml_multi' or training_mode == 'train_ml_multi_ls' or training_mode == 'train_ml_multi_nols':
            temp_weight += unnormalized_weight
    
    if training_mode == 'train_ml_multi_elbo':
        loss_bound = temp_weight / num_samples
    elif training_mode == 'train_ml_multi' or training_mode == 'train_ml_multi_ls':
        loss_bound = logsumexp( temp_weight, dim=0 ) - torch.log(torch.ones([1]) * num_samples)
    elif training_mode == 'train_ml_multi_nols':
        loss_bound = torch.sum( normalized_weight * temp_weight, dim=0 )
    
    loss_bound /= epochs

    loss_bound = torch.Tensor([loss_bound])
    spikenum_hid_inference = torch.Tensor([spikenum_hid_inference])
    spikenum_output_inference = torch.Tensor([spikenum_output_inference])

    return loss_bound, spikenum_hid_inference, spikenum_output_inference


def get_distance(network, input_sequence, output_sequence, num_samples, training_mode, writer=None):
    """
    Compute distance between the desired output sequence and generated output sequences 
    """
  
    network.set_mode('test-ll')
    network.reset_internal_state()

    S_prime = input_sequence.shape[-1]
    epochs = input_sequence.shape[0]
    S = S_prime * epochs

    # compute distance(A,B) = 1/constant * sum_t (filtered_output_A(t) - filtered_output_B(t))^2
    distance_avg = 0
    distance_ref = 0
    distance_fix = 0

    network.set_mode('inference')
    network.reset_internal_state()
    
    # Duplicate the networks 
    network_list = duplicate_networks(network, num_samples)
    training_params_list = list()
    for k in range(num_samples):
        training_params_list.append(TrainingParams(k))        
    
    output_fb_trace_target = torch.zeros([network.n_output_neurons, network.alphabet_size, S_prime])
    output_fb_trace_reference = torch.zeros([network.n_output_neurons, network.alphabet_size, S_prime])
    output_fb_trace_inference = torch.zeros([num_samples, network.n_output_neurons, network.alphabet_size, S_prime])
    spikenum_hid_inference = 0
    spikenum_output_inference = 0
    spikenum_hid_ref = 0
    spikenum_hid_fix = 0
    output_spikes_inference = torch.zeros([num_samples, network.n_output_neurons, network.alphabet_size, S_prime])
    output_spikes_reference = torch.zeros([network.n_output_neurons, network.alphabet_size, S_prime])
    hid_spikes_inference = torch.zeros([num_samples, network.n_hidden_neurons, network.alphabet_size, S_prime])
    hid_spikes_reference = torch.zeros([network.n_hidden_neurons, network.alphabet_size, S_prime])
        
    for s in range(S):
         
        output_fb_trace_target[:, :, s % S_prime] = network.compute_fb_trace(output_sequence[int(s / S_prime), :, :, max(0, s%S_prime -9):s%S_prime+1]).reshape(network.n_output_neurons, network.alphabet_size)
        
        for k in range(num_samples):
            # reset K networks and internal states for each example
            if s % S_prime == 0:
                network_list[k].reset_internal_state()
                
            training_params_list[k].log_proba = network_list[k](input_sequence[int(s / S_prime), :, :, s % S_prime])
            output_spikes_inference[k, :, :, s % S_prime] = network_list[k].spiking_history[network_list[k].output_neurons, :, -1]
            hid_spikes_inference[k, :, :, s % S_prime] = network_list[k].spiking_history[network_list[k].hidden_neurons, :, -1]

            output_fb_trace_inference[k, :, :, s % S_prime] = network_list[k].compute_fb_trace(network_list[k].spiking_history)[network_list[k].output_neurons, :].reshape(network.n_output_neurons, network.alphabet_size)
            
            spikenum_hid_inference += torch.nonzero(network_list[k].spiking_history[network_list[k].hidden_neurons, :, -1]).shape[0]
            spikenum_output_inference += torch.nonzero(network_list[k].spiking_history[network_list[k].output_neurons, :, -1]).shape[0]
        
            # average over K outputs
            if s % S_prime == S_prime - 1:
                distance_avg += torch.sum(torch.pow(output_fb_trace_target[:, :, :] - output_fb_trace_inference[k, :, :, :],2), dim=(0,-1))
               
        output_spikes_reference[:, :, s % S_prime] = (torch.sum(output_spikes_inference[:, :, :, s % S_prime], dim=0) > int(num_samples/2)).float()
        output_fb_trace_reference[:, :, s % S_prime] = network.compute_fb_trace(output_spikes_reference[:, :, max(0, s%S_prime -9):s%S_prime+1]).reshape(network.n_output_neurons, network.alphabet_size)
        hid_spikes_reference[:, :, s % S_prime] = (torch.sum(hid_spikes_inference[:, :, :, s % S_prime], dim=0) > int(num_samples/2)).float()
    
        spikenum_hid_ref += torch.nonzero(hid_spikes_reference[:, :, s % S_prime]).shape[0]
        spikenum_hid_fix += torch.nonzero(network_list[0].spiking_history[network_list[0].hidden_neurons, :, -1]).shape[0]

        if s % S_prime == S_prime - 1:
            distance_ref += torch.sum(torch.pow(output_fb_trace_target - output_fb_trace_reference, 2), dim=(0,-1))
            distance_fix += torch.sum(torch.pow(output_fb_trace_target[:, :, :] - output_fb_trace_inference[0, :, :, :],2), dim=(0,-1))
    
    distance_avg = distance_avg/(S*network.n_output_neurons*num_samples)
    distance_ref = distance_ref/(S*network.n_output_neurons)
    distance_fix = distance_fix/(S*network.n_output_neurons)
    
    distance_avg = torch.Tensor([distance_avg])
    distance_ref = torch.Tensor([distance_ref])
    distance_fix = torch.Tensor([distance_fix])
    spikenum_hid_inference = torch.Tensor([spikenum_hid_inference])
    spikenum_output_inference = torch.Tensor([spikenum_output_inference])
    spikenum_hid_ref = torch.Tensor([spikenum_hid_ref])
    spikenum_hid_fix = torch.Tensor([spikenum_hid_fix])

    return distance_avg, distance_ref, distance_fix, spikenum_hid_inference, spikenum_output_inference, spikenum_hid_ref, spikenum_hid_fix


def get_inference(network, input_sequence, output_sequence, dec_type, num_ll_est, writer=None):
    """
    Perform inference (given input_sequence, network) with num_ll_est (number of samples) and compute spike number of hidden and output neurons, and doing decision rule

    """

    network.set_mode('inference')
    network.reset_internal_state()

    S_prime = input_sequence.shape[-1]
    epochs = input_sequence.shape[0]

    S = S_prime * epochs
    acc_inference = 0
    spikenum_hid_inference = 0
    spikenum_output_inference = 0
    img_size = int(np.sqrt(input_sequence.shape[1]*2))

    if dec_type == 'single':
        num_ll_est = 1

    outputs_inference = torch.zeros([num_ll_est, epochs, network.n_output_neurons])
    img_sequence = torch.zeros([num_ll_est, S_prime, img_size, img_size])
    img_true = torch.zeros([num_ll_est, S_prime, img_size, img_size])
    img_outputs_inference = torch.zeros([num_ll_est, network.n_output_neurons, network.alphabet_size, S_prime])

    for nl in range(num_ll_est): # TODO: make duplicated networks
        network.reset_internal_state()

        for s in range(S):
            if s % S_prime == 0:
                network.reset_internal_state()

            log_proba_inference = network(input_sequence[int(s / S_prime), :, :, s % S_prime])
            outputs_inference[nl, int(s / S_prime), :] += torch.sum(network.spiking_history[network.output_neurons, :, -1], dim=-2)
            img_outputs_inference[nl, :, :, s % S_prime] = network.spiking_history[network.output_neurons, :, -1]

            if s % S_prime == S_prime-1:
                img_sequence[:, :, np.arange(int(img_size/2)), :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(input_sequence[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_ll_est,1,1,1)    
                img_sequence[:, :, int(img_size/2):, :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(img_outputs_inference.reshape(num_ll_est, img_size, int(img_size/2), S_prime), 270, (1,2)), (0,3,1,2)))
                img_true[:, :, np.arange(int(img_size/2)), :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(input_sequence[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_ll_est,1,1,1)    
                img_true[:, :, int(img_size/2):, :] = \
                    torch.from_numpy(np.transpose(ndimage.rotate(output_sequence[int(s/S_prime), :, :, :].reshape(img_size, int(img_size/2), S_prime), 270), (2,0,1))).repeat(num_ll_est,1,1,1)

                if int(s / S_prime) % 10 == 0 and nl == 0:
                    writer.add_images('generated images (inference)', img_sequence[0, :, :, :].reshape(S_prime, 1, img_size, img_size), s+1, dataformats='NCHW')
                    writer.add_images('true images (inference)', img_true[0, :, :, :].reshape(S_prime, 1, img_size, img_size), s+1, dataformats='NCHW')

            spikenum_hid_inference += torch.nonzero(network.spiking_history[network.hidden_neurons, :, -1]).shape[0]
            spikenum_output_inference += torch.nonzero(network.spiking_history[network.output_neurons, :, -1]).shape[0]

    if (dec_type == 'majority'):
        tempout, tempcount = torch.unique(torch.max(outputs_inference, dim=-1).indices, sorted=True, return_counts=True, dim=0)
        predictions = tempout[torch.max(tempcount, dim=-1).indices]
    elif (dec_type == 'maxnum') | (dec_type == 'single'):
        predictions = torch.max(torch.sum(outputs_inference, dim=0), dim=-1).indices

    true_classes = torch.max(torch.sum(output_sequence, dim=(-1,-2)), dim=-1).indices
    acc_inference = float(torch.sum(predictions == true_classes, dtype=torch.float) / len(predictions))
    
    return acc_inference, spikenum_hid_inference, spikenum_output_inference, outputs_inference