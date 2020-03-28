import os
import torch
import numpy as np 
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("..") 
from utils.inference_utils import get_distance, get_loss_bound
from SNN import SNNetwork

Z = 1.96  # For 95% confidence interval

def get_hprofs_from_files(path_dir):

    files_in_path = os.listdir(path_dir)
    hprofs = list()

    for f in files_in_path:
        filename = os.path.splitext(f)  # split filename and ext

        if not filename[1] == '.pth':  # read only pth files
            continue
        
        # store the hyperparameters from filename to dictionary
        hp_dict = dict()
        for v in filename[0].split('_'):
            hp_dict['filename']=os.path.join(path_dir, f)
            k = v.split('-')
            hp_dict[k[0]] = k[1]
        hprofs.append(hp_dict)
    return hprofs


def search_hprofs(hprof_list, query_hprof, time_range=None):
    """ return hprofs which is the same as input_hprof and satisfies time range """

    hprof_dict_by_time = dict()

    for hprof in hprof_list:
        if compare_hprofs(hprof, query_hprof):
            if not hprof['time'] in hprof_dict_by_time.keys():
                hprof_dict_by_time[hprof['time']] = list()
            hprof_dict_by_time[hprof['time']].append(hprof)

    if len(hprof_dict_by_time) == 0:
        print('There is no matched file', query_hprof)
        return None

    if time_range == None:  # return latest ones
        latest_key = sorted(hprof_dict_by_time.keys())[-1]
        return hprof_dict_by_time[latest_key]
    else:
        for t in reversed(sorted(hprof_dict_by_time.keys())):
            if time_range[0] <= t <= time_range[1]:
                return hprof_dict_by_time[t]

    print('There is no matched file', query_hprof, time_range)
    return None


def compare_hprofs(x, y):
    """ Check if hprof x and y are the same except for time and niter """

    x_keys = set(x.keys())
    y_keys = set(y.keys())
    intersect_keys = x_keys.intersection(y_keys)
    keys_diff = x_keys.union(y_keys) - intersect_keys

    # check if x and y incldue different hyperparam except for filename, niter, and time
    if len(keys_diff - {'filename','niter', 'time'}) > 0:
        return False

    # check all values are same except for niter and time
    for k in intersect_keys - {'niter', 'time'}:
        if not x[k] == y[k]:
            return False
    
    return True


def get_stats_from_hprof(hprofs, metrics):

    if hprofs == None:
        return None

    ret = dict()
    eval_data = dict()

    for i, h in enumerate(hprofs):
        load_data = torch.load(h['filename'])
        for m in metrics:
            d = load_data[m].numpy()  # TODO

            if i == 0:
                timestamp = load_data['timestamp'].numpy()
                len_timestep = len(timestamp)
                timestamp = timestamp.reshape(len_timestep)
                
                if len(timestamp) != len(d):
                    timestamp = np.append([], timestamp[-1])
                    len_timestep = len(d)

                eval_data[m] =  np.empty((len_timestep,0))
            
            eval_data[m] = np.concatenate((eval_data[m], d), axis=-1)
    
    for m in metrics:
        ret[m] = dict()
        ret[m]['num_of_iter'] = len(hprofs)
        ret[m]['mertic'] = m
        ret[m]['timestep'] = timestamp
        ret[m]['mean'] = np.mean(eval_data[m], axis=-1)
        ret[m]['var'] = np.var(eval_data[m], axis=-1)
        ret[m]['std'] = np.std(eval_data[m], axis=-1)
        ret[m]['confidence_interval'] = Z * np.sqrt(ret[m]['var']) / np.sqrt(ret[m]['num_of_iter'])

    return ret


def get_result(hp_list, sweep, metrics, input_hp, time_range=None):

    results = list()
    label = list()
    for s in sweep['value']:
        input_hp[sweep['hyperparam']] = str(s)

        target_hprofs = search_hprofs(hp_list, input_hp, time_range)
        if target_hprofs == None:
            continue
        results.append(get_stats_from_hprof(target_hprofs, metrics))
        label.append(sweep['hyperparam']+'_'+str(s))

    # print(results)
    plot_result(results, metrics, label)

    return 0


def plot_result(r_list, metrics, label=None):

    if not label == None and not len(r_list) == len(label):
        print('Number of label is not matched to data')
        label = None

    num_metric = len(metrics)
    fig_width = 3.6 * num_metric
    fig_height = 2.4 * 2
    fig, axs = plt.subplots(2, num_metric, figsize=[fig_width, fig_height], squeeze=False)

    for i, m in enumerate(metrics):
        for j, r in enumerate(r_list):
            x = r[m]['timestep']
            y = r[m]['mean']

            if label == None:
                axs[0][i].plot(x, y) 
            else:
                axs[0][i].plot(x, y, 'o-', label=label[j])
                axs[0][i].legend()

            lb = r[m]['mean'] - r[m]['confidence_interval']
            ub = r[m]['mean'] + r[m]['confidence_interval']
            axs[0][i].fill_between(x, lb, ub, alpha=0.4)

            axs[1][i].bar(j, r[m]['mean'][-1], yerr = r[m]['confidence_interval'][-1], align='center', ecolor='black', capsize=3)

            print('%s: %s, %f +- %f' % (m, label[j], r[m]['mean'][-1], r[m]['confidence_interval'][-1]), flush=True)
            
            axs[0][i].title.set_text(m)
            
    plt.show(block=False)


def get_result_2(hp_list, sweep1, sweep2, metrics, input_hp, time_range=None):

    result_dict = dict()
    label = list()
    for s1 in sweep1['value']:
        input_hp[sweep1['hyperparam']] = str(s1)
        result_dict[s1] = dict()
        for s2 in sweep2['value']:
            input_hp[sweep2['hyperparam']] = str(s2)

            target_hprofs = search_hprofs(hp_list, input_hp, time_range)
            if target_hprofs == None:
                continue
            result_dict[s1][s2] = get_stats_from_hprof(target_hprofs, metrics)

    plot_one_feature_2(result_dict, metrics)

    return 0


def plot_one_feature_2(r_dict, metrics, label=None):

    color_list = [u'b', u'g', u'r', u'c', u'm', u'y', u'k']

    num_metric = len(metrics)
    fig_width = 3.6 * num_metric
    fig_height = 2.4 * 2
    fig, axs = plt.subplots(2, num_metric, figsize=[fig_width, fig_height], squeeze=False)

    for i, m in enumerate(metrics):
        for j, (k1, v) in enumerate(r_dict.items()):
            
            w = 1/(len(v)+1)
            for k, (k2, r) in enumerate(v.items()):

                if r[m]['timestep'].size > 1:
                    x = r[m]['timestep']
                    y = r[m]['mean']

                    axs[0][i].plot(x, y, label=str(k1)+'_'+str(k2)) 
                    
                    lb = r[m]['mean'] - r[m]['confidence_interval']
                    ub = r[m]['mean'] + r[m]['confidence_interval']
                    axs[0][i].fill_between(x, lb, ub, alpha=0.4)

                    if j == 0:
                        axs[1][i].bar(j+k*w, r[m]['mean'][-1], yerr = r[m]['confidence_interval'][-1], align='center', color=color_list[k], width=w, ecolor='black', capsize=3, label=str(k2))
                    else:
                        axs[1][i].bar(j+k*w, r[m]['mean'][-1], yerr = r[m]['confidence_interval'][-1], align='center', color=color_list[k], width=w, ecolor='black', capsize=3)

                    print('%s: %s, %f +- %f' % (m, str(k1)+'_'+str(k2), r[m]['mean'][-1], r[m]['confidence_interval'][-1]), flush=True)

                elif r[m]['timestep'].size == 1:
                    if j == 0:
                        axs[1][i].bar(j+k*w, r[m]['mean'], yerr = r[m]['confidence_interval'], align='center', color=color_list[k], width=w, ecolor='black', capsize=3, label=str(k2))
                    else:
                        axs[1][i].bar(j+k*w, r[m]['mean'], yerr = r[m]['confidence_interval'], align='center', color=color_list[k], width=w, ecolor='black', capsize=3)
    
                    print('%s: %s, %f +- %f' % (m, str(k1)+'_'+str(k2), r[m]['mean'], r[m]['confidence_interval']), flush=True)
                    
        axs[1][i].legend()        
        axs[1][i].set_xticks(np.arange(len(r_dict)))
        axs[1][i].set_xticklabels(r_dict.keys())
               
        axs[0][i].title.set_text(m)
        axs[0][i].legend()

    plt.show(block=False)


def get_inference_eval(hp_list, sweep, metrics, input_hp, input_sequence, output_sequence, time_range=None):

    results = list()
    label = list()
    
    for s in sweep['value']:
        input_hp[sweep['hyperparam']] = str(s)
        target_hprofs = search_hprofs(hp_list, input_hp, time_range)
        
        if target_hprofs == None:
            continue
        
        results.append(get_inference_from_hprof(target_hprofs, metrics, input_sequence, output_sequence))
        label.append(sweep['hyperparam']+'_'+str(s))

    #print(results)
    plot_inference_result(results, metrics, label)

    return 0


def plot_inference_result(r_list, metrics, label=None):

    if not label == None and not len(r_list) == len(label):
        print('Number of label is not matched to data')
        label = None

    num_metric = len(metrics)
    fig_width = 3.6 * num_metric
    fig_height = 2.4 * 2
    fig, axs = plt.subplots(2, num_metric, figsize=[fig_width, fig_height], squeeze=False)

    for i, m in enumerate(metrics):
        x_line = list()
        y_line = list()
        y_ci_line = list()
        
        for j, r in enumerate(r_list):
            if r[m]['timestep'].size > 1:
                x = r[m]['timestep']
                y = r[m]['mean']

                if label == None:
                    axs[0][i].plot(x, y) 
                else:
                    axs[0][i].plot(x, y, label=label[j])
                    axs[0][i].legend()

                lb = r[m]['mean'] - r[m]['confidence_interval']
                ub = r[m]['mean'] + r[m]['confidence_interval']
                axs[0][i].fill_between(x, lb, ub, alpha=0.4)

                axs[1][i].bar(j, r[m]['mean'][-1], yerr = r[m]['confidence_interval'][-1], align='center', ecolor='black', capsize=3)

                print('%s: %s, %f +- %f' % (m, label[j], r[m]['mean'][-1], r[m]['confidence_interval'][-1]), flush=True)
            
            elif r[m]['timestep'].size == 1:
                axs[1][i].bar(j, r[m]['mean'], yerr = r[m]['confidence_interval'], align='center', ecolor='black', capsize=3)
                print('%s: %s, %f +- %f' % (m, label[j], r[m]['mean'], r[m]['confidence_interval']), flush=True)
                
                x_line.append(r[m]['timestep'][0])
                y_line.append(r[m]['mean'])
                y_ci_line.append(r[m]['confidence_interval'])

        print(' ', flush=True)

        axs[0][i].title.set_text(m)           
        axs[0][i].plot(x_line, y_line) 
        axs[0][i].fill_between(x_line, np.array(y_line)-np.array(y_ci_line), np.array(y_line)+np.array(y_ci_line), alpha=0.4)
            
    plt.show(block=False)


def get_inference_from_hprof(hprofs, metrics, input_sequence, output_sequence):
    
    if hprofs == None:
        return None
    
    ret = dict()
    eval_data = dict()
    
    for i, h in enumerate(hprofs):
        load_data = torch.load(h['filename'])
        Nk = int(h['Nk'])
        
        if h['mode'] == 'mb':
            mode = 'train_ml_multi_elbo'
        elif h['mode'] == 'iw':
            mode = 'train_ml_multi'
        elif h['mode'] == 'iwb':
            mode = 'train_ml_multi_ls'
        elif h['mode'] == 'gem':
            mode = 'train_ml_multi_nols'
        elif h['mode'] == 'mb1':
            mode = 'train_ml'
        
        for m in metrics:
            network = load_data['network']
            network.set_tau(20)
            #print('network filter memory length is %d' %(network.tau_fb))
            
            if i == 0:
                timestep = load_data['timestep']
                timestep = np.append([], timestep)
                eval_data[m] = np.empty((1,0))
            
            # inference
            if m == 'loss_bound':
                d, _, _ = get_loss_bound(network, input_sequence, output_sequence, Nk, mode)
            elif m == 'distance_avg':
                d, _, _, _ = get_distance(network, input_sequence, output_sequence, Nk, mode)
            elif m == 'distance_majority':
                _, d, _, _ = get_distance(network, input_sequence, output_sequence, Nk, mode)
            elif m == 'spikenum_hid':
                _, _, d, _ = get_distance(network, input_sequence, output_sequence, Nk, mode)
                
            eval_data[m] = np.append(eval_data[m], d)
        
    for m in metrics:
        ret[m] = dict()
        ret[m]['num_of_iter'] = len(hprofs)
        ret[m]['inference_metric'] = m
        ret[m]['timestep'] = timestep
        ret[m]['mean'] = np.mean(eval_data[m], axis=-1)
        ret[m]['var'] = np.var(eval_data[m], axis=-1)
        ret[m]['std'] = np.std(eval_data[m], axis=-1)
        ret[m]['confidence_interval'] = Z * np.sqrt(ret[m]['var']) / np.sqrt(ret[m]['num_of_iter'])
            
    #print(ret)
    print("done with get_inference_from_hprof", flush=True)
    
    return ret


if __name__ == "__main__":

    input_hprof_1 = { 
    'task': 'prediction',
    'digits': '2',
    'ex': 'single',
    'dec': 'majority',
    'eptrain': '10',
    'eptest': '30',
    'mode': 'iw',
    'niter': '3',
    'Nh': '2',
    'Nk': '10',
    'lr': '0.05',
    'lrconst': '1.5',
    'kappa': '0.05',
    'Nb': '8',
    'nll': '10',
    'time': '03131600'
    }    
    input_hprof_2 = { 
    'task': 'prediction',
    'digits': '2',
    'ex': 'single',
    'dec': 'majority',
    'eptrain': '30',
    'eptest': '30',
    'mode': 'iw',
    'niter': '3',
    'Nh': '2',
    'Nk': '10',
    'lr': '0.05',
    'lrconst': '1.5',
    'kappa': '0.05',
    'Nb': '8',
    'nll': '10'
    }    

    metrics = ['loss_output_train_t', 'acc_train_t']
    time_range = ['03131200', '03131830']
        
    hprof_file_list = get_hprofs_from_files('res_example')

    # target_hprofs_1 = get_hprof_separate_iter(hprof_file_list, input_hprof_1, time_range)
    # r_1 = get_stat_for_hprof(target_hprofs_1, metrics)

    # target_hprofs_2 = get_hprof_separate_iter(hprof_file_list, input_hprof_2)
    # r_2 = get_stat_for_hprof(target_hprofs_2, metrics)

    # plot_one_feature([r_1, r_2], metrics)

    sweep = {'hyperparam':'eptrain', 'value': [10, 30, 40]}
    get_result(hprof_file_list, sweep, metrics, input_hprof_1, time_range)
    
    # sweep1 = {'hyperparam':'eptrain', 'value': [10, 30, 40]}
    # sweep2 = {'hyperparam':'eptest', 'value': [10, 30, 40]}
    # get_result_2(hprof_file_list, sweep1, sweep2, metrics, input_hprof_1, time_range)
