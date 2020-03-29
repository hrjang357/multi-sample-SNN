import os
import numpy as np
import tables
import torch
from utils.inference_utils import get_distance, get_loss_bound

def merge_eval_files(path_dir):

    files_in_path = os.listdir(path_dir)
    eval_dict = dict()
    
    for f in files_in_path:
        filename = os.path.splitext(f)  # split filename and ext
        if not filename[1] == '.eval':  # read only eval files
            continue
        
        if not '_t-' in f:
            continue 

        filename_t = filename[0].split('_t-')
        file_key = filename_t[0]
        timestep = filename_t[1]
        if not file_key in eval_dict.keys():
            eval_dict[file_key] = dict()
        eval_dict[file_key][int(timestep)] = f
    
    for file_key, file_value in eval_dict.items():
        result_dict = dict()
        for timestep, filename in sorted(file_value.items()):
            load_filename=os.path.join(path_dir, filename)
            load_data = torch.load(load_filename)

            if len(result_dict.keys()) == 0:
                # result_dict = load_data
                result_dict['timestamp'] = torch.Tensor([[timestep]])
                for k, v in load_data.items():
                    result_dict[k] = torch.Tensor([[v]])


            else:
                result_dict['timestamp'] = torch.cat((result_dict['timestamp'], torch.Tensor([[timestep]])), 0)
                for k, v in load_data.items():
                    result_dict[k] = torch.cat((result_dict[k], torch.Tensor([[v]])), 0)

        save_filename = file_key+'_eval-eval.pth'
        save_filename = os.path.join(path_dir, save_filename)
        torch.save(result_dict, save_filename)
        print(save_filename)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluation with inference.')
    parser.add_argument('-p', '--path', default='results', help="Path of eval result files (pth files)")
    args = parser.parse_args()

    merge_eval_files(args.path)
