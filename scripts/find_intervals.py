import argparse
import copy
import yaml
import os

import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='This script takes ' +
                                                 'path to input folder, with cohort files '+
                                                 'and fins intervals for all parameters.' +
                                                 '\n' + 'If output folder does not exist, script will create it.')
    parser.add_argument('-i', '--input_folder', type=str,
                        help='full path to input folder with points data')

    parser.add_argument('-o', '--output_path', type=str, help='full path to output .tsv file')
    parser.add_argument('-c', '--config', type=str, help='path to config file')

    return parser.parse_args()


def find_interval(input_filename,
                  arg1_name: str,
                  arg2_name: str,
                  threshold=0.95):

    res_df = pd.read_csv(input_filename, sep='\t')

    max_value = res_df['R2'].max()
    if max_value < threshold: # bad fitting
        return None

    opt_row = res_df.iloc[np.argmax(res_df['R2'])]
    res_df_above_threshold = res_df[res_df['R2'] > threshold]

    res_dict = {}
    arg2name = {'arg1': arg1_name,
                'arg2': arg2_name,
                'opt_scale': 'opt_scale'}

    for arg_name in ['arg1', 'arg2', 'opt_scale']:
        # min
        cur_key = f'{arg2name[arg_name]}_min'
        cur_value = res_df_above_threshold[arg_name].min()
        res_dict[cur_key] = cur_value

        # max
        cur_key = f'{arg2name[arg_name]}_max'
        cur_value = res_df_above_threshold[arg_name].max()
        res_dict[cur_key] = cur_value

        # optimal
        cur_key = f'{arg2name[arg_name]}_optimal'
        cur_value = float(opt_row[arg_name])
        res_dict[cur_key] = cur_value

    # R2 in optimal point
    cur_key = f'R2 optimal'
    cur_value = opt_row['R2']
    res_dict[cur_key] = cur_value

    return res_dict


if __name__ == '__main__':

    args = parse_args()
    config_path = args.config
    with open(config_path, 'r') as config_file:
        config_data = yaml.load(config_file, Loader=yaml.FullLoader)

    res_dict = dict()

    for file_name in tqdm(sorted(os.listdir(args.input_folder))):
        cohort_name = file_name.split('.')[0]

        full_input_file_name = os.path.join(args.input_folder, file_name)

        cur_dict = find_interval(full_input_file_name,
                                 config_data['arg1'],
                                 config_data['arg2'],
                                 config_data['threshold'])
        if cur_dict is not None:
            res_dict[cohort_name] = cur_dict
        else:
            res_dict[cohort_name] = None

    final_df = pd.DataFrame.from_dict(res_dict)
    final_df.to_csv(args.output_path, sep='\t')
    final_df.to_excel(args.output_path.replace('.tsv', '.xlsx'))
