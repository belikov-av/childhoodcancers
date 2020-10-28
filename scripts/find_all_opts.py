import argparse
import yaml
import os

import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='This script takes ' +
                                                 'path to input folder, with cohort folders '+
                                                 'and finds optimal values for all parameters.' +
                                                 '\n' + 'If output folder does not exist, script will create it.')
    parser.add_argument('-i', '--input_folder', type=str,
                        help='full path to input folder with points data')

    parser.add_argument('-o', '--output_path', type=str, help='full path to output .tsv file')

    return parser.parse_args()


def find_interval(input_filename):

    res_df = pd.read_csv(input_filename, sep='\t')

    opt_row = res_df.iloc[np.argmax(res_df['R2'])]

    res_dict = {}

    for arg_name in ['arg1', 'arg2', 'opt_scale']:

        # optimal
        cur_key = f'{arg_name}_optimal'
        cur_value = float(opt_row[arg_name])
        res_dict[cur_key] = cur_value

    # R2 in optimal point
    cur_key = f'R2 optimal'
    cur_value = opt_row['R2']
    res_dict[cur_key] = cur_value

    return res_dict


if __name__ == '__main__':

    args = parse_args()

    res_dict = dict()
    distr_names = ['logistic', 'weibull', 'extreme_value', 'normal', 'erlang']

    for distr_name in distr_names:
        folder_path = os.path.join(args.input_folder, f'{distr_name}_child_r2')

        for file_name in tqdm(sorted(os.listdir(folder_path))):
            cohort_name = file_name.split('.')[0]
            dict_key = f'{distr_name}, {cohort_name}'

            full_input_file_name = os.path.join(folder_path, file_name)

            cur_dict = find_interval(full_input_file_name)

            res_dict[dict_key] = cur_dict


    final_df = pd.DataFrame.from_dict(res_dict)
    final_df.insert(loc=0, column='value_name', value=final_df.index)
    final_df.to_csv(args.output_path, sep='\t', index=False)

    #final_df.to_excel(args.output_path.replace('.tsv', '.xlsx'))
