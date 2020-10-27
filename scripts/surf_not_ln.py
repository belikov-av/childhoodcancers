from multiprocessing import Pool
import argparse
import copy
import os, yaml

from scipy.special import gamma
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from math import pi


def parse_args():
    parser = argparse.ArgumentParser(description='This script takes ' +
                                                 '"Childhood Incidence Data.xlsx" file, and fits Erlang.' +
                                                 '\n' + 'If output folder does not exist, script will create it.')
    parser.add_argument('-i', '--input_path', type=str,
                        help='full path to input "Childhood Incidence Data.xlsx" file')
    parser.add_argument('-o', '--output_folder', type=str, help='full path to output folder')
    parser.add_argument('-c', '--config', type=str, help='path to .yml config file for current fitting')
    parser.add_argument('-n', '--num_workers', type=int, help='num workers to run in parallel', default=1)


    return parser.parse_args()


def MSE_loss(true_probas, predicted_probas, weights_mode='equal'):
    N = len(true_probas)
    if weights_mode == 'equal':
        weights = np.ones(N)
    elif weights_mode == 'linear':
        weights = np.arange(N) / N
    elif weights_mode == 'exponential':
        weights = np.exp(np.arange(N) / N)
    else:
        raise Exception(f'weights mode is not recognized: {weights_mode}')
    true_probas = np.array(true_probas)
    predicted_probas = np.array(predicted_probas)

    return np.mean(np.power(weights * (true_probas - predicted_probas), 2))

## Probability density functions for different distributions

def Extreme_value_pdf_numpy(mu, b, t):
    return np.exp((mu - t) / b) * (1/b) * np.exp(-np.exp((mu - t) / b))


def Logistic_pdf_numpy(mu, b, t):
    return (1/b) * np.exp((t - mu) / b) / np.square(1 + np.exp((t - mu) / b))


def Normal_pdf_numpy(mu, b, t):
    return (1/(b * np.sqrt(2 * pi))) * np.exp(-0.5 * np.square((t - mu)/b))


def Weibull_pdf_numpy(k, b, t):
    return (k/b) * (t/b)**(k-1) * np.exp(-(t/b)**k)


def Erlang_pdf_float(k, b, t):
    return ( t**(k-1) * np.exp(-t/float(b)) ) / (b**k * gamma(k))


def find_mse_not_ln(predicted_pdf, true_pdf, scale):

    scaled_pred = scale * predicted_pdf
    return MSE_loss(true_pdf, scaled_pred)


def find_R2_not_ln(predicted_pdf, true_pdf, scale):

    scaled_pred = scale * predicted_pdf
    return r2_score(true_pdf, scaled_pred)


def find_optimal_scale_golder_section_not_ln(predicted_pdf, true_pdf, r=0.382, argument_abs_tol=1e-6):

    # scale is 'x'
    # mse if 'f(x)'

    # the idea is taken from: https://web.stanford.edu/class/msande311/lecture10.pdf

    ## 1) INITIALIZATION

    init_scale = np.sum(true_pdf) * 2.
    x_left = init_scale / 100.
    x_right = init_scale * 100

    n_steps = 0

    while True:

        n_steps += 1

        ## 2) FIND INNER VALUES

        x_left_hat = x_left + r * (x_right - x_left)
        x_right_hat = x_left + (1-r) * (x_right - x_left)

        # we put '-', as we want to minimize
        f_left_hat = -find_R2_not_ln(predicted_pdf, true_pdf, x_left_hat)
        f_right_hat = -find_R2_not_ln(predicted_pdf, true_pdf, x_right_hat)

        ## 3) UPDATE BORDERS

        if f_left_hat < f_right_hat:
            x_right = copy.deepcopy(x_right_hat)
        else:
            x_left = copy.deepcopy(x_left_hat)

        ## STOPPING CRITERIA
        if x_right - x_left < argument_abs_tol:

            x_optimal = (x_left + x_right) / 2
            final_r2 = find_R2_not_ln(predicted_pdf, true_pdf, x_optimal)
            return x_optimal, final_r2


def run(cohort):

    print(cohort, 'processing...')

    EPS=1e-6
    not_nan_bool = ~df[cohort].isna()
    cohort_array = df[cohort][not_nan_bool]
    non_nan_t = df['Age, years'][not_nan_bool]

    bad_ixs = np.where(cohort_array<EPS)[0]
    if len(bad_ixs) > 0:
        first_nonzero_ix = np.max(bad_ixs)+1
    else:
        first_nonzero_ix = 0

    non_zero_t = non_nan_t[first_nonzero_ix:]
    true_pdf = cohort_array[first_nonzero_ix:]

    #arg1_list = np.linspace(0.1, 20, 20)
    #arg2_list = np.linspace(0.1, 20, 20)

    result_dict ={'arg1' : [],
                  'arg2': [],
                  'opt_scale': [],
                  'R2': []}

    for arg1 in arg1_list:
        for arg2 in arg2_list:
            predicted_pdf = pdf_func(arg1, arg2, non_zero_t)
            if np.any(np.isnan(predicted_pdf)) or np.any(np.isinf(predicted_pdf)):
                current_opt_scale = 100
                current_r2 = -1
            else:
                current_opt_scale, current_r2 = \
                find_optimal_scale_golder_section_not_ln(
                    predicted_pdf,
                    true_pdf
                    )

            # save result
            result_dict['arg1'].append(arg1)
            result_dict['arg2'].append(arg2)
            result_dict['opt_scale'].append(current_opt_scale)
            result_dict['R2'].append(current_r2)

    res_df = pd.DataFrame.from_dict(result_dict)
    output_path = os.path.join(args.output_folder, cohort + '.tsv')
    res_df.to_csv(output_path, sep='\t', index=False)

    print(cohort, 'finished')


if __name__=='__main__':

    args = parse_args()
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)
    print('results will be saved to:', args.output_folder)

    df = pd.read_excel(args.input_path)
    all_cohorts = list(df.columns[1:])

    yaml_path = args.config
    with open(yaml_path, 'r') as yaml_file:
        configs_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        distr_name = configs_dict['distr_name']

        arg1_start = configs_dict['arg1_start']
        arg1_stop = configs_dict['arg1_stop']
        arg1_num_points = configs_dict['arg1_num_points']
        arg1_list = np.linspace(arg1_start, arg1_stop, arg1_num_points)

        arg2_start = configs_dict['arg2_start']
        arg2_stop = configs_dict['arg2_stop']
        arg2_num_points = configs_dict['arg2_num_points']
        arg2_list = np.linspace(arg2_start, arg2_stop, arg2_num_points)

    if distr_name == 'extreme_value':
        pdf_func = lambda arg1, arg2, t: Extreme_value_pdf_numpy(arg1, arg2, t)
    elif distr_name == 'logistic':
        pdf_func = lambda arg1, arg2, t: Logistic_pdf_numpy(arg1, arg2, t)
    elif distr_name == 'normal':
        pdf_func = lambda arg1, arg2, t: Normal_pdf_numpy(arg1, arg2, t)
    elif distr_name == 'weibull':
        pdf_func = lambda arg1, arg2, t: Weibull_pdf_numpy(arg1, arg2, t)
    elif distr_name == 'erlang':
        pdf_func = lambda arg1, arg2, t: Erlang_pdf_float(arg1, arg2, t)

    with Pool(args.num_workers) as p:
        p.map(run, all_cohorts)
