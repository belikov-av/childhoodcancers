import argparse
import copy
import os

import pandas as pd
import numpy as np
#from matplotlib import cm
#import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yaml

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='This script takes ' +
                                                 'path to input folder, with cohort files '+
                                                 'and save the result to output folder.' +
                                                 '\n' + 'If output folder does not exist, script will create it.')
    parser.add_argument('-i', '--input_folder', type=str,
                        help='full path to input folder with points data')
    parser.add_argument('-c', '--config_path', type=str,
                            help='full path to .yml config')

    parser.add_argument('-o', '--output_folder', type=str, help='full path to output folder')

    return parser.parse_args()


def get_optimal(df):

    opt_row = df.iloc[np.argmax(df['R2'])]
    return float(opt_row['arg1']), float(opt_row['arg2'])

def plot_cohort(input_filename, output_filename, config_path):
    # 0. Read config file
    with open(config_path, 'r') as config_file:
        conf_data = yaml.load(config_file, Loader=yaml.FullLoader)

    # 1. read as many arguments as we can
    ## height and width
    if 'height' in conf_data and 'width' in conf_data:
        height = conf_data['height']
        width = conf_data['width']
    else:
        height = 1000
        width = 1000
    # fontsize
    if 'fontsize' in conf_data:
        fontsize = conf_data['fontsize']
    else:
        fontsize = 16

    # colorbar data
    if 'is_colorbar' in conf_data:
        is_colorbar = conf_data['is_colorbar']
    else:
        is_colorbar = True

    # colorbar thickness
    if 'colorbar_thickness' in conf_data:
        colorbar_thickness = conf_data['colorbar_thickness']
    else:
        colorbar_thickness = 25

    # optimal value inscription
    if 'is_opt_value' in conf_data:
        is_opt_value = conf_data['is_opt_value']
    else:
        is_opt_value = True

    # show distribution name
    if 'show_distr_name' in conf_data:
        show_distr_name = conf_data['show_distr_name']
    else:
        show_distr_name = True

    # margins
    if len(set(conf_data.keys()) & set(['margin_l', 'margin_r', 'margin_b', 'margin_t'])) == 4:
        margin_l = conf_data['margin_l']
        margin_r = conf_data['margin_r']
        margin_b = conf_data['margin_b']
        margin_t = conf_data['margin_t']
    else:
        margin_l = 70
        margin_r = 10
        margin_b = 10
        margin_t = 70

    # opt_point_size:
    if 'opt_size' in conf_data:
        opt_size = conf_data['opt_size']
    else:
        opt_size = 5

    x_size = conf_data['arg1_size']
    x_name = conf_data['arg1_name']

    y_size = conf_data['arg2_size']
    y_name = conf_data['arg2_name']

    cohort = os.path.basename(input_filename).split('.')[0]
    res_df = pd.read_csv(input_filename, sep='\t')

    opt_k, opt_b = get_optimal(res_df)

    x_min, x_max = res_df['arg1'].values.min(), res_df['arg1'].values.max()
    y_min, y_max = res_df['arg2'].values.min(), res_df['arg2'].values.max()

    x = np.linspace(x_min, x_max, x_size)
    y = np.linspace(y_min, y_max, y_size)

    lower_value = 0.9
    z = np.nan_to_num(res_df['R2'].values, lower_value).reshape((x_size, y_size)).T

    # define colorbar dict
    if is_colorbar:
        colorbar_dict = dict(
            title=r'$R^2$', # title here
            xpad=20,
            titleside='right',
            thickness=colorbar_thickness,
            thicknessmode='pixels',
            len=0.7,
            lenmode='fraction',
            outlinewidth=0
        )
    else:
        colorbar_dict = None

    fig = go.Figure(data =
    go.Contour(
        z=z,
        x=x, # horizontal axis
        y=y, # vertical axis
        colorscale='Portland',
        contours=dict(
            start=0.9,
            end=1.00,
            size=0.01,
        ),
        colorbar=colorbar_dict,
        showscale=is_colorbar
    ))

    fig.add_trace(
    go.Scatter(x=[opt_k],
               y=[opt_b],
               mode='markers',
               marker=dict(
                   color='Black',
                   size=opt_size,
                   ),
               name='Optimal <br> value',
               showlegend=is_opt_value
        )
    )

    if show_distr_name:
        title_text = f'{cohort}, <br>{conf_data["distribution"]} distribution, ' + \
        f'{x_name}={opt_k:.2f}, ' + \
        f'{y_name}={opt_b:.2f}'
    else:
        title_text = f'{cohort}, ' + f'{x_name}={opt_k:.2f}, ' + f'{y_name}={opt_b:.2f}'

    num_ticks = conf_data['num_ticks']
    dxtick = x_max / num_ticks
    x_ticks = [number for number in np.linspace(dxtick, x_max, num_ticks)]
    if np.isclose(dxtick, round(dxtick), atol=1e-4):
        x_ticks_str = [f'{number:.0f}' for number in x_ticks]
    else:
        x_ticks_str = [f'{number:.1f}' for number in x_ticks]

    if x_name in ['mu', 'lambda', 'sigma', 'beta', 'theta']:
        x_name = f'$\{x_name}$'
    if y_name in ['mu', 'lambda', 'sigma', 'beta', 'theta']:
        y_name = f'$\{y_name}$'

    fig.update_layout(
        autosize=False,
        title=title_text,
        xaxis_title=x_name,
        yaxis_title=y_name,

        yaxis = dict(
            tickmode = 'linear',
            tick0 = 0.0,
            dtick = y_max / num_ticks
        ),

        xaxis = dict(
            tickmode = 'array',
            tickvals = x_ticks,
            ticktext = x_ticks_str
        ),

        font=dict(
            family="Avenir Next Condensed",
            size=fontsize
        ),

        height=height,
        width=width,

        margin=dict(l=margin_l,
                    r=margin_r,
                    b=margin_b,
                    t=margin_t,
                    pad=0),
        paper_bgcolor="white",
    )

    fig.update_yaxes(range=[0, y_max+0.1])
    fig.update_xaxes(range=[x_min, x_max])

    fig.write_image(output_filename)


if __name__=='__main__':

    args = parse_args()

    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    for file_name in tqdm(os.listdir(args.input_folder)):
        cohort_name = file_name.split('.')[0]

        full_input_file_name = os.path.join(args.input_folder, file_name)
        full_output_file_name = os.path.join(args.output_folder, cohort_name + '.pdf')
        try:
            plot_cohort(full_input_file_name, full_output_file_name, args.config_path)
        except:
            raise Exception(f' bad filename: {full_input_file_name}')
