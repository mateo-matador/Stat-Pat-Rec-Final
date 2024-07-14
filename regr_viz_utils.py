import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os

from FeatureTransformPolynomial import PolynomialFeatureTransform

src_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(src_dir), 'traffic_data')

def load_dataset(seed=123, data_dir=DATA_DIR):
    # Reading in Data / Prep
    # Read in data from Excel
    og_data = pd.read_excel("Stat_Pat_Rec_Project3.xlsx", sheet_name=1)
    # Deletes rows before Y_1 variable time series begins
    og_data = og_data.iloc[312:]
    # Deletes column Consumer Price Index for All Urban Consumers: Rent of Primary (too many NaN values)
    og_data = og_data.drop(['CUUR0000SEHA'], axis=1)
    # Renames column labels
    og_data = og_data.rename(columns={'AAA': 'Moodys Seasoned Aaa Corporate Bond Yield', 'CPIAUCNS': 'Consumer Price Index for All Urban Consumers: All Items in U.S. City', 'CURRCIR': 'Currency in Circulation','IPB50001N': 'Industrial Production: Total Index', 'PAYNSA': 'All Employees, Total Nonfarm', 'PPIACO': 'Producer Price Index by Commodity: All Commodities', 'TB3MS': '3-Month Treasury Bill Secondary Market Rate, Discount Basis'})
    # Delete last row with NaN value
    og_data.drop(og_data.tail(1).index,inplace=True)

    # Set DATE as index
    if not og_data.index.name == 'DATE':
        og_data.set_index('DATE', inplace=True)

    # Standardize data
    for curr_name in og_data.keys():
        og_data[curr_name] = (og_data[curr_name] - og_data[curr_name].mean()) / og_data[curr_name].std()

    # Splitting into Training, Validation, Testing
    x_train = og_data.loc['1939-01-01':'1999-12-01'] # end_row+1 = '1999-12-01' + 1
    t_train = og_data['3-Month Treasury Bill Secondary Market Rate, Discount Basis'].loc['1939-02-01':'2000-01-01']

    x_valid = og_data.loc['2000-01-01':'2010-12-01']
    t_valid = og_data['3-Month Treasury Bill Secondary Market Rate, Discount Basis'].loc['2000-02-01':'2011-01-01']

    x_test = og_data.loc['2011-01-01':'2024-01-01']
    t_test = og_data['3-Month Treasury Bill Secondary Market Rate, Discount Basis'].loc['2011-02-01':'2024-02-01']

    return x_train, t_train, x_valid, t_valid, x_test, t_test

def make_fig_for_estimator(
        dimension, Estimator,
        order_list, alpha_list, beta_list,
        x_train_ND, t_train_N,
        x_test_ND, t_test_N,
        test_scores_list=None,
        color='b',
        num_stddev=2,
        legend_label='MAP prediction',
        ):
    ''' Create figure showing estimator's predictions across orders

    Returns
    -------
    None

    Post Condition
    --------------
    Creates matplotlib figure
    '''
    fig, axgrid = prepare_x_vs_t_fig(order_list)
    xgrid_G1 = prepare_xgrid_G1(x_train_ND)

    # Loop over order of polynomial features
    # and associated axes of our plots
    for fig_col_id in range(len(order_list)):
        order = order_list[fig_col_id]
        alpha = alpha_list[fig_col_id]
        beta = beta_list[fig_col_id]
        cur_ax = axgrid[0, fig_col_id]

        feature_transformer = PolynomialFeatureTransform(
            order=order, input_dim=dimension)

        estimator = Estimator(
            feature_transformer, alpha=alpha, beta=beta)
        estimator.fit(x_train_ND, t_train_N)

        # Obtain predicted mean and stddev for estimator
        # at each x value in provided dense grid of size G
        mean_G = estimator.predict(xgrid_G1)
        var_G = estimator.predict_variance(xgrid_G1)

        plot_predicted_mean_with_filled_stddev_interval(
            cur_ax, # plot on figure's current axes
            xgrid_G1, mean_G, np.sqrt(var_G),
            num_stddev=num_stddev,
            color=color,
            legend_label=legend_label)
    finalize_x_vs_t_plot(
        axgrid, x_train_ND, t_train_N, x_test_ND, t_test_N,
        order_list, alpha_list, beta_list)

def prepare_x_vs_t_fig(
        order_list,
        ):
    ''' Prepare figure for visualizing predictions on top of train data

    Returns
    -------
    fig : figure handle object
    axgrid : axis grid object
    '''
    nrows = 1
    ncols = len(order_list)
    Wpanel = 4
    Hpanel = 3
    fig1, fig1_axgrid = plt.subplots(
        nrows=nrows, ncols=ncols,
        sharex=True, sharey=True, squeeze=False,
        figsize=(Wpanel * ncols, Hpanel * nrows))
    return fig1, fig1_axgrid

def prepare_xgrid_G1(
        x_train_ND,
        G=301,
        extrapolation_width_factor=0.5,
        ):
    '''
    
    Returns
    -------
    xgrid_G1 : 2D array, shape (G, 1)
        Grid of x points for making predictions 
    '''

    # To visualize prediction function learned from data,            
    # Create dense grid of G values between x.min() - R, x.max() + R
    # Basically, 2x as wide as the observed data values for 'x'
    xmin = x_train_ND.iloc[:,0].min()
    xmax = x_train_ND.iloc[:,0].max()
    R = extrapolation_width_factor * (xmax - xmin)
    xgrid_G = np.linspace(xmin - R, xmax + R, G)
    xgrid_G1 = np.reshape(xgrid_G, (G, 1))
    
    return xgrid_G1

def plot_predicted_mean_with_filled_stddev_interval(
        ax, xgrid_G1, t_mean_G, t_stddev_G,
        num_stddev=3,
        color='b',
        legend_label='MAP prediction',
        ):
    xgrid_G = np.squeeze(xgrid_G1)
    # Plot predicted mean and +/- 3 std dev interval
    ax.fill_between(
        xgrid_G,
        t_mean_G - num_stddev * t_stddev_G,
        t_mean_G + num_stddev * t_stddev_G,
        facecolor=color, alpha=0.2)
    ax.plot(
        xgrid_G, t_mean_G, 
        linestyle='-',
        color=color,
        label=legend_label)

def finalize_x_vs_t_plot(
        axgrid, 
        x_train_ND, t_train_N,
        x_test_ND, t_test_N,
        order_list=None,
        alpha_list=None,
        beta_list=None,
        transparency_level=0.2):
    # Make figure beautiful
    for ii, ax in enumerate(axgrid.flatten()):
        ax.plot(x_test_ND, t_test_N, 
            'r.', markersize=6, label='test data', mew=0, markeredgecolor='none',
            alpha=transparency_level)

        ax.plot(x_train_ND, t_train_N,
            'k.', markersize=9, label='train data', mew=0, markeredgecolor='none')

        alpha = alpha_list[ii]
        beta = beta_list[ii]
        order = order_list[ii]
        ax.set_title(
            "order = %d \n alpha=% .3g  beta=% .3g" % (
            order, alpha, beta))
        ax.set_ylim([-2, 12])
        ax.set_xlim([-1.75, 1.6]) # little to left for legend
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels([-24, -12, 0, 12, 24])
        ax.set_yticks([0, 3, 6, 9])
        ax.grid(axis='y')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        #ax.set_aspect('equal', 'box')
        ax.set_xlabel("hours since Mon midnight")
        if ii == 0:
            ax.set_ylabel("traffic count (1000s)")
            ax.legend(loc='upper left', fontsize=0.9*plt.rcParams['legend.fontsize'])
    plt.subplots_adjust(top=0.85, bottom=0.19)