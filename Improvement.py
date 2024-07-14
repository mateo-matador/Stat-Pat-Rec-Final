'''
Summary
-------
1. Select best hyperparameters (alpha, beta) of linear regression via a grid search
-- Use the score function of MAPEstimator on heldout set (average across K=5 folds).
2. Plot the best score found vs. polynomial feature order.
-- Normalize scale of log probabilities by dividing by train size N
3. Report test set performance of best overall model (alpha, beta, order)
4. Report overall time required for model selection

'''
import numpy as np
import pandas as pd
import time
import copy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_context("notebook")

import regr_viz_utils
from FeatureTransformPolynomial import PolynomialFeatureTransform
from LinearRegressionMAPEstimator import LinearRegressionMAPEstimator

from statsmodels.tsa.arima.model import ARIMA
import scipy.stats

def main():
    x_train, t_train, x_valid, t_valid, x_test, t_test = regr_viz_utils.load_dataset()

    # 3 sizes of train set to explore
    F = 7        # number of features
    TR = 732     # number of observations in training
    VA = 132     # number of observations in validation
    TE = 157     # number of observations in testing

    # Logs the best score for each month
    best_score_per_num_months = []

    # # Expand the feature matrix to include data from multiple months
    # x_train_expanded = expand_features(x_train, num_months)
    # x_valid_expanded = expand_features(x_valid, num_months)
    # x_test_expanded = expand_features(x_test, num_months)

    # TRAINING: Isolating exogenous (non y-value) features
    exog_train = x_train
    # TRAINING: Isolating endogenous (y-value) feature and shifting by 1 month so index lines up with exog_train (rate is for 1 month after index says now)
    endog_train = t_train
    endog_train.index = pd.to_datetime(endog_train.index) - pd.DateOffset(months=1)

    # VALID: Isolating exogenous (non y-value) features
    exog_valid = x_valid
    # VALID: Isolating endogenous (y-value) feature and shifting by 1 month so index lines up with exog_train (rate is for 1 month after index says now)
    endog_valid = t_valid
    endog_valid.index = pd.to_datetime(endog_valid.index) - pd.DateOffset(months=1)

    # TEST: Isolating exogenous (non y-value) features
    exog_test = x_test
    # TEST: Isolating endogenous (y-value) feature and shifting by 1 month so index lines up with exog_train (rate is for 1 month after index says now)
    endog_test = t_test
    endog_test.index = pd.to_datetime(endog_test.index) - pd.DateOffset(months=1)

    # Original (comprehensive) hyperparameter search
    # hypers_to_search = dict(
    #                 p = [1, 2, 3, 4, 5, 6, 7, 10, 12],              # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] p: The lag order, representing the number of lag observations incorporated in the model.
    #                 d = [0, 1, 2, 3],                                   # d: [0, 1, 2, 3, 4, 5, 6] Degree of differencing, denoting the number of times raw observations undergo differencing.
    #                 q = [1, 2, 3, 4, 5, 6, 7, 10, 12],     # q: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] Order of moving average, indicating the size of the moving average window.                                           
    #                 beta = np.logspace(-1, 1, 5).tolist()              # np.logspace(-1, 1, 3 + 24).tolist(): 0.1, 0.119, 0.143, 0.17, 0.203, 0.242, 0.289, 0.346, 0.412, 0.492, 0.588, 0.702,
    #                                                                 # 0.838, 1, 1.19, 1.43, 1.7, 2.03, 2.42, 2.89, 3.46, 4.12, 4.92, 5.88, 7.02, 8.38, 10
    #                 )

    # Top scoring hyperparameter combination (for faster runtime)
    hypers_to_search = dict(
                    p = [10],             
                    d = [0],                                  
                    q = [6],                                            
                    beta = [10]              
                    )

    combos = len(hypers_to_search['p']) * len(hypers_to_search['d']) * len(hypers_to_search['q']) * len(hypers_to_search['beta'])

    curr_combo = 0

    best_score = -np.inf
    best_params = None

    # Initialize ARIMA models with each combo of hyperparameters
    for l, beta in enumerate(hypers_to_search['beta']):
        for i, p in enumerate(hypers_to_search['p']):
            for j, d in enumerate(hypers_to_search['d']):
                for k, q in enumerate(hypers_to_search['q']):
                    curr_combo = curr_combo + 1
                    print(curr_combo)
                    
                    model = ARIMA(endog=endog_train, exog=exog_train, order=(p,d,q), freq='MS')

                    model_fit = model.fit()

                    # # Make predictions on the validation set
                    valid_predictions = model_fit.forecast(len(exog_valid), exog=exog_valid)

                    # ______________________

                    # ;;;;;;;;;;;;;;;;;;;;;; 
                    # ADD-ON 1: Plots predictions vs actual in validation set
                    # plt.plot(valid_predictions, label='Predicted Yield')
                    # plt.plot(endog_valid.index, endog_valid, label='True Yield')
                    # plt.legend(loc = "upper left")
                    # plt.title("Validation Set Predictions vs True Values")
                    # plt.show()
                    # ;;;;;;;;;;;;;;;;;;;;;; 

                    # ______________________

                    # Calculate log likelihood
                    total_log_proba = scipy.stats.norm.logpdf(
                    np.array(valid_predictions), np.array(endog_valid), 1.0/np.sqrt(beta))
                    score = np.sum(total_log_proba) / VA

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'beta': beta,
                            'p': p,
                            'd': d,
                            'q': q
                        }

    print(f"\nBest validation score: {best_score}")
    print(f"Best hyperparameters: {best_params}")

    # Combine training and validation sets
    x_train_full = pd.concat([x_train, x_valid])
    t_train_full = pd.concat([t_train, t_valid])

    # Train the best estimator on the full training set
    model = ARIMA(endog=endog_train, exog=exog_train, order=(best_params['p'],best_params['d'],best_params['q']), freq='MS')
    model_fit = model.fit()

    test_predictions = model_fit.forecast(len(exog_test), exog=exog_test)

    # Evaluate the performance on the test set
    total_log_proba = scipy.stats.norm.logpdf(
    np.array(test_predictions), np.array(endog_test), 1.0/np.sqrt(best_params['beta']))
    test_score = np.sum(total_log_proba) / VA
    print(f"Test set score: {test_score}")


def expand_features(features, num_months):
    """
    Expand the feature matrix to include data from multiple consecutive months.

    Args:
    - features (pd.DataFrame): Original feature matrix where each row represents one month
    - num_months (int): Number of consecutive months to include in each prediction

    Returns:
    - expanded_features (pd.DataFrame): Expanded feature matrix
    """
    expanded_features = []

    for i in range(len(features) - num_months + 1):
        expanded_feature = []
        for j in range(num_months):
            expanded_feature.extend(features.iloc[i + j])
        expanded_features.append(expanded_feature)

    return pd.DataFrame(expanded_features, columns=[f'month_{i+1}_feature_{j+1}' for i in range(num_months) for j in range(len(features.columns))])

def plot_scores(num_months_list, best_score_per_num_months):
    plt.figure(figsize=(8, 6))
    plt.plot(num_months_list, best_score_per_num_months, marker='o', linestyle='-')
    plt.title('Best Test Set Score vs. Number of Months')
    plt.xlabel('Number of Months')
    plt.ylabel('Best Test Set Score')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()