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

def main():
    # Load data into separate train, validation, test dataframes
    x_train, t_train, x_valid, t_valid, x_test, t_test = regr_viz_utils.load_dataset()

    # 3 sizes of train set to explore
    F = 7        # number of features
    TR = 733     # number of observations in training
    VA = 133     # number of observations in validation
    TE = 158     # number of observations in testing

    # Define the number of months to include in each prediction
    top_num_months = 7  # Change this value as needed

    # Define # months for which you want to record all scores for every order/alpha/beta combo (see ADD-ON 1)
    goal_months = 4 # Change this value as needed
    if (goal_months < 1) or (goal_months >= top_num_months):
        print("Error: goal_months has to be within the range 1 to the top_num_months to zoom in on data points for said goal # of months input.")

    # Saves array the best score for each # months input
    best_score_per_num_months = []

    # Hyperparameter search grid
    hypers_to_search = dict(
        order = [0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12],     #  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        alpha = np.logspace(-8, 1, 10).tolist(),                  # 1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10
        beta = [10.0, 15.0, 20.0, 25.0, 35.0, 50.0, 75.0, 100.0],               # 0.1, 0.119, 0.143, 0.17, 0.203, 0.242, 0.289, 0.346, 0.412, 0.492, 0.588, 0.702,
                                                                # 0.838, 1, 1.19, 1.43, 1.7, 2.03, 2.42, 2.89, 3.46, 4.12, 4.92, 5.88, 7.02, 8.38, 10
        )
    
    # ______________________

    # ADD-ON 1: See printing section below
    # Create a 3D numpy array to store scores for each hyperparameter combination
    hyperparameter_scores = np.zeros((len(hypers_to_search['order']),
                                    len(hypers_to_search['alpha']),
                                    len(hypers_to_search['beta'])))
    hyperparameter_alphas = np.zeros((len(hypers_to_search['order']),
                                    len(hypers_to_search['alpha']),
                                    len(hypers_to_search['beta'])))
    hyperparameter_betas = np.zeros((len(hypers_to_search['order']),
                                    len(hypers_to_search['alpha']),
                                    len(hypers_to_search['beta'])))
    hyperparameter_order = np.zeros((len(hypers_to_search['order']),
                                    len(hypers_to_search['alpha']),
                                    len(hypers_to_search['beta'])))

    # Fix # months and observe optimal score per order (with corresponding alpha and beta)
    max_val_score_per_order_given_goal_months = []
    order_corresponding_alpha = []
    order_corresponding_beta = []

    # Fix # months and observe optimal score per alpha (with corresponding order and beta)
    max_val_score_per_alpha_given_goal_months = []
    alpha_corresponding_order = []
    alpha_corresponding_beta = []

    # Fix # months and observe optimal score per alpha (with corresponding order and beta)
    max_val_score_per_beta_given_goal_months = []
    beta_corresponding_order = []
    beta_corresponding_alpha = []

    # ______________________

    for num_months in range(1, top_num_months):

        # Expand the feature matrix to include data from multiple months
        x_train_expanded = expand_features(x_train, num_months)
        x_valid_expanded = expand_features(x_valid, num_months)
        x_test_expanded = expand_features(x_test, num_months)

        # Adjust t sets accordingly
        # Delete the first two rows
        t_train_exp = t_train.drop(t_train.index[:num_months - 1])
        # Delete the first two rows
        t_valid_exp = t_valid.drop(t_valid.index[:num_months - 1])
        # Delete the first two rows
        t_test_exp = t_test.drop(t_test.index[:num_months - 1])

        # 3 sizes of train set to explore
        exp_F = F * num_months  # Each month has F=7 features

        best_score = -np.inf
        best_params = None

        for i, order in enumerate(hypers_to_search['order']):
            feature_transformer = PolynomialFeatureTransform(order=order, input_dim=exp_F)

            for j, alpha in enumerate(hypers_to_search['alpha']):
                for k, beta in enumerate(hypers_to_search['beta']):
                    estimator = LinearRegressionMAPEstimator(feature_transformer, alpha=alpha, beta=beta)
                    estimator.fit(x_train_expanded, t_train_exp)
                    score = estimator.score(x_valid_expanded, t_valid_exp)

                    if num_months == goal_months:
                        # i = order, j = alpha, k = beta
                        hyperparameter_scores[i, j, k] = score
                        hyperparameter_alphas[i, j, k] = alpha
                        hyperparameter_betas[i, j, k] = beta
                        hyperparameter_order[i, j, k] = order

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'order': order,
                            'alpha': alpha,
                            'beta': beta,
                        }

        print(f"\nBest validation score: {best_score}")
        print(f"Best hyperparameters: {best_params}")

        # Combine training and validation sets
        x_train_full = pd.concat([x_train_expanded, x_valid_expanded], axis=0)
        t_train_full = pd.concat([t_train_exp, t_valid_exp], axis=0)

        # Train the best estimator on the full training set
        best_feature_transformer = PolynomialFeatureTransform(order=best_params['order'], input_dim=exp_F)
        best_estimator = LinearRegressionMAPEstimator(best_feature_transformer, alpha=best_params['alpha'], beta=best_params['beta'])
        best_estimator.fit(x_train_full, t_train_full)

        # Evaluate the performance on the test set
        test_score = best_estimator.score(x_test_expanded, t_test_exp)
        print(f"Test set score: {test_score}")

        best_score_per_num_months.append(test_score)

    
    # ______________________

    # ;;;;;;;;;;;;;;;;;;;;;; 
    # ADD-ON 1: Gets best scores for all combinations of hyperparam X and Y given Z

    # Best scores for all combinations of hyperparam alpha and beta given order
    for i, order in enumerate(hypers_to_search['order']):
        # # Extracting a 2D slice (scores for each beta, alpha) for a specific order 
        order_index = i  # Index of the desired order in the 'order' list
        order_slice = hyperparameter_scores[order_index, :, :]
        # Get the indices of the maximum value
        max_index = np.unravel_index(np.argmax(order_slice), order_slice.shape)
        max_val_score_per_order_given_goal_months.append(np.max(order_slice))

        # Useful print statements if needed
        # print("_______________")
        # print("ORDER: ", order)
        # print("ALPHA: ", hyperparameter_alphas[order_index, max_index[0], max_index[1]])
        # print("BETA: ", hyperparameter_betas[order_index, max_index[0], max_index[1]])
        # print("SCORE: ", hyperparameter_scores[order_index, max_index[0], max_index[1]])
        # print("_______________")

        order_corresponding_alpha.append(hyperparameter_alphas[order_index, max_index[0], max_index[1]])
        order_corresponding_beta.append(hyperparameter_betas[order_index, max_index[0], max_index[1]])

    # Best scores for all combinations of hyperparam order and beta given alpha
    # for j, alpha in enumerate(hypers_to_search['alpha']):
    #     # # Extracting a 2D slice (scores for each order, beta) for a specific alpha
    #     alpha_index = j  # Index of the desired alpha in the 'alpha' list
    #     alpha_slice = hyperparameter_scores[:, alpha_index, :]
    #     max_index = np.unravel_index(np.argmax(alpha_slice), alpha_slice.shape)
    #     max_val_score_per_alpha_given_goal_months.append(np.max(alpha_slice))
    #     alpha_corresponding_order.append(hyperparameter_order[max_index[0], alpha_index, max_index[1]])
    #     alpha_corresponding_beta.append(hyperparameter_betas[max_index[0], alpha_index, max_index[1]])

    # Best scores for all combinations of hyperparam alpha and order given beta
    # for k, beta in enumerate(hypers_to_search['beta']):
    #     # # Extracting a 2D slice (scores for each order, beta) for a specific alpha
    #     beta_index = k  # Index of the desired alpha in the 'alpha' list
    #     beta_slice = hyperparameter_scores[:, :, beta_index]
    #     max_index = np.unravel_index(np.argmax(beta_slice), beta_slice.shape)
    #     max_val_score_per_beta_given_goal_months.append(np.max(beta_slice))
    #     beta_corresponding_order.append(hyperparameter_order[max_index[0], max_index[1], beta_index])
    #     beta_corresponding_alpha.append(hyperparameter_betas[max_index[0], max_index[1], beta_index])
    # ;;;;;;;;;;;;;;;;;;;;;; 

    # ______________________

    # ;;;;;;;;;;;;;;;;;;;;;; 
    # ADD-ON 2: Plots graph with # months input vs. best validation log likelihood score 
    # See Figure 2 in report
    # plot_scores(range(1, top_num_months), best_score_per_num_months)
    # key_order_list = hypers_to_search['order']
    # regr_viz_utils.make_fig_for_estimator(
    #     dimension = exp_F, 
    #     Estimator=LinearRegressionMAPEstimator,
    #     order_list=key_order_list,
    #     alpha_list=order_corresponding_alpha,
    #     beta_list=order_corresponding_beta,
    #     x_train_ND=x_train_full,
    #     t_train_N=t_train_full,
    #     x_test_ND=x_test,
    #     t_test_N=t_test,
    #     num_stddev=2,
    #     color='g',
    #     legend_label='MAP +/- 2 stddev')
    # plt.savefig('fig2a-our-version.pdf',bbox_inches='tight', pad_inches=0)
    # ;;;;;;;;;;;;;;;;;;;;;; 

    # ______________________

    # ;;;;;;;;;;;;;;;;;;;;;; 
    # ADD-ON 1: Creates 4 plots comparing best scoring model per order
    # SWITCH: Can change to create plots based on alpha or beta. To do so, uncomment ADD-ON 1 loops above (lines 154 - 197)
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(25, 5))

    for i, order in enumerate([0, 1, 2, 3]): # 0, 1, 2, 3 represent orders, change if needed
        # Get the predictions and standard deviations for the current degree
        feature_transformer = PolynomialFeatureTransform(order=order, input_dim=exp_F)
        estimator = LinearRegressionMAPEstimator(feature_transformer, alpha=order_corresponding_alpha[i], beta=order_corresponding_beta[i])
        estimator.fit(x_train_full, t_train_full)
        predictions = estimator.predict(x_test_expanded)
        stddev = 1 / np.sqrt(order_corresponding_beta[i])

        # Plot on the current subplot
        ax = axes[i]
        ax.plot(predictions, label='Predicted Mean' if i == 0 else None)
        ax.scatter(range(len(t_test_exp)), t_test_exp, label='True Values' if i == 0 else None, marker='o', color='r', s=5)
        ax.fill_between(range(len(predictions)),
                        predictions - 2 * stddev,
                        predictions + 2 * stddev,
                        alpha=0.3,
                        label=r'$\pm 2\sigma$' if i == 0 else None)

        # Add labels and title
        ax.set_xlabel('Data Point Index')
        ax.set_ylabel('Value')
        ax.set_title(f'Polynomial Order: {order} \n alpha: {order_corresponding_alpha[i]} \n beta: {order_corresponding_beta[i]}')
        ax.legend()

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.5)

    # Show the plot
    plt.show()
    # ;;;;;;;;;;;;;;;;;;;;;; 

    # ______________________



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
    plt.title('Best Validation Score for Each # Months Input')
    plt.xlabel('# Months Input')
    plt.ylabel('Best Log Likelihod')
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    main()