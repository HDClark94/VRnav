import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from Modelling.no_kalman_model import *
from Modelling.Stochastic_angelaki import *
from scipy import stats
from scipy.optimize import least_squares
from Modelling.params import Parameters, Model_Parameters

# we set the seed so results that rely on random number generation can be consistently reproduced.
np.random.seed(64)
model_params = Model_Parameters()

'''
this script was written to model fit target response data to the Angelaki dynamic baseysian observer model
add reference (2018)

this script takes processed_results pandas dataframes created in concantenate_data.py
see /../summarize/

some functions within this script includes early attempts to fit stochastic model, this is still mostly wrong
'''

def model_stochastic_2(target_distances, likelihood_width=1, expon_coef=1, lambda_coef=1, k=1):

    velocity_arange = np.arange(0, params.velocity_max, 0.01) # to sweep over potential velocity range
    target_width = 13.8
    gt_velocities = np.ones(params.N_iterations)
    gt_positions = np.append(0, np.cumsum(gt_velocities)[:-1])

    response_distances = []
    for j in range(len(target_distances)):
        not_passed_target = True
        # Initialization of state matrices
        X = np.array([0, 0.0])     # state vector position and velocity
        i=0 # iterator

        # we create a vector of n posterior velocities, this resembles a drift diffusion equation about the means of the liklihood function
        posterior_velocity_estimates = posterior_velocity(gt_velocities[i], expon_coef, likelihood_width, velocity_arange)
        posterior_position_estimates = np.cumsum(posterior_velocity_estimates)

        # Applying the Kalman Filter
        while(not_passed_target):
            #TODO consider using recursion to speed this shit up
            X = np.array([X[0]+posterior_velocity_estimates[i], posterior_velocity_estimates[i]])  # pos vel perceived vector

            tmp = gt_positions[i]*gt_positions[i]/X # this step extrapolates the response location for the current gt location
            Optimal_response, _ = position_uncertainty_recursive(tmp, lambda_coef=lambda_coef, target_width=target_width, k=k)
            i+=1

            if gt_positions[i] >= target_distances[j]:
                response_distances.append(Optimal_response)
                not_passed_target = False

    response_distances = np.array(response_distances)
    return response_distances


def compute_sigmas(x, y):
    '''
    :param target_distances: numpy array 1d
    :param response_distances: numpy array 1d
    :return: numpy array 1d of standard deviation calculated at target distances
    '''
    sigmas = np.zeros(len(x))
    for length_x in np.unique(x):
        std__at_length_x = np.std(y[x==length_x])
        sigmas[x==length_x] = std__at_length_x
    return sigmas

def compute_means(x, y):
    '''
    :param target_distances: numpy array 1d
    :param response_distances: numpy array 1d
    :return: numpy array 1d of means calculated at target distances
    '''
    means = np.zeros(len(x))
    for length_x in np.unique(x):
        mean_at_length_x = np.mean(y[x==length_x])
        means[x==length_x] = mean_at_length_x
    return means

def simple_variance_model(target_distances, likelihood_width):
    target_width = 13.8
    response_distances = []

    for target_distance in target_distances:
        X = np.array([target_distance, 1])

        # we add a new term "likelihood width" to determine how true the prior gain is for every trial
        prior_gain = np.random.normal(loc=model_params.gamma, scale=likelihood_width)

        Optimal_response, error = position_uncertainty(X, lambda_coef=model_params.lambda_coef, target_width=target_width, k=model_params.k)
        Optimal_response = prior_gain*Optimal_response
        response_distances.append(Optimal_response)

    sigmas = compute_sigmas(target_distances, np.array(response_distances))

    return sigmas # we want to output the variance at all distance as predicted by the model


def simple_variance_model_full(target_distances, likelihood_width):
    target_width = 13.8
    response_distances = []

    for target_distance in target_distances:
        X = np.array([target_distance, 1])

        # we add a new term "likelihood width" to determine how true the prior gain is for every trial
        prior_gain = np.random.normal(loc=model_params.gamma, scale=likelihood_width)

        Optimal_response, error = position_uncertainty(X, lambda_coef=model_params.lambda_coef, target_width=target_width, k=model_params.k)
        Optimal_response = prior_gain*Optimal_response
        response_distances.append(Optimal_response)

    sigmas = compute_sigmas(target_distances, np.array(response_distances))

    return np.array(response_distances), sigmas

def simple_stochastic_model(target_distances,  likelihood_width=1, prior_gain=1, lambda_coef=1, k=1):
    target_width = 13.8
    response_distances = []
    prior_gain_og = prior_gain
    for target_distance in target_distances:
        X = np.array([target_distance, 1])

        # we add a new term "likelihood width" to determine how true the prior gain is for every trial
        prior_gain = np.random.normal(loc=prior_gain, scale=likelihood_width)
        prior_gain=prior_gain_og

        Optimal_response, error = position_uncertainty(X, lambda_coef=lambda_coef, target_width=target_width, k=k)
        Optimal_response = prior_gain*Optimal_response
        response_distances.append(Optimal_response)

    response_distances = np.array(response_distances)

    return response_distances

def fit_parameters_to_Stochastic_model(x, y, sigmas, starters=[(0,1),(0,2),(0,2),(0,1)]):
    _popt = []
    minimal = 1e16
    minimal_i = 0

    for i in range(3):
        # random assignment of starter parameter value
        p0_1 = np.random.uniform(low=starters[0][0], high=starters[0][1])
        p0_2 = np.random.uniform(low=starters[1][0], high=starters[1][1])
        p0_3 = np.random.uniform(low=starters[2][0], high=starters[2][1])
        p0_4 = np.random.uniform(low=starters[3][0], high=starters[3][1])

        popt, pcov = curve_fit(simple_stochastic_model, x, y, sigma=1/sigmas, absolute_sigma=True, p0=[p0_1, p0_2, p0_3, p0_4])
        sq_sum_error = np.sum(np.square(simple_stochastic_model(x, likelihood_width=popt[0], prior_gain=popt[1], lambda_coef=popt[2], k=popt[3]) - y))

        if sq_sum_error < minimal:
            minimal = sq_sum_error
            minimal_i = i
            print("New minimum found")
        _popt.append(popt)

    model_params = _popt[minimal_i]
    print("estimate of model parameters ", model_params)
    return model_params


def fit_variance_to_model(x, data_sigmas, starters=[(0,0.5)]):
    _popt = []
    minimal = 1e16
    minimal_i = 0

    for i in range(100):
        # random assignment of starter parameter value
        p0_1 = np.random.uniform(low=starters[0][0], high=starters[0][1])
        popt, pcov = curve_fit(simple_variance_model, x, data_sigmas, p0=[p0_1])
        model_sigmas = simple_variance_model(x, likelihood_width=popt[0])
        sq_sum_error = np.sum(np.square(model_sigmas - data_sigmas))

        if sq_sum_error < minimal:
            minimal = sq_sum_error
            minimal_i = i
            print("New minimum found")
        _popt.append(popt)

    var_param = _popt[minimal_i]
    print("estimate of var parameters ", var_param)
    return var_param

def remove_dots(value_string):
    tmp = value_string.split(".")
    return "".join(tmp)

def get_std_with_id(human_data, human_std_data, y_column):
    y_std = []
    tmp_columns = human_std_data.columns.to_list()
    tmp_columns.remove(y_column)

    for index, row in human_data.iterrows():
        row = row.to_frame().T.reset_index(drop=True)

        tt = row["Trial type"].iloc[0]
        il = row["integration_length"].iloc[0]
        ppid = row["ppid"].iloc[0]

        corresponding_row = human_std_data[((human_std_data["Trial type"] == tt) &
                                            (human_std_data["integration_length"] == il) &
                                            (human_std_data["ppid"] == ppid))]

        std = corresponding_row[y_column].iloc[0]

        y_std.append(std)
    human_data["y_std"] = y_std
    return human_data

def get_std_with_group(human_data, human_std_data, y_column):
    y_std = []
    tmp_columns = human_std_data.columns.to_list()
    tmp_columns.remove(y_column)

    for index, row in human_data.iterrows():
        row = row.to_frame().T.reset_index(drop=True)

        tt = row["Trial type"].iloc[0]
        il = row["integration_length"].iloc[0]

        corresponding_row = human_std_data[((human_std_data["Trial type"] == tt) &
                                            (human_std_data["integration_length"] == il))]

        std = corresponding_row[y_column].iloc[0]
        y_std.append(std)

    human_data["y_std"] = y_std
    return human_data

def fit_not_so_stochastic(human_data_original, save_path, trial_type, model_parameters_df, constraints, params):
    # written to fit 4 parameters but we hold the likelihood width at 0 so its not stochastic at all,
    test_x = np.arange(0,400, 20)

    collumn_values = [""]
    if constraints is not "":
        collumn_values = np.unique(human_data_original[constraints])

        for collumn_value in collumn_values:

            # only filter results by the constraint if a constraint is parsed
            if collumn_value is not "":
                human_data = human_data_original[(human_data_original[constraints] == collumn_value)]
            else:
                human_data = human_data_original

            human_data = human_data.dropna()
            human_mean_data = human_data.groupby(['Trial type', 'integration_length'])['first_stop_location'].mean().reset_index()
            human_mean_data_with_id = human_data.groupby(['Trial type', 'integration_length', "ppid"])['first_stop_location'].mean().reset_index()
            human_std_data_with_id = human_data.groupby(['Trial type', 'integration_length', "ppid"])['first_stop_location'].std().reset_index()
            human_std_data_group = human_data.groupby(['Trial type', 'integration_length'])['first_stop_location'].std().reset_index()
            human_mean_with_id_x = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type)]['integration_length'])
            human_mean_with_id_y = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type)]['first_stop_location'])
            human_mean_data_x = np.asarray(human_mean_data[(human_mean_data["Trial type"] == trial_type)]['integration_length'])
            human_mean_data_y = np.asarray(human_mean_data[(human_mean_data["Trial type"] == trial_type)]['first_stop_location'])
            human_data_x = np.asarray(human_data[(human_data["Trial type"] == trial_type)]['integration_length'])
            human_data_y = np.asarray(human_data[(human_data["Trial type"] == trial_type)]['first_stop_location'])
            human_data_with_std_individual = get_std_with_id(human_data, human_std_data_with_id, y_column='first_stop_location')
            human_data_with_std_group = get_std_with_group(human_data, human_std_data_group,  y_column='first_stop_location')
            human_data_std_id = np.asarray(human_data_with_std_individual[(human_data_with_std_individual["Trial type"] == trial_type)]['y_std'])
            human_data_std_group = np.asarray(human_data_with_std_group[(human_data_with_std_group["Trial type"] == trial_type)]['y_std'])
            group_model_params = fit_parameters_to_Stochastic_model(human_data_x, human_data_y, human_data_std_group)

            # plotting model fit
            best_fit_responses = simple_stochastic_model(test_x, likelihood_width=group_model_params[0], prior_gain=group_model_params[1],
                                                  lambda_coef=group_model_params[2], k=group_model_params[3])
            # plot optimised response target
            fig = plt.figure(figsize = (6,6))
            ax = fig.add_subplot(1,1,1) #stops per trial
            plt.title("All subjects", fontsize="20")
            plt.scatter(human_mean_with_id_x, human_mean_with_id_y, color="r", marker="o")
            plt.plot(human_mean_data_x, human_mean_data_y, "r", label="data")
            plt.plot(test_x, best_fit_responses, "g", label="model")
            plt.plot(np.arange(0,400), np.arange(0,400), "k--", label="Unity")
            plt.xlabel("Target (VU)", fontsize=20)
            plt.xlim((0,400))
            plt.ylim((0,400))
            plt.ylabel("Response (VU)", fontsize=20)
            plt.subplots_adjust(left=0.2)
            ax.tick_params(axis='both', which='major', labelsize=15)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            textstr = '\n'.join((
                r'$\Gamma=%.2f$' % (group_model_params[0], ),
                r'$\lambda=%.2f$' % (group_model_params[1], ),
                r'$\mathrm{k}=%.2f$' % (group_model_params[2],),
                r'$\L=%.2f$'         % (group_model_params[3],)))
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
            plt.legend(loc="upper left")

            if constraints is not "":
                plt.savefig(save_path+"\\"+trial_type+"_"+constraints+remove_dots(str(collumn_value))+"_group_model_fit.png")
            else:
                plt.savefig(save_path+"\\"+trial_type+"_"+constraints+"_group_model_fit.png")
            plt.show()
            plt.close()

            # now we do it per subject
            ppids = np.unique(human_data["ppid"])
            subjects_model_params = np.zeros((len(ppids), 4)) # fitting 3 parameters
            for j in range(len(ppids)):
                subject_data_mean_x = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type) & (human_mean_data_with_id["ppid"] == ppids[j])]['integration_length'])
                subject_data_mean_y = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type) & (human_mean_data_with_id["ppid"] == ppids[j])]['first_stop_location'])

                subject_data_x = np.asarray(human_data[(human_data["Trial type"] == trial_type) & (human_data["ppid"] == ppids[j])]['integration_length'])
                subject_data_y = np.asarray(human_data[(human_data["Trial type"] == trial_type) & (human_data["ppid"] == ppids[j])]["first_stop_location"])
                subject_human_data_with_std_individual = np.asarray(human_data_with_std_individual[(human_data_with_std_individual["Trial type"] == trial_type) &
                                                                                                   (human_data_with_std_individual["ppid"] == ppids[j])]["y_std"])


                subject_model_params = fit_parameters_to_Stochastic_model(subject_data_x, subject_data_y, subject_human_data_with_std_individual)
                subjects_model_params[j] = subject_model_params

                # plotting model fit
                best_fit_responses = simple_stochastic_model(test_x, likelihood_width=subject_model_params[0], 
                                                             prior_gain=subject_model_params[1],
                                                             lambda_coef=subject_model_params[2], 
                                                             k=subject_model_params[3])
                # plot optimised response target
                fig = plt.figure(figsize = (6,6))
                ax = fig.add_subplot(1,1,1) #stops per trial
                plt.title(ppids[j], fontsize="20")
                plt.scatter(subject_data_x, subject_data_y, color="r", marker="o")
                plt.plot(subject_data_mean_x, subject_data_mean_y, "r", label="data")
                plt.plot(test_x, best_fit_responses, "g", label="model")
                plt.plot(np.arange(0,400), np.arange(0,400), "k--", label="Unity")
                plt.xlabel("Target", fontsize=20)
                plt.xlim((0,400))
                plt.ylim((0,400))
                plt.ylabel("Optimal Response", fontsize=20)
                plt.subplots_adjust(left=0.2)
                ax.tick_params(axis='both', which='major', labelsize=15)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                textstr = '\n'.join((
                    r'$\Gamma=%.2f$' % (group_model_params[0], ),
                    r'$\lambda=%.2f$' % (group_model_params[1], ),
                    r'$\mathrm{k}=%.2f$' % (group_model_params[2],),
                    r'$\L=%.2f$'         % (group_model_params[3],)))
                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
                plt.legend(loc="upper left")
                plt.savefig(save_path+"\\"+trial_type+"_"+ppids[j]+"_model_stochastic_fit.png")
                plt.show()
                plt.close()

    return model_parameters_df

def fit_variance(human_data_original, save_path, trial_type, model_params_df, constraints):
    # written to fit 4 parameters but we hold the likelihood width at 0 so its not stochastic at all,
    test_x = np.arange(0,400, 20)

    collumn_values = [""]
    if constraints is not "":
        collumn_values = np.unique(human_data_original[constraints])

        for collumn_value in collumn_values:

            # only filter results by the constraint if a constraint is parsed
            if collumn_value is not "":
                human_data = human_data_original[(human_data_original[constraints] == collumn_value)]
            else:
                human_data = human_data_original

            group_parameter_fits = model_params_df[(model_params_df["ppid"] == "group") & (model_params_df["trial_type"] == trial_type) & (model_params_df[constraints] == collumn_value)]
            model_params.likelihood_width = 0
            model_params.gamma = group_parameter_fits["gamma"].iloc[0]
            model_params.lambda_coef = group_parameter_fits["lambda"].iloc[0]
            model_params.k = group_parameter_fits["k"].iloc[0]

            human_data = human_data.dropna()
            human_mean_data = human_data.groupby(['Trial type', 'integration_length'])['first_stop_location'].mean().reset_index()
            human_mean_data_with_id = human_data.groupby(['Trial type', 'integration_length', "ppid"])['first_stop_location'].mean().reset_index()

            human_std_data_with_id = human_data.groupby(['Trial type', 'integration_length', "ppid"])['first_stop_location'].std().reset_index()
            human_std_data_group = human_data.groupby(['Trial type', 'integration_length'])['first_stop_location'].std().reset_index()

            human_mean_with_id_x = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type)]['integration_length'])
            human_mean_with_id_y = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type)]['first_stop_location'])
            human_mean_data_x = np.asarray(human_mean_data[(human_mean_data["Trial type"] == trial_type)]['integration_length'])
            human_mean_data_y = np.asarray(human_mean_data[(human_mean_data["Trial type"] == trial_type)]['first_stop_location'])
            human_data_x = np.asarray(human_data[(human_data["Trial type"] == trial_type)]['integration_length'])
            human_data_y = np.asarray(human_data[(human_data["Trial type"] == trial_type)]['first_stop_location'])

            human_data_y_std = compute_sigmas(human_data_x, human_data_y)

            #human_data_with_std_individual = get_std_with_id(human_data, human_std_data_with_id, y_column='first_stop_location')
            #human_data_with_std_group = get_std_with_group(human_data, human_std_data_group,  y_column='first_stop_location')
            #human_data_std_id = np.asarray(human_data_with_std_individual[(human_data_with_std_individual["Trial type"] == trial_type)]['y_std'])
            #human_data_std_group = np.asarray(human_data_with_std_group[(human_data_with_std_group["Trial type"] == trial_type)]['y_std'])
            var_param = fit_variance_to_model(human_data_x, human_data_y_std)

            # plotting model fit

            test_x = np.sort(human_data_x)
            best_fit_responses, best_fit_sigmas = simple_variance_model_full(test_x, likelihood_width=var_param[0])

            # plot optimised response target
            fig = plt.figure(figsize = (6,6))
            ax = fig.add_subplot(1,1,1) #stops per trial
            plt.title("All subjects", fontsize="20")
            plt.scatter(human_mean_with_id_x, human_mean_with_id_y, color="r", marker="o")
            plt.plot(human_mean_data_x, human_mean_data_y, "r", label="data")

            _, unique_idx = np.unique(test_x, return_index=True)
            unique_mask = create_mask(indices=unique_idx, size=len(test_x))
            model_data_x_means, model_data_y_std_means = sort_by_other_array(first_array_orderby= test_x[unique_mask],
                                                                             second_array=compute_means(test_x, best_fit_responses)[unique_mask])
            plt.plot(model_data_x_means, model_data_y_std_means, "g", label="model")

            plt.plot(np.arange(0,400), np.arange(0,400), "k--", label="Unity")
            plt.xlabel("Target (VU)", fontsize=20)
            plt.xlim((0,400))
            plt.ylim((0,400))
            plt.ylabel("Response (VU)", fontsize=20)
            plt.subplots_adjust(left=0.2)
            ax.tick_params(axis='both', which='major', labelsize=15)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            textstr = '\n'.join((
                r'$\Gamma=%.2f$' % (model_params.gamma, ),
                r'$\lambda=%.2f$' % (model_params.lambda_coef, ),
                r'$\mathrm{k}=%.2f$' % (model_params.k,),
                r'$\L=%.2f$'         % (var_param[0],)))
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
            plt.legend(loc="upper left")

            if constraints is not "":
                plt.savefig(save_path+"\\"+trial_type+"_"+constraints+remove_dots(str(collumn_value))+"_group_model_fit.png")
            else:
                plt.savefig(save_path+"\\"+trial_type+"_"+constraints+"_group_model_fit.png")
            plt.show()
            plt.close()

            # plot optimised variance target
            fig = plt.figure(figsize = (6,6))
            ax = fig.add_subplot(1,1,1) #stops per trial
            plt.title("All subjects", fontsize="20")
            plt.scatter(human_data_x, human_data_y_std, color="r", marker="o")

            _, unique_idx = np.unique(human_data_x, return_index=True)
            unique_mask = create_mask(indices=unique_idx, size=len(human_data_x))
            human_data_x_means, human_data_y_std_means = sort_by_other_array(first_array_orderby= human_data_x[unique_mask],
                                                    second_array=compute_means(human_data_x, human_data_y_std)[unique_mask])
            plt.plot(human_data_x_means, human_data_y_std_means, "r", label="data")

            _, unique_idx = np.unique(test_x, return_index=True)
            unique_mask = create_mask(indices=unique_idx, size=len(test_x))
            model_data_x_means, model_data_y_std_means = sort_by_other_array(first_array_orderby= test_x[unique_mask],
                                                                             second_array=compute_means(test_x, best_fit_sigmas)[unique_mask])

            plt.plot(model_data_x_means, model_data_y_std_means, "g", label="model")
            plt.xlabel("Target (VU)", fontsize=20)
            plt.xlim((0,400))
            plt.ylim((0,100))
            plt.ylabel("Response STD (VU)", fontsize=20)
            plt.subplots_adjust(left=0.2)
            ax.tick_params(axis='both', which='major', labelsize=15)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            textstr = '\n'.join((
                r'$\Gamma=%.2f$' % (model_params.gamma, ),
                r'$\lambda=%.2f$' % (model_params.lambda_coef, ),
                r'$\mathrm{k}=%.2f$' % (model_params.k,),
                r'$\L=%.2f$'         % (var_param[0],)))
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
            plt.legend(loc="upper left")

            if constraints is not "":
                plt.savefig(save_path+"\\"+trial_type+"_"+constraints+remove_dots(str(collumn_value))+"_group_model_variance_fit.png")
            else:
                plt.savefig(save_path+"\\"+trial_type+"_"+constraints+"_group_model_variance_fit.png")
            plt.show()
            plt.close()

            '''

            # now we do it per subject
            ppids = np.unique(human_data["ppid"])
            subjects_model_params = np.zeros((len(ppids), 4)) # fitting 3 parameters
            for j in range(len(ppids)):

                subject_parameter_fits = model_params_df[(model_params_df["ppid"] == ppids[j]) & (model_params_df["trial_type"] == trial_type) & (model_params_df[constraints] == collumn_value)]
                model_params.likelihood_width = 0
                model_params.gamma = subject_parameter_fits["gamma"].iloc[0]
                model_params.lambda_coef = subject_parameter_fits["lambda"].iloc[0]
                model_params.k = subject_parameter_fits["k"].iloc[0]

                subject_data_mean_x = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type) & (human_mean_data_with_id["ppid"] == ppids[j])]['integration_length'])
                subject_data_mean_y = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type) & (human_mean_data_with_id["ppid"] == ppids[j])]['first_stop_location'])

                subject_data_x = np.asarray(human_data[(human_data["Trial type"] == trial_type) & (human_data["ppid"] == ppids[j])]['integration_length'])
                subject_data_y = np.asarray(human_data[(human_data["Trial type"] == trial_type) & (human_data["ppid"] == ppids[j])]["first_stop_location"])

                subject_data_y_std = compute_sigmas(subject_data_x, subject_data_y)

                subject_human_data_with_std_individual = np.asarray(human_data_with_std_individual[(human_data_with_std_individual["Trial type"] == trial_type) &
                                                                                                   (human_data_with_std_individual["ppid"] == ppids[j])]["y_std"])

                sub_var_param = fit_variance_to_model(subject_data_x, subject_data_y_std)
                subjects_model_params[j] = sub_var_param

                # plotting model fit
                best_fit_responses, best_fit_sigmas = simple_variance_model_full(test_x, likelihood_width=sub_var_param[0])
                # plot optimised response target
                fig = plt.figure(figsize = (6,6))
                ax = fig.add_subplot(1,1,1) #stops per trial
                plt.title(ppids[j], fontsize="20")
                plt.scatter(subject_data_x, subject_data_y, color="r", marker="o")
                plt.plot(subject_data_mean_x, subject_data_mean_y, "r", label="data")
                plt.plot(test_x, best_fit_responses, "g", label="model")
                plt.plot(np.arange(0,400), np.arange(0,400), "k--", label="Unity")
                plt.xlabel("Target", fontsize=20)
                plt.xlim((0,400))
                plt.ylim((0,400))
                plt.ylabel("Optimal Response", fontsize=20)
                plt.subplots_adjust(left=0.2)
                ax.tick_params(axis='both', which='major', labelsize=15)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                textstr = '\n'.join((
                    r'$\Gamma=%.2f$' % (model_params.gamma, ),
                    r'$\lambda=%.2f$' % (model_params.lambda_coef, ),
                    r'$\mathrm{k}=%.2f$' % (model_params.k,),
                    r'$\L=%.2f$'         % (sub_var_param[0],)))
                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
                plt.legend(loc="upper left")
                plt.savefig(save_path+"\\"+trial_type+"_"+ppids[j]+"_model_stochastic_fit.png")
                plt.show()
                plt.close()


                # plot optimised variance target
                fig = plt.figure(figsize = (6,6))
                ax = fig.add_subplot(1,1,1) #stops per trial
                plt.title(ppids[j], fontsize="20")
                plt.scatter(subject_data_x, subject_data_y_std, color="r", marker="o")

                _, unique_idx = np.unique(subject_data_x, return_index=True)
                unique_mask = create_mask(indices=unique_idx, size=len(subject_data_x))
                subject_data_x_means, subject_data_y_std_means = sort_by_other_array(first_array_orderby= subject_data_x[unique_mask],
                                                                                 second_array=compute_means(subject_data_x, subject_data_y_std)[unique_mask])

                plt.plot(subject_data_x_means, subject_data_y_std_means, "r", label="data")
                plt.plot(test_x, best_fit_sigmas, "g", label="model")
                #plt.plot(np.arange(0,400), np.arange(0,400), "k--", label="Unity")
                plt.xlabel("Target (VU)", fontsize=20)
                plt.xlim((0,400))
                plt.ylim((0,100))
                plt.ylabel("Response SD (VU)", fontsize=20)
                plt.subplots_adjust(left=0.2)
                ax.tick_params(axis='both', which='major', labelsize=15)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                textstr = '\n'.join((
                    r'$\Gamma=%.2f$' % (model_params.gamma, ),
                    r'$\lambda=%.2f$' % (model_params.lambda_coef, ),
                    r'$\mathrm{k}=%.2f$' % (model_params.k,),
                    r'$\L=%.2f$'         % (sub_var_param[0],)))
                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
                plt.legend(loc="upper left")

                if constraints is not "":
                    plt.savefig(save_path+"\\"+trial_type+"_"+constraints+remove_dots(str(collumn_value))+"_group_model_variance_fit.png")
                else:
                    plt.savefig(save_path+"\\"+trial_type+"_"+constraints+"_group_model_variance_fit.png")
                plt.show()
                plt.close()
                
            '''

    return model_params_df

def create_mask(indices, size):
    mask = np.zeros(size, dtype=bool)
    for idx in indices:
        mask[idx] = 1
    return mask

def sort_by_other_array(first_array_orderby, second_array):
    inds = first_array_orderby.argsort()
    first_array_ordered = first_array_orderby[inds]
    second_array_ordered = second_array[inds]

    return first_array_ordered, second_array_ordered

def main():
    print("run something here")
    save_path = r"Z:\ActiveProjects\Harry\OculusVR\Figures\stochastic_angelaki_model"

    model_params_df = pd.read_pickle(r"Z:\ActiveProjects\Harry\OculusVR\Figures\angelaki_model\model_parameters.pkl")
    human_data = pd.read_pickle(r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV3.0\vr_recordings_Emre\processed_results.pkl")
    model_params = fit_variance(human_data, save_path, trial_type="non_beaconed", model_params_df=model_params_df, constraints="gain_std")
    model_params = fit_variance(human_data, save_path, trial_type="beaconed", model_params_df=model_params_df, constraints="gain_std")
    model_params.to_pickle(save_path+"/model_parameters.pkl")

    '''
    print("run something here")
    save_path = r"Z:\ActiveProjects\Harry\OculusVR\Figures\stochastic_angelaki_model"

    model_params = pd.DataFrame()
    human_data = pd.read_pickle(r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV3.0\vr_recordings_Emre\processed_results.pkl")
    model_params = fit_not_so_stochastic(human_data, save_path, trial_type="beaconed", model_parameters_df=model_params, constraints="gain_std", params=params)
    model_params = fit_not_so_stochastic(human_data, save_path, trial_type="non_beaconed", model_parameters_df=model_params, constraints="gain_std", params=params)

    model_params = pd.DataFrame()
    human_data = pd.read_pickle(r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV3.0\vr_recordings_Maya\processed_results.pkl")
    model_params = fit_stochastic(human_data, save_path, trial_type="beaconed", model_parameters_df=model_params, constraints="movement_mechanism")
    model_params = fit_stochastic(human_data, save_path, trial_type="non_beaconed", model_parameters_df=model_params, constraints="movement_mechanism")
    model_params.to_pickle(save_path+"/model_parameters.pkl")
    '''

if __name__ == '__main__':
    main()