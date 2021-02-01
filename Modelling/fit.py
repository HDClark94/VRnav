import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from Modelling.no_kalman_model import *
from Modelling.Stochastic_angelaki import *
from scipy import stats
from scipy.optimize import least_squares
from Modelling.params import Parameters

# we set the seed so results that rely on random number generation can be consistently reproduced.
np.random.seed(64)

'''
this script was written to model fit target response data to the Angelaki dynamic baseysian observer model
add reference (2018)

this script takes processed_results pandas dataframes created in concantenate_data.py
see /../summarize/

some functions within this script includes early attempts to fit stochastic model, this is still mostly wrong
'''


def fit_parameters_to_model(x, y, starters=[(0,2),(0,2),(0,0.1)]):
    _popt = []
    minimal = 1e16
    minimal_i = 0

    for i in range(50):
        # random assignment of starter parameter value
        p0_1 = np.random.uniform(low=starters[0][0], high=starters[0][1])
        p0_2 = np.random.uniform(low=starters[1][0], high=starters[1][1])
        p0_3 = np.random.uniform(low=starters[2][0], high=starters[2][1])

        popt, pcov = curve_fit(model, x, y, p0=[p0_1, p0_2, p0_3])
        sq_sum_error = np.sum(np.square(model(x, prior_gain=popt[0], lambda_coef=popt[1], k=popt[2]) - y))

        if sq_sum_error < minimal:
            minimal = sq_sum_error
            minimal_i = i
            print("New minimum found")
        _popt.append(popt)

    model_params = _popt[minimal_i]
    print("estimate of model parameters ", model_params)
    return model_params

def fit_parameters_to_Stochastic_model(x, y, starters=[(0,1),(0,1),(0,1),(0,1)]):
    params = Parameters(gain=1, dt=1, n_trials=10, N_iterations=500, velocity_max = 2, target_width=0.1)

    _popt = []
    minimal = 1e16
    minimal_i = 0

    for i in range(50):
        # random assignment of starter parameter value
        p0_1 = np.random.uniform(low=starters[0][0], high=starters[0][1])
        p0_2 = np.random.uniform(low=starters[1][0], high=starters[1][1])
        p0_3 = np.random.uniform(low=starters[2][0], high=starters[2][1])
        p0_4 = np.random.uniform(low=starters[3][0], high=starters[3][1])

        popt, pcov = curve_fit(model_stochastic, x, y, p0=[p0_1, p0_2, p0_3, p0_4])
        sq_sum_error = np.sum(np.square(model_stochastic(x, likelihood_width=popt[0], expon_coef=popt[1], lambda_coef=popt[2], k=p0_4) - y))

        if sq_sum_error < minimal:
            minimal = sq_sum_error
            minimal_i = i
            print("New minimum found")
        _popt.append(popt)

    model_params = _popt[minimal_i]
    print("estimate of model parameters ", model_params)
    return model_params

def fit(human_data, save_path, trial_type, model_parameters_df=None):
    '''
    :param human_data: processed results pandas dataframe
    :param save_path: the path to save dataframes and figures to
    :param trial_type: the trial type to model fit
    :return: model fit plots
            model fit parameter dataframe
    '''
    test_x = np.arange(0,400, 20)

    # un comment this to remove trials completed with the button tap method # reccommended
    human_data = human_data[human_data["movement_mechanism"]== "analogue"]

    human_data = human_data.dropna()
    human_mean_data = human_data.groupby(['Trial type', 'integration_length'])['first_stop_location'].mean().reset_index()
    human_mean_data_with_id = human_data.groupby(['Trial type', 'integration_length', "ppid"])['first_stop_location'].mean().reset_index()
    human_mean_with_id_x = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type)]['integration_length'])
    human_mean_with_id_y = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type)]['first_stop_location'])
    human_mean_data_x = np.asarray(human_mean_data[(human_mean_data["Trial type"] == trial_type)]['integration_length'])
    human_mean_data_y = np.asarray(human_mean_data[(human_mean_data["Trial type"] == trial_type)]['first_stop_location'])
    human_data_x = np.asarray(human_data[(human_data["Trial type"] == trial_type)]['integration_length'])
    human_data_y = np.asarray(human_data[(human_data["Trial type"] == trial_type)]['first_stop_location'])
    group_model_params = fit_parameters_to_model(human_mean_data_x, human_mean_data_y)

    # plotting model fit
    best_fit_responses = model(test_x, prior_gain=group_model_params[0], lambda_coef=group_model_params[1], k=group_model_params[2])
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
        r'$\mathrm{k}=%.2f$' % (group_model_params[2], )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
    plt.legend(loc="upper left")
    plt.savefig(save_path+"/"+trial_type+"_group_model_fit.png")
    plt.show()
    plt.close()

    ppids = np.unique(human_data["ppid"])
    subjects_model_params = np.zeros((len(ppids), 3)) # fitting 3 parameters

    if model_parameters_df is None:
        model_parameters_df = pd.DataFrame()

    for j in range(len(ppids)):
        subject_model_parameters_df = pd.DataFrame()

        subject_data_mean_x = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type) & (human_mean_data_with_id["ppid"] == ppids[j])]['integration_length'])
        subject_data_mean_y = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type) & (human_mean_data_with_id["ppid"] == ppids[j])]['first_stop_location'])

        subject_data_x = np.asarray(human_data[(human_data["Trial type"] == trial_type) & (human_data["ppid"] == ppids[j])]['integration_length'])
        subject_data_y = np.asarray(human_data[(human_data["Trial type"] == trial_type) & (human_data["ppid"] == ppids[j])]["first_stop_location"])

        subject_model_params = fit_parameters_to_model(subject_data_mean_x, subject_data_mean_y)
        subjects_model_params[j] = subject_model_params

        subject_model_parameters_df["ppid"] = [ppids[j]]
        subject_model_parameters_df["gamma"] = [subject_model_params [0]]
        subject_model_parameters_df["lambda"] = [subject_model_params [1]]
        subject_model_parameters_df["k"] = [subject_model_params [2]]
        subject_model_parameters_df["trial_type"] = [trial_type]
        model_parameters_df = pd.concat([model_parameters_df, subject_model_parameters_df], ignore_index=True)

        # plotting model fit
        best_fit_responses = model(test_x, prior_gain=subject_model_params[0], lambda_coef=subject_model_params[1], k=subject_model_params[2])
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
            r'$\Gamma=%.2f$' % (subject_model_params[0], ),
            r'$\lambda=%.2f$' % (subject_model_params[1], ),
            r'$\mathrm{k}=%.2f$' % (subject_model_params[2], )))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
        plt.legend(loc="upper left")
        plt.savefig(save_path+"/"+trial_type+"_"+ppids[j]+"_model_fit.png")
        plt.show()
        plt.close()

    print("for Gamma, mean = ", np.mean(subjects_model_params[:,0]))
    print("for lambda, mean = ", np.mean(subjects_model_params[:,1]))
    print("for k, mean = ", np.mean(subjects_model_params[:,2]))

    print("for Gamma, std = ", np.std(subjects_model_params[:,0]))
    print("for lambda, std = ", np.std(subjects_model_params[:,1]))
    print("for k, std = ", np.std(subjects_model_params[:,2]))

    print("for Gamma, p=", stats.ttest_1samp(subjects_model_params[:,0],1)[1]/2)
    print("for lambda, p=", stats.ttest_1samp(subjects_model_params[:,2],1)[1]/2)
    return model_parameters_df


def fit_stochastic(human_data, save_path, trial_type):
    test_x = np.arange(0,400, 20)

    human_data = human_data.dropna()
    human_mean_data = human_data.groupby(['Trial type', 'integration_length'])['first_stop_location'].mean().reset_index()
    human_mean_data_with_id = human_data.groupby(['Trial type', 'integration_length', "ppid"])['first_stop_location'].mean().reset_index()
    human_mean_with_id_x = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type)]['integration_length'])
    human_mean_with_id_y = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type)]['first_stop_location'])
    human_mean_data_x = np.asarray(human_mean_data[(human_mean_data["Trial type"] == trial_type)]['integration_length'])
    human_mean_data_y = np.asarray(human_mean_data[(human_mean_data["Trial type"] == trial_type)]['first_stop_location'])
    human_data_x = np.asarray(human_data[(human_data["Trial type"] == trial_type)]['integration_length'])
    human_data_y = np.asarray(human_data[(human_data["Trial type"] == trial_type)]['first_stop_location'])
    group_model_params = fit_parameters_to_Stochastic_model(human_data_x, human_data_y)

    # plotting model fit
    best_fit_responses = model_stochastic(test_x, likelihood_width=group_model_params[0], expon_coef=group_model_params[1],
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
    #textstr = '\n'.join((
    #    r'$\Gamma=%.2f$' % (group_model_params[0], ),
    #    r'$\lambda=%.2f$' % (group_model_params[1], ),
    #    r'$\mathrm{k}=%.2f$' % (group_model_params[2], )))
    #
    #props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    #ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
    plt.legend(loc="upper left")
    plt.savefig(save_path+"/"+trial_type+"_group_model_fit.png")
    plt.show()
    plt.close()

    ppids = np.unique(human_data["ppid"])
    subjects_model_params = np.zeros((len(ppids), 3)) # fitting 3 parameters
    for j in range(len(ppids)):
        subject_data_mean_x = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type) & (human_mean_data_with_id["ppid"] == ppids[j])]['integration_length'])
        subject_data_mean_y = np.asarray(human_mean_data_with_id[(human_mean_data_with_id["Trial type"] == trial_type) & (human_mean_data_with_id["ppid"] == ppids[j])]['first_stop_location'])

        subject_data_x = np.asarray(human_data[(human_data["Trial type"] == trial_type) & (human_data["ppid"] == ppids[j])]['integration_length'])
        subject_data_y = np.asarray(human_data[(human_data["Trial type"] == trial_type) & (human_data["ppid"] == ppids[j])]["first_stop_location"])

        subject_model_params = fit_parameters_to_model(subject_data_mean_x, subject_data_mean_y)
        subjects_model_params[j] = subject_model_params

        # plotting model fit
        best_fit_responses = model(test_x, prior_gain=subject_model_params[0], lambda_coef=subject_model_params[1], k=subject_model_params[2])
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
        #textstr = '\n'.join((
        #    r'$\Gamma=%.2f$' % (subject_model_params[0], ),
        #    r'$\lambda=%.2f$' % (subject_model_params[1], ),
        #    r'$\mathrm{k}=%.2f$' % (subject_model_params[2], )))
        #props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        #ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
        plt.legend(loc="upper left")
        plt.savefig(save_path+"/"+trial_type+"_"+ppids[j]+"_model_fit.png")
        plt.show()
        plt.close()

def main():
    '''

    :return:
    '''

    print("run something here")

    human_data = pd.read_pickle(r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV3.0\vr_recordings_Emre\processed_results.pkl")
    save_path = r"Z:\ActiveProjects\Harry\OculusVR\Figures\angelaki_model"

    model_params = pd.DataFrame()
    model_params = fit(human_data, save_path, trial_type="beaconed", model_parameters_df=model_params)
    model_params = fit(human_data, save_path, trial_type="non_beaconed", model_parameters_df=model_params)

    human_data = pd.read_pickle(r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV3.0\vr_recordings_Maya\processed_results.pkl")
    save_path = r"Z:\ActiveProjects\Harry\OculusVR\Figures\angelaki_model"
    model_params = fit(human_data, save_path, trial_type="beaconed", model_parameters_df=model_params)
    model_params = fit(human_data, save_path, trial_type="non_beaconed", model_parameters_df=model_params)

    # the model parameters dataframe is saved here
    model_params.to_pickle(save_path+"/model_parameters.pkl")

    # this isn't important right now
    #save_path = r"Z:\ActiveProjects\Harry\OculusVR\Figures\stochastic_angelaki_model"
    #fit_stochastic(human_data, save_path, trial_type="non_beaconed")
    #fit_stochastic(human_data, save_path, trial_type="beaconed")

if __name__ == '__main__':
    main()