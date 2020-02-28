# this script will pull relavent features from each recording and compile them into a large dataframe from which you can analyse in whatever way you like.

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from summarize.plotting import *
import json
from summarize.common import *
from sklearn.linear_model import LinearRegression
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from summarize.tidy import *


def plot_mean_error(recording_folder_path, error_type, trial_type):
    results = pd.read_pickle(recording_folder_path+"\processed_results.pkl")
    results = results.dropna()

    mean_error = results.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].mean().reset_index()
    sd_error =  results.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].std().reset_index()

    fig, ax = plt.subplots()

    for experiment in np.unique(mean_error.experiment):
        if "hab" not in experiment:
            lengths = mean_error[(mean_error.experiment == experiment) & (mean_error["Trial type"] == trial_type)]['integration_length']
            exp_mu_error = mean_error[(mean_error.experiment == experiment) & (mean_error["Trial type"] == trial_type)][error_type]
            exp_sd_error = sd_error[(sd_error.experiment == experiment) & (sd_error["Trial type"] == trial_type)][error_type]
            ax.plot(lengths, exp_mu_error, label = experiment)
            ax.fill_between(lengths, exp_mu_error-exp_sd_error, exp_mu_error+exp_sd_error, alpha=0.3)

    plt.ylabel(error_type, fontsize=12, labelpad=10)
    plt.xlabel('Target (m)', fontsize=12, labelpad=10)
    plt.title(trial_type_title(trial_type))
    #plt.xlim(0, max(mean_error['integration_length']))
    #plt.ylim(0, max(mean_error[error_type]))
    plt.legend()
    plt.show()


def plot_every_mean_error(recording_folder_path, error_type, trial_type, researcher):
    results = pd.read_pickle(recording_folder_path+"\processed_results.pkl")
    results = results.dropna()

    if researcher == "Emre":
        results = results[(results.experiment == 'basic_settings_cued_RLincrem_ana_1s_0') |
                          (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_05') |
                          (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_2')]
    else:
        results = results[(results.experiment == 'basic_settings_non_cued_RLincrem_button_tap') |
                          (results.experiment == 'basic_settings_RLimcrem_analogue')]

    mean_error = results.groupby(['ppid', 'experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].mean().reset_index()
    sd_error =  results.groupby(['ppid', 'experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].std().reset_index()
    mean_error = mean_error.dropna()
    sd_error = sd_error.dropna()

    fig, ax = plt.subplots()

    colours = ['red', 'blue', 'green']
    i=0
    for experiment in np.unique(mean_error.experiment):
        c = colours[i]

        exp_mean_errors = []
        for ppid in np.unique(mean_error.ppid):
            if "hab" not in experiment:
                lengths = mean_error[(mean_error.experiment == experiment) &
                                     (mean_error["Trial type"] == trial_type) &
                                     (mean_error.ppid == ppid)]['integration_length']
                exp_mu_error = mean_error[(mean_error.experiment == experiment) &
                                          (mean_error["Trial type"] == trial_type) &
                                          (mean_error.ppid == ppid)][error_type]
                exp_sd_error = sd_error[(sd_error.experiment == experiment) &
                                        (sd_error.ppid == ppid) &
                                        (sd_error["Trial type"] == trial_type)][error_type]

                if len(np.array(exp_mu_error)) == 5:
                    ax.plot(lengths, exp_mu_error, color=c, alpha=0.14)
                    exp_mean_errors.append(np.asarray(exp_mu_error))

        ax.plot(lengths, np.mean(np.stack(exp_mean_errors), axis=0), color=c, alpha=1, label=legend_title_experiment(experiment))
        ax.fill_between(lengths,
                        np.mean(np.stack(exp_mean_errors), axis=0)-np.std(np.stack(exp_mean_errors), axis=0),
                        np.mean(np.stack(exp_mean_errors), axis=0)+np.std((exp_mean_errors), axis=0), alpha=0.3, color=c)
        i+=1

    plt.ylabel(yaxis_mean_error_title(error_type), fontsize=12, labelpad=10)
    plt.xlabel('Target (m)', fontsize=12, labelpad=10)
    plt.title(trial_type_title(trial_type))
    #plt.xlim(0, max(mean_error['integration_length']))
    plt.ylim(-75, 100)
    plt.legend()
    plt.show()

def plot_every_sd_error(recording_folder_path, error_type, trial_type, researcher):
    results = pd.read_pickle(recording_folder_path+"\processed_results.pkl")
    results = results.dropna()

    if researcher == "Emre":
        results = results[(results.experiment == 'basic_settings_cued_RLincrem_ana_1s_0') |
                          (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_05') |
                          (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_2')]
    else:
        results = results[(results.experiment == 'basic_settings_non_cued_RLincrem_button_tap') |
                          (results.experiment == 'basic_settings_RLimcrem_analogue')]

    mean_error = results.groupby(['ppid', 'experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].mean().reset_index()
    sd_error =  results.groupby(['ppid', 'experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].std().reset_index()
    mean_error = mean_error.dropna()
    sd_error = sd_error.dropna()

    fig, ax = plt.subplots()

    colours = ['red', 'blue', 'green']
    i=0
    for experiment in np.unique(mean_error.experiment):
        c = colours[i]

        exp_sd_errors = []
        for ppid in np.unique(mean_error.ppid):
            if "hab" not in experiment:
                lengths = mean_error[(mean_error.experiment == experiment) &
                                     (mean_error["Trial type"] == trial_type) &
                                     (mean_error.ppid == ppid)]['integration_length']

                exp_sd_error = sd_error[(sd_error.experiment == experiment) &
                                        (sd_error.ppid == ppid) &
                                        (sd_error["Trial type"] == trial_type)][error_type]

                if len(np.array(exp_sd_error)) == 5:
                    ax.plot(lengths, exp_sd_error, color=c, alpha=0.14)
                    exp_sd_errors.append(np.asarray(exp_sd_error))

        ax.plot(lengths, np.mean(np.stack(exp_sd_errors), axis=0), color=c, alpha=1, label=legend_title_experiment(experiment))
        ax.fill_between(lengths,
                        np.mean(np.stack(exp_sd_errors), axis=0)-np.std(np.stack(exp_sd_errors), axis=0),
                        np.mean(np.stack(exp_sd_errors), axis=0)+np.std((exp_sd_errors), axis=0), alpha=0.3, color=c)
        i+=1

    plt.ylabel(yaxis_variance_error_title(error_type), fontsize=12, labelpad=10)
    plt.xlabel('Target (m)', fontsize=12, labelpad=10)
    plt.title(trial_type_title(trial_type))
    #plt.xlim(0, max(mean_error['integration_length']))
    plt.ylim(0, 100)
    plt.legend()
    plt.show()

def plot_variance_of_error(recording_folder_path, error_type, trial_type, researcher):
    results = pd.read_pickle(recording_folder_path+"\processed_results.pkl")
    results = results.dropna()

    if researcher == "Emre":
        results = results[(results.experiment == 'basic_settings_cued_RLincrem_ana_1s_0') |
                          (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_05') |
                          (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_2')]

    elif researcher == "Maja":
        results = results[(results.experiment == 'basic_settings_non_cued_RLincrem_button_tap') |
                          (results.experiment == 'basic_settings_RLimcrem_analogue')]

    sd_error =  results.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].std().reset_index()

    ppid_sd_error = results.groupby(['ppid', 'experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].std().reset_index()
    sd_sd_error = ppid_sd_error.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].std().reset_index()

    fig, ax = plt.subplots()
    colours = ['red', 'blue', 'green']

    i=0
    for experiment in np.unique(sd_error.experiment):
        c = colours[i]
        if "hab" not in experiment:
            lengths = sd_error[(sd_error.experiment == experiment) & (sd_error["Trial type"] == trial_type)]['integration_length']
            exp_var_error = sd_error[(sd_error.experiment == experiment) & (sd_error["Trial type"] == trial_type)][error_type]
            exp_var_var_error = sd_sd_error[(sd_sd_error.experiment == experiment) & (sd_sd_error["Trial type"] == trial_type)][error_type]

            ax.plot(lengths, exp_var_error, label= legend_title_experiment(experiment), color=c)
            ax.fill_between(lengths, exp_var_error-exp_var_var_error, exp_var_error+exp_var_var_error, alpha=0.3, color=c)
            i+=1

    plt.ylabel(yaxis_variance_error_title(error_type), fontsize=12, labelpad=10)
    plt.xlabel('Target (m)', fontsize=12, labelpad=10)
    plt.title(trial_type_title(trial_type))
    #plt.xlim(0, max(mean_error['integration_length']))
    #plt.ylim(0, max(mean_error[error_type]))
    plt.legend()
    plt.show()



def process(recording_folder_path, researcher, trial_type):
    results = pd.read_pickle(recording_folder_path+"\processed_results.pkl")

    if researcher == "Emre":
        results = results[(results.experiment == 'basic_settings_cued_RLincrem_ana_1s_0') |
                         (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_05') |
                         (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_2')]

    elif researcher == "Maja":
        results = results[(results.experiment == 'basic_settings_non_cued_RLincrem_button_tap') |
                          (results.experiment == 'basic_settings_RLimcrem_analogue')]

    # specify one trial type
    results = results[(results['Trial type']  == trial_type)]

    stops_on_track = plt.figure(figsize=(6, 6))
    #ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    fig, ax = plt.subplots()

    distances = np.arange(0,400)
    reward_zone_size = 13.801
    ax.fill_between(distances, distances-reward_zone_size/2, distances+reward_zone_size/2, facecolor="k", alpha=0.3)
    if researcher == "Emre":
        cm = plt.cm.get_cmap('viridis')
        dd = ax.scatter(results["target"], results["first_stop_location_relative_to_ip"], c=results["gain_std"], cmap=cm)
        cbar = plt.colorbar(dd)
        cbar.ax.locator_params(nbins=5)
        cbar.set_label('Gain SD', rotation=270, labelpad=12)

    elif researcher == "Maja":
        cm = ListedColormap(['yellow', 'blue'])
        results.loc[results.movement_mechanism == "analogue", 'movement_mechanism'] = 0
        results.loc[results.movement_mechanism == "button_tap", 'movement_mechanism'] = 1

        dd = ax.scatter(results["target"], results["first_stop_location_relative_to_ip"], c=results["movement_mechanism"], cmap=cm)

        yellow_patch = mpatches.Patch(color='yellow', label='Button')
        blue_patch = mpatches.Patch(color='blue', label='Joystick')
        plt.legend(handles=[yellow_patch, blue_patch])

    plt.ylabel('Response (m)', fontsize=12, labelpad=10)
    plt.xlabel('Target (m)', fontsize=12, labelpad=10)
    #stops_on_track.colorbar(im, ax=ax)
    plt.xlim(0, max(distances))
    plt.ylim(0, max(distances)+100)
    plt.title(trial_type_title(trial_type))

    x = results["target"].values
    y = results["first_stop_location_relative_to_ip"].values
    x = x[~np.isnan(y)].reshape(-1, 1)
    y = y[~np.isnan(y)].reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x,y)  # perform linear regression
    Y_pred = linear_regressor.predict(distances.reshape(-1, 1))  # make predictions
    plt.plot(distances, Y_pred, color='red', label="Linear Regression fit")

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree = 2)
    X_poly = poly.fit_transform(x)
    poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)
    plt.plot(distances, lin2.predict(poly.fit_transform(distances.reshape(-1, 1))), color = 'magenta', label="Polynomial fit, k=2")

    poly = PolynomialFeatures(degree = 3)
    X_poly = poly.fit_transform(x)
    poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)
    plt.plot(distances, lin2.predict(poly.fit_transform(distances.reshape(-1, 1))), color = 'darkviolet', label="Polynomial fit, k=3")

    poly = PolynomialFeatures(degree = 4)
    X_poly = poly.fit_transform(x)
    poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)
    plt.plot(distances, lin2.predict(poly.fit_transform(distances.reshape(-1, 1))), color = 'blue', label="Polynomial fit, k=4")

    poly = PolynomialFeatures(degree = 5)
    X_poly = poly.fit_transform(x)
    poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)
    plt.plot(distances, lin2.predict(poly.fit_transform(distances.reshape(-1, 1))), color = 'deepskyblue', label="Polynomial fit, k=5")

    #ax.yaxis.set_ticks_position('left')
    #ax.xaxis.set_ticks_position('bottom')
    plt.legend()
    plt.savefig(recording_folder_path + '/bias.png', dpi=200)
    plt.show()
    plt.close()


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # type path name in here with similar structure to this r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    #results_path = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    #process(results_path, "Emre")
    #results_path = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Maya"
    #process(results_path, "Maja")


    results_path = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    researcher = "Emre"
    trial_type = "non_beaconed"
    plot_every_mean_error(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)
    plot_every_mean_error(results_path, error_type="absolute_first_stop_error", trial_type=trial_type, researcher=researcher)
    plot_every_sd_error(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)
    #plot_every_sd_error(results_path, error_type="absolute_first_stop_error", trial_type=trial_type, researcher=researcher)
    process(results_path, researcher=researcher, trial_type=trial_type)

    trial_type = "beaconed"
    plot_every_mean_error(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)
    plot_every_mean_error(results_path, error_type="absolute_first_stop_error", trial_type=trial_type, researcher=researcher)
    plot_every_sd_error(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)
    #plot_every_sd_error(results_path, error_type="absolute_first_stop_error", trial_type=trial_type, researcher=researcher)
    process(results_path, researcher=researcher, trial_type=trial_type)

    '''
    results_path = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Maya"
    researcher = "Maja"
    trial_type = "non_beaconed"
    plot_every_mean_error(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)
    #plot_every_mean_error(results_path, error_type="absolute_first_stop_error", trial_type=trial_type, researcher=researcher)
    plot_every_sd_error(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)
    #plot_every_sd_error(results_path, error_type="absolute_first_stop_error", trial_type=trial_type, researcher=researcher)
    process(results_path, researcher=researcher, trial_type=trial_type)

    trial_type = "beaconed"
    plot_every_mean_error(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)
    #plot_every_mean_error(results_path, error_type="absolute_first_stop_error", trial_type=trial_type, researcher=researcher)
    plot_every_sd_error(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)
    #plot_every_sd_error(results_path, error_type="absolute_first_stop_error", trial_type=trial_type, researcher=researcher)
    process(results_path, researcher=researcher, trial_type=trial_type)
    '''
if __name__ == '__main__':
    main()