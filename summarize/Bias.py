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
    fig, ax = plt.subplots()
    results.dropna()
    results = results[(results["Trial type"] == trial_type)]

    if researcher == "Emre":
        results = results[(results.experiment == 'basic_settings_cued_RLincrem_ana_1s_0') |
                          (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_05') |
                          (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_2')]

        mean_error = results.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].mean().reset_index()
        sem_error =  results.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].sem().reset_index()

        mean_error_ppid = results.groupby(['ppid', 'experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].mean().reset_index()
        sem_error_ppid =  results.groupby(['ppid', 'experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].sem().reset_index()


        e1_mean = mean_error[(mean_error['gain_std'] == 0)]
        e1_sem = sem_error[(sem_error['gain_std'] == 0)]
        e2_mean = mean_error[(mean_error['gain_std'] == 0.5)]
        e2_sem = sem_error[(sem_error['gain_std'] == 0.5)]
        e3_mean = mean_error[(mean_error['gain_std'] == 2)]
        e3_sem = sem_error[(sem_error['gain_std'] == 2)]

        e1_mean_ppid = mean_error_ppid[(mean_error_ppid['gain_std'] == 0)]
        e1_sem_ppid = sem_error_ppid[(sem_error_ppid['gain_std'] == 0)]
        e2_mean_ppid = mean_error_ppid[(mean_error_ppid['gain_std'] == 0.5)]
        e2_sem_ppid = sem_error_ppid[(sem_error_ppid['gain_std'] == 0.5)]
        e3_mean_ppid = mean_error_ppid[(mean_error_ppid['gain_std'] == 2)]
        e3_sem_ppid = sem_error_ppid[(sem_error_ppid['gain_std'] == 2)]

        ax.plot(e1_mean_ppid["integration_length"], e1_mean_ppid[error_type], color="blue", alpha=0.3, linewidth=1)
        ax.plot(e2_mean_ppid["integration_length"], e2_mean_ppid[error_type], color="orange", alpha=0.3, linewidth=1)
        ax.plot(e3_mean_ppid["integration_length"], e3_mean_ppid[error_type], color="green", alpha=0.3, linewidth=1)

        ax.errorbar(e1_mean["integration_length"], e1_mean[error_type], yerr=e1_sem[error_type], capsize=10, color="blue", markersize="10")
        ax.errorbar(e2_mean["integration_length"], e2_mean[error_type], yerr=e2_sem[error_type], capsize=10, color="orange", markersize="10")
        ax.errorbar(e3_mean["integration_length"], e3_mean[error_type], yerr=e3_sem[error_type], capsize=10, color="green", markersize="10")

    else:
        results = results[(results.experiment == 'basic_settings_non_cued_RLincrem_button_tap') |
                          (results.experiment == 'basic_settings_RLimcrem_analogue')]
        mean_error = results.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].mean().reset_index()
        sem_error =  results.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].sem().reset_index()

        mean_error_ppid = results.groupby(['ppid', 'experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].mean().reset_index()
        sem_error_ppid =  results.groupby(['ppid', 'experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].sem().reset_index()


        e1_mean = mean_error[(mean_error['movement_mechanism'] == "analogue")]
        e1_sem = sem_error[(sem_error['movement_mechanism'] == "analogue")]
        e2_mean = mean_error[(mean_error['movement_mechanism'] == "button_tap")]
        e2_sem = sem_error[(sem_error['movement_mechanism'] == "button_tap")]

        e1_mean_ppid = mean_error_ppid[(mean_error_ppid['movement_mechanism'] == "analogue")]
        e1_sem_ppid = sem_error_ppid[(sem_error_ppid['movement_mechanism'] == "analogue")]
        e2_mean_ppid = mean_error_ppid[(mean_error_ppid['movement_mechanism'] == "button_tap")]
        e2_sem_ppid = sem_error_ppid[(sem_error_ppid['movement_mechanism'] == "button_tap")]

        ax.plot(e1_mean_ppid["integration_length"], e1_mean_ppid[error_type], color="blue", alpha=0.3, linewidth=1)
        ax.plot(e2_mean_ppid["integration_length"], e2_mean_ppid[error_type], color="orange", alpha=0.3, linewidth=1)

        ax.errorbar(e1_mean["integration_length"], e1_mean[error_type], yerr=e1_sem[error_type], capsize=10, color="blue", markersize="10")
        ax.errorbar(e2_mean["integration_length"], e2_mean[error_type], yerr=e2_sem[error_type], capsize=10, color="orange", markersize="10")


    plt.ylabel(yaxis_mean_error_title(error_type), fontsize=20, labelpad=10)
    plt.xlabel('Target (VU)', fontsize=20, labelpad=10)
    plt.title(trial_type_title(trial_type), fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlim(0, max(mean_error['integration_length'])+20)

    plt.hlines(0, 0, max(mean_error['integration_length'])+20, colors='k', linestyles='dashed')
    plt.ylim(-75, 100)
    #plt.legend(prop={"size":15})
    plt.show()


def plot_every_mean_error_pooled(recording_folder_path, error_type, researcher):
    results = pd.read_pickle(recording_folder_path+"\processed_results.pkl")
    results = results.dropna()
    fig, ax = plt.subplots()
    results.dropna()

    if researcher == "Emre":
        results = results[(results.experiment == 'basic_settings_cued_RLincrem_ana_1s_0') |
                          (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_05') |
                          (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_2')]

        mean_error = results.groupby(['Trial type', 'integration_length'])[error_type].mean().reset_index()
        sem_error =  results.groupby(['Trial type', 'integration_length'])[error_type].sem().reset_index()

        mean_error_ppid = results.groupby(['ppid', 'Trial type', 'integration_length'])[error_type].mean().reset_index()
        sem_error_ppid =  results.groupby(['ppid', 'Trial type', 'integration_length'])[error_type].sem().reset_index()


        eb_mean = mean_error[(mean_error["Trial type"] == "beaconed")]
        eb_sem = sem_error[(sem_error["Trial type"] == "beaconed")]
        enb_mean = mean_error[(mean_error["Trial type"] == "non_beaconed")]
        enb_sem = sem_error[(sem_error["Trial type"] == "non_beaconed")]

        eb_mean_ppid = mean_error_ppid[(mean_error_ppid["Trial type"] == "beaconed")]
        eb_sem_ppid = sem_error_ppid[(sem_error_ppid["Trial type"] == "beaconed")]
        enb_mean_ppid = mean_error_ppid[(mean_error_ppid["Trial type"] == "non_beaconed")]
        enb_sem_ppid = sem_error_ppid[(sem_error_ppid["Trial type"] == "non_beaconed")]

        ax.plot(eb_mean_ppid["integration_length"], eb_mean_ppid[error_type], color="blue", alpha=0.3, linewidth=1)
        ax.plot(enb_mean_ppid["integration_length"], enb_mean_ppid[error_type], color="red", alpha=0.3, linewidth=1)

        ax.errorbar(eb_mean["integration_length"], eb_mean[error_type], yerr=eb_sem[error_type], capsize=10, color="blue", markersize="10", marker="s")
        ax.errorbar(enb_mean["integration_length"], enb_mean[error_type], yerr=enb_sem[error_type], capsize=10, color="red", markersize="10", marker="^")

    else:
        results = results[(results.experiment == 'basic_settings_non_cued_RLincrem_button_tap') |
                          (results.experiment == 'basic_settings_RLimcrem_analogue')]

        mean_error = results.groupby(['Trial type', 'integration_length'])[error_type].mean().reset_index()
        sem_error =  results.groupby(['Trial type', 'integration_length'])[error_type].sem().reset_index()

        mean_error_ppid = results.groupby(['ppid', 'Trial type', 'integration_length'])[error_type].mean().reset_index()
        sem_error_ppid =  results.groupby(['ppid', 'Trial type', 'integration_length'])[error_type].sem().reset_index()


        eb_mean = mean_error[(mean_error["Trial type"] == "beaconed")]
        eb_sem = sem_error[(sem_error["Trial type"] == "beaconed")]
        enb_mean = mean_error[(mean_error["Trial type"] == "non_beaconed")]
        enb_sem = sem_error[(sem_error["Trial type"] == "non_beaconed")]

        eb_mean_ppid = mean_error_ppid[(mean_error_ppid["Trial type"] == "beaconed")]
        eb_sem_ppid = sem_error_ppid[(sem_error_ppid["Trial type"] == "beaconed")]
        enb_mean_ppid = mean_error_ppid[(mean_error_ppid["Trial type"] == "non_beaconed")]
        enb_sem_ppid = sem_error_ppid[(sem_error_ppid["Trial type"] == "non_beaconed")]

        ax.plot(eb_mean_ppid["integration_length"], eb_mean_ppid[error_type], color="blue", alpha=0.3, linewidth=1)
        ax.plot(enb_mean_ppid["integration_length"], enb_mean_ppid[error_type], color="red", alpha=0.3, linewidth=1)

        ax.errorbar(eb_mean["integration_length"], eb_mean[error_type], yerr=eb_sem[error_type], capsize=10, color="blue", markersize="10", marker="s")
        ax.errorbar(enb_mean["integration_length"], enb_mean[error_type], yerr=enb_sem[error_type], capsize=10, color="red", markersize="10", marker="^")

    plt.ylabel(yaxis_mean_error_title(error_type), fontsize=20, labelpad=10)
    plt.xlabel('Target (VU)', fontsize=20, labelpad=10)
    #plt.title(trial_type_title(trial_type), fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlim(0, max(mean_error['integration_length'])+20)

    plt.hlines(0, 0, max(mean_error['integration_length'])+20, colors='k', linestyles='dashed')
    plt.ylim(-75, 100)
    #plt.legend(prop={"size":15})
    plt.show()


def plot_every_sd_error2(recording_folder_path, error_type, trial_type, researcher):
    results = pd.read_pickle(recording_folder_path+"\processed_results.pkl")
    results = results.dropna()
    fig, ax = plt.subplots()
    results.dropna()
    results = results[(results["Trial type"] == trial_type)]

    if researcher == "Emre":
        results = results[(results.experiment == 'basic_settings_cued_RLincrem_ana_1s_0') |
                          (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_05') |
                          (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_2')]

        sd_error_ppid =  results.groupby(['ppid', 'experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].std().reset_index()
        e1_sd_ppid = sd_error_ppid[(sd_error_ppid['gain_std'] == 0)]
        e2_sd_ppid = sd_error_ppid[(sd_error_ppid['gain_std'] == 0.5)]
        e3_sd_ppid = sd_error_ppid[(sd_error_ppid['gain_std'] == 2)]

        e1_mean_sd_ppid = e1_sd_ppid.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].mean().reset_index()
        e2_mean_sd_ppid = e2_sd_ppid.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].mean().reset_index()
        e3_mean_sd_ppid = e3_sd_ppid.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].mean().reset_index()

        e1_sem_sd_ppid = e1_sd_ppid.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].sem().reset_index()
        e2_sem_sd_ppid = e1_sd_ppid.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].sem().reset_index()
        e3_sem_sd_ppid = e1_sd_ppid.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].sem().reset_index()

        ax.plot(e1_sd_ppid["integration_length"], e1_sd_ppid[error_type], color="blue", alpha=0.3, linewidth=1)
        ax.plot(e2_sd_ppid["integration_length"], e2_sd_ppid[error_type], color="orange", alpha=0.3, linewidth=1)
        ax.plot(e3_sd_ppid["integration_length"], e3_sd_ppid[error_type], color="green", alpha=0.3, linewidth=1)

        ax.errorbar(e1_mean_sd_ppid["integration_length"], e1_mean_sd_ppid[error_type], yerr=e1_sem_sd_ppid[error_type], capsize=10, color="blue", markersize="10")
        ax.errorbar(e2_mean_sd_ppid["integration_length"], e2_mean_sd_ppid[error_type], yerr=e2_sem_sd_ppid[error_type], capsize=10, color="orange", markersize="10")
        ax.errorbar(e3_mean_sd_ppid["integration_length"], e3_mean_sd_ppid[error_type], yerr=e3_sem_sd_ppid[error_type], capsize=10, color="green", markersize="10")

    elif researcher == "Maja":

        results = results[(results.experiment == 'basic_settings_non_cued_RLincrem_button_tap') |
                          (results.experiment == 'basic_settings_RLimcrem_analogue')]

        sd_error_ppid =  results.groupby(['ppid', 'experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].std().reset_index()
        e1_sd_ppid = sd_error_ppid[(sd_error_ppid['movement_mechanism'] == "analogue")]
        e2_sd_ppid = sd_error_ppid[(sd_error_ppid['movement_mechanism'] == "button_tap")]

        e1_mean_sd_ppid = e1_sd_ppid.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].mean().reset_index()
        e2_mean_sd_ppid = e2_sd_ppid.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].mean().reset_index()

        e1_sem_sd_ppid = e1_sd_ppid.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].sem().reset_index()
        e2_sem_sd_ppid = e1_sd_ppid.groupby(['experiment', 'Trial type', 'session_num', 'gain_std', 'integration_length', 'movement_mechanism'])[error_type].sem().reset_index()

        ax.plot(e1_sd_ppid["integration_length"], e1_sd_ppid[error_type], color="blue", alpha=0.3, linewidth=1)
        ax.plot(e2_sd_ppid["integration_length"], e2_sd_ppid[error_type], color="orange", alpha=0.3, linewidth=1)

        ax.errorbar(e1_mean_sd_ppid["integration_length"], e1_mean_sd_ppid[error_type], yerr=e1_sem_sd_ppid[error_type], capsize=10, color="blue", markersize="10")
        ax.errorbar(e2_mean_sd_ppid["integration_length"], e2_mean_sd_ppid[error_type], yerr=e2_sem_sd_ppid[error_type], capsize=10, color="orange", markersize="10")

    plt.ylabel(yaxis_variance_error_title(error_type), fontsize=20, labelpad=10)
    plt.xlabel('Target (VU)', fontsize=20, labelpad=10)
    plt.title(trial_type_title(trial_type), fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.xlim(0, max(mean_error['integration_length']))
    plt.ylim(0, 100)
    #plt.legend(prop={"size":15})
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

    colours = ['blue', 'orange', 'green']
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

    plt.ylabel(yaxis_variance_error_title(error_type), fontsize=20, labelpad=10)
    plt.xlabel('Target (VU)', fontsize=20, labelpad=10)
    plt.title(trial_type_title(trial_type), fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.xlim(0, max(mean_error['integration_length']))
    plt.ylim(0, 100)
    #plt.legend()
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
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
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

    print("trials = ", len(results))
    print("participants = ", len(np.unique(results["ppid"])))
    # specify one trial type

    results = results[(results['Trial type'] == trial_type)]

    stops_on_track = plt.figure(figsize=(6, 6))
    fig, ax = plt.subplots()
    distances = np.arange(0,400)
    reward_zone_size = 13.801
    ax.fill_between(distances, distances-reward_zone_size/2, distances+reward_zone_size/2, facecolor="k", alpha=0.3)

    if researcher == "Emre":

        mean_correct = results.groupby(["gain_std", 'integration_length'])['first_stop_location_relative_to_ip'].mean().reset_index()
        sem_correct = results.groupby(["gain_std", 'integration_length'])['first_stop_location_relative_to_ip'].sem().reset_index()

        e1_mean = mean_correct[(mean_correct['gain_std'] == 0)]
        e1_sem = sem_correct[(sem_correct['gain_std'] == 0)]
        e2_mean = mean_correct[(mean_correct['gain_std'] == 0.5)]
        e2_sem = sem_correct[(sem_correct['gain_std'] == 0.5)]
        e3_mean = mean_correct[(mean_correct['gain_std'] == 2)]
        e3_sem = sem_correct[(sem_correct['gain_std'] == 2)]

        mean_correct_ppid = results.groupby(['ppid', 'integration_length', "gain_std"])['first_stop_location_relative_to_ip'].mean().reset_index()
        sem_correct_ppid = results.groupby(['ppid', 'integration_length', "gain_std"])['first_stop_location_relative_to_ip'].sem().reset_index()
        e1_mean_ppid = mean_correct_ppid[(mean_correct_ppid['gain_std'] == 0)]
        e1_sem_ppid = sem_correct_ppid[(sem_correct_ppid['gain_std'] == 0)]
        e2_mean_ppid = mean_correct_ppid[(mean_correct_ppid['gain_std'] == 0.5)]
        e2_sem_ppid = sem_correct_ppid[(sem_correct_ppid['gain_std'] == 0.5)]
        e3_mean_ppid = mean_correct_ppid[(mean_correct_ppid['gain_std'] == 2)]
        e3_sem_ppid = sem_correct_ppid[(sem_correct_ppid['gain_std'] == 2)]

        ax.plot(e1_mean_ppid["integration_length"], e1_mean_ppid["first_stop_location_relative_to_ip"], color="blue", alpha=0.3, linewidth=1)
        ax.plot(e2_mean_ppid["integration_length"], e2_mean_ppid["first_stop_location_relative_to_ip"], color="orange", alpha=0.3, linewidth=1)
        ax.plot(e3_mean_ppid["integration_length"], e3_mean_ppid["first_stop_location_relative_to_ip"], color="green", alpha=0.3, linewidth=1)

        ax.errorbar(e1_mean["integration_length"], e1_mean["first_stop_location_relative_to_ip"], yerr=e1_sem["first_stop_location_relative_to_ip"], capsize=10, color="blue", markersize="10")
        ax.errorbar(e2_mean["integration_length"], e2_mean["first_stop_location_relative_to_ip"], yerr=e2_sem["first_stop_location_relative_to_ip"], capsize=10, color="orange", markersize="10")
        ax.errorbar(e3_mean["integration_length"], e3_mean["first_stop_location_relative_to_ip"], yerr=e3_sem["first_stop_location_relative_to_ip"], capsize=10, color="green", markersize="10")

        '''       
        ax.scatter(results[(results.gain_std == 0.5)]["target"]-5, results[(results.gain_std == 0.5)]["first_stop_location_relative_to_ip"], color="orange")
        ax.scatter(results[(results.gain_std == 0)]["target"], results[(results.gain_std == 0)]["first_stop_location_relative_to_ip"], color="blue")
        ax.scatter(results[(results.gain_std == 2)]["target"]+5, results[(results.gain_std == 2)]["first_stop_location_relative_to_ip"], color="green")
        '''

        x = results[(results.gain_std == 0)]["target"].values
        y = results[(results.gain_std == 0)]["first_stop_location_relative_to_ip"].values
        x = x[~np.isnan(y)].reshape(-1, 1)
        y = y[~np.isnan(y)].reshape(-1, 1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(x,y)  # perform linear regression
        Y_pred = linear_regressor.predict(distances.reshape(-1, 1))  # make predictions
        plt.plot(distances, Y_pred, color='blue', linestyle=":", label="No uncertainty linear regression")

        x = results[(results.gain_std == 0.5)]["target"].values
        y = results[(results.gain_std == 0.5)]["first_stop_location_relative_to_ip"].values
        x = x[~np.isnan(y)].reshape(-1, 1)
        y = y[~np.isnan(y)].reshape(-1, 1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(x,y)  # perform linear regression
        Y_pred = linear_regressor.predict(distances.reshape(-1, 1))  # make predictions
        plt.plot(distances, Y_pred, color='orange', linestyle=":", label="Low uncertainty linear regression")

        x = results[(results.gain_std == 2)]["target"].values
        y = results[(results.gain_std == 2)]["first_stop_location_relative_to_ip"].values
        x = x[~np.isnan(y)].reshape(-1, 1)
        y = y[~np.isnan(y)].reshape(-1, 1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(x,y)  # perform linear regression
        Y_pred = linear_regressor.predict(distances.reshape(-1, 1))  # make predictions
        plt.plot(distances, Y_pred, color='green', linestyle=":", label="High uncertainty linear regression")








    elif researcher == "Maja":

        mean_correct = results.groupby(["movement_mechanism", 'integration_length'])['first_stop_location_relative_to_ip'].mean().reset_index()
        sem_correct = results.groupby(["movement_mechanism", 'integration_length'])['first_stop_location_relative_to_ip'].sem().reset_index()

        e1_mean = mean_correct[(mean_correct['movement_mechanism'] == "analogue")]
        e1_sem = sem_correct[(sem_correct['movement_mechanism'] == "analogue")]
        e2_mean = mean_correct[(mean_correct['movement_mechanism'] == "button_tap")]
        e2_sem = sem_correct[(sem_correct['movement_mechanism'] == "button_tap")]

        mean_correct_ppid = results.groupby(['ppid', 'integration_length', "movement_mechanism"])['first_stop_location_relative_to_ip'].mean().reset_index()
        sem_correct_ppid = results.groupby(['ppid', 'integration_length', "movement_mechanism"])['first_stop_location_relative_to_ip'].sem().reset_index()
        e1_mean_ppid = mean_correct_ppid[(mean_correct_ppid['movement_mechanism'] == "analogue")]
        e1_sem_ppid = sem_correct_ppid[(sem_correct_ppid['movement_mechanism'] == "analogue")]
        e2_mean_ppid = mean_correct_ppid[(mean_correct_ppid['movement_mechanism'] == "button_tap")]
        e2_sem_ppid = sem_correct_ppid[(sem_correct_ppid['movement_mechanism'] == "button_tap")]

        ax.plot(e1_mean_ppid["integration_length"], e1_mean_ppid["first_stop_location_relative_to_ip"], color="blue", alpha=0.3, linewidth=1)
        ax.plot(e2_mean_ppid["integration_length"], e2_mean_ppid["first_stop_location_relative_to_ip"], color="orange", alpha=0.3, linewidth=1)

        ax.errorbar(e1_mean["integration_length"], e1_mean["first_stop_location_relative_to_ip"], yerr=e1_sem["first_stop_location_relative_to_ip"], capsize=10, color="blue", markersize="10")
        ax.errorbar(e2_mean["integration_length"], e2_mean["first_stop_location_relative_to_ip"], yerr=e2_sem["first_stop_location_relative_to_ip"], capsize=10, color="orange", markersize="10")

        '''
        x = results[(results.movement_mechanism == "analogue")]["target"].values
        y = results[(results.movement_mechanism == "analogue")]["first_stop_location_relative_to_ip"].values
        x = x[~np.isnan(y)].reshape(-1, 1)
        y = y[~np.isnan(y)].reshape(-1, 1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(x,y)  # perform linear regression
        Y_pred = linear_regressor.predict(distances.reshape(-1, 1))  # make predictions
        plt.plot(distances, Y_pred, color='blue', linestyle=":", label="Joystick linear regression")

        x = results[(results.movement_mechanism == "button_tap")]["target"].values
        y = results[(results.movement_mechanism == "button_tap")]["first_stop_location_relative_to_ip"].values
        x = x[~np.isnan(y)].reshape(-1, 1)
        y = y[~np.isnan(y)].reshape(-1, 1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(x,y)  # perform linear regression
        Y_pred = linear_regressor.predict(distances.reshape(-1, 1))  # make predictions
        plt.plot(distances, Y_pred, color='orange', linestyle=":", label="Button linear regression")
        '''

        '''
        ax.scatter(results[(results.movement_mechanism == "analogue")]["target"]-3, results[(results.movement_mechanism == "analogue")]["first_stop_location_relative_to_ip"], c="blue")
        ax.scatter(results[(results.movement_mechanism == "button_tap")]["target"]+3, results[(results.movement_mechanism == "button_tap")]["first_stop_location_relative_to_ip"], c="orange")

        x = results[(results.movement_mechanism == "button_tap")]["target"].values
        y = results[(results.movement_mechanism == "button_tap")]["first_stop_location_relative_to_ip"].values
        x = x[~np.isnan(y)].reshape(-1, 1)
        y = y[~np.isnan(y)].reshape(-1, 1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(x,y)  # perform linear regression
        Y_pred = linear_regressor.predict(distances.reshape(-1, 1))  # make predictions
        plt.plot(distances, Y_pred, color='orange')

        x = results[(results.movement_mechanism == "analogue")]["target"].values
        y = results[(results.movement_mechanism == "analogue")]["first_stop_location_relative_to_ip"].values
        x = x[~np.isnan(y)].reshape(-1, 1)
        y = y[~np.isnan(y)].reshape(-1, 1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(x,y)  # perform linear regression
        Y_pred = linear_regressor.predict(distances.reshape(-1, 1))  # make predictions
        plt.plot(distances, Y_pred, color='blue')
        '''

    plt.ylabel('Response (VU)', fontsize=20, labelpad=10)
    plt.xlabel('Target (VU)', fontsize=20, labelpad=10)
    #stops_on_track.colorbar(im, ax=ax)
    plt.xlim(0, max(distances))
    plt.ylim(0, max(distances)+100)
    plt.title(trial_type_title(trial_type), fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    #ax.yaxis.set_ticks_position('left')
    #ax.xaxis.set_ticks_position('bottom')
    plt.legend(prop={"size":10})
    plt.savefig(recording_folder_path + '/bias.png', dpi=200)
    plt.show()
    plt.close()


def process_pooled(recording_folder_path, researcher):
    results = pd.read_pickle(recording_folder_path+"\processed_results.pkl")

    if researcher == "Emre":
        results = results[(results.experiment == 'basic_settings_cued_RLincrem_ana_1s_0') |
                          (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_05') |
                          (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_2')]

    elif researcher == "Maja":
        results = results[(results.experiment == 'basic_settings_non_cued_RLincrem_button_tap') |
                          (results.experiment == 'basic_settings_RLimcrem_analogue')]

    print("trials = ", len(results))
    print("participants = ", len(np.unique(results["ppid"])))
    # specify one trial type


    fig, ax = plt.subplots()
    distances = np.arange(0,400)
    reward_zone_size = 13.801
    ax.fill_between(distances, distances-reward_zone_size/2, distances+reward_zone_size/2, facecolor="k", alpha=0.3)

    mean_correct = results.groupby(['Trial type', 'integration_length'])['first_stop_location_relative_to_ip'].mean().reset_index()
    sem_correct = results.groupby(['Trial type', 'integration_length'])['first_stop_location_relative_to_ip'].sem().reset_index()
    e1_b = mean_correct[(mean_correct['Trial type'] == "beaconed")]
    e1_nb = mean_correct[(mean_correct['Trial type'] == "non_beaconed")]
    e1_b_sem = sem_correct[(sem_correct['Trial type'] == "beaconed")]
    e1_nb_sem = sem_correct[(sem_correct['Trial type'] == "non_beaconed")]

    mean_correct_ppid = results.groupby(['ppid', 'Trial type', 'integration_length'])['first_stop_location_relative_to_ip'].mean().reset_index()
    sem_correct_ppid = results.groupby(['ppid', 'Trial type', 'integration_length'])['first_stop_location_relative_to_ip'].sem().reset_index()
    e1_b_ppid = mean_correct_ppid[(mean_correct_ppid['Trial type'] == "beaconed")]
    e1_nb_ppid = mean_correct_ppid[(mean_correct_ppid['Trial type'] == "non_beaconed")]
    e1_b_sem_ppid = sem_correct_ppid[(sem_correct_ppid['Trial type'] == "beaconed")]
    e1_nb_sem_ppid = sem_correct_ppid[(sem_correct_ppid['Trial type'] == "non_beaconed")]

    ax.errorbar(e1_b["integration_length"], e1_b["first_stop_location_relative_to_ip"], yerr=e1_b_sem["first_stop_location_relative_to_ip"], label= "Beaconed", capsize=10, color="blue", marker="s", markersize="10")
    ax.errorbar(e1_nb["integration_length"], e1_nb["first_stop_location_relative_to_ip"], yerr=e1_nb_sem["first_stop_location_relative_to_ip"], label= "Non Beaconed", capsize=10, color="red", marker="^", markersize="10")
    ax.plot(e1_b_ppid["integration_length"], e1_b_ppid["first_stop_location_relative_to_ip"], color="blue", alpha=0.3, linewidth=1)
    ax.plot(e1_nb_ppid["integration_length"], e1_nb_ppid["first_stop_location_relative_to_ip"], color="red", alpha=0.3, linewidth=1)


    #ax.scatter(results["target"], results["first_stop_location_relative_to_ip"], color="orange")
    x = results[(results['Trial type'] == "beaconed")]["target"].values
    y = results[(results['Trial type'] == "beaconed")]["first_stop_location_relative_to_ip"].values
    x = x[~np.isnan(y)].reshape(-1, 1)
    y = y[~np.isnan(y)].reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x,y)  # perform linear regression
    Y_pred = linear_regressor.predict(distances.reshape(-1, 1))  # make predictions
    plt.plot(distances, Y_pred, color='blue', label="Linear Regression Beaconed", linestyle=":")

    x = results[(results['Trial type'] == "non_beaconed")]["target"].values
    y = results[(results['Trial type'] == "non_beaconed")]["first_stop_location_relative_to_ip"].values
    x = x[~np.isnan(y)].reshape(-1, 1)
    y = y[~np.isnan(y)].reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(x,y)  # perform linear regression
    Y_pred = linear_regressor.predict(distances.reshape(-1, 1))  # make predictions
    plt.plot(distances, Y_pred, color='red', label="Linear Regression Non Beaconed", linestyle=":")
    plt.legend(prop={"size":13})


    plt.ylabel('Response (VU)', fontsize=20, labelpad=10)
    plt.xlabel('Target (VU)', fontsize=20, labelpad=10)
    plt.xlim(0, max(distances))
    plt.ylim(0, max(distances)+100)
    #plt.title(trial_type_title(trial_type), fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.legend(prop={"size":15})
    #plt.savefig(recording_folder_path + '/bias.png', dpi=200)
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
    plot_every_sd_error2(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)
    process(results_path, researcher=researcher, trial_type=trial_type)

    trial_type = "beaconed"
    plot_every_mean_error(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)
    plot_every_sd_error2(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)

    plot_every_mean_error_pooled(results_path, error_type="first_stop_error", researcher=researcher)
    process(results_path, researcher=researcher, trial_type=trial_type)

    process_pooled(results_path, researcher=researcher)

    results_path = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Maya"
    researcher = "Maja"
    trial_type = "non_beaconed"
    plot_every_mean_error(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)
    plot_every_sd_error2(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)
    process(results_path, researcher=researcher, trial_type=trial_type)

    trial_type = "beaconed"
    plot_every_mean_error(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)
    plot_every_sd_error2(results_path, error_type="first_stop_error", trial_type=trial_type, researcher=researcher)
    process(results_path, researcher=researcher, trial_type=trial_type)

    plot_every_mean_error_pooled(results_path, error_type="first_stop_error", researcher=researcher)
    process_pooled(results_path, researcher=researcher)



if __name__ == '__main__':
    main()