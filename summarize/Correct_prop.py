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


def process_both(recording_folder_path, recording_folder_path2):
    results = pd.read_pickle(recording_folder_path+"\processed_results.pkl")
    results2 = pd.read_pickle(recording_folder_path2+"\processed_results.pkl")
    results = results.dropna()
    results2 = results2.dropna()

    results = results[(results.experiment == 'basic_settings_cued_RLincrem_ana_1s_0') |
                      (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_05') |
                      (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_2')]
    results2 = results2[(results2.experiment == 'basic_settings_RLimcrem_analogue')]
    results = pd.concat([results, results2], ignore_index=True)

    fig, ax = plt.subplots()
    df = calculate_correct(results)
    df['correct_proportion'] = df['correct_proportion']*100

    mean_correct = df.groupby(['trial_type', 'session', 'distance'])['correct_proportion'].mean().reset_index()
    sem_correct = df.groupby(['trial_type', 'session', 'distance'])['correct_proportion'].sem().reset_index()
    e1_b = mean_correct[(mean_correct.trial_type == "beaconed")]
    e1_nb = mean_correct[(mean_correct.trial_type == "non_beaconed")]
    e1_b_sem = sem_correct[(sem_correct.trial_type == "beaconed")]
    e1_nb_sem = sem_correct[(sem_correct.trial_type == "non_beaconed")]
    ax.errorbar(e1_b["distance"], e1_b["correct_proportion"], yerr=e1_b_sem["correct_proportion"], label= "Beaconed", capsize=10, color="blue", marker="s", markersize="10")
    ax.errorbar(e1_nb["distance"], e1_nb["correct_proportion"], yerr=e1_nb_sem["correct_proportion"], label= "Non Beaconed", capsize=10, color="red", marker="^", markersize="10")

    mean_correct_ppid = df.groupby(['ppid', 'trial_type', 'session', 'distance'])['correct_proportion'].mean().reset_index()
    sem_correct_ppid = df.groupby(['ppid', 'trial_type', 'session', 'distance'])['correct_proportion'].sem().reset_index()
    e1_b_ppid = mean_correct_ppid[(mean_correct_ppid.trial_type == "beaconed")]
    e1_nb_ppid = mean_correct_ppid[(mean_correct_ppid.trial_type == "non_beaconed")]
    e1_b_sem_ppid = sem_correct_ppid[(sem_correct_ppid.trial_type == "beaconed")]
    e1_nb_sem_ppid = sem_correct_ppid[(sem_correct_ppid.trial_type == "non_beaconed")]
    ax.plot(e1_b_ppid["distance"], e1_b_ppid["correct_proportion"], color="blue", alpha=0.3, linewidth=1)
    ax.plot(e1_nb_ppid["distance"], e1_nb_ppid["correct_proportion"], color="red", alpha=0.3, linewidth=1)

    plt.ylabel('Correct Trials %', fontsize=20, labelpad=10)
    plt.xlabel('Distance (VU)', fontsize=20, labelpad=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    #plt.legend()

    plt.xlim(0, max(e1_b_ppid["distance"]))
    plt.ylim(0, 100)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.savefig(results_path + '/bias.png', dpi=200)
    plt.show()
    plt.close()

    print("trials = ", len(results))
    print("participants = ", len(np.unique(results["ppid"])))




def process(recording_folder_path):
    results = pd.read_pickle(recording_folder_path+"\processed_results.pkl")
    results = results.dropna()

    results = results[(results.experiment == 'basic_settings_cued_RLincrem_ana_1s_0') |
                    (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_05') |
                    (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_2')]

    fig, ax = plt.subplots()
    df = calculate_correct(results)
    df['correct_proportion'] = df['correct_proportion']*100

    mean_correct = df.groupby(['trial_type', 'session', 'distance'])['correct_proportion'].mean().reset_index()
    sem_correct = df.groupby(['trial_type', 'session', 'distance'])['correct_proportion'].sem().reset_index()
    e1_b = mean_correct[(mean_correct.trial_type == "beaconed")]
    e1_nb = mean_correct[(mean_correct.trial_type == "non_beaconed")]
    e1_b_sem = sem_correct[(sem_correct.trial_type == "beaconed")]
    e1_nb_sem = sem_correct[(sem_correct.trial_type == "non_beaconed")]
    ax.errorbar(e1_b["distance"], e1_b["correct_proportion"], yerr=e1_b_sem["correct_proportion"], label= "Beaconed", capsize=10, color="blue", marker="s", markersize="10")
    ax.errorbar(e1_nb["distance"], e1_nb["correct_proportion"], yerr=e1_nb_sem["correct_proportion"], label= "Non Beaconed", capsize=10, color="red", marker="^", markersize="10")

    mean_correct_ppid = df.groupby(['ppid', 'trial_type', 'session', 'distance'])['correct_proportion'].mean().reset_index()
    sem_correct_ppid = df.groupby(['ppid', 'trial_type', 'session', 'distance'])['correct_proportion'].sem().reset_index()
    e1_b_ppid = mean_correct_ppid[(mean_correct_ppid.trial_type == "beaconed")]
    e1_nb_ppid = mean_correct_ppid[(mean_correct_ppid.trial_type == "non_beaconed")]
    e1_b_sem_ppid = sem_correct_ppid[(sem_correct_ppid.trial_type == "beaconed")]
    e1_nb_sem_ppid = sem_correct_ppid[(sem_correct_ppid.trial_type == "non_beaconed")]
    ax.plot(e1_b_ppid["distance"], e1_b_ppid["correct_proportion"], color="blue", alpha=0.3, linewidth=1)
    ax.plot(e1_nb_ppid["distance"], e1_nb_ppid["correct_proportion"], color="red", alpha=0.3, linewidth=1)

    plt.ylabel('Correct Trials %', fontsize=20, labelpad=10)
    plt.xlabel('Distance (VU)', fontsize=20, labelpad=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    #plt.legend()

    plt.xlim(0, max(e1_b_ppid["distance"])+10)
    plt.ylim(0, 100)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    #plt.savefig(results_path + '/bias.png', dpi=200)
    plt.show()
    plt.close()

    print("trials = ", len(results))
    print("participants = ", len(np.unique(results["ppid"])))


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # type path name in here with similar structure to this r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    results_path = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    process(results_path)
    results_path2 = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Maya"
    process_both(results_path, results_path2)

if __name__ == '__main__':
    main()