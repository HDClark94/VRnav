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

def process_joystick(recording_folder_path, recording_folder_path2, trial_type='non_beaconed'):

    results1 = pd.read_pickle(recording_folder_path+"\processed_results.pkl")
    results2 = pd.read_pickle(recording_folder_path2+"\processed_results.pkl")

    results1 = results1[(results1.experiment == 'basic_settings_cued_RLincrem_ana_1s_05') |
                      (results1.experiment == 'basic_settings_cued_RLincrem_ana_1s_2')]
    results2 = results2[(results2.experiment == 'basic_settings_RLimcrem_analogue')]

    results = pd.concat([results1, results2], ignore_index=True)
    results = results.dropna()

    print("trials = ", len(results))

    stops_on_track = plt.figure(figsize=(6, 6))
    #ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    for length in np.unique(results["integration_length"]):
        length_results = results[(results["integration_length"] == length) &
                                    (results["Trial type"] == trial_type)]

        fig, ax = plt.subplots()
        ax.scatter(length_results["gain"], length_results["first_stop_error"], marker="o", facecolors='none', edgecolors='k')
        plot_regression(ax, length_results["gain"], length_results["first_stop_error"])
        plt.ylabel('Error (VU)', fontsize=20, labelpad=10)
        plt.xlabel('Gain', fontsize=20, labelpad=10)
        fig.suptitle(trial_type+", Length = "+str(length)+"VU", fontsize=20)
        plt.xlim(0, 2)
        plt.ylim(-150, 150)
        plt.subplots_adjust(top=0.2)
        plt.tick_params(labelsize=20)

    #plt.savefig(recording_folder_path + '/bias_gain.png', dpi=200)
    plt.show()
    #plt.close()

def process_button(recording_folder_path, trial_type='non_beaconed'):

    results = pd.read_pickle(recording_folder_path+"\processed_results.pkl")
    results = results[(results.experiment == 'basic_settings_non_cued_RLincrem_button_tap')]
    results = results.dropna()

    print("trials = ", len(results))

    stops_on_track = plt.figure(figsize=(6, 6))
    #ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    for length in np.unique(results["integration_length"]):
        length_results = results[(results["integration_length"] == length) &
                                 (results["Trial type"] == trial_type)]

        fig, ax = plt.subplots()
        ax.scatter(length_results["gain"], length_results["first_stop_error"], marker="o", facecolors='none', edgecolors='k')
        plot_regression(ax, length_results["gain"], length_results["first_stop_error"])
        plt.ylabel('Error (VU)', fontsize=20, labelpad=10)
        plt.xlabel('Gain', fontsize=20, labelpad=10)
        fig.suptitle(trial_type+", Length = "+str(length)+"VU", fontsize=20)
        plt.xlim(0.5, 1.5)
        plt.ylim(-150, 150)
        plt.subplots_adjust(top=0.2)
        plt.tick_params(labelsize=20)


    #plt.savefig(recording_folder_path + '/bias_gain.png', dpi=200)
    plt.show()
    #plt.close()


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # type path name in here with similar structure to this r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    results_path = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    results_path2 = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Maya"
    process_joystick(results_path, results_path2, trial_type="beaconed")
    process_joystick(results_path, results_path2, trial_type="non_beaconed")

    process_button(results_path2, trial_type="beaconed")
    process_button(results_path2, trial_type="non_beaconed")

if __name__ == '__main__':
    main()