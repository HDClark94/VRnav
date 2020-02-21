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


def calculate_correct(results):
    df = pd.DataFrame()

    unique_experiments = np.unique(results['experiment'])
    for experiment in unique_experiments:
        experiment_results = results[results['experiment'] == experiment]
        unique_participants = np.unique(results['ppid'])
        for id in unique_participants:
            participant_results = experiment_results[experiment_results['ppid'] == id]
            unique_sessions = np.unique(participant_results['session_num'])
            for session in unique_sessions:
                session_results = participant_results[participant_results['session_num'] == session]
                unique_trial_types = np.unique(session_results["Trial type"])
                for trial_type in unique_trial_types:
                    trial_type_results = session_results[session_results['Trial type'] == trial_type]
                    unique_movement_mechs = np.unique(trial_type_results["movement_mechanism"])
                    for movement_mech in unique_movement_mechs:
                        movement_mech_results = trial_type_results[trial_type_results["movement_mechanism"] == movement_mech]
                        unique_track_lengths = np.unique(movement_mech_results["integration_length"])
                        for track_length in unique_track_lengths:
                            track_length_results = movement_mech_results[movement_mech_results["integration_length"] == track_length]

                            df = df.append(pd.DataFrame([{'experiment': experiment,
                                                         'ppid': id,
                                                         'session': int(session),
                                                         'trial_type': trial_type,
                                                         'gain sd': track_length_results["gain_std"].iloc[0], # bodgy but gain sd is always the same within a session
                                                         'distance': track_length,
                                                         'n_trial': len(track_length_results),
                                                         'n_correct': sum(track_length_results["Trial Scored"]),
                                                         'correct_proportion': sum(track_length_results["Trial Scored"])/len(track_length_results)}]))

    return df

def process(results_path, researcher):
    results = pd.read_pickle(results_path)
    results = results.dropna()

    if researcher == "Emre":
        results = results[(results.experiment == 'basic_settings_cued_RLincrem_ana_1s_0') |
                         (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_05') |
                         (results.experiment == 'basic_settings_cued_RLincrem_ana_1s_2')]

    elif researcher == "Maja":
        results = results[(results.experiment == 'basic_settings_non_cued_RLincrem_button_tap') |
                          (results.experiment == 'basic_settings_RLimcrem_analogue')]

    fig, ax = plt.subplots()

    df = calculate_correct(results)

    mean_correct = df.groupby(['experiment', 'trial_type', 'session', 'gain sd', 'distance'])['correct_proportion'].mean().reset_index()
    std_correct = df.groupby(['experiment', 'trial_type', 'session', 'gain sd', 'distance'])['correct_proportion'].std().reset_index()

    if researcher == "Emre":
        e1_b = mean_correct[(std_correct.experiment == "basic_settings_cued_RLincrem_ana_1s_0") & (std_correct.trial_type == "beaconed")]
        e1_nb = mean_correct[(std_correct.experiment == "basic_settings_cued_RLincrem_ana_1s_0") & (std_correct.trial_type == "non_beaconed")]
        e2_b = mean_correct[(std_correct.experiment == "basic_settings_cued_RLincrem_ana_1s_05") & (std_correct.trial_type == "beaconed")]
        e2_nb = mean_correct[(std_correct.experiment == "basic_settings_cued_RLincrem_ana_1s_05") & (std_correct.trial_type == "non_beaconed")]
        e3_b = mean_correct[(std_correct.experiment == "basic_settings_cued_RLincrem_ana_1s_2") & (std_correct.trial_type == "beaconed")]
        e3_nb = mean_correct[(std_correct.experiment == "basic_settings_cued_RLincrem_ana_1s_2") & (std_correct.trial_type == "non_beaconed")]

        e1_b_sd = std_correct[(std_correct.experiment == "basic_settings_cued_RLincrem_ana_1s_0") & (std_correct.trial_type == "beaconed")]
        e1_nb_sd = std_correct[(std_correct.experiment == "basic_settings_cued_RLincrem_ana_1s_0") & (std_correct.trial_type == "non_beaconed")]
        e2_b_sd = std_correct[(std_correct.experiment == "basic_settings_cued_RLincrem_ana_1s_05") & (std_correct.trial_type == "beaconed")]
        e2_nb_sd = std_correct[(std_correct.experiment == "basic_settings_cued_RLincrem_ana_1s_05") & (std_correct.trial_type == "non_beaconed")]
        e3_b_sd = std_correct[(std_correct.experiment == "basic_settings_cued_RLincrem_ana_1s_2") & (std_correct.trial_type == "beaconed")]
        e3_nb_sd = std_correct[(std_correct.experiment == "basic_settings_cued_RLincrem_ana_1s_2") & (std_correct.trial_type == "non_beaconed")]

        ax.errorbar(e1_b["distance"], e1_b["correct_proportion"], yerr=e1_b_sd["correct_proportion"], label= "gain = 0.0, beaconed", capsize=10)
        ax.errorbar(e2_b["distance"], e2_b["correct_proportion"], yerr=e2_b_sd["correct_proportion"], label= "gain = 0.5, beaconed", capsize=10)
        ax.errorbar(e3_b["distance"], e3_b["correct_proportion"], yerr=e3_b_sd["correct_proportion"], label= "gain = 2.0, beaconed", capsize=10)
        ax.errorbar(e1_nb["distance"], e1_nb["correct_proportion"], yerr=e1_nb_sd["correct_proportion"], label= "gain = 0.0, non beaconed", capsize=10)
        ax.errorbar(e2_nb["distance"], e2_nb["correct_proportion"], yerr=e2_nb_sd["correct_proportion"], label= "gain = 0.5, non beaconed", capsize=10)
        ax.errorbar(e3_nb["distance"], e3_nb["correct_proportion"], yerr=e3_nb_sd["correct_proportion"], label= "gain = 2.0, non beaconed", capsize=10)

    elif researcher == "Maja":

        e1_b = mean_correct[(std_correct.experiment == "basic_settings_non_cued_RLincrem_button_tap") & (std_correct.trial_type == "beaconed")]
        e1_nb = mean_correct[(std_correct.experiment == "basic_settings_non_cued_RLincrem_button_tap") & (std_correct.trial_type == "non_beaconed")]
        e2_b = mean_correct[(std_correct.experiment == "basic_settings_RLimcrem_analogue") & (std_correct.trial_type == "beaconed")]
        e2_nb = mean_correct[(std_correct.experiment == "basic_settings_RLimcrem_analogue") & (std_correct.trial_type == "non_beaconed")]

        e1_b_sd = std_correct[(std_correct.experiment == "basic_settings_non_cued_RLincrem_button_tap") & (std_correct.trial_type == "beaconed")]
        e1_nb_sd = std_correct[(std_correct.experiment == "basic_settings_non_cued_RLincrem_button_tap") & (std_correct.trial_type == "non_beaconed")]
        e2_b_sd = std_correct[(std_correct.experiment == "basic_settings_RLimcrem_analogue") & (std_correct.trial_type == "beaconed")]
        e2_nb_sd = std_correct[(std_correct.experiment == "basic_settings_RLimcrem_analogue") & (std_correct.trial_type == "non_beaconed")]

        ax.errorbar(e1_b["distance"], e1_b["correct_proportion"], yerr=e1_b_sd["correct_proportion"], label= "Button, beaconed", capsize=10)
        ax.errorbar(e2_b["distance"], e2_b["correct_proportion"], yerr=e2_b_sd["correct_proportion"], label= "Joystick, beaconed", capsize=10)
        ax.errorbar(e1_nb["distance"], e1_nb["correct_proportion"], yerr=e1_nb_sd["correct_proportion"], label= "Button, non beaconed", capsize=10)
        ax.errorbar(e2_nb["distance"], e2_nb["correct_proportion"], yerr=e2_nb_sd["correct_proportion"], label= "Joystick, non beaconed", capsize=10)

    plt.ylabel('Correct prob', fontsize=12, labelpad=10)
    plt.xlabel('Distance (VU)', fontsize=12, labelpad=10)
    plt.legend()

    #plt.xlim(0, max(distances))
    plt.ylim(0, 1)

    #ax.yaxis.set_ticks_position('left')
    #ax.xaxis.set_ticks_position('bottom')
    #plt.savefig(results_path + '/bias.png', dpi=200)
    plt.show()
    plt.close()


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # type path name in here with similar structure to this r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    results_path = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre\processed_results.pkl"
    process(results_path, "Emre")
    results_path = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Maya\processed_results.pkl"
    process(results_path, "Maja")

if __name__ == '__main__':
    main()