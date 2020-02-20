# this script will pull relavent features from each recording and compile them into a large dataframe from which you can analyse in whatever way you like.

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from summarize.plotting import *
import json
from summarize.common import *


'''
This script can be used to concatenate the data across subjects into a long format dataframe that you can analyse in python or in Rs
'''


def concatenate_dataframe(dataframe_paths, save_path):
    concat_dataframe = pd.DataFrame()

    for i in range(len(dataframe_paths)):
        df = pd.read_pickle(dataframe_paths[i])
        concat_dataframe.append(df)

    concat_dataframe.to_pickle(save_path + '/concatenated_results.pkl')


def process(recording_folder_path):
    # makes a pandas dataframe and saves it in long form for downstream analysis
    results = pd.DataFrame()

    setting_dir = [f.path for f in os.scandir(recording_folder_path) if f.is_dir()]

    # loop over settings
    for setting in setting_dir:
        participant_dir = [f.path for f in os.scandir(setting) if f.is_dir()]
        # loop over participants
        for participant in participant_dir:
            session_dir = [f.path for f in os.scandir(participant) if f.is_dir()]
            # loop over sessions
            for session_path in session_dir:
                participant_id = participant.split("\\")[-1]
                session_number = session_path.split("\\")[-1]

                trial_results = pd.read_csv(session_path+"/trial_results.csv")

                trial_results = extract_summary(trial_results,session_path)

                with open(session_path+'/settings.json') as f:
                    session_config = json.load(f)

                # add from trial results
                df1 = trial_results[['experiment','ppid', 'session_num', 'trial_num','block_num', 'trial_num_in_block', 'trial_duration',
                                     'Trial type', 'Trial Scored', 'Cummulative Score', 'Cue Boundary Min', 'Cue Boundary Max', 'Transparency',
                                     'Reward Boundary Min', 'Reward Boundary Max', 'Track Start', 'Track End', 'Cue Sight', 'Rewarded Location',
                                     'Teleport from', 'Teleport to', 'Acceleration', 'first_stop_location', 'first_stop_location_post_cue',
                                     'first_stop_location_relative_to_ip', 'integration_length', 'first_stop_error', 'absolute_first_stop_error',
                                     'first_stop_post_cue_error', 'absolute_first_stop_post_cue_error']]

                # add from config file
                df1["move_cue"] = np.repeat(session_config["move_cue"], len(trial_results))
                df1["variable_cue_contrast"] = np.repeat(session_config["variable_cue_contrast"], len(trial_results))
                df1["variable_cue_sight"]  = np.repeat(session_config["variable_cue_sight"], len(trial_results))
                df1["Probe_probability"]  = np.repeat(session_config["Probe_probability"], len(trial_results))
                df1["Non_beaconed_probability"]  = np.repeat(session_config["Non_beaconed_probability"], len(trial_results))
                df1["track_length_difficulty"]  = np.repeat(session_config["track_length_difficulty"], len(trial_results))
                df1["task"]  = np.repeat(session_config["task"], len(trial_results))
                df1["probe_criteria"]  = np.repeat(session_config["probe_criteria"], len(trial_results))
                df1["gain_std"]  = np.repeat(session_config["gain_std"], len(trial_results))
                df1["show_correct"]  = np.repeat(session_config["show_correct"], len(trial_results))
                df1["movement_mechanism"]  = np.repeat(session_config["movement_mechanism"], len(trial_results))
                df1["n_scorable_stops"]  = np.repeat(session_config["n_scorable_stops"], len(trial_results))
                df1["n_trials_per_track_length"]  = np.repeat(session_config["n_trials_per_track_length"], len(trial_results))

                # append to multi-subject dataframe
                results = results.append(df1)

    # now save dataframe somewhere useful
    results.to_pickle(recording_folder_path + '/processed_results.pkl')




def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # type path name in here with similar structure to this r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    recording_folder_path = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Emre"
    process(recording_folder_path)
    recording_folder_path = r"Z:\ActiveProjects\Harry\OculusVR\vr_recordings_Maya"
    process(recording_folder_path)

if __name__ == '__main__':
    main()