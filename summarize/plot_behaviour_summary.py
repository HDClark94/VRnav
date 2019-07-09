import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from summarize.plotting import *

def update_summary_plots(recording_folder_path, override=False):
    '''
    This functions looks into the recording folder and creates summary plots for if no plots are found or updates plots that are present
    :param recording_folder_path: this should be the directory path before the setting directories eg. a level up from basic_settings_experiment_1
    :param override: when true, all summary plots are updated in the folder regardless
    :return:
    '''

    setting_dir = [f.path for f in os.scandir(recording_folder_path) if f.is_dir()]

    # loop over settings
    for setting in setting_dir:
        participant_dir = [f.path for f in os.scandir(setting) if f.is_dir()]
        # loop over partipants
        for participant in participant_dir:
            session_dir = [f.path for f in os.scandir(participant) if f.is_dir()]
            #loop over sessions
            for session in session_dir:

                if 'summary_plot.png' not in os.listdir(session) or override==True:
                    try:
                        plot_summary(session)
                        
                    except IndexError:
                        print('Error')


def split_stop_data_by_trial_type(session_dataframe):
    '''
    :param session_dataframe:
    :return: dataframe with selected trial types
    '''

    beaconed = session_dataframe[session_dataframe['Trial type'] == "beaconed"]
    non_beaconed = session_dataframe[session_dataframe['Trial type'] == "non_beaconed"]
    probe = session_dataframe[session_dataframe['Trial type'] == "probe"]
    return beaconed, non_beaconed, probe

def split_stop_data_by_block(session_dataframe, block):
    return session_dataframe[session_dataframe["block_num"] == block]


def correct_time(trial_dataframe):
    # time always starts at zero each trial
    trial_dataframe["time"] = trial_dataframe["time"]-trial_dataframe["time"][0]
    return trial_dataframe

def extract_speeds_same_length(trial_results, session_path):
    speeds = []

    for index, _ in trial_results.iterrows():
        player_movement = pd.read_csv(session_path + "/" + str(trial_results["player_movement_filename"][index]))

        location_diff = np.diff(np.array(player_movement["pos_x"]))
        time_diff = np.diff(np.array(player_movement["time"]))

        speed = location_diff / time_diff  # this is speed in vu per second
        speed = np.concatenate([[speed[0]], speed])
        speeds.append(speed)

    trial_results["speeds"] = speeds

    # returns speeds with same length as movement steps
    return trial_results


def extract_speeds(trial_results, session_path):
    speeds = []

    for index, _ in trial_results.iterrows():
        player_movement = pd.read_csv(session_path + "/" + str(trial_results["player_movement_filename"][index]))

        location_diff = np.diff(np.array(player_movement["pos_x"]))*-1
        time_diff = np.diff(np.array(player_movement["time"]))

        speed = location_diff/time_diff # this is speed in vu per second
        speeds.append(speed)

    trial_results["speeds"] = speeds

    return trial_results

def extract_speed_by_spatial_bins(trial_results, session_path):
   # TODO
   return trial_results


def extract_speed_by_time_bins(trial_results, session_path):
    #TODO
    return trial_results

def extract_stops(trial_results, session_path):

    stop_locations = []
    stop_times = []

    for index, _ in trial_results.iterrows():
        player_movement = pd.read_csv(session_path+"/"+str(trial_results["player_movement_filename"][index]))
        stop_idx = np.where(np.diff(np.array(player_movement["pos_x"]))==0)

        trial_stop_locations = np.array(player_movement["pos_x"])[stop_idx]
        trial_stop_locations = np.unique(trial_stop_locations)  # ignores duplicates

        player_movement = correct_time(player_movement)
        trial_stop_times = np.array(player_movement["time"])[stop_idx]

        stop_locations.append(trial_stop_locations)
        stop_times.append(trial_stop_times)

    trial_results["stop_locations"] = stop_locations
    trial_results["stop_times"] = stop_times

    return trial_results

def plot_summary(session_path):
    '''
    This function creates a summary plot for the session
    :param session: path of session directory
    :return:

    '''

    trial_results = pd.read_csv(session_path+"/trial_results.csv")
    #trial_results = split_stop_data_by_block(trial_results, block=2)  # only use block 2, this ejects habituation block 1
    trial_results = extract_stops(trial_results, session_path) # add stop times and locations to dataframe
    trial_results = extract_speeds(trial_results, session_path) # adds speeds to dataframe

    plot_stops_on_track(trial_results, session_path)

    #plot_stops_in_time(trial_results,session_path)


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')


    #recording_folder_path = '/home/harry/local_ard/Harry/Oculus VR/test_recordings' # for ardbeg
    recording_folder_path = '/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Harry/Oculus VR/test_vr_recordings'
    #recording_folder_path = '/home/harry/Documents/test_vr_recordings'

    #recording_folder_path = '/home/harry/Harry_ard/test_recordings'
    update_summary_plots(recording_folder_path, override=True)


if __name__ == '__main__':
    main()



