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
                #trial_results = pd.read_csv(session+"/trial_results.csv")

                if 'summary_plot.png' not in os.listdir(session) or override==True:
                    plot_summary(session)


def split_stop_data_by_trial_type(session_dataframe):
    '''
    :param session_dataframe:
    :return: dataframe with selected trial types
    '''

    beaconed = session_dataframe[session_dataframe['Trial type'] == "beaconed"]
    non_beaconed = session_dataframe[session_dataframe['Trial type'] == "non_beaconed"]
    probe = session_dataframe[session_dataframe['Trial type'] == "probe"]
    return beaconed, non_beaconed, probe

def plot_stops_in_time(trial_results, session_path):
    stops_in_time = plt.figure(figsize=(6, 6))
    ax = stops_in_time.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    beaconed, non_beaconed, probe = split_stop_data_by_trial_type(trial_results)


def plot_stops_on_track(trial_results, session_path):
    stops_on_track = plt.figure(figsize=(6, 6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed, non_beaconed, probe = split_stop_data_by_trial_type(trial_results)

    #cue_length =

    for index, _ in beaconed.iterrows():
        b_stops = (np.array(beaconed["stop_locations"][index])*-1)+beaconed["Cue Boundary Max"][index]
        b_trial_num = np.array(beaconed["trial_num"][index])
        b_trials = b_trial_num*np.ones(len(b_stops))

        ax.plot((np.linspace(beaconed["Track Start"][index], beaconed["Track End"][index], 2)*-1)+beaconed["Cue Boundary Max"][index], np.array([b_trial_num, b_trial_num]), color="y") # marks out track area
        ax.plot(b_stops, b_trials, 'o', color='0.5', markersize=2)

    for index, _ in non_beaconed.iterrows():
        nb_stops = (np.array(non_beaconed["stop_locations"][index])*-1)+non_beaconed["Cue Boundary Max"][index]
        nb_trial_num = np.array(non_beaconed["trial_num"][index])
        nb_trials = nb_trial_num * np.ones(len(nb_stops))

        ax.plot((np.linspace(non_beaconed["Track Start"][index], non_beaconed["Track End"][index], 2)*-1)+non_beaconed["Cue Boundary Max"][index], np.array([nb_trial_num,nb_trial_num]), color="y")  # marks out track area
        ax.plot(nb_stops, nb_trials, 'o', color='red', markersize=2)

    for index, _ in probe.iterrows():
        p_stops = (np.array(probe["stop_locations"][index])*-1)+probe["Cue Boundary Max"][index]
        p_trial_num = np.array(probe["trial_num"][index])
        p_trials = p_trial_num * np.ones(len(p_stops))

        ax.plot((np.linspace(probe["Track Start"][index], probe["Track End"][index], 2)*-1)+probe["Cue Boundary Max"][index], np.array([p_trial_num, p_trial_num]), color="y")  # marks out track area
        ax.plot(p_stops, p_trials, 'o', color='blue', markersize=2)

    plt.ylabel('Stops on trials', fontsize=12, labelpad=10)
    plt.xlabel('Location (vu)', fontsize=12, labelpad=10)
    # plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    #plt.xlim(0, 200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    style_track_plot(ax, beaconed)
    style_vr_plot(ax)  # can be any trialtype example

    plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.2, left=0.12, right=0.87, top=0.92)
    plt.savefig(session_path + '/summary_plot.png', dpi=200)
    #plt.savefig('/home/harry/aa/plot_summary.png', dpi=200)   # TODO change this to ardbeg when I have permission to write with Linux
    plt.show()
    plt.close()

def append_calculated_speed():
    pass

def correct_time(trial_dataframe):
    # time always starts at zero each trial
    trial_dataframe["time"] = trial_dataframe["time"]-trial_dataframe["time"][0]
    return trial_dataframe

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
    append_calculated_speed()
    trial_results = extract_stops(trial_results, session_path) # add stop times and locations to dataframe

    plot_stops_on_track(trial_results, session_path)
    #plot_stops_in_time(trial_results,session_path)




    '''
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    raster(behaviour, trialtype_log, ax1)
    # accum_reward(behaviour, ax2)
    speed_of_last(behaviour, trialtype_log, ax3)
    average_ep_reward(behaviour, ax2)
    # actions_of_last(behaviour, actions, ax5)
    value_fn_of_last(behaviour, values, ax4)
    f.tight_layout()
    # plt.show()

    f.savefig(save_path + title)

    f.clf()
    '''









#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')


    #recording_folder_path = '/home/harry/local_ard/Harry/Oculus VR/test_recordings' # for ardbeg
    recording_folder_path = '/home/harry/Harry_ard/test_recordings'

    update_summary_plots(recording_folder_path, override=True)


if __name__ == '__main__':
    main()



