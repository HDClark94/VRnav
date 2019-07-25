import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from summarize.plotting import *
from summarize.fig1C_function import *

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
                        
                    except KeyError:
                        print('There was mostly a key error, nothing to worry about, not all files have the same collumn titles in trial results')


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
    #recording_folder_path = '/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Harry/Oculus VR/test_vr_recordings'
    #recording_folder_path =  '/Volumes/cmvm/ActiveProjects/Harry/OculusVR/test_vr_recordings'
    recording_folder_path = '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings'

    #recording_folder_path = '/home/harry/Harry_ard/test_recordings'
    update_summary_plots(recording_folder_path, override=True)


if __name__ == '__main__':
    main()



