import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from summarize.plotting import *
from summarize.fig1C_function import *
from summarize.common import *
import sys
import traceback

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
                        print("successful with session, ", session)
                    except Exception as ex:
                        print('This is what Python says happened:')
                        print(ex)
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        traceback.print_tb(exc_traceback)


def plot_summary(session_path):
    '''
    This function creates a summary plot for the session
    :param session: path of session directory
    :return:
    '''

    trial_results = pd.read_csv(session_path+"/trial_results.csv")
    trial_results = extract_summary(trial_results,session_path)

    plot_stops_on_track(trial_results, session_path)
    error_longer_tracks(trial_results, session_path, error_collumn="first_stop_error")
    error_longer_tracks(trial_results, session_path, error_collumn="absolute_first_stop_error")
    error_longer_tracks(trial_results, session_path, error_collumn="absolute_first_stop_post_cue_error")
    variance_longer_tracks(trial_results, session_path, error_collumn="first_stop_error")
    variance_longer_tracks(trial_results, session_path, error_collumn="absolute_first_stop_error")
    variance_longer_tracks(trial_results, session_path, error_collumn="absolute_first_stop_post_cue_error")
    plot_target_response(trial_results, session_path)
    #stop_histogram(trial_results, session_path, cummulative=True, first_stop=True)
    #stop_histogram(trial_results, session_path, cummulative=True, first_stop=False)
    #stop_histogram(trial_results, session_path, cummulative=False, first_stop=True)
    #stop_histogram(trial_results, session_path, cummulative=False, first_stop=False)
    #plot_stops_in_time(trial_results,session_path)

#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    recording_folder_path = r"Z:\ActiveProjects\Harry\OculusVR\TrenchRunV2.0\recordings"
    update_summary_plots(recording_folder_path, override=True)

if __name__ == '__main__':
    main()



