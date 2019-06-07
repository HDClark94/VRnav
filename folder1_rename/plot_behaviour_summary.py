import os
import matplotlib.pyplot as plot
import pandas as pd


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
        for particpant in participant_dir:
            session_dir = [f.path for f in os.scandir(particpant) if f.is_dir()]
            #loop over sessions
            for session in session_dir:
                #trial_results = pd.read_csv(session+"/trial_results.csv")

                if 'summary_plot.png' not in os.listdir(session) or override==True:
                    plot_summary(session)


def plot_summary(session):
    '''
    This function creates a summary plot for the session
    :param session: path of session directory
    :return:
    '''

    trial_results = pd.read_csv(session+"/trial_results.csv")






#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')


    recording_folder_path = '/home/harry/local_ard/Harry/Oculus VR/test_recordings'

    update_summary_plots(recording_folder_path)


if __name__ == '__main__':
    main()



