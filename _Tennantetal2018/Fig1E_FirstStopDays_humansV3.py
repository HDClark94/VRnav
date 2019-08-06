# -*- coding: utf-8 -*-
"""

# Calculates the proportion of times an animal stops first in each location bin of the track per training session

For each of the days and mice specified, location along the track is split into 10 cm bins and the proportion of trials in which animals stop in each location bin is calculated. This is then averaged over days for each animal, then over animals. The average proportion is then plotted against track location.

"""

# Import packages and functions
from Functions_Core_0100 import extractstops_HUMAN,filterstops, create_srdata, makebinarray, extractrewards, FirstStops, FirstStops_humans, readhdfdata, adjust_spines, maketrialarray, shuffle_analysis_pertrial3
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform
from summarize.map2legacy import *

# Load raw data: specify the HDF5 file to read data from
#session_path = '/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Harry/Oculus VR/test_vr_recordings/basic_settings_short/P_190717101640/S001'
#saraharray, track_start, track_end = translate_to_legacy_format(session_path)

#filename = '/home/Data_Input/Behaviour_DataFiles/Task13_0300.h5' # raw data files
#filename = '/home/harry/Downloads/Task13_0300.h5'


def main():

    print('-------------------------------------------------------------')

    print('-------------------------------------------------------------')
    '''
    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190729100324/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190726112925/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190726153910/S001']
    plot_fig1E(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1E_S_Gain_AvgFirstStop.png')
    '''
    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190729100324/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190726112925/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190726153910/S001']
    plot_fig1E(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1E_M_Gain_AvgFirstStop.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190729100324/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190726112925/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190726153910/S001']
    plot_fig1E(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1E_L_Gain_AvgFirstStop.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190729100324/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190726112925/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190726153910/S001']
    plot_fig1E(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1E_CM_Gain_AvgFirstStop.png')

    #session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190729100324/S001',
    #                 '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190726112925/S001',
    #                 '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190726153910/S001']
    #plot_fig1E(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1E_H_Gain_FirstStop.png')



def plot_fig1E(session_paths, save_path):


    days = [1]
    # specify mouse/mice and day/s to analyse
     # choose specific day/s
    number_of_trials = 1000 # arbitrarily large amound of trials so we can catch all the trials

    # Arrays for storing data (output)
    firststopstorebeac = np.zeros(((number_of_trials), len(session_paths)))
    firststopstorenbeac = np.zeros(((number_of_trials), len(session_paths)))
    firststopstoreprobe = np.zeros(((number_of_trials), len(session_paths)))
    firststopstorebeac[:,:] = np.nan
    firststopstorenbeac[:,:] = np.nan
    firststopstoreprobe[:,:] = np.nan

    sd_con_FirstStopsstorebeac = np.zeros(((number_of_trials), len(session_paths)))
    sd_con_FirstStopsstorenbeac = np.zeros(((number_of_trials), len(session_paths)))
    sd_con_FirstStopsstoreprobe = np.zeros(((number_of_trials), len(session_paths)))
    sd_con_FirstStopsstorebeac[:, :] = np.nan
    sd_con_FirstStopsstorenbeac[:, :] = np.nan
    sd_con_FirstStopsstoreprobe[:, :] = np.nan

    # For each day and mouse, pull raw data, calculate first stops and store data
    for dcount,day in enumerate(days):
        session_count = 0
        for session_path in session_paths:
            saraharray, track_start, track_end = translate_to_legacy_format(session_path)

            rz_start = saraharray[0, 11]
            rz_end = saraharray[0, 12]

            # make array of trial number for each row in dataset
            trialarray = maketrialarray(saraharray) # write array of trial per row in datafile
            saraharray[:,9] = trialarray[:,0] # replace trial column in dataset *see README for why this is done*

            # split data by trial type
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)

            # get stops
            stops_b = extractstops_HUMAN(dailymouse_b)
            stops_nb = extractstops_HUMAN(dailymouse_nb)
            stops_p= extractstops_HUMAN(dailymouse_p)

            #filterstops
            stops_b = filterstops(stops_b)
            stops_nb = filterstops(stops_nb)
            stops_p= filterstops(stops_p)

            # get first stop for each trial

            trarray = np.unique(trialarray[:, 0])  # get unique trial numbers
            stops_f_b = FirstStops_humans(trarray, stops_b, track_start, track_end)  # get locations of first stop for each trial
            #beac = np.nanmean(stops_f_b, axis=0)  # find average first stop location
            #sdbeac = np.nanstd(stops_f_b, axis=0)  # get sd of first stop location

            if stops_nb.size > 0:
                stops_f_nb = FirstStops_humans(trarray, stops_nb, track_start, track_end)  # get locations of first stop for each trial
                #nbeac = np.nanmean(stops_f_nb, axis=0)  # find average first stop location
                #sdnbeac = np.nanstd(stops_f_nb, axis=0)  # get sd of first stop location
            if stops_p.size > 0:
                stops_f_p = FirstStops_humans(trarray, stops_p, track_start, track_end)  # get locations of first stop for each trial
                #probe = np.nanmean(stops_f_p, axis=0)  # find average first stop location
                #sdprobe = np.nanstd(stops_f_p, axis=0)  # get sd of first stop location

            # store data
            for i in range(len(stops_f_b)):
                firststopstorebeac[i, session_count] = stops_f_b[i][0]  # x 10 to convert virtual units to cm
                #sd_con_FirstStopsstorebeac[dcount, session_count] = beac[0]

            if stops_nb.size > 0:
                for i in range(len(stops_f_nb)):
                    firststopstorenbeac[i, session_count] = stops_f_nb[i][0]  # x 10 to convert virtual units to cm
                    #sd_con_FirstStopsstorenbeac[dcount, session_count] = nbeac[0]

            if stops_p.size > 0:
                for i in range(len(stops_f_p)):
                    firststopstoreprobe[i, session_count] = stops_f_p[i][0]  # x 10 to convert virtual units to cm
                    #sd_con_FirstStopsstoreprobe[dcount, session_count] = probe[0]

            session_count += 1


            # Average over days for all mice

    con_beac = np.nanmean(((firststopstorebeac)), axis=1)
    con_beac = con_beac[~np.isnan(con_beac)]
    con_nbeac = np.nanmean(((firststopstorenbeac)), axis=1)
    con_nbeac = con_nbeac[~np.isnan(con_nbeac)]
    con_probe = np.nanmean(((firststopstoreprobe)), axis=1)
    con_probe = con_probe[~np.isnan(con_probe)]
    sd_con_beac = np.nanstd(((firststopstorebeac)), axis=1) / math.sqrt(session_count)
    sd_con_beac = sd_con_beac[~np.isnan(sd_con_beac)]
    sd_con_nbeac = np.nanstd(((firststopstorenbeac)), axis=1) / math.sqrt(session_count)
    sd_con_nbeac = sd_con_nbeac[~np.isnan(sd_con_nbeac)]
    sd_con_probe = np.nanstd(((firststopstoreprobe)), axis=1) / math.sqrt(session_count)
    sd_con_probe = sd_con_probe[~np.isnan(sd_con_probe)]

    # PLOT GRAPHS

    b_trial_max = np.arange(0,len(con_beac),1)
    nb_trial_max = np.arange(0,len(con_nbeac),1)
    p_trial_max = np.arange(0,len(sd_con_probe),1)

    # array of days
    #x = con_beac[0]

    # plot average first stop over days for all mice
    fig = plt.figure(figsize = (12,5))  # make figure, this shape (width, height)
    ax = fig.add_subplot(1,3,1)
    ax.set_title('Beaconed', fontsize=20, verticalalignment='bottom', style='italic') #title
    ax.axhspan(rz_start, rz_end, facecolor='g', alpha=0.15, hatch='/',linewidth=0)  # green box of reward zone
    ax.axhspan(0, 4, facecolor='k', alpha=0.15, hatch='/', linewidth=0)  # black box (4 is used for appearance)
    ax.axhline(4, linewidth=1, ls='--', color='black')  # mark black box border
    ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
    ax.axhline(2, linewidth=3, color='black')  # bold line on the x axis

    ax.plot(b_trial_max, con_beac, 'o',color = '0.3', label = 'Non reward zone score', linewidth = 2, markersize = 6, markeredgecolor = 'black')
    ax.errorbar(b_trial_max,con_beac,sd_con_beac, fmt = 'o', color = '0.3', capsize = 1.5, markersize = 2, elinewidth = 1.5)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =15) # tick parameters: pad is tick label relative to
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =15)
    ax.set_xlim(0,len(b_trial_max))
    ax.set_ylim(2,rz_end+2)
    adjust_spines(ax, ['left','bottom']) # remove right and top axis
    plt.locator_params(nbins=7, axis ='y') # set tick number on y axis
    plt.locator_params(nbins=3, axis ='x') # set tick number on x axis
    ax = plt.ylabel('Location (VU)', fontsize=16, labelpad = 18)

    ax = fig.add_subplot(1,3,2)
    ax.set_title('Non-Beaconed', fontsize=20, verticalalignment='bottom', style='italic')
    ax.axhspan(rz_start, rz_end, facecolor='g', alpha=0.15, hatch='/', linewidth=0)  # green box spanning the rewardzone - to mark reward zone
    ax.axhspan(0, 4, facecolor='k', alpha=0.15, hatch='/', linewidth=0)  # black box
    ax.axhline(4, linewidth=1, ls='--', color='black')  # mark black box border
    ax.axvline(0, linewidth = 3, color = 'black')# bold line on the y axis
    ax.axhline(2, linewidth = 3, color = 'black')# bold line on the x axis

    ax.plot(nb_trial_max, con_nbeac, 'o', color = '0.3', label = 'Non reward zone score', linewidth = 2, markersize = 6, markeredgecolor = 'black')
    ax.errorbar(nb_trial_max,con_nbeac,sd_con_nbeac, fmt = 'o', color = '0.3', capsize = 1.5, markersize = 2, elinewidth = 1.5)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =15)
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =15)
    ax.set_xlim(0,len(nb_trial_max))
    ax.set_ylim(2,rz_end+2)
    adjust_spines(ax, ['left','bottom']) # remove right and top axis
    plt.locator_params(nbins=7, axis ='y') # set tick number on y axis
    plt.locator_params(nbins=3, axis ='x') # set tick number on x axis
    ax = plt.xlabel('Number of trials', fontsize=16,labelpad=18)

    ax = fig.add_subplot(1,3,3)
    ax.set_title('Probe', fontsize=20, verticalalignment='bottom', style='italic')  # title
    ax.axhspan(rz_start, rz_end, facecolor='g', alpha=0.15, hatch='/', linewidth=0)  # green box of reward zone
    ax.axhspan(0, 4, facecolor='k', alpha=0.15, hatch='/', linewidth=0)  # black box (4 is used for appearance)
    ax.axhline(4, linewidth=1, ls='--', color='black')  # mark black box border
    ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
    ax.axhline(2, linewidth=3, color='black')  # bold line on the x axis

    ax.plot(p_trial_max, con_probe, 'o', color = '0.3', label = 'Non reward zone score', linewidth = 2, markersize = 6, markeredgecolor = 'black')
    ax.errorbar(p_trial_max,con_probe,sd_con_probe, fmt = 'o', color = '0.3', capsize = 1.5, markersize = 2, elinewidth = 1.5)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =15)
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =15)
    ax.set_xlim(0,len(p_trial_max))
    ax.set_ylim(2,rz_end+2)
    adjust_spines(ax, ['left','bottom']) # remove right and top axis
    plt.locator_params(nbins=7, axis ='y') # set tick number on y axis
    plt.locator_params(nbins=3, axis ='x') # set tick number on x axis

    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.15, right = 0.82, top = 0.92)

    fig.savefig(save_path, dpi=200)
    plt.close()

if __name__ == '__main__':
    main()