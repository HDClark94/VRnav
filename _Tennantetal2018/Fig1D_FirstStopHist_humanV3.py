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

    #SESSION PATHS FOR CONSTANT SPEED PARTICIPANTS
    '''
    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190722150558/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190723100511/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190723111425/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190731113135/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190731153240/S001']
    plot_fig1D(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1D_S_FirstStopHist.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190722150558/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190723100511/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190723111425/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190731113135/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190731153240/S001']
    plot_fig1D(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1D_M_FirstStopHist.png')
    
    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190722150558/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190723100511/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190723111425/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190731113135/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190731153240/S001']
    plot_fig1D(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1D_L_FirstStopHist.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium/P_190722150558/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium/P_190723100511/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium/P_190723111425/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium/P_190731113135/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium/P_190731153240/S001']
    plot_fig1D(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1D_CM_FirstStopHist.png')
    
    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation/P_190722150558/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation/P_190723100511/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation/P_190723111425/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation/P_190731113135/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation/P_190731153240/S001']
    plot_fig1D(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1D_H_FirstStopHist.png')


    #SESSION PATHS FOR GAIN PARTICIPANTS

    '''
    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190724151650/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190725130212/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190726112925/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190726153910/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190729100324/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190801113837/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190726140934/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190726165828/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190802100142/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190805133756/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190807100354/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190807111732/S001']

    plot_fig1D(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1D_S_Gain_FirstStopHist.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190724151650/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190725130212/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190726112925/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190726153910/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190729100324/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190801113837/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190726140934/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190726165828/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190802100142/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190805133756/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190807100354/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190807111732/S001']
    plot_fig1D(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1D_M_Gain_FirstStopHist.png')
    
    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190724151650/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190725130212/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190726112925/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190726153910/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190729100324/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190801113837/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190726140934/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190726165828/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190802100142/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190805133756/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190807100354/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190807111732/S001']
    plot_fig1D(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1D_L_Gain_FirstStopHist.png')
    
    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190724151650/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190725130212/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190726112925/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190726153910/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190729100324/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190801113837/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190726140934/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190726165828/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190802100142/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190805133756/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190807100354/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190807111732/S001']
    plot_fig1D(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1D_CM_Gain_FirstStopHist.png')

    '''
    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190724151650/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190725130212/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190726112925/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190726153910/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190729100324/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190801113837/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190726140934/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190726165828/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190802100142/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190805133756/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190807100354/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190807111732/S001']
    plot_fig1D(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1D_H_Gain_FirstStopHist.png')
    '''

def plot_fig1D(session_paths, save_path):


    days = [1]
    n_bins = 20
    # specify mouse/mice and day/s to analyse
    days = ['Day' + str(int(x)) for x in np.arange(1)]

    # Arrays for storing data (output)
    firststopstorebeac = np.zeros((len(days), len(session_paths), n_bins,2))
    firststopstorenbeac = np.zeros((len(days), len(session_paths), n_bins,2))
    firststopstoreprobe = np.zeros((len(days), len(session_paths), n_bins,2))
    firststopstorebeac[:,:,:,:] = np.nan
    firststopstorenbeac[:,:,:,:] = np.nan
    firststopstoreprobe[:,:,:,:] = np.nan

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

            trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]+0.1),1)# array of trial numbers
            beac=[]; nbeac=[]; probe=[] # make empty arrays to store data
            trialids_b = np.unique(stops_b[:, 2]) # find unique trial numbers
            stops_f_b = FirstStops_humans( trarray,stops_b, track_start, track_end) # get locations of first stop for each trial
            stops_f_b = create_srdata( stops_f_b, trialids_b ) # bin first stop data
            beac = np.nanmean(stops_f_b, axis = 0) # average times mouse stops first in each bin
            if stops_nb.size >0 :
                trialids_nb = np.unique(stops_nb[:, 2])
                stops_f_nb = FirstStops_humans( trarray,stops_nb, track_start, track_end)# get locations of first stop for each trial
                stops_f_nb = create_srdata( stops_f_nb, trialids_nb )# bin first stop data
                nbeac = np.nanmean(stops_f_nb, axis = 0)# average times mouse stops first in each bin
            if stops_p.size >0 :
                trialids_p = np.unique(stops_p[:, 2])
                stops_f_p = FirstStops_humans( trarray,stops_p, track_start, track_end)# get locations of first stop for each trial
                stops_f_p = create_srdata( stops_f_p, trialids_p )# bin first stop data
                probe = np.nanmean(stops_f_p, axis = 0)# average times mouse stops first in each bin

            # store data
            #if mcount == 3 or mcount == 5 or mcount == 6 or mcount == 7 or mcount == 8:
            firststopstorebeac[dcount,session_count,:,0] = beac # store first stop data
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stops_b, trialids_b ) # get average stops per location bin
            firststopstorebeac[dcount,session_count,:,1] = srbin_mean # store stops data
            if stops_nb.size >0 :
                firststopstorenbeac[dcount, session_count,:,0] = nbeac# store first stop data
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stops_nb, trialids_nb )# get average stops per location bin
                firststopstorenbeac[dcount,session_count,:,1] = srbin_mean# store stops data
            if stops_p.size >0:
                firststopstoreprobe[dcount, session_count,:,0] = probe# store first stop data
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stops_p, trialids_p )# get average stops per location bin
                firststopstoreprobe[dcount,session_count,:,1] = srbin_mean# store stops data
            session_count +=1



    # Average over days for all mice

    # week 1 (first stop)
    con_beac_w1 = np.nanmean(((firststopstorebeac[0,:,:,0])), axis = 0)
    con_nbeac_w1 = np.nanmean(((firststopstorenbeac[0,:,:,0])), axis =0)
    con_probe_w1 = np.nanmean(((firststopstoreprobe[0,:,:,0])), axis = 0)
    sd_con_beac_w1 = np.nanstd(((firststopstorebeac[0,:,:,0])), axis = 0)/math.sqrt(session_count)
    sd_con_nbeac_w1 = np.nanstd(((firststopstorenbeac[0,:,:,0])), axis = 0)/math.sqrt(session_count)
    sd_con_probe_w1 = np.nanstd(((firststopstoreprobe[0,:,:,0])), axis = 0)/math.sqrt(session_count)
    # week 1 (all stops)
    con_beac1_w1 = np.nanmean(((firststopstorebeac[0,:,:,1])), axis = 0)
    con_nbeac1_w1 = np.nanmean(((firststopstorenbeac[0,:,:,1])), axis =0)
    con_probe1_w1 = np.nanmean(((firststopstoreprobe[0,:,:,1])), axis = 0)
    sd_con_beac1_w1 = np.nanstd(((firststopstorebeac[0,:,:,1])), axis = 0)/math.sqrt(session_count)
    sd_con_nbeac1_w1 = np.nanstd(((firststopstorenbeac[0,:,:,1])), axis =0)/math.sqrt(session_count)
    sd_con_probe1_w1 = np.nanstd(((firststopstoreprobe[0,:,:,1])), axis = 0)/math.sqrt(session_count)

    '''
    # week 4 (first stop)
    con_beac_w4 = np.nanmean(np.nanmean(((firststopstorebeac[18:22,:,:,0])), axis = 0), axis = 0)
    con_nbeac_w4 = np.nanmean(np.nanmean(((firststopstorenbeac[18:22,:,:,0])), axis =0), axis = 0)
    con_probe_w4 = np.nanmean(np.nanmean(((firststopstoreprobe[18:22,:,:,0])), axis = 0), axis = 0)
    sd_con_beac_w4 = np.nanstd(np.nanmean(((firststopstorebeac[18:22,:,:,0])), axis = 0), axis = 0)/math.sqrt(8)
    sd_con_nbeac_w4 = np.nanstd(np.nanmean(((firststopstorenbeac[18:22,:,:,0])), axis =0), axis = 0)/math.sqrt(8)
    sd_con_probe_w4 = np.nanstd(np.nanmean(((firststopstoreprobe[18:22,:,:,0])), axis = 0), axis = 0)/math.sqrt(8)
    # week 4 (all stops)
    con_beac1_w4 = np.nanmean(np.nanmean(((firststopstorebeac[18:22,:,:,1])), axis = 0), axis = 0)
    con_nbeac1_w4 = np.nanmean(np.nanmean(((firststopstorenbeac[18:22,:,:,1])), axis =0), axis = 0)
    con_probe1_w4 = np.nanmean(np.nanmean(((firststopstoreprobe[18:22,:,:,1])), axis = 0), axis = 0)
    sd_con_beac1_w4 = np.nanstd(np.nanmean(((firststopstorebeac[18:22,:,:,1])), axis = 0), axis = 0)/math.sqrt(8)
    sd_con_nbeac1_w4 = np.nanstd(np.nanmean(((firststopstorenbeac[18:22,:,:,1])), axis =0), axis = 0)/math.sqrt(8)
    sd_con_probe1_w4 = np.nanstd(np.nanmean(((firststopstoreprobe[18:22,:,:,1])), axis = 0), axis = 0)/math.sqrt(8)
    '''

    # PLOT GRAPHS

    bins = np.arange(0.5,n_bins+0.5,1)

    # first stop histogram
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,3,1) #stops per trial
    ax.set_title('Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
    ax.axvspan(rz_start, rz_end, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
    ax.axvspan(0, track_start, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
    ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
    ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
    ax.plot(bins,con_beac_w1,color = 'blue',label = 'Beaconed', linewidth = 2) #plot becaoned trials
    ax.fill_between(bins,con_beac_w1-sd_con_beac_w1,con_beac_w1+sd_con_beac_w1, facecolor = 'blue', alpha = 0.3)
    #ax.plot(bins,con_beac_w4,color = 'red',label = 'Beaconed', linewidth = 2) #plot becaoned trials
    #ax.fill_between(bins,con_beac_w4-sd_con_beac_w4,con_beac_w4+sd_con_beac_w4, facecolor = 'red', alpha = 0.3)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
    ax.set_xlim(0,20)
    ax.set_ylim(0,0.95)
    adjust_spines(ax, ['left','bottom']) # removes top and right spines
    ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
    ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
    ax.set_xticklabels(['0', '10', '20'])
    ax.set_yticklabels(['0','0.3','0.6','0.9'])
    ax.set_ylabel('1st stop probability', fontsize=16, labelpad = 18)

    ax = fig.add_subplot(1,3,2) #stops per trial
    ax.set_title('Non-Beaconed', fontsize=20, verticalalignment='bottom', style='italic')  # title
    ax.axvspan(rz_start, rz_end, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
    ax.axvspan(0, track_start, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
    ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
    ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
    ax.plot(bins,con_nbeac_w1,color = 'blue', linewidth = 2) #plot becaoned trials
    ax.fill_between(bins,con_nbeac_w1-sd_con_nbeac_w1,con_nbeac_w1+sd_con_nbeac_w1, facecolor = 'blue', alpha = 0.3)
    #ax.plot(bins,con_nbeac_w4,color = 'red', linewidth = 2) #plot becaoned trials
    #ax.fill_between(bins,con_nbeac_w4-sd_con_nbeac_w4,con_nbeac_w4+sd_con_nbeac_w4, facecolor = 'red', alpha = 0.3)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
    ax.set_xlim(0,20)
    ax.set_ylim(0,0.95)
    adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
    ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
    ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
    ax.set_xticklabels(['0', '10', '20'])
    ax.set_yticklabels(['', '', '',''])
    ax.set_xlabel('Distance (VU)', fontsize=16, labelpad=18)

    ax = fig.add_subplot(1,3,3) #stops per trial
    ax.set_title('Probe', fontsize=20, verticalalignment='bottom', style='italic')  # title
    ax.axvspan(rz_start, rz_end, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
    ax.axvspan(0, track_start, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
    ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
    ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
    ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
    ax.plot(bins,con_probe_w1,color = 'blue', label = 'Beaconed', linewidth = 2) #plot becaoned trials
    ax.fill_between(bins,con_probe_w1-sd_con_probe_w1,con_probe_w1+sd_con_probe_w1, facecolor = 'blue', alpha = 0.3)
    #ax.plot(bins,con_probe_w4,color = 'red', label = 'Beaconed', linewidth = 2) #plot becaoned trials
    #ax.fill_between(bins,con_probe_w4-sd_con_probe_w4,con_probe_w4+sd_con_probe_w4, facecolor = 'red', alpha = 0.3)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
    ax.set_xlim(0,20)
    ax.set_ylim(0,0.95)
    adjust_spines(ax, ['left','bottom']) # removes top and right spines
    ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
    ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
    ax.set_yticklabels(['', '', '',''])
    ax.set_xticklabels(['0', '10', '20'])

    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.15, right = 0.82, top = 0.85)

    fig.savefig(save_path,  dpi = 200)
    plt.close()

if __name__ == '__main__':
    main()