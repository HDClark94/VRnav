# -*- coding: utf-8 -*-
"""

### Calculates Z-scores for each location bin along the track
- Location bins are 10 cm
- Z-scores calculated for each mouse in last training week then averaged over mice


"""

# import packages and functions
from Functions_Core_0100 import extractstops_HUMAN, filterstops, create_srdata, makebinarray, shuffle_analysis_pertrial3, z_score1, adjust_spines,readhdfdata,maketrialarray
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform
from summarize.map2legacy import *

#--------------------------------------------------------------------------------------------------------------#

# First half of script gets data for first training week

#--------------------------------------------------------------------------------------------------------------#

def main():

    print('-------------------------------------------------------------')

    print('-------------------------------------------------------------')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190729100324/S001', '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190726112925/S001', '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190726153910/S001']
    plot_fig1F(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/S_Gain_ZscoreHist.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190729100324/S001', '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190726112925/S001', '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190726153910/S001']
    plot_fig1F(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/M_Gain_ZscoreHist.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190729100324/S001', '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190726112925/S001', '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190726153910/S001']
    plot_fig1F(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/L_Gain_ZscoreHist.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190729100324/S001', '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190726112925/S001', '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190726153910/S001']
    plot_fig1F(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/CM_Gain_ZScoreHist.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190729100324/S001', '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190726112925/S001', '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190726153910/S001']
    plot_fig1F(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/H_Gain_ZScoreHist.png')



def plot_fig1F(session_paths, save_path):


    # specify mouse/mice and day/s to analyse
    days = [1]
    n_bins = 20
    # specify mouse/mice and day/s to analyse
    days = ['Day' + str(int(x)) for x in np.arange(1)]

    # empty arrays for storing data
    firststopstorebeac = np.zeros((len(days), len(session_paths), n_bins, 2))
    firststopstorenbeac = np.zeros((len(days), len(session_paths), n_bins, 2))
    firststopstoreprobe = np.zeros((len(days), len(session_paths), n_bins, 2))
    firststopstorebeac[:,:,:] = np.nan
    firststopstorenbeac[:,:,:] = np.nan
    firststopstoreprobe[:,:,:] = np.nan

    #loop days and mice to collect data
    for dcount,day in enumerate(days):
        session_count = 0
        for session_path in session_paths:
            saraharray, track_start, track_end = translate_to_legacy_format(session_path)

            rz_start = saraharray[0, 11]
            rz_end = saraharray[0, 12]

            trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
            saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)

            # split data by trial type
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)# delete all data not on probe tracks

            #extract stops
            stopsdata_b = extractstops_HUMAN(dailymouse_b)
            stopsdata_nb = extractstops_HUMAN(dailymouse_nb)
            stopsdata_p = extractstops_HUMAN(dailymouse_p)

            # filter stops
            stopsdata_b = filterstops(stopsdata_b)
            stopsdata_nb = filterstops(stopsdata_nb)
            stopsdata_p = filterstops(stopsdata_p)

            trialids_b = np.unique(stopsdata_b[:, 2])  # make array of unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stopsdata_b,trialids_b)  # get average real stops & shuffled stops per lcoation bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
            firststopstorebeac[dcount, session_count,:,1] = zscore_b  # store zscores
            if stopsdata_nb.size > 0:  # if there is non-beaconed data
                trialids_nb = np.unique(stopsdata_nb[:, 2])  # make array of unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stopsdata_nb, trialids_nb)  # get average real stops & shuffled stops per lcoation bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
                firststopstorenbeac[dcount, session_count,:,1] = zscore_nb  # store zscores
            if stopsdata_p.size > 0:  # if there is probe data
                trialids_p = np.unique(stopsdata_p[:, 2])  # make array of unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stopsdata_p, trialids_p)  # get average real stops & shuffled stops per lcoation bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
                firststopstoreprobe[dcount, session_count,:,1] = zscore_p  # store zscores
                session_count += 1
            '''
            trialids_b = np.unique(stopsdata_b[:, 2])  # get array of trial numbers for beaconed
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stopsdata_b, trialids_b)  # get average real stops & shuffled stops per lcoation bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
            firststopstorebeac2[dcount, session_count,:,1] = zscore_b  # store zscores
            if stopsdata_nb.size > 0:
                trialids_nb = np.unique(stopsdata_nb[:, 2])  # get array of trial numbers for non-beaconed
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stopsdata_nb, trialids_nb)  # get average real stops & shuffled stops per lcoation bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
                firststopstorenbeac2[dcount, session_count,:,1] = zscore_nb  # store zscores
            if stopsdata_p.size > 0:
                trialids_p = np.unique(stopsdata_p[:, 2])  # get array of trial numbers for probe
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stopsdata_p, trialids_p)  # get average real stops & shuffled stops per lcoation bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
                firststopstoreprobe2[dcount, session_count,:,1] = zscore_p  # store zscores
                session_count += 1
            '''


    # AVERAGE DATA FOR PLOTS
    # stack experiments then average over days then mice
    con_b1 = np.nanmean(((firststopstorebeac[0, :, :,1])), axis=0)
    con_nb1 = np.nanmean(((firststopstorenbeac[0, :, :,1])), axis=0)
    con_p1 = np.nanmean(((firststopstoreprobe[0,:, :, 1])), axis=0)
    sdcon_b1 = np.nanstd(((firststopstorebeac[0, :, :,1])), axis=0)/math.sqrt(session_count)
    sdcon_nb1 = np.nanstd(((firststopstorenbeac[0, :, :,1])), axis=0)/ math.sqrt(session_count)
    sdcon_p1 = np.nanstd(((firststopstoreprobe[0, :, :,1])), axis=0)/ math.sqrt(session_count)


    # PLOT GRAPHS

    bins = np.arange(0.5,n_bins+0.5,1) # track bins

    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,3,1)
    ax.axvspan(rz_start, rz_end, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
    ax.axvspan(0, track_start, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
    ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
    ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
    ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
    ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
    #ax.plot(bins*5,con_b,color = 'red',label = 'AAV-fl-GFP', linewidth = 2) #plot becaoned trials
    #ax.fill_between(bins*5,con_b-sdcon_b,con_b+sdcon_b, facecolor = 'red', alpha = 0.3)
    ax.plot(bins,con_b1,'',color = 'blue',label = 'AAV-fl-GFP', linewidth = 2) #plot becaoned trials
    ax.fill_between(bins,con_b1-sdcon_b1,con_b1+sdcon_b1, facecolor = 'blue', alpha = 0.3)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.set_xlim(0,20)
    ax.set_ylim(-10,20)
    adjust_spines(ax, ['left','bottom']) # removes top and right spines
    ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
    ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
    ax.set_xticklabels(['0', '10', '20'])
    ax.set_yticklabels(['-10','0','10','20'])
    ax = plt.ylabel('Stops (Zscore)', fontsize=16, labelpad = 18)

    ax = fig.add_subplot(1,3,2) #stops per trial
    ax.axvspan(rz_start, rz_end, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
    ax.axvspan(0, track_start, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
    ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
    ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
    ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
    ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
    #ax.plot(bins*5,con_nb,color = 'red', linewidth = 2) #plot becaoned trials
    #ax.fill_between(bins*5,con_nb-sdcon_nb,con_nb+sdcon_nb, facecolor = 'red', alpha = 0.3)
    ax.plot(bins,con_nb1,color = 'blue', linewidth = 2) #plot becaoned trials
    ax.fill_between(bins,con_nb1-sdcon_nb1,con_nb1+sdcon_nb1, facecolor = 'blue', alpha = 0.3)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.set_xlim(0,20)
    ax.set_ylim(-10,20)
    adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
    ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
    ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
    ax.set_xticklabels(['0', '10', '20'])
    ax.set_yticklabels(['','','',''])
    ax = plt.xlabel('Location (VU)', fontsize=16, labelpad = 18)

    ax = fig.add_subplot(1,3,3) #stops per trial
    ax.axvspan(rz_start, rz_end, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
    ax.axvspan(0, track_start, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
    ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
    ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
    ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
    ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
    #ax.plot(bins*5,con_p,color = 'red', label = 'Beaconed', linewidth = 2) #plot becaoned trials
    #ax.fill_between(bins*5,con_p-sdcon_p,con_p+sdcon_p, facecolor = 'red', alpha = 0.3)
    ax.plot(bins*5,con_p1,color = 'blue', label = 'Beaconed', linewidth = 2) #plot becaoned trials
    ax.fill_between(bins*5,con_p1-sdcon_p1,con_p1+sdcon_p1, facecolor = 'blue', alpha = 0.3)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.set_xlim(0,20)
    ax.set_ylim(-10,20)
    adjust_spines(ax, ['left','bottom']) # removes top and right spines
    ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
    ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
    ax.set_xticklabels(['0', '10', '20'])
    ax.set_yticklabels(['','','',''])

    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.15, right = 0.82, top = 0.92)

    fig.savefig(save_path, dpi=200)
    plt.close()

if __name__ == '__main__':
    main()




