# -*- coding: utf-8 -*-
"""

@author: Sarah Tennant


### Calculates the difference between Z-scores in the black box and reward zone for each session - plots over days


"""

# Import packages and functions
from Functions_Core_0100 import extractstops_HUMAN,filterstops, create_srdata, makebinarray, shuffle_analysis_pertrial3, z_score1, adjust_spines, makelegend2,readhdfdata,maketrialarray
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform
from math import floor
from summarize.map2legacy import *


def main():
    print('-------------------------------------------------------------')

    print('-------------------------------------------------------------')

    # SESSION PATHS FOR CONSTANT SPEED PARTICIPANTS

    session_paths = [
        '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190722150558/S001',
        '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190723100511/S001',
        '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190723111425/S001',
        '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190731113135/S001',
        '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190731153240/S001']
    plot_fig1G(session_paths,save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1G_S_AvgZScore.png')
    '''
    session_paths = [
        '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190722150558/S001',
        '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190723100511/S001',
        '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190723111425/S001',
        '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190731113135/S001',
        '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190731153240/S001']
    plot_fig1G(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1G_M_AvgZScore.png')
    
    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190722150558/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190723100511/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190723111425/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190731113135/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190731153240/S001']
    plot_fig1G(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1G_L_AvgZScore.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium/P_190722150558/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium/P_190723100511/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium/P_190723111425/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium/P_190731113135/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium/P_190731153240/S001']
    plot_fig1G(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1G_CM_AvgZScore.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation/P_190722150558/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation/P_190723100511/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation/P_190723111425/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation/P_190731113135/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation/P_190731153240/S001']
    plot_fig1G(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1G_H_AvgZScore.png')


    #SESSION PATHS FOR GAIN PARTICIPANTS


    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190726112925/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190726153910/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190726165828/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190729100324/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190801113837/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short_gain/P_190802100142/S001']
    plot_fig1G(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1G_S_Gain_AvgZScore.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190726112925/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190726153910/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190726165828/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190729100324/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190801113837/S001',
                      '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium_gain/P_190802100142/S001']
    plot_fig1G(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1G_M_Gain_AvgZScore.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190726112925/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190726153910/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190726165828/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190729100324/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190801113837/S001',
                     '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long_gain/P_190802100142/S001']
    plot_fig1G(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1G_L_Gain_AvgZScore.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190726112925/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190726153910/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190726165828/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190729100324/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190801113837/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium_gain/P_190802100142/S001']
    plot_fig1G(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1E_CM_Gain_AvgZScore.png')

    session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190726112925/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190726153910/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190726165828/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190729100324/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190801113837/S001',
                    '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation_gain/P_190802100142/S001']
    plot_fig1G(session_paths, save_path='/Users/emmamather-pike/PycharmProjects/data/plots/Fig1DE_H_Gain_AvgZScore.png') 

    '''


def plot_fig1G (session_paths, save_path):

    days = [1] #
    number_of_trials = 1000 # arbitrarily large amount of trials so we can catch all the trials

    # Arrays for storing data (output)
    firststopstorebeac = np.zeros(((number_of_trials), len(session_paths)))
    firststopstorenbeac = np.zeros(((number_of_trials), len(session_paths)))
    firststopstoreprobe = np.zeros(((number_of_trials), len(session_paths)))
    firststopstorebeac[:,:] = np.nan
    firststopstorenbeac[:,:] = np.nan
    firststopstoreprobe[:,:] = np.nan

    sd_con_firststopstorebeac = np.zeros(((number_of_trials), len(session_paths)))
    sd_con_firststopstorenbeac = np.zeros(((number_of_trials), len(session_paths)))
    sd_con_firststopstoreprobe = np.zeros(((number_of_trials), len(session_paths)))
    sd_con_firststopstorebeac[:,:] = np.nan
    sd_con_firststopstorenbeac[:,:] = np.nan
    sd_con_firststopstoreprobe[:,:] = np.nan

    # For each day and mouse, pull raw data, calculate first stops and store data
    for dcount, day in enumerate(days):
        session_count = 0
        for session_path in session_paths:
            saraharray, track_start, track_end = translate_to_legacy_format(session_path)

            rz_start = saraharray[0, 11]
            rz_end = saraharray[0, 12]

            # make array of trial number for each row in dataset
            trialarray = maketrialarray(saraharray) # write array of trial per row in datafile
            saraharray[:,9] = trialarray[:,0] # replace trial column in dataset *see README for why this is done*

            # get stops and trial arrays
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)# delete all data not on

            # get stops
            stops_b = extractstops_HUMAN(dailymouse_b)
            stops_nb = extractstops_HUMAN(dailymouse_nb)
            stops_p= extractstops_HUMAN(dailymouse_p)

            # filter stops
            stops_b = filterstops(stops_b)
            stops_nb = filterstops(stops_nb)
            stops_p = filterstops(stops_p)

            if stops_b.size > 0:
                trialids_b = np.unique(stops_b[:, 2])  # make array of unique trial numbers

                for i in range(len(stops_b)):
                    srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stops_b,trialids_b)  # get average real stops & shuffled stops per lcoation bin
                shuff_beac = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
                bb = shuff_beac[3]; rz = shuff_beac[9]  # black box - reward zone zscore
                score = rz - bb


            if stops_nb.size > 0:
                trialids_nb = np.unique(stops_nb[:, 2])  # make array of unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stops_nb,trialids_nb)  # get average real stops & shuffled stops per lcoation bin
                shuff_nbeac = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
                bb = shuff_nbeac[3];
                rz = shuff_nbeac[9]  # black box - reward zone zscore
                score = rz - bb
            for i in range(len(stops_nb)):
                firststopstorebeac[i, session_count] = score[i][0]

            if stops_p.size > 0:
                trialids_p = np.unique(stops_p[:, 2])  # make array of unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stops_p,trialids_p)  # get average real stops & shuffled stops per lcoation bin
                shuff_probe = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
                bb = shuff_probe[3];
                rz = shuff_probe[9]  # black box - reward zone zscore
                score = rz - bb
                firststopstoreprobe[dcount, mcount] = score  # store data

                mcount += 1

         # get average real stops & shuffled stops per location bin
                zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
            for i in range(len(stops_b)):
                firststopstorebeac[i, session_count] = zscore_b[i][0]

            if stops_nb.size>0:
                trialids_nb = np.unique(stops_nb[:, 2])
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stops_nb,trialids_nb)  # get average real stops & shuffled stops per location bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
            for i in range(len(stops_nb)):
                firststopstorebeac[i, session_count] = zscore_nb[i][0]

            if stops_p.nb.size>0:
                trialids_p = np.unique(stops_p[:, 2])
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stops_p,trialids_p)  # get average real stops & shuffled stops per location bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
            for i in range(len(stops_p)):
                firststopstorebeac[i, session_count] = zscore_p[i][0]

                session_count +=1


    # stack experiments then average over days for each mouse
    con_b = np.nanmean(((firststopstorebeac)), axis = 1)
    con_b = con_b[~np.isnan(con_b)]
    con_nb = np.nanmean(((firststopstorenbeac)), axis =1)
    con_nb = con_nb[~np.isnan(con_nb)]
    con_p = np.nanmean(((firststopstoreprobe)), axis = 1)
    con_p = con_p[~np.isnan(con_nb)]
    sd_con_b = np.nanstd(((firststopstorebeac)), axis=1)/math.sqrt(session_count)
    sd_con_b = sd_con_b[~np.isnan(sd_con_b)]
    sd_con_nb = np.nanstd(((firststopstorenbeac)), axis=1)/math.sqrt(session_count)
    sd_con_nb = sd_con_nb[~np.isnan(sd_con_nb)]
    sd_con_p = np.nanstd(((firststopstoreprobe)), axis=1)/math.sqrt(session_count)
    sd_con_p = sd_con_p[~np.isnan(sd_con_p)]


    # PLOT GRAPHS

    bins = np.arange(0.5,21.5+1e-6,1) # track bins

    fig = plt.figure(figsize = (12,3))
    ax = fig.add_subplot(1,3,1)
    ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
    ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
    ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
    ax.plot(bins,con_b,'o',color = 'Black',label = 'AAV-fl-GFP', linewidth = 2, markersize = 6, markeredgecolor = 'black') #plot becaoned trials
    ax.errorbar(bins,con_b,sd_con_b, fmt = 'o', color = 'black', capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.set_xlim(0,22)
    ax.set_ylim(-10,20)
    adjust_spines(ax, ['left','bottom']) # removes top and right spines
    ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
    ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
    ax = plt.ylabel('Training day)', fontsize=16, labelpad = 18)

    ax = fig.add_subplot(1,3,2) #stops per trial
    ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
    ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
    ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
    ax.plot(bins,con_nb,'o', color = 'Black', linewidth = 2, markersize = 6, markeredgecolor = 'black') #plot becaoned trials
    ax.errorbar(bins,con_nb,sd_con_nb, fmt = 'o', color = 'black',  capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.set_xlim(0,22)
    ax.set_ylim(-10,20)
    adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
    ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
    ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
    ax = plt.xlabel('Training day', fontsize=16, labelpad = 18)

    ax = fig.add_subplot(1,3,3) #stops per trial
    ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
    ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
    ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
    ax.plot(bins,con_p, 'o', color = 'Black',label = 'Beaconed', linewidth = 2, markersize = 6, markeredgecolor = 'black') #plot becaoned trials
    ax.errorbar(bins,con_p,sd_con_p, fmt = 'o', color = 'black',  capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
    ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
    ax.set_xlim(0,22)
    ax.set_ylim(-10,20)
    adjust_spines(ax, ['left','bottom']) # removes top and right spines
    ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
    ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
    ax = plt.xlabel('Training day', fontsize=16, labelpad = 18)

    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.15, right = 0.82, top = 0.85)

    fig.savefig(save_path, dpi=200)
    plt.close()

if __name__ == '__main__':
    main()
