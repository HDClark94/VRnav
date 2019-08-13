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

# Load SMALL TRACK data:
# CONSTANT SPEED
session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190722150558/S001',
                 '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190723100511/S001',
                 '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190723111425/S001',
                 '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190731113135/S001',
                 '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190731153240/S001']

# GAIN

days = [1]
n_bins = 20

days = ['Day' + str(int(x)) for x in np.arange(1)]

# empty arrays for storing data
firststopstorebeac = np.zeros((len(days), len(session_paths), n_bins, 2))
firststopstorenbeac = np.zeros((len(days), len(session_paths), n_bins, 2))
firststopstoreprobe = np.zeros((len(days), len(session_paths), n_bins, 2))
firststopstorebeac[:,:,:] = np.nan
firststopstorenbeac[:,:,:] = np.nan
firststopstoreprobe[:,:,:] = np.nan

#loop days and mice to collect data
for dcount, day in enumerate(days):
    session_count = 0
    for session_path in session_paths:
        saraharray, track_start, track_end = translate_to_legacy_format(session_path)

        rz_start_s = saraharray[0, 11]
        rz_end_s = saraharray[0, 12]

        trialarray = maketrialarray(saraharray)  # make array of trial number same size as saraharray
        saraharray[:, 9] = trialarray[:, 0]  # replace trial number because of increment error (see README.py)

        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)  # delete all data not on beaconed tracks
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)  # delete all data not on non beaconed tracks
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)  # delete all data not on probe tracks

        # extract stops
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
        firststopstorebeac[dcount, session_count, :, 1] = zscore_b  # store zscores

        if stopsdata_nb.size > 0:  # if there is non-beaconed data
            trialids_nb = np.unique(stopsdata_nb[:, 2])  # make array of unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stopsdata_nb,trialids_nb)  # get average real stops & shuffled stops per lcoation bin
            zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
            firststopstorenbeac[dcount, session_count, :, 1] = zscore_nb  # store zscores

        if stopsdata_p.size > 0:  # if there is probe data
            trialids_p = np.unique(stopsdata_p[:, 2])  # make array of unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stopsdata_p,trialids_p)  # get average real stops & shuffled stops per lcoation bin
            zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
            firststopstoreprobe[dcount, session_count, :, 1] = zscore_p  # store zscores
            session_count += 1


# Load MEDIUM TRACK data:
session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190722150558/S001',
                '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190723100511/S001',
                '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190723111425/S001',
                '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190731113135/S001',
                '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190731153240/S001']

days = [1]
n_bins = 20

days = ['Day' + str(int(x)) for x in np.arange(1)]

# empty arrays for storing data
firststopstorebeac2 = np.zeros((len(days), len(session_paths), n_bins, 2))
firststopstorenbeac2 = np.zeros((len(days), len(session_paths), n_bins, 2))
firststopstoreprobe2 = np.zeros((len(days), len(session_paths), n_bins, 2))
firststopstorebeac2[:,:,:] = np.nan
firststopstorenbeac2[:,:,:] = np.nan
firststopstoreprobe2[:,:,:] = np.nan

#loop days and mice to collect data
for dcount, day in enumerate(days):
    session_count = 0
    for session_path in session_paths:
        saraharray, track_start, track_end = translate_to_legacy_format(session_path)

        rz_start_m = saraharray[0, 11]
        rz_end_m = saraharray[0, 12]  # end of reward zone
        # make location bins

        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray)  # make array of trial number same size as saraharray
        saraharray[:, 9] = trialarray[:, 0]  # replace trial number because of increment error (see README.py)

        trialarray = maketrialarray(saraharray)  # make array of trial number same size as saraharray
        saraharray[:, 9] = trialarray[:, 0]  # replace trial number because of increment error (see README.py)

        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)  # delete all data not on beaconed tracks
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)  # delete all data not on non beaconed tracks
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)  # delete all data not on probe tracks

        # extract stops
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
        firststopstorebeac2[dcount, session_count, :, 1] = zscore_b  # store zscores

        if stopsdata_nb.size > 0:  # if there is non-beaconed data
            trialids_nb = np.unique(stopsdata_nb[:, 2])  # make array of unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stopsdata_nb,trialids_nb)  # get average real stops & shuffled stops per lcoation bin
            zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
            firststopstorenbeac2[dcount, session_count, :, 1] = zscore_nb  # store zscores

        if stopsdata_p.size > 0:  # if there is probe data
            trialids_p = np.unique(stopsdata_p[:, 2])  # make array of unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stopsdata_p,trialids_p)  # get average real stops & shuffled stops per lcoation bin
            zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
            firststopstoreprobe2[dcount, session_count, :, 1] = zscore_p  # store zscores
            session_count += 1

# Load LONG TRACK data:
session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190722150558/S001',
                 '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190723100511/S001',
                 '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190723111425/S001',
                 '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190731113135/S001',
                 '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190731153240/S001']

days = [1]
n_bins = 20

days = ['Day' + str(int(x)) for x in np.arange(1)]

# empty arrays for storing data
firststopstorebeac3 = np.zeros((len(days), len(session_paths), n_bins, 2))
firststopstorenbeac3 = np.zeros((len(days), len(session_paths), n_bins, 2))
firststopstoreprobe3 = np.zeros((len(days), len(session_paths), n_bins, 2))
firststopstorebeac3[:,:,:] = np.nan
firststopstorenbeac3[:,:,:] = np.nan
firststopstoreprobe3[:,:,:] = np.nan

# loop days and mice to collect data
for dcount, day in enumerate(days):
    session_count = 0
    for session_path in session_paths:
        saraharray, track_start, track_end = translate_to_legacy_format(session_path)

        rz_start_l = saraharray[0, 11]
        rz_end_l = saraharray[0, 12]

        trialarray = maketrialarray(saraharray)  # make array of trial number same size as saraharray
        saraharray[:, 9] = trialarray[:, 0]  # replace trial number because of increment error (see README.py)

        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)  # delete all data not on beaconed tracks
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)  # delete all data not on non beaconed tracks
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)  # delete all data not on probe tracks

        # extract stops
        stopsdata_b = extractstops_HUMAN(dailymouse_b)
        stopsdata_nb = extractstops_HUMAN(dailymouse_nb)
        stopsdata_p = extractstops_HUMAN(dailymouse_p)

        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)

        trialids_b = np.unique(stopsdata_b[:, 2])  # make array of unique trial numbers
        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stopsdata_b, trialids_b)  # get average real stops & shuffled stops per lcoation bin
        zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
        firststopstorebeac3[dcount, session_count, :, 1] = zscore_b  # store zscores

        if stopsdata_nb.size > 0:  # if there is non-beaconed data
            trialids_nb = np.unique(stopsdata_nb[:, 2])  # make array of unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stopsdata_nb, trialids_nb)  # get average real stops & shuffled stops per lcoation bin
            zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
            firststopstorenbeac3[dcount, session_count, :, 1] = zscore_nb  # store zscores

        if stopsdata_p.size > 0:  # if there is probe data
            trialids_p = np.unique(stopsdata_p[:, 2])  # make array of unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stopsdata_p, trialids_p)  # get average real stops & shuffled stops per lcoation bin
            zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
            firststopstoreprobe3[dcount, session_count, :, 1] = zscore_p  # store zscores
            session_count += 1

# stack experiments for short trials then average over days then session_counts
con_b_s = np.nanmean(((firststopstorebeac[0, :, :,1])), axis=0)
con_nb_s = np.nanmean(((firststopstorenbeac[0, :, :,1])), axis=0)
con_p_s = np.nanmean(((firststopstoreprobe[0,:, :, 1])), axis=0)
sd_con_b_s = np.nanstd(((firststopstorebeac[0, :, :,1])), axis=0)/math.sqrt(session_count)
sd_con_nb_s = np.nanstd(((firststopstorenbeac[0, :, :,1])), axis=0)/ math.sqrt(session_count)
sd_con_p_s = np.nanstd(((firststopstoreprobe[0, :, :,1])), axis=0)/ math.sqrt(session_count)

# stack experiments for medium trials then average over days then mice
con_b_m = np.nanmean(((firststopstorebeac2[0, :, :,1])), axis=0)
con_nb_m = np.nanmean(((firststopstorenbeac2[0, :, :,1])), axis=0)
con_p_m = np.nanmean(((firststopstoreprobe2[0,:, :, 1])), axis=0)
sd_con_b_m = np.nanstd(((firststopstorebeac2[0, :, :,1])), axis=0)/math.sqrt(session_count)
sd_con_nb_m = np.nanstd(((firststopstorenbeac2[0, :, :,1])), axis=0)/ math.sqrt(session_count)
sd_con_p_m = np.nanstd(((firststopstoreprobe2[0, :, :,1])), axis=0)/ math.sqrt(session_count)

# stack experiments for long trials then average over days then mice
con_b_l = np.nanmean(((firststopstorebeac3[0, :, :,1])), axis=0)
con_nb_l = np.nanmean(((firststopstorenbeac3[0, :, :,1])), axis=0)
con_p_l = np.nanmean(((firststopstoreprobe3[0,:, :, 1])), axis=0)
sd_con_b_l = np.nanstd(((firststopstorebeac3[0, :, :,1])), axis=0)/math.sqrt(session_count)
sd_con_nb_l = np.nanstd(((firststopstorenbeac3[0, :, :,1])), axis=0)/ math.sqrt(session_count)
sd_con_p_l = np.nanstd(((firststopstoreprobe3[0, :, :,1])), axis=0)/ math.sqrt(session_count)


# PLOT GRAPHS

bins = np.arange(0.5,n_bins+0.5,1) # track bins # track bins

fig = plt.figure(figsize = (13,5))
ax = fig.add_subplot(1,3,1)
ax.set_title('Small', fontsize=20, verticalalignment='bottom', style='italic')  # title
ax.axvspan(rz_start_s, rz_end_s, facecolor='g', alpha=0.25, hatch='/',linewidth=0)  # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, track_start, facecolor='k', alpha=0.15, hatch='/', linewidth=0)  # black box
ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, hatch='/', linewidth=0)  # black box
ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
ax.axhline(-10, linewidth=3, color='black')  # bold line on the x axis
ax.axhline(0, linewidth=1, ls='--', color='black')  # bold line on the x axis

ax.plot(bins,con_b_s,color = 'blue',label = 'Beaconed', linewidth = 2) #plot beaconed trials
ax.fill_between(bins , con_b_s-sd_con_b_s, con_b_s+sd_con_b_s, facecolor = 'blue', alpha = 0.3)
ax.plot(bins, con_p_s, '', color='red', label='Probe', linewidth=2)  # plot probe trials
ax.fill_between(bins, con_p_s-sd_con_p_s, con_p_s+sd_con_p_s, facecolor='red', alpha=0.3)

ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.set_xlim(0, 20)
ax.set_ylim(-10, 30)
adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
makelegend(fig, ax)
ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
ax.locator_params(axis='y', nbins=5)  # set number of ticks on y axis
ax.set_xticklabels(['0', '10', '20'])
ax.set_yticklabels(['-10', '0', '10', '20', '30'])
ax = plt.ylabel('Z-Score', fontsize=16, labelpad=18)


ax = fig.add_subplot(1,3,2)
ax.set_title('Medium', fontsize=20, verticalalignment='bottom', style='italic')  # title
ax.axvspan(rz_start_m, rz_end_m, facecolor='g', alpha=0.25, hatch='/',linewidth=0)  # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, track_start, facecolor='k', alpha=0.15, hatch='/', linewidth=0)  # black box
ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, hatch='/', linewidth=0)  # black box
ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
ax.axhline(-10, linewidth=3, color='black')  # bold line on the x axis
ax.axhline(0, linewidth=1, ls='--', color='black')  # bold line on the x axis

ax.plot(bins,con_b_m,color = 'blue',label = 'Beaconed', linewidth = 2) #plot beaconed trials
ax.fill_between(bins, con_b_m-sd_con_b_m, con_b_m+sd_con_b_m, facecolor = 'blue', alpha = 0.3)
ax.plot(bins, con_p_m, '', color='red', label='Probe', linewidth=2)  # plot probe trials
ax.fill_between(bins, con_p_m-sd_con_p_m, con_p_m+sd_con_p_m, facecolor='red', alpha=0.3)

ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.set_xlim(0, 20)
ax.set_ylim(-10, 30)
adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
ax.locator_params(axis='y', nbins=5)  # set number of ticks on y axis
ax.set_xticklabels(['0', '10', '20'])
ax.set_yticklabels(['', '', '', '', ''])
ax = plt.xlabel('Location (VU)', fontsize=16, labelpad=18)


ax = fig.add_subplot(1,3,3)
ax.set_title('Long', fontsize=20, verticalalignment='bottom', style='italic')  # title
ax.axvspan(rz_start_l, rz_end_l, facecolor='g', alpha=0.25, hatch='/',linewidth=0)  # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, track_start, facecolor='k', alpha=0.15, hatch='/', linewidth=0)  # black box
ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, hatch='/', linewidth=0)  # black box
ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
ax.axhline(-10, linewidth=3, color='black')  # bold line on the x axis
ax.axhline(0, linewidth=1, ls='--', color='black')  # bold line on the x axis

ax.plot(bins,con_b_l,color = 'blue',label = 'Beaconed', linewidth = 2) #plot beaconed trials
ax.fill_between(bins, con_b_l-sd_con_b_l, con_b_l+sd_con_b_l, facecolor = 'blue', alpha = 0.3)
ax.plot(bins, con_p_l, '', color='red', label='Probe', linewidth=2)  # plot probe trials
ax.fill_between(bins, con_p_l-sd_con_p_l, con_p_l+sd_con_p_l, facecolor='red', alpha=0.3)

ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.set_xlim(0, 20)
ax.set_ylim(-10, 30)
adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
ax.locator_params(axis='y', nbins=5)  # set number of ticks on y axis
ax.set_xticklabels(['0', '10', '20'])
ax.set_yticklabels(['', '', '', '', ''])

plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.25, left = 0.15, right = 0.82, top = 0.85)


fig.savefig('/Users/emmamather-pike/PycharmProjects/data/plots/Fig1F2_S_ZScoreHist_BP'+'.png', dpi = 200)
plt.close()




