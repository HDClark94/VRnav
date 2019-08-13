"""


# Calculates Zscores for each location bin along the track for increasing track lengths


"""

# import packages and functions
from Functions_Core_0100 import extractstops, filterstops, create_srdata, makebinarray, speed_per_trial, makelegend, \
    makelegend2, makelegend3, makelegend4, shuffle_analysis_pertrial3, z_score1, shuffle_analysis_pertrial_tracks, \
    adjust_spines, makelegend2, readhdfdata, maketrialarray
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
from scipy.stats import uniform
import random
from summarize.map2legacy import *

#LOAD DATA FOR SMALL TRACKS
#CONSTANT SPEED PARTICIPANTS
session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190812133404/S001',
                 '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190812150450/S001',
                 '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190813133909/S001']
                 #'/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190731113135/S001',
                 #'/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190731153240/S001']


session_paths = ['/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Harry/OculusVR/TrenchRunV2.2_disparate/short/P_190812150450/S001',
                 '/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Harry/OculusVR/TrenchRunV2.2_disparate/short/P_190812133404/S001']
                 #'/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Harry/OculusVR/TrenchRunV2.2_disparate/short/P_190813133909/S001',
                 #'/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Harry/OculusVR/TrenchRunV2.2_disparate/short/P_190809155704/S001']

#GAIN SPEED PARTICIPANTS

days = [1]
bin_length_s = 20

# empty arrays for storing data
track_s_b = np.zeros((len(session_paths), len(days), bin_length_s)) #number need changing 
track_s_nb = np.zeros((len(session_paths), len(days), bin_length_s)) #number need changing 
track_s_p = np.zeros((len(session_paths), len(days), bin_length_s)) # number need changing 
track_s_b[:, :, :] = np.nan
track_s_nb[:, :, :] = np.nan
track_s_p[:, :, :] = np.nan

for dcount, day in enumerate(days):  # load mouse and day
    session_count = 0
    for session_path in session_paths:
        saraharray, track_start, track_end = translate_to_legacy_format(session_path)

        # define track length parameters
        rz_start_s = saraharray[0, 11]
        rz_end_s = saraharray[0, 12]
        tracklength_s = track_end - track_start  # tracklength on HDF5 file
        binmin = 0
        binmax = tracklength_s
        interval = tracklength_s / 20  # make location bins
        bins = np.arange(binmin, binmax + 1e-6, interval)  # make location bins

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
        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks(stopsdata_b,trialids_b, bin_length_s) # get average real stops & shuffled stops per lcoation bin
        zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
        track_s_b[session_count, dcount, :] = zscore_b  # store zscores

        if stopsdata_nb.size > 0:  # if there is non-beaconed data
            trialids_nb = np.unique(stopsdata_nb[:, 2])  # make array of unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks(stopsdata_nb,trialids_nb, bin_length_s)  # get average real stops & shuffled stops per lcoation bin
            zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
            track_s_nb[session_count, dcount, :] = zscore_nb  # store zscores

        if stopsdata_p.size > 0:  # if there is probe data
            trialids_p = np.unique(stopsdata_p[:, 2])  # make array of unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks(stopsdata_p,trialids_p, bin_length_s)  # get average real stops & shuffled stops per lcoation bin
            zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
            track_s_p[session_count, dcount, :] = zscore_p  # store zscores
            session_count += 1


#LOAD DATA FOR MEDIUM TRACKS
#CONSTANT SPEED PARTICIPANTS
session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190812133404/S001',
                '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190812150450/S001']
                #'/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190813133909/S001'
                #'/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190731113135/S001',
                #'/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190731153240/S001']

session_paths = ['/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Harry/OculusVR/TrenchRunV2.2_disparate/medium/P_190809155704/S001',
                 '/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Harry/OculusVR/TrenchRunV2.2_disparate/medium/P_190731153240/S001']

#GAIN SPEED PARTICIPANTS

days = [1]
bin_length_m = 24  # approximation of number of bins proportionate to s track length size ie. np.round(track_length_m/track_length_s *bin_length_s)


# empty arrays for storing data
track_m_b =  np.zeros((len(session_paths), len(days), bin_length_m))
track_m_nb = np.zeros((len(session_paths), len(days), bin_length_m))
track_m_p =  np.zeros((len(session_paths), len(days), bin_length_m))
track_m_b[:, :, :] = np.nan
track_m_nb[:, :, :] = np.nan
track_m_p[:, :, :] = np.nan

for dcount, day in enumerate(days):  # load mouse and day
    session_count = 0
    for session_path in session_paths:
        saraharray, track_start, track_end = translate_to_legacy_format(session_path)

        # define track length parameters
        rz_start_m = saraharray[0, 11]
        rz_end_m = saraharray[0, 12]
        tracklength_m = track_end - track_start  # tracklength on HDF5 file
        binmin = 0
        binmax = tracklength_m
        interval = tracklength_m / 20  # make location bins
        bins = np.arange(binmin, binmax + 1e-6, interval)  # make location bins

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
        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks(stopsdata_b,trialids_b, bin_length_m)  # get average real stops & shuffled stops per lcoation bin
        zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
        track_m_b[session_count, dcount, :] = zscore_b  # store zscores

        if stopsdata_nb.size > 0:  # if there is non-beaconed data
            trialids_nb = np.unique(stopsdata_nb[:, 2])  # make array of unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks(stopsdata_b,trialids_b, bin_length_m)  # get average real stops & shuffled stops per lcoation bin
            zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
            track_m_nb[session_count, dcount, :] = zscore_nb  # store zscores

        if stopsdata_p.size > 0:  # if there is probe data
            trialids_p = np.unique(stopsdata_p[:, 2])  # make array of unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks(stopsdata_b,trialids_b, bin_length_m)  # get average real stops & shuffled stops per lcoation bin
            zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
            track_m_p[session_count, dcount, :] = zscore_p  # store zscores
            session_count += 1

#LOAD DATA FOR LONG TRACKS
#CONSTANT SPEED PARTICIPANTS


session_paths = ['/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190812133404/S001',
                 '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190812150450/S001',
                 '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190813133909/S001']
                 #'/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190731113135/S001',
                 #'/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190731153240/S001']

session_paths = ['/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Harry/OculusVR/TrenchRunV2.2_disparate/long/P_190809155704/S001',
                 '/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Harry/OculusVR/TrenchRunV2.2_disparate/long/P_190731153240/S001']


days = [1]
bin_length_l = 31 # approximation of number of bins proportionate to s track length size ie. np.round(tracklength_l/track_length_s *bin_length_s)

# empty arrays for storing data
track_l_b =  np.zeros((len(session_paths), len(days), bin_length_l))
track_l_nb = np.zeros((len(session_paths), len(days), bin_length_l))
track_l_p =  np.zeros((len(session_paths), len(days), bin_length_l))
track_l_b[:, :, :] = np.nan
track_l_nb[:, :, :] = np.nan
track_l_p[:, :, :] = np.nan

for dcount, day in enumerate(days):  # load mouse and day
    session_count = 0
    for session_path in session_paths:
        saraharray, track_start, track_end = translate_to_legacy_format(session_path)


        # define track length parameters
        rz_start_l = saraharray[0, 11]
        rz_end_l = saraharray[0, 12]
        tracklength_l = track_end - track_start  # tracklength on HDF5 file
        binmin = 0
        binmax = tracklength_l
        interval = tracklength_l / 20  # make location bins
        bins = np.arange(binmin, binmax + 1e-6, interval)  # make location bins

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


        if stopsdata_nb.size > 0:  # if there is non-beaconed data
            trialids_nb = np.unique(stopsdata_nb[:, 2])  # make array of unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks(stopsdata_b,
                                                                                                  trialids_b,
                                                                                                  bin_length_l)  # get average real stops & shuffled stops per lcoation bin
            zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
            track_l_nb[session_count, dcount, :] = zscore_nb  # store zscores

        if stopsdata_p.size > 0:  # if there is probe data
            trialids_p = np.unique(stopsdata_p[:, 2])  # make array of unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks(stopsdata_b,
                                                                                                  trialids_b,
                                                                                                  bin_length_l)  # get average real stops & shuffled stops per lcoation bin
            zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)  # calculate z-scores
            track_l_p[session_count, dcount, :] = zscore_p  # store zscores
            session_count += 1


# stack experiments then average

tracks_s_b = np.nanmean((track_s_b[0, :, :, 1]), axis=0)
tracks_s_nb = np.nanmean((track_s_nb[0, :, :, 1]), axis=0)
tracks_s_p = np.nanmean((track_s_p[0, :, :, 1]), axis=0)

sdtracks_s_b = np.nanmean((track_s_b[0, :, :, 1]), axis=0)/math.sqrt(session_count)
sdtracks_s_nb = np.nanmean((track_s_nb[0, :, :, 1]), axis=0)/math.sqrt(session_count)
sdtracks_s_p = np.nanmean((track_s_p[0, :, :, 1]), axis=0)/math.sqrt(session_count)

tracks_m_b = np.nanmean(((track_m_b[0, :, :, 1])), axis=0)
tracks_m_nb = np.nanmean(((track_m_nb[0, :, :, 1])), axis=0)
tracks_m_p = np.nanmean(((track_m_p[0, :, :, 1])), axis=0)

sdtracks_m_b = np.nanmean(((track_m_b[0, :, :, 1])), axis=0)/math.sqrt(session_count)
sdtracks_m_nb = np.nanmean(((track_m_nb[0, :, :, 1])), axis=0)/math.sqrt(session_count)
sdtracks_m_p = np.nanmean(((track_m_p[0, :, :, 1])), axis=0)/math.sqrt(session_count)

tracks_l_b = np.nanmean(((track_l_b[0, :, :, 1])), axis=0)
tracks_l_nb = np.nanmean(((track_l_nb[0, :, :, 1])), axis=0)
tracks_l_p = np.nanmean(((track_l_p[0, :, :, 1])), axis=0)

sdtracks_l_b = np.nanmean(((track_l_p[0, :, :, 1])), axis=0)/math.sqrt(session_count)
sdtracks_l_nb = np.nanmean(((track_l_nb[0, :, :, 1])), axis=0)/math.sqrt(session_count)
sdtracks_l_p = np.nanmean(((track_l_p[0, :, :, 1])), axis=0)/math.sqrt(session_count)


tracks1_b = np.nanmean(np.vstack((np.nanmean(track1_b, axis=1), np.nanmean(track1_b1, axis=1))), axis=0)
tracks1_nb = np.nanmean(np.vstack((np.nanmean(track1_nb, axis=1), np.nanmean(track1_nb1, axis=1))), axis=0)
tracks1_p = np.nanmean(np.vstack((np.nanmean(track1_p, axis=1), np.nanmean(track1_p1, axis=1))), axis=0)

sdtracks1_b = np.nanstd(np.vstack((np.nanmean(track1_b, axis=1), np.nanmean(track1_b1, axis=1))), axis=0)
sdtracks1_nb = np.nanstd(np.vstack((np.nanmean(track1_nb, axis=1), np.nanmean(track1_nb1, axis=1))), axis=0)
sdtracks1_p = np.nanstd(np.vstack((np.nanmean(track1_p, axis=1), np.nanmean(track1_p1, axis=1))), axis=0)

tracks2_b = np.nanmean(np.vstack((np.nanmean(track2_b, axis=1), np.nanmean(track2_b1, axis=1))), axis=0)
tracks2_nb = np.nanmean(np.vstack((np.nanmean(track2_nb, axis=1), np.nanmean(track2_nb1, axis=1))), axis=0)
tracks2_p = np.nanmean(np.vstack((np.nanmean(track2_p, axis=1), np.nanmean(track2_p1, axis=1))), axis=0)

sdtracks2_b = np.nanstd(np.vstack((np.nanmean(track2_b, axis=1), np.nanmean(track2_b1, axis=1))), axis=0)
sdtracks2_nb = np.nanstd(np.vstack((np.nanmean(track2_nb, axis=1), np.nanmean(track2_nb1, axis=1))), axis=0)
sdtracks2_p = np.nanstd(np.vstack((np.nanmean(track2_p, axis=1), np.nanmean(track2_p1, axis=1))), axis=0)

tracks3_b = np.nanmean(np.vstack((np.nanmean(track3_b, axis=1), np.nanmean(track3_b1, axis=1))), axis=0)
tracks3_nb = np.nanmean(np.vstack((np.nanmean(track3_nb, axis=1), np.nanmean(track3_nb1, axis=1))), axis=0)
tracks3_p = np.nanmean(np.vstack((np.nanmean(track3_p, axis=1), np.nanmean(track3_p1, axis=1))), axis=0)

sdtracks3_b = np.nanstd(np.vstack((np.nanmean(track3_b, axis=1), np.nanmean(track3_b1, axis=1))), axis=0)
sdtracks3_nb = np.nanstd(np.vstack((np.nanmean(track3_nb, axis=1), np.nanmean(track3_nb1, axis=1))), axis=0)
sdtracks3_p = np.nanstd(np.vstack((np.nanmean(track3_p, axis=1), np.nanmean(track3_p1, axis=1))), axis=0)



# plot data

bins1 = np.arange(0.5, 20.5, 1)

fig = plt.figure(figsize=(12, 3))  # make figure, this shape (width, height)
ax = fig.add_subplot(1, 3, 1)
ax.axvspan(rz_start_s, rz_end_s, facecolor='g', alpha=0.2, hatch='/', linewidth=0)  # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, track_start, facecolor='k', alpha=0.1, hatch='/', linewidth=0)  # black box
ax.axvspan(track_end, tracklength_s, facecolor='k', alpha=0.1, hatch='/', linewidth=0)  # black box
ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
ax.axhline(0, linewidth=3, color='black')  # bold line on the x axis
ax.plot(bins1, tracks_s_b, color='Black', label='Track 1')  # plot becaoned trials
ax.fill_between(bins1, tracks_s_b - sdtracks_s_b, tracks_s_b + sdtracks_s_b, facecolor='Black', alpha=0.3)
ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.set_xlim(0, tracklength_s)
ax.set_ylim(0, 1)
adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
ax.locator_params(axis='x', nbins=6)  # set number of ticks on x axis
ax.locator_params(axis='y', nbins=2)  # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
#ax.set_ylabel('Zscore', fontsize=16, labelpad=18)

ax = fig.add_subplot(1, 3, 2)
ax.axvspan(8.8, 8.8 + 2.2, facecolor='g', alpha=0.2, hatch='/',
           linewidth=0)  # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch='/', linewidth=0)  # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch='/', linewidth=0)  # black box
ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
ax.axhline(0, linewidth=3, color='black')  # bold line on the x axis
ax.plot(bins1, tracks1_nb, color='Black', label='Track 1')  # plot becaoned trials
ax.fill_between(bins1, tracks1_nb - sdtracks1_nb, tracks1_nb + sdtracks1_nb, facecolor='Black', alpha=0.3)
ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.set_xlim(0, 20)
# ax.set_ylim(0,1)
adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
ax.locator_params(axis='x', nbins=6)  # set number of ticks on x axis
ax.locator_params(axis='y', nbins=2)  # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1, 3, 3)
ax.axvspan(8.8, 8.8 + 2.2, facecolor='g', alpha=0.25, hatch='/',
           linewidth=0)  # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch='/', linewidth=0)  # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.15, hatch='/', linewidth=0)  # black box
ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
ax.axhline(-0.25, linewidth=3, color='black')  # bold line on the x axis
ax.plot(bins1, tracks1_b, color='blue', label='Track 1')  # plot becaoned trials
ax.fill_between(bins1, tracks1_b - sdtracks1_b, tracks1_b + sdtracks1_b, facecolor='blue', alpha=0.3)
ax.plot(bins1, tracks1_p, color='red', label='Track 1')  # plot becaoned trials
ax.fill_between(bins1, tracks1_p - sdtracks1_p, tracks1_p + sdtracks1_p, facecolor='red', alpha=0.3)
ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.set_xlim(0, 20)
ax.set_ylim(-0.25, 0.65)
adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
ax.locator_params(axis='x', nbins=6)  # set number of ticks on x axis
ax.locator_params(axis='y', nbins=6)  # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.15, left=0.07, right=0.82, top=0.92)
fig.savefig('Plots/Figure3/Task18_StopHist_Tracks1' + '_0200.png', dpi=200)
plt.close()

bins1 = np.arange(0.5, 24.5, 1)
fig = plt.figure(figsize=(13.8, 3))  # make figure, this shape (width, height)
ax = fig.add_subplot(1, 3, 1)
ax.axvspan(10.261, 10.261 + 1.91, facecolor='g', alpha=0.2, hatch='/',
           linewidth=0)  # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 2.6, facecolor='k', alpha=0.2, hatch='/', linewidth=0)  # black box
ax.axvspan(20 - 2.6, 20, facecolor='k', alpha=0.2, hatch='/', linewidth=0)  # black box
ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
ax.axhline(0, linewidth=3, color='black')  # bold line on the x axis
ax.plot(bins1, tracks2_b, color='Black', label='Track 2')  # plot becaoned trials
ax.fill_between(bins1, tracks2_b - sdtracks2_b, tracks2_b + sdtracks2_b, facecolor='Black', alpha=0.3)
ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.set_xlim(0, 20)
adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
ax.locator_params(axis='x', nbins=6)  # set number of ticks on x axis
ax.locator_params(axis='y', nbins=2)  # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_ylabel('Stops (cm)', fontsize=16, labelpad=18)

ax = fig.add_subplot(1, 3, 2)
ax.axvspan(10.261, 10.261 + 1.91, facecolor='g', alpha=0.2, hatch='/',
           linewidth=0)  # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 2.6, facecolor='k', alpha=0.2, hatch='/', linewidth=0)  # black box
ax.axvspan(20 - 2.6, 20, facecolor='k', alpha=0.2, hatch='/', linewidth=0)  # black box
ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
ax.axhline(0, linewidth=3, color='black')  # bold line on the x axis
ax.plot(bins1, tracks2_nb, color='Black', label='Track 2')  # plot becaoned trials
ax.fill_between(bins1, tracks2_nb - sdtracks2_nb, tracks2_nb + sdtracks2_nb, facecolor='Black', alpha=0.3)
ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.set_xlim(0, 20)
adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
ax.locator_params(axis='x', nbins=6)  # set number of ticks on x axis
ax.locator_params(axis='y', nbins=2)  # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1, 3, 3)
ax.axvspan(10.261, 10.261 + 1.91, facecolor='g', alpha=0.2, hatch='/',
           linewidth=0)  # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 2.6, facecolor='k', alpha=0.2, hatch='/', linewidth=0)  # black box
ax.axvspan(20 - 2.6, 20, facecolor='k', alpha=0.2, hatch='/', linewidth=0)  # black box
ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
ax.plot(bins1, tracks2_b, color='blue', label='Track 2')  # plot becaoned trials
ax.fill_between(bins1, tracks2_b - sdtracks2_b, tracks2_b + sdtracks2_b, facecolor='blue', alpha=0.3)
ax.plot(bins1, tracks2_p, color='red', label='Track 2')  # plot becaoned trials
ax.fill_between(bins1, tracks2_p - sdtracks2_p, tracks2_p + sdtracks2_p, facecolor='red', alpha=0.3)
ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.set_xlim(0, 23)
adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
ax.locator_params(axis='x', nbins=6)  # set number of ticks on x axis
ax.locator_params(axis='y', nbins=2)  # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.15, left=0.07, right=0.82, top=0.92)
fig.savefig('Plots/Figure3/Task18_StopHist_Tracks2' + '_0200.png', dpi=200)
plt.close()

bins1 = np.arange(0.5, 31.5, 1)
fig = plt.figure(figsize=(16.5, 3))  # make figure, this shape (width, height)
ax = fig.add_subplot(1, 3, 1)
ax.axvspan(11.241, 11.241 + 1.517, facecolor='g', alpha=0.2, hatch='/',
           linewidth=0)  # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 2.1, facecolor='k', alpha=0.2, hatch='/', linewidth=0)  # black box
ax.axvspan(20 - 2.1, 20, facecolor='k', alpha=0.2, hatch='/', linewidth=0)  # black box
ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
ax.axhline(0, linewidth=3, color='black')  # bold line on the x axis
ax.plot(bins1, tracks3_b, color='Black', label='Track 3')  # plot becaoned trials
ax.fill_between(bins1, tracks3_b - sdtracks3_b, tracks3_b + sdtracks3_b, facecolor='Black', alpha=0.3)
ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.set_xlim(0, 20)
adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
ax.locator_params(axis='x', nbins=6)  # set number of ticks on x axis
ax.locator_params(axis='y', nbins=2)  # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_ylabel('Stops (cm)', fontsize=16, labelpad=18)

ax = fig.add_subplot(1, 3, 2)
ax.axvspan(11.241, 11.241 + 1.517, facecolor='g', alpha=0.2, hatch='/',
           linewidth=0)  # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 2.1, facecolor='k', alpha=0.2, hatch='/', linewidth=0)  # black box
ax.axvspan(20 - 2.1, 20, facecolor='k', alpha=0.2, hatch='/', linewidth=0)  # black box
ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
ax.axhline(0, linewidth=3, color='black')  # bold line on the x axis
ax.plot(bins1, tracks3_nb, color='Black', label='Track 3')  # plot becaoned trials
ax.fill_between(bins1, tracks3_nb - sdtracks3_nb, tracks3_nb + sdtracks3_nb, facecolor='Black', alpha=0.3)
ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.set_xlim(0, 20)
# ax.set_ylim(0,1)
adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
ax.locator_params(axis='x', nbins=6)  # set number of ticks on x axis
ax.locator_params(axis='y', nbins=2)  # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1, 3, 3)
ax.axvspan(16.3, 16.3 + 2.2, facecolor='g', alpha=0.25, hatch='/',
           linewidth=0)  # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch='/', linewidth=0)  # black box
ax.axvspan(29 - 3, 29, facecolor='k', alpha=0.15, hatch='/', linewidth=0)  # black box
ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
ax.axhline(-0.25, linewidth=3, color='black')  # bold line on the x axis
ax.plot(bins1, tracks3_b, color='blue', label='Track 4')  # plot becaoned trials
ax.fill_between(bins1, tracks3_b - sdtracks3_b, tracks3_b + sdtracks3_b, facecolor='blue', alpha=0.3)
ax.plot(bins1, tracks3_p, color='red', label='Track 4')  # plot becaoned trials
ax.fill_between(bins1, tracks3_p - sdtracks3_p, tracks3_p + sdtracks3_p, facecolor='red', alpha=0.3)
ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=16)
ax.set_xlim(0, 29)
ax.set_ylim(-0.25, 0.65)
adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
ax.locator_params(axis='x', nbins=6)  # set number of ticks on x axis
ax.locator_params(axis='y', nbins=6)  # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.15, left=0.07, right=0.82, top=0.92)
fig.savefig('Plots/Figure3/Task18_StopHist_Tracks3' + '_0200.png', dpi=200)
plt.close()


fig.savefig('/Users/emmamather-pike/PycharmProjects/data/plots/Fig1F2_S_ZScoreHist_BP'+'.png' + '_.png', dpi=200)
plt.close()

# save data for R

