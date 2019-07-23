"""

# PLOTS EXAMPLE TRAINING SESSIONS

For each mouse and day specified, the location of stops for each trial is calculated. From this, average stops per 10 cm location bins is plotted. In addition, the average speed along the track is plotted. One figure is produced for each mouse and day specified.

"""
# Import functions and packages
from _Tennantetal2018.Functions_Core_0100 import extractstops, extractstops_HUMAN, filterstops, create_srdata, makebinarray, speed_per_trial, makelegend, \
    makelegend2, makelegend3, makelegend4, shuffle_analysis_pertrial3, extractrewards, adjust_spines, readhdfdata, \
    FirstStops, maketrialarray
from summarize.map2legacy import *
import numpy as np

def plot_fig1C(session_path):

    bins = np.arange(0.5, 20.5, 1)  # array of bins for location

    saraharray, track_start, track_end = translate_to_legacy_format(session_path)

    rz_start = saraharray[0, 11]
    rz_end = saraharray[0, 12]

    trialno = np.max(saraharray[:, 9])  # total number of trials for that day and mouse (used later for defining y axis max)


    # Extract data for beaconed, non-beaconed, probe
    dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0),
                             0)  # delete all data not on beaconed tracks
    dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10),
                              0)  # delete all data not on non beaconed tracks
    dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)  # delete all data not on probe tracks

    # extract stops
    stopsdata_b = extractstops_HUMAN(dailymouse_b)
    stopsdata_nb = extractstops_HUMAN(dailymouse_nb)
    stopsdata_p = extractstops_HUMAN(dailymouse_p)

    # get location and trial number of rewards
    reward_beac = extractrewards(dailymouse_b)
    reward_nbeac = extractrewards(dailymouse_nb)

    # filter stops (removes stops 0.5 cm after a stop)
    stopsdata_b = filterstops(stopsdata_b)
    stopsdata_nb = filterstops(stopsdata_nb)
    stopsdata_p = filterstops(stopsdata_p)

    # extract trial numbers from data (only unique ones)
    trialids_b = np.unique(stopsdata_b[:, 2])
    trialids_nb = np.unique(stopsdata_nb[:, 2])
    if stopsdata_p.size > 0:  # if there are probe trials
        trialids_p = np.unique(stopsdata_p[:, 2])

    # get mean stops per bin for real and shuffled data
    srbin_mean_b, srbin_std_b, shuffled_mean_b, shuffled_std_b = shuffle_analysis_pertrial3(stopsdata_b, trialids_b)
    srbin_mean_nb, srbin_std_nb, shuffled_mean_nb, shuffled_std_nb = shuffle_analysis_pertrial3(stopsdata_nb,
                                                                                                trialids_nb)
    if stopsdata_p.size > 0:
        srbin_mean_p, srbin_std_p, shuffled_mean_p, shuffled_std_p = shuffle_analysis_pertrial3(stopsdata_p,
                                                                                                trialids_p)

    # calculate average speed
    speed_beaconed = speed_per_trial(bins, saraharray, trialids_b)
    speed_nbeaconed = speed_per_trial(bins, saraharray, trialids_nb)
    if stopsdata_p.size > 0:  # if there are probe trials
        speed_probe = speed_per_trial(bins, saraharray, trialids_p)
        sd_speed_probe = np.nanstd(speed_probe, axis=1)
    sd_speed_beaconed = np.nanstd(speed_beaconed, axis=1)
    sd_speed_nbeaconed = np.nanstd(speed_nbeaconed, axis=1)
    speed_beaconed = np.nanmean(speed_beaconed, axis=1)
    speed_nbeaconed = np.nanmean(speed_nbeaconed, axis=1)
    if stopsdata_p.size > 0:  # if there are probe trials
        speed_probe = np.nanmean(speed_probe, axis=1)

    # plot graphs:
    if stopsdata_p.size > 0:  # if there are probe trials, plot 3x3 subplots
        # MAKE FIGURE: 1st row: stop rasters. 2nd row: average stops. 3rd row: speed. Columns are beaconed/non-beaconed/probe
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(3, 3, 1)  # stops per trial
        ax.set_title('Beaconed trials', fontsize=22, verticalalignment='bottom', style='italic')
        ax.axvspan(rz_start, rz_end, facecolor='DarkGreen', alpha=0.25,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start, facecolor='k', linewidth=0, alpha=0.15)  # black box
        ax.axvspan(track_end, 20, facecolor='k', linewidth=0, alpha=0.15)  # black box
        ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
        ax.axhline(0, linewidth=3, color='black')  # bold line on the x axis
        try:
            ax.plot(stopsdata_b[:, 0], stopsdata_b[:, 2], 'o', color='Black',
                    markersize=4.5)  # plot becaoned trials
        except IndexError:
            print("Error exception because theres no stops to plot")

        try:
            ax.plot(reward_beac[:, 0], reward_beac[:, 2], '>', color='Red', markersize=6)  # plot becaoned reward trials
        except IndexError:
            print("Error exception because theres no rewards to plot")
        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, trialno + 0.5)
        adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
        makelegend(fig, ax)  # make legend
        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=4)  # set number of ticks on y axis
        ax.set_xticklabels(['', '', ''])
        ax.set_ylabel('Trial Number', fontsize=18, labelpad=20)
        #ax.set(ylabel='Trial Number')

        ax = fig.add_subplot(3, 3, 2)  # stops per trial
        ax.set_title('Non-beaconed trials', fontsize=22, style='italic', verticalalignment='bottom')
        ax.axvspan(rz_start, rz_end, facecolor='DarkGreen', alpha=0.25,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
        ax.axhline(0, linewidth=3, color='black')  # bold line on the x axis
        try:
            ax.plot(stopsdata_nb[:, 0], stopsdata_nb[:, 2], 'o', color='Black',
                    markersize=4.5)  # plot becaoned trials
        except IndexError:
            print("Error exception because theres no stops to plot")

        try:
            ax.plot(reward_nbeac[:, 0], reward_nbeac[:, 2], '>', color='Red', markersize=6)  # plot becaoned trials
        except IndexError:
            print("Error exception because theres no rewards to plot")

        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, trialno + 0.5)
        adjust_spines(ax, ['left', 'bottom'])  # re;moves top and right spines
        makelegend(fig, ax)  # makes legend
        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=4)  # set number of ticks on y axis
        ax.set_xticklabels(['', '', ''])
        ax.set_yticklabels(['', '', ''])

        ax = fig.add_subplot(3, 3, 3)  # stops per trial
        ax.set_title('Probe trials', fontsize=22, style='italic', verticalalignment='bottom')
        ax.axvspan(rz_start, rz_end, facecolor='DarkGreen', alpha=0.25,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
        ax.axhline(0, linewidth=3, color='black')  # bold line on the x axis
        try:
            ax.plot(stopsdata_p[:, 0], stopsdata_p[:, 2], 'o', color='Black', markersize=4.5,
                    label='Non - beaconed')  # plot becaoned trials
        except IndexError:
            print("error as theres no stops to plots!")

        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, trialno + 0.5)
        adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=4)  # set number of ticks on y axis
        ax.set_xticklabels(['', '', ''])
        ax.set_yticklabels(['', '', ''])

        ax = fig.add_subplot(3, 3, 4)
        ax.axvspan(rz_start, rz_end, facecolor='DarkGreen', alpha=0.25,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvline(0, linewidth=3, color='black')
        ax.axhline(-0.05, linewidth=3, color='black')
        ax.plot(bins, srbin_mean_b, color='red', linewidth=2)  # plot becaoned trials
        ax.fill_between(bins, srbin_mean_b - srbin_std_b, srbin_mean_b + srbin_std_b, facecolor='red',
                        alpha=0.3)
        ax.plot(bins, shuffled_mean_b, '--', color='Black', linewidth=2)  # plot becaoned trials
        ax.fill_between(bins, shuffled_mean_b - shuffled_std_b, shuffled_mean_b + shuffled_std_b,
                        facecolor='Black', alpha=0.3)
        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.set_xlim(0, 20)
        ax.set_ylim(-0.05, 1.25)
        adjust_spines(ax, ['left', 'bottom'])
        ax.set_ylabel('Stops (VU/Trial)', fontsize=18, labelpad=23)
        #ax.set(ylabel='Stops (VU/Trial)')
        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=4)  # set number of ticks on y axis
        ax.set_xticklabels(['', '', ''])

        # avg stops histogram - non beaconed
        ax = fig.add_subplot(3, 3, 5)
        ax.axvspan(rz_start, rz_end, facecolor='DarkGreen', alpha=0.25,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvline(0, linewidth=3, color='black')
        ax.axhline(-0.05, linewidth=3, color='black')
        ax.plot(bins, srbin_mean_nb, color='red', linewidth=2, label='Real')  # plot becaoned trials
        ax.fill_between(bins, srbin_mean_nb - srbin_std_nb, srbin_mean_nb + srbin_std_nb, facecolor='red',
                        alpha=0.3)
        ax.plot(bins, shuffled_mean_nb, '--', color='Black', linewidth=2,
                label='Shuffled')  # plot becaoned trials
        ax.fill_between(bins, shuffled_mean_nb - shuffled_std_nb, shuffled_mean_nb + shuffled_std_nb,
                        facecolor='Black', alpha=0.3)
        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        adjust_spines(ax, ['left', 'bottom'])
        makelegend2(fig, ax)  # makes legend
        ax.set_xlim(0, 20)
        ax.set_ylim(-0.05, 1.25)
        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=4)  # set number of ticks on y axis
        ax.set_xticklabels(['', '', ''])
        ax.set_yticklabels(['', '', '', ''])

        ax = fig.add_subplot(3, 3, 6)
        ax.axvspan(rz_start, rz_end, facecolor='DarkGreen', alpha=0.25,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axhline(-0.05, linewidth=3, color='black')
        ax.axvline(0, linewidth=3, color='black')
        ax.plot(bins, srbin_mean_p, color='red', linewidth=2)  # plot becaoned trials
        ax.fill_between(bins, srbin_mean_p - srbin_std_p, srbin_mean_p + srbin_std_p, facecolor='red',
                        alpha=0.3)
        ax.plot(bins, shuffled_mean_p, '--', color='Black', linewidth=2)  # plot becaoned trials
        ax.fill_between(bins, shuffled_mean_p - shuffled_std_p, shuffled_mean_p + shuffled_std_p,
                        facecolor='Black', alpha=0.3)
        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        adjust_spines(ax, ['left', 'bottom'])
        ax.set_xlim(0, 20)
        ax.set_ylim(-0.05, 1.25)
        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=4)  # set number of ticks on y axis
        ax.set_xticklabels(['', '', ''])
        ax.set_yticklabels(['', '', '', ''])

        ax = fig.add_subplot(3, 3, 7)
        ax.axvspan(rz_start*5, rz_end*5, facecolor='g', alpha=0.25,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start*5, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvspan(track_end*5, 100, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvline(0, linewidth=3, color='black')
        ax.axhline(0, linewidth=3, color='black')
        ax.plot(bins * 5, speed_beaconed, '-', markersize=2, color='Black', linewidth=1)  # plot becaoned trials
        ax.fill_between(bins * 5, speed_beaconed - sd_speed_beaconed, speed_beaconed + sd_speed_beaconed,
                        facecolor='Black', alpha=0.3)
        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.set_xlim(0, 101)
        ax.set_ylim(0, 4.5)
        adjust_spines(ax, ['left', 'bottom'])
        ax.set_ylabel('Speed (VU/s)', fontsize=18, labelpad=20)
        #ax.set(ylabel='Speed (VU/s)')

        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=3)  # set number of ticks on y axis
        ax.set_xticklabels(['0', '10', '20'])
        ax.set_yticklabels(['0','2','4'])

        # avg stops histogram - non beaconed
        ax = fig.add_subplot(3, 3, 8)
        ax.axvspan(rz_start*5, rz_end*5, facecolor='g', alpha=0.2,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start*5, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvspan(track_end*5, 100, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvline(0, linewidth=3, color='black')
        ax.axhline(0, linewidth=3, color='black')
        ax.plot(bins * 5, speed_nbeaconed, '-', markersize=2, color='Black',
                linewidth=1)  # plot becaoned trials
        ax.fill_between(bins * 5, speed_nbeaconed - sd_speed_nbeaconed, speed_nbeaconed + sd_speed_nbeaconed,
                        facecolor='Black', alpha=0.3)
        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        adjust_spines(ax, ['left', 'bottom'])
        ax.set_xlim(0, 101)
        ax.set_ylim(0, 4.5)
        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=3)  # set number of ticks on y axis
        ax.set_yticklabels(['', '', ''])
        ax.set_xticklabels(['0', '10', '20'])
        ax.set_xlabel('Location (VU)', fontsize=18, labelpad=20)

        ax = fig.add_subplot(3, 3, 9)
        ax.axvspan(rz_start*5, rz_end*5, facecolor='g', alpha=0.3,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start*5, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvspan(track_end*5, 100, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvline(0, linewidth=3, color='black')
        ax.axhline(0, linewidth=3, color='black')
        ax.plot(bins * 5, speed_probe, '-', markersize=2, color='Black', linewidth=1)  # plot becaoned trials
        ax.fill_between(bins * 5, speed_probe - sd_speed_probe, speed_probe + sd_speed_probe, facecolor='Black',
                        alpha=0.3)
        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        adjust_spines(ax, ['left', 'bottom'])
        ax.set_xlim(0, 101)
        ax.set_ylim(0, 4.5)
        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=3)  # set number of ticks on y axis
        ax.set_yticklabels(['', '', ''])
        ax.set_xticklabels(['0', '10', '20'])

        plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.15, left=0.17, right=0.82, top=0.92)
        fig.savefig(session_path+'/ExampleData.png',
                    dpi=200)
        plt.close()

    else:  # if there is no probe trials, plot 2x3 subplots -> stops per trial, average stops, speed for beaconed and non-beaconed trials
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(3, 3, 1)  # stops per trial
        ax.set_title('Beaconed trials', fontsize=18, verticalalignment='bottom', style='italic')
        ax.axvspan(rz_start, rz_end, facecolor='DarkGreen', alpha=0.25,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
        ax.axhline(0, linewidth=3, color='black')  # bold line on the x axis
        try:
            ax.plot(stopsdata_b[:, 0], stopsdata_b[:, 2], 'o', color='Black', markersize=4.5,
                    label='Stop')  # plot beaconed trials
        except IndexError:
            print("no stops to be plotted")

        try:
            ax.plot(reward_beac[:, 0], reward_beac[:, 2], '>', color='Red', markersize=6,
                    label='Reward')  # plot beaconed trials
        except IndexError:
            print("no rewards to be plotted")

        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, trialno + 0.5)
        adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
        ax.set_ylabel('Trial Number', fontsize=18, labelpad=20)

        makelegend3(fig, ax)
        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=4)  # set number of ticks on y axis
        ax.set_xticklabels(['', '', ''])


        ax = fig.add_subplot(3, 3, 2)  # stops per trial
        ax.set_title('Non-beaconed trials', fontsize=18, style='italic', verticalalignment='bottom')
        ax.axvspan(rz_start, rz_end, facecolor='DarkGreen', alpha=0.25,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvspan(track_end, 20, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
        ax.axhline(0, linewidth=3, color='black')  # bold line on the x axis

        try:
            ax.plot(stopsdata_nb[:, 0], stopsdata_nb[:, 2], 'o', color='Black',
                    markersize=4.5)  # plot becaoned trials
        except IndexError:
            print("Error exception because there's no stops to plot")

        try:
            ax.plot(reward_nbeac[:, 0], reward_nbeac[:, 2], '>', color='Red', markersize=6,
                    label='Reward')  # plot becaoned trials
        except IndexError:
            print("Error exception because theres no rewards to plot")

        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, trialno + 0.5)
        adjust_spines(ax, ['left', 'bottom'])  # removes top and right spines
        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=4)  # set xsnumber of ticks on y axis
        ax.set_xticklabels(['', '', ''])
        ax.set_yticklabels(['', '', '',''])

        ax = fig.add_subplot(3, 3, 4)
        ax.axvspan(rz_start * 5, rz_end * 5, facecolor='DarkGreen', alpha=0.25,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start * 5, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvspan(track_end * 5, 100, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvline(0, linewidth=3, color='black')
        ax.axhline(-0.05, linewidth=3, color='black')
        ax.plot(bins * 5, srbin_mean_b, color='red', linewidth=2)  # plot becaoned trials
        ax.fill_between(bins * 5, srbin_mean_b - srbin_std_b, srbin_mean_b + srbin_std_b, facecolor='red',
                        alpha=0.3)
        ax.plot(bins * 5, shuffled_mean_b, '--', color='Black', linewidth=2)  # plot becaoned trials
        ax.fill_between(bins * 5, shuffled_mean_b - shuffled_std_b, shuffled_mean_b + shuffled_std_b,
                        facecolor='Black', alpha=0.3)
        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.set_xlim(0, 101)
        ax.set_ylim(-0.05, 1.25)
        adjust_spines(ax, ['left', 'bottom'])
        ax.set_ylabel('Stops (VU/Trial)', fontsize=18, labelpad=20)
        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=4)  # set number of ticks on y axis
        ax.set_xticklabels(['', '', ''])

        # avg stops histogram - non beaconed
        ax = fig.add_subplot(3, 3, 5)
        ax.axvspan(rz_start * 5, rz_end * 5, facecolor='DarkGreen', alpha=0.25,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start * 5, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvspan(track_end * 5, 100, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvline(0, linewidth=3, color='black')
        ax.axhline(-0.05, linewidth=3, color='black')
        ax.plot(bins * 5, srbin_mean_nb, color='red', linewidth=2, label='Real')  # plot becaoned trials
        ax.fill_between(bins * 5, srbin_mean_nb - srbin_std_nb, srbin_mean_nb + srbin_std_nb, facecolor='red',
                        alpha=0.3)
        ax.plot(bins * 5, shuffled_mean_nb, '--', color='Black', linewidth=2,
                label='Shuffled')  # plot becaoned trials
        ax.fill_between(bins * 5, shuffled_mean_nb - shuffled_std_nb, shuffled_mean_nb + shuffled_std_nb,
                        facecolor='Black', alpha=0.3)
        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        adjust_spines(ax, ['left', 'bottom'])
        ax.set_yticklabels(['', '', ''])
        makelegend4(fig, ax)  # makes legend
        ax.set_xlim(0, 101)
        ax.set_ylim(-0.05, 1.25)
        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=4)  # set number of ticks on y axis
        ax.set_xticklabels(['', '', ''])
        ax.set_yticklabels(['', '', ''])

        ax = fig.add_subplot(3, 3, 7)
        ax.axvspan(rz_start * 5, rz_end * 5, facecolor='g', alpha=0.25,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start * 5, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvspan(track_end * 5, 100, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvline(0, linewidth=3, color='black')
        ax.axhline(0, linewidth=3, color='black')
        ax.plot(bins * 5, speed_beaconed, '-', markersize=2, color='Black', linewidth=1)  # plot becaoned trials
        ax.fill_between(bins * 5, speed_beaconed - sd_speed_beaconed, speed_beaconed + sd_speed_beaconed,
                        facecolor='Black', alpha=0.3)
        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.set_xlim(0, 101)
        ax.set_ylim(0, 4.5)
        adjust_spines(ax, ['left', 'bottom'])
        ax.set_ylabel('Speed (VU/s)', fontsize=18, labelpad=20)
        # ax.set(ylabel='Speed (VU/s)')

        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=3)  # set number of ticks on y axis
        ax.set_xticklabels(['0', '10', '20'])
        ax.set_yticklabels(['0', '2', '4'])
        #ax.set_yticklabels(['0', '1', '2', '3'])

        # avg stops histogram - non beaconed
        ax = fig.add_subplot(3, 3, 8)
        ax.axvspan(rz_start * 5, rz_end * 5, facecolor='g', alpha=0.2,
                   linewidth=0)  # green box spanning the rewardzone to mark reward zone
        ax.axvspan(0, track_start * 5, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvspan(track_end * 5, 100, facecolor='k', alpha=0.15, linewidth=0)  # black box
        ax.axvline(0, linewidth=3, color='black')
        ax.axhline(0, linewidth=3, color='black')
        ax.plot(bins * 5, speed_nbeaconed, '-', markersize=2, color='Black',
                linewidth=1)  # plot becaoned trials
        ax.fill_between(bins * 5, speed_nbeaconed - sd_speed_nbeaconed, speed_nbeaconed + sd_speed_nbeaconed,
                        facecolor='Black', alpha=0.3)
        ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=8,
                       labelsize=18)
        adjust_spines(ax, ['left', 'bottom'])
        ax.set_xlim(0, 101)
        ax.set_ylim(0, 4.5)
        ax.locator_params(axis='x', nbins=3)  # set number of ticks on x axis
        ax.locator_params(axis='y', nbins=3)  # set number of ticks on y axis
        ax.set_yticklabels(['', '', ''])
        ax.set_xticklabels(['0', '10', '20'])
        ax.set_xlabel('Location (VU)', fontsize=18, labelpad=20)

        plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.15, left=0.15, right=0.82, top=0.92)
        fig.savefig(session_path+'/ExampleData.png',
                    dpi=200)
        plt.close()

#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    #session_path = '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_habituation/P_190722150558/S001'
    session_path = '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_short/P_190717113448/S001'
    #session_path = '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_medium/P_190717101640/S001'
    #session_path = '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_long/P_190717101640/S001'
    #session_path = '/Users/emmamather-pike/PycharmProjects/data/test_vr_recordings/basic_settings_move_cue_medium/P_190717101640/S001'

    plot_fig1C(session_path)

if __name__ == '__main__':
    main()