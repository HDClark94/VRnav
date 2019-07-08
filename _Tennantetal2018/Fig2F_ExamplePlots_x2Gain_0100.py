
"""


# Plots example data for gain modulation trials


"""


# import packages and functions
from Functions_Core_0100 import extractstops,filterstops, create_srdata, makebinarray, speed_per_trial, makelegend, makelegend2, makelegend3, makelegend4, shuffle_analysis_pertrial3, extractrewards, adjust_spines,readhdfdata, FirstStops, maketrialarray, speed_per_trial_gain, shuffle_analysis_pertrial_tracks, readhdfdata, makebinarray, adjust_spines
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
from scipy.stats import uniform

#Load data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task19_0100.h5'

#specify mouse and days to analyse
days = ['Day' + str(int(x)) for x in [30]]
mice = ['M' + str(int(x)) for x in [2]]

# for each day and mouse, calculate stops per trial and plot graph
for dcount,day in enumerate(days): #load mouse and day
    for mcount,mouse in enumerate(mice):
        print ('Processing...',day,mouse)
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data') #load HDF5 data set for that day and mouse
        except KeyError:
            print ('Error, no file')
            continue
        
        #define track length parameters
        tracklength = np.max(saraharray[:,1]) # max length of track
        trialno = np.max(saraharray[:,9]) # total number of trials for that day and mouse
        rewardzonestart = saraharray[1,11] # start of reward zone
        rewardzoneend = saraharray[1,12] # end of reward zone
        
        # make array of trial number for each row in dataset
        trialarray = maketrialarray(saraharray) # write array of trial per row in datafile
        saraharray[:,9] = trialarray[:,0] # replace trial column in dataset *see README for why this is done*

        # if gain modulation trial
        if tracklength >22 and tracklength <28:
            tracklength = 245
        binmin = np.min(saraharray[:,1]);binmax = np.max(saraharray[:,1]);interval = 0.1 # i.e if track is 20, 0.2 interval gives 100 bins

        bins = np.arange(0,tracklength/10,interval) # add 1e-6 so that last point included - array of bins for location
        sbins = np.arange(binmin,binmax+1e-6,0.1)
        # Extract data for beaconed, non-beaconed, probe

        # seperate data according to trial type
        if tracklength == 245:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != -10), 0)# delete all data not on probe tracks
            dailymouse = np.delete(saraharray, np.where(saraharray[:, 8] == 0), 0)# delete all data on beaconed tracks
            dailymouse_nb = np.delete(dailymouse, np.where(dailymouse[:, 8] == -10), 0)# delete all data not on non beaconed tracks
        else:
            break

        #extract stops
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)

        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)

        # get array of trial numbers
        trialids_b = np.unique(stopsdata_b[:, 2])
        trialids_nb = np.unique(stopsdata_nb[:, 2])
        trialids_p = np.unique(stopsdata_p[:, 2])

        #get location and trial number of rewards
        reward_beac = extractrewards(dailymouse_b)
        reward_nbeac = extractrewards(dailymouse_nb)

        # real and shuffled stop average
        srbin_mean_b, srbin_std_b,shuffled_mean_b, shuffled_std_b = shuffle_analysis_pertrial3(stopsdata_b, trialids_b) # get average real and shuffled stops
        srbin_mean_nb, srbin_std_nb, shuffled_mean_nb, shuffled_std_nb = shuffle_analysis_pertrial3(stopsdata_nb, trialids_nb) # get average real and shuffled stops
        if stopsdata_p.size > 0:
            srbin_mean_p, srbin_std_p, shuffled_mean_p, shuffled_std_p = shuffle_analysis_pertrial_tracks(stopsdata_p, trialids_p, tracklength/10) # get average real and shuffled stops
    
        # calculate average speed
        speed_beaconed = speed_per_trial(bins,saraharray,trialids_b)
        speed_nbeaconed = speed_per_trial(bins,saraharray,trialids_nb)
        if stopsdata_p.size>0:
            speed_probe = speed_per_trial(bins,saraharray,trialids_p)
            sd_speed_probe = np.nanstd(speed_probe,axis = 1)
        sd_speed_beaconed = np.nanstd(speed_beaconed,axis = 1)
        sd_speed_nbeaconed = np.nanstd(speed_nbeaconed,axis = 1)
        speed_beaconed = np.nanmean(speed_beaconed,axis = 1)
        speed_nbeaconed = np.nanmean(speed_nbeaconed,axis = 1)
        if stopsdata_p.size>0:
            speed_probe = np.nanmean(speed_probe,axis = 1)

        # plot graph
        if tracklength == 245:
            if stopsdata_p.size > 0: # if theres probe trials, plot 3x3 subplots -> stops per trial, average stops, speed for beaconed, non-beaconed and probe trials
                # make figure
                fig = plt.figure(figsize = (12,12))
                ax = fig.add_subplot(3,3,1) #stops per trial
                ax.set_title('Beaconed trials', fontsize = 22,verticalalignment = 'bottom', style = 'italic')
                ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(20-3, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_b[:,0],stopsdata_b[:,2], 'o', color = 'Black', markersize =4.5, label = 'Stop') #plot becaoned trials
                #ax.plot(rewardloc_beac,rewardtrial_beac, '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
                #ax.plot(lickloc_beac,licktrial_beac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,20)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                makelegend(fig,ax) # make legend
                ax.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_ylabel('Trial number', fontsize=18, labelpad = 20)
                
                ax = fig.add_subplot(3,3,2) #stops per trial
                ax.set_title('Non-beaconed trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(20-3, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_nb[:,0],stopsdata_nb[:,2], 'o',color = 'Black', markersize = 4.5) #plot becaoned trials
                #ax.plot(rewardloc_nbeac,rewardtrial_nbeac, '>', color = 'Red', markersize = 6) #plot becaoned trials
                #ax.plot(lickloc_nbeac,licktrial_nbeac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,20)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
                makelegend(fig,ax) # makes legend
                ax.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax = fig.add_subplot(3,3,3) #stops per trial
                ax.set_title('Gain trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(15, 15+2.2, facecolor='Orange', alpha=0.35, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
                ax.axvspan(26.5-3, 26.5, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_p[:,0],stopsdata_p[:,2], 'o',color = 'Black', markersize = 4.5, label = 'Non - beaconed') #plot becaoned trials
                #ax.plot(lickloc_probe,licktrial_probe, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                ax.set_xlim(0,26.5)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                ax.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax1 = fig.add_subplot(3,3,4)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                #ax1.plot(bins*5, srbin_mean_b, color = 'Black',linewidth=1) #plot becaoned trials
                #ax1.fill_between(bins*5,srbin_mean_b-srbin_std_b,srbin_mean_b+srbin_std_b, facecolor = 'Black', alpha = 0.3)
                #ax1.plot(bins*5, shuffled_mean_b, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                #ax1.fill_between(bins*5,shuffled_mean_b-shuffled_std_b,shuffled_mean_b+shuffled_std_b, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,0.65)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Stops (cm/trial)', fontsize=18, labelpad = 23)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                
                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,5)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.2, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                #ax1.plot(bins*5,srbin_mean_nb, color = 'Black',linewidth=1, label = 'Real') #plot becaoned trials
                #ax1.fill_between(bins*5,srbin_mean_nb-srbin_std_nb,srbin_mean_nb+srbin_std_nb, facecolor = 'Black', alpha = 0.3)
                #ax1.plot(bins*5, shuffled_mean_nb, color = 'DodgerBlue',linewidth=1, label = 'Shuffled') #plot becaoned trials
                #ax1.fill_between(bins*5,shuffled_mean_nb-shuffled_std_nb,shuffled_mean_nb+shuffled_std_nb, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                makelegend2(fig,ax1) # makes legend
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])
                
                ax1 = fig.add_subplot(3,3,6)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(75, 75+12, facecolor='b', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax1.axvspan(132.5-15, 132.5, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                #ax1.plot(bins*5,srbin_mean_p,color = 'Black',linewidth=1) #plot becaoned trials
                #ax1.fill_between(bins*5,srbin_mean_p-srbin_std_p,srbin_mean_p+srbin_std_p, facecolor = 'Black', alpha = 0.3)
                #ax1.plot(bins*5, shuffled_mean_p, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                #ax1.fill_between(bins*5,shuffled_mean_p-shuffled_std_p,shuffled_mean_p+shuffled_std_p, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 132.5)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])
                
                ax1 = fig.add_subplot(3,3,7)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                #ax1.plot(sbins*5,speed_beaconed,'-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                #ax1.fill_between(sbins*5,speed_beaconed-sd_speed_beaconed,speed_beaconed+sd_speed_beaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,120)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Speed (cm/s)', fontsize=18, labelpad = 20)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                #ax1.set_xticklabels(['0', '100', '200'])
                
                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,8)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                #ax1.plot(sbins*5,speed_nbeaconed, '-',markersize = 2,color = 'Black',linewidth = 1) #plot becaoned trials
                #ax1.fill_between(sbins*5,speed_nbeaconed-sd_speed_nbeaconed,speed_nbeaconed+sd_speed_nbeaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,120)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                ax1.set_xlabel('Location (cm)', fontsize=18, labelpad = 20)
                
                ax1 = fig.add_subplot(3,3,9)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(75, 75+12, facecolor='r', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
                ax1.axvspan(132.5-15, 132.5, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                #ax1.plot(sbins*5,speed_probe, '-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                #ax1.fill_between(sbins*5,speed_probe-sd_speed_probe,speed_probe+sd_speed_probe, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =22)
                ax1.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =22)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 132.5)
                ax1.set_ylim(0,120)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                
                plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)

                fig.savefig('Plots/Figure2/ExampleData/Example' + 'Data' + '_' + str(mouse) + '_' + str(day) + '.png',  dpi = 200)
                plt.close()



