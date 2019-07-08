
"""

@author: Sarah Anne Tennant

# FIGURE 1: EXAMPLE PLOTS


"""


#IMPORT FUNCTIONS AND PACKAGES
from Functions_Core_0100 import extractstops,filterstops, create_srdata, makebinarray, speed_per_trial, makelegend, makelegend2, makelegend3, makelegend4, shuffle_analysis_pertrial3, extractrewards, adjust_spines,readhdfdata, FirstStops, maketrialarray
import matplotlib.pyplot as plt
import numpy as np
import math


# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task15_0100.h5' # raw data files

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in [17]]
mice = ['M' + str(int(x)) for x in [2]]# specific day/s

print ('Loading data from...' + filename)
track = 200 # specify track size in cm


# loop thorugh mice and days to get data
for dcount,day in enumerate(days): #load mouse and day
    for mcount,mouse in enumerate(mice):
        print ('Processing...',mcount,day)
        #load HDF5 data set for that day and mouse
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue

        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)

        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]),1) #array of trial number
        trialno = np.max(saraharray[:,9]) # total number of trials for that day and mouse
        binmin = np.min(saraharray[:,1]);binmax = np.max(saraharray[:,1]);interval = 0.1 # i.e if track is 20, 0.2 interval gives 100 bins
        bins = np.arange(0.5,20.5,1) # array of bins for location
        
        # Extract data for beaconed, non-beaconed, probe
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        
        #extract stops
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)
        
        #get location and trial number of rewards
        reward_beac = extractrewards(dailymouse_b)
        reward_nbeac = extractrewards(dailymouse_nb)
        
        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)
        
        # extract trial numbers from data (only unique ones)
        trialids_b = np.unique(stopsdata_b[:, 2])
        if stopsdata_nb.size > 0:
            trialids_nb = np.unique(stopsdata_nb[:, 2])
        if stopsdata_p.size > 0:
            trialids_p = np.unique(stopsdata_p[:, 2])
        firststop = FirstStops( trialids_b , stopsdata_b)
        
        # get mean stops per bin for real and shuffled data
        srbin_mean_b, srbin_std_b,shuffled_mean_b, shuffled_std_b = shuffle_analysis_pertrial3(stopsdata_b, trialids_b)
        srbin_mean_nb, srbin_std_nb, shuffled_mean_nb, shuffled_std_nb = shuffle_analysis_pertrial3(stopsdata_nb, trialids_nb)
        if stopsdata_p.size > 0:
            srbin_mean_p, srbin_std_p, shuffled_mean_p, shuffled_std_p = shuffle_analysis_pertrial3(stopsdata_p, trialids_p)
                
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
        
        if stopsdata_b.size >0:
            if stopsdata_p.size > 0: # if there are probe trials, plot 3x3 subplots
                # MAKE FIGURE: 1st row: stop rasters. 2nd row: average stops. 3rd row: speed.
                fig = plt.figure(figsize = (12,12))
                ax = fig.add_subplot(3,3,1) #stops per trial
                ax.set_title('Beaconed trials', fontsize = 22,verticalalignment = 'bottom', style = 'italic')
                ax.axvspan(8.8, 8.8+2.2, facecolor='DarkGreen', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone to mark reward zone
                ax.axvspan(0, 3, facecolor='k', hatch = '/', linewidth =0, alpha=0.15) # black box
                ax.axvspan(20-3, 20, facecolor='k', hatch = '/', linewidth =0, alpha=0.15)# black box
                ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_b[:,0],stopsdata_b[:,2], 'o', color = 'Black', markersize =4.5, label = 'Stop') #plot becaoned trials
                ax.plot(reward_beac[:,0],reward_beac[:,2], '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                ax.set_xlim(0,20)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                makelegend(fig,ax) # make legend
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_ylabel('Trial number', fontsize=18, labelpad = 20)
                
                ax = fig.add_subplot(3,3,2) #stops per trial
                ax.set_title('Non-beaconed trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(8.8, 8.8+2.2, facecolor='DarkGreen', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
                ax.axvspan(20-3, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
                if stopsdata_nb.size >0:
                    ax.plot(stopsdata_nb[:,0],stopsdata_nb[:,2], 'o',color = 'Black', markersize = 4.5) #plot becaoned trials
                    ax.plot(reward_nbeac[:,0],reward_nbeac[:,2], '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                ax.set_xlim(0,20)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
                makelegend(fig,ax) # makes legend
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax = fig.add_subplot(3,3,3) #stops per trial
                ax.set_title('Probe trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(8.8, 8.8+2.2, facecolor='DarkGreen', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
                ax.axvspan(20-3, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_p[:,0],stopsdata_p[:,2], 'o',color = 'Black', markersize = 4.5, label = 'Non - beaconed') #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                ax.set_xlim(0,20)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax = fig.add_subplot(3,3,4)
                ax.axvspan(44, 44+11, facecolor='DarkGreen', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 15, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(100-15, 100, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black')
                ax.axhline(0, linewidth = 3, color = 'black')
                ax.plot(bins*5, srbin_mean_b, color = 'red',linewidth=2) #plot becaoned trials
                ax.fill_between(bins*5,srbin_mean_b-srbin_std_b,srbin_mean_b+srbin_std_b, facecolor = 'red', alpha = 0.3)
                ax.plot(bins*5, shuffled_mean_b, '--',color = 'Black',linewidth=2) #plot becaoned trials
                ax.fill_between(bins*5,shuffled_mean_b-shuffled_std_b,shuffled_mean_b+shuffled_std_b, facecolor = 'Black', alpha = 0.3)
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                ax.set_xlim(0, 101)
                ax.set_ylim(0,1.25)
                adjust_spines(ax, ['left','bottom'])
                ax.set_ylabel('Stops (cm/trial)', fontsize=18, labelpad = 23)
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                
                #avg stops histogram - non beaconed
                ax = fig.add_subplot(3,3,5)
                ax.axvspan(44, 44+11, facecolor='DarkGreen', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
                ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black')
                ax.axhline(0, linewidth = 3, color = 'black')
                ax.plot(bins*5,srbin_mean_nb, color = 'red',linewidth=2, label = 'Real') #plot becaoned trials
                ax.fill_between(bins*5,srbin_mean_nb-srbin_std_nb,srbin_mean_nb+srbin_std_nb, facecolor = 'red', alpha = 0.3)
                ax.plot(bins*5, shuffled_mean_nb,'--', color = 'Black',linewidth=2, label = 'Shuffled') #plot becaoned trials
                ax.fill_between(bins*5,shuffled_mean_nb-shuffled_std_nb,shuffled_mean_nb+shuffled_std_nb, facecolor = 'Black', alpha = 0.3)
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                adjust_spines(ax, ['left','bottom'])
                makelegend2(fig,ax) # makes legend
                ax.set_xlim(0, 101)
                ax.set_ylim(0,1.25)
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', '',''])
                
                ax = fig.add_subplot(3,3,6)
                ax.axvspan(44, 44+11, facecolor='DarkGreen', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone to mark reward zone
                ax.axvspan(0, 15, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(100-15, 100, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black')
                ax.axhline(0, linewidth = 3, color = 'black')
                ax.plot(bins*5,srbin_mean_p,color = 'red',linewidth=2) #plot becaoned trials
                ax.fill_between(bins*5,srbin_mean_p-srbin_std_p,srbin_mean_p+srbin_std_p, facecolor = 'red', alpha = 0.3)
                ax.plot(bins*5, shuffled_mean_p, '--', color = 'Black',linewidth=2) #plot becaoned trials
                ax.fill_between(bins*5,shuffled_mean_p-shuffled_std_p,shuffled_mean_p+shuffled_std_p, facecolor = 'Black', alpha = 0.3)
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                adjust_spines(ax, ['left','bottom'])
                ax.set_xlim(0, 101)
                ax.set_ylim(0,1.25)
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', '',''])
                
                ax = fig.add_subplot(3,3,7)
                ax.axvspan(44, 44+11, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone to mark reward zone
                ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
                ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black')
                ax.axhline(0, linewidth = 3, color = 'black')
                ax.plot(bins*5,speed_beaconed,'-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax.fill_between(bins*5,speed_beaconed-sd_speed_beaconed,speed_beaconed+sd_speed_beaconed, facecolor = 'Black', alpha = 0.3)
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                ax.set_xlim(0, 101)
                ax.set_ylim(0)
                adjust_spines(ax, ['left','bottom'])
                ax.set_ylabel('Speed (cm/s)', fontsize=18, labelpad = 20)
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['-30', '100', '170'])
                
                #avg stops histogram - non beaconed
                ax = fig.add_subplot(3,3,8)
                ax.axvspan(44, 44+11, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone to mark reward zone
                ax.axvspan(0, 15, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(100-15, 100, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black')
                ax.axhline(0, linewidth = 3, color = 'black')
                ax.plot(bins*5,speed_nbeaconed, '-',markersize = 2,color = 'Black',linewidth = 1) #plot becaoned trials
                ax.fill_between(bins*5,speed_nbeaconed-sd_speed_nbeaconed,speed_nbeaconed+sd_speed_nbeaconed, facecolor = 'Black', alpha = 0.3)
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                adjust_spines(ax, ['left','bottom'])
                ax.set_xlim(0, 101)
                ax.set_ylim(0)
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_yticklabels(['', '', ''])
                ax.set_xlabel('Location (cm)', fontsize=18, labelpad = 20)
                
                ax = fig.add_subplot(3,3,9)
                ax.axvspan(44, 44+11, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone to mark reward zone
                ax.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black')
                ax.axhline(0, linewidth = 3, color = 'black')
                ax.plot(bins*5,speed_probe, '-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax.fill_between(bins*5,speed_probe-sd_speed_probe,speed_probe+sd_speed_probe, facecolor = 'Black', alpha = 0.3)
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                adjust_spines(ax, ['left','bottom'])
                ax.set_xlim(0, 101)
                ax.set_ylim(0)
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_yticklabels(['', '', ''])
                
                plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
                fig.savefig('Plots/Figure5/ExampleData/Example' + 'Data' + '_' + str(mouse) + '_' + str(day) + '_4.png',  dpi = 200)
                plt.close()
            
            else: # if there is not probe trials, plot 2x3 subplots -> stops per trial, average stops, speed for beaconed and non-beaconed trials
                fig = plt.figure(figsize = (12,12))
                ax = fig.add_subplot(3,3,1) #stops per trial
                ax.set_title('Beaconed trials', fontsize = 18,verticalalignment = 'bottom', style = 'italic')
                ax.axvspan(8.8, 8.8+2.2, facecolor='DarkGreen', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
                ax.axvspan(20-3, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_b[:,0],stopsdata_b[:,2], 'o', color = 'Black', markersize =4.5, label = 'Stop') #plot becaoned trials
                ax.plot(reward_beac[:,0],reward_beac[:,2], '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                ax.set_xlim(0,20)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                makelegend3(fig,ax)
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_ylabel('Trial number', fontsize=18, labelpad = 20)
                
                ax = fig.add_subplot(3,3,2) #stops per trial
                ax.set_title('Non-beaconed trials', fontsize = 18, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(8.8, 8.8+2.2, facecolor='DarkGreen', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
                ax.axvspan(20-3, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_nb[:,0],stopsdata_nb[:,2], 'o',color = 'Black', markersize = 4.5) #plot becaoned trials
                if reward_nbeac.size >0:
                    ax.plot(reward_nbeac[:,0],reward_nbeac[:,2], '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                ax.set_xlim(0,20)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
                makelegend3(fig,ax) # makes legend
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set xsnumber of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax = fig.add_subplot(3,3,4)
                ax.axvspan(44, 44+11, facecolor='DarkGreen', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone to mark reward zone
                ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
                ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black')
                ax.axhline(0, linewidth = 3, color = 'black')
                ax.plot(bins*5, srbin_mean_b, color = 'blue',linewidth=2) #plot becaoned trials
                ax.fill_between(bins*5,srbin_mean_b-srbin_std_b,srbin_mean_b+srbin_std_b, facecolor = 'blue', alpha = 0.3)
                ax.plot(bins*5, shuffled_mean_b,'--' ,color = 'Black',linewidth=2) #plot becaoned trials
                ax.fill_between(bins*5,shuffled_mean_b-shuffled_std_b,shuffled_mean_b+shuffled_std_b, facecolor = 'Black', alpha = 0.3)
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                ax.set_xlim(0, 101)
                ax.set_ylim(0,1.25)
                adjust_spines(ax, ['left','bottom'])
                ax.set_ylabel('Stops (cm/trial)', fontsize=18, labelpad = 20)
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                
                #avg stops histogram - non beaconed
                ax = fig.add_subplot(3,3,5)
                ax.axvspan(44, 44+11, facecolor='DarkGreen', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone to mark reward zone
                ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
                ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black')
                ax.axhline(0, linewidth = 3, color = 'black')
                ax.plot(bins*5,srbin_mean_nb, color = 'blue',linewidth=2, label = 'Real') #plot becaoned trials
                ax.fill_between(bins*5,srbin_mean_nb-srbin_std_nb,srbin_mean_nb+srbin_std_nb, facecolor = 'blue', alpha = 0.3)
                ax.plot(bins*5, shuffled_mean_nb, '--',color = 'Black',linewidth=2, label = 'Shuffled') #plot becaoned trials
                ax.fill_between(bins*5,shuffled_mean_nb-shuffled_std_nb,shuffled_mean_nb+shuffled_std_nb, facecolor = 'Black', alpha = 0.3)
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                adjust_spines(ax, ['left','bottom'])
                ax.set_yticklabels(['', '', ''])
                makelegend4(fig,ax) # makes legend
                ax.set_xlim(0, 101)
                ax.set_ylim(0,1.25)
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax = fig.add_subplot(3,3,7)
                ax.axvspan(44, 44+11, facecolor='DarkGreen', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone to mark reward zone
                ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
                ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black')
                ax.axhline(0, linewidth = 3, color = 'black')
                ax.plot(bins*5,speed_beaconed,'-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax.fill_between(bins*5,speed_beaconed-sd_speed_beaconed,speed_beaconed+sd_speed_beaconed, facecolor = 'Black', alpha = 0.3)
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                ax.set_xlim(0, 101)
                ax.set_ylim(0,120)
                adjust_spines(ax, ['left','bottom'])
                ax.set_ylabel('Speed (cm/s)', fontsize=18, labelpad = 20)
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['-30', '100', '170'])
                
                #avg stops histogram - non beaconed
                ax = fig.add_subplot(3,3,8)
                ax.axvspan(44, 44+11, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone to mark reward zone
                ax.axvspan(0, 15, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(100-15, 100, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 3, color = 'black')
                ax.axhline(0, linewidth = 3, color = 'black')
                ax.plot(bins*5,speed_nbeaconed, '-',markersize = 2,color = 'Black',linewidth = 1) #plot becaoned trials
                ax.fill_between(bins*5,speed_nbeaconed-sd_speed_nbeaconed,speed_nbeaconed+sd_speed_nbeaconed, facecolor = 'Black', alpha = 0.3)
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 8, labelsize =18)
                adjust_spines(ax, ['left','bottom'])
                ax.set_yticklabels(['', '', ''])
                ax.set_xlim(0, 101)
                ax.set_ylim(0,120)
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['0', '100', '200'])
                ax.set_xlabel('Location (cm)', fontsize=18, labelpad = 20)
                ax.set_yticklabels(['', '', ''])
                
                
                plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
                fig.savefig('Plots/Figure5/ExampleData/Example' + 'Data' + '_' + str(mouse) + '_' + str(day) + '_4.png',  dpi = 200)
                plt.close()
  
  


