
"""

# Plots example training session for gain modulation trials


"""


# import packages and functions
from Functions_Core_0100 import extractstops,filterstops, create_srdata, makebinarray, speed_per_trial, makelegend, makelegend2, makelegend3, makelegend4, shuffle_analysis_pertrial3, extractrewards, adjust_spines,readhdfdata, FirstStops, maketrialarray, speed_per_trial_gain, readhdfdata, adjust_spines, makebinarray
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
from scipy.stats import uniform

# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task12_0600.h5'

# specify mouse/mice and day/s to analyse
mice = ['M' + str(int(x)) for x in [6]]# specific day/s
days = ['Day' + str(int(x)) for x in [32]]# specific day/s

# for each day and mouse, calculate stops per trial and plot graph
for dcount,day in enumerate(days): #load mouse and day
    for mcount,mouse in enumerate(mice):
        print ('Processing...',day,mouse)
        #load HDF5 data set for that day and mouse
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        # make array of trial number for each row in dataset
        trialarray = maketrialarray(saraharray) # write array of trial per row in datafile
        saraharray[:,9] = trialarray[:,0] # replace trial column in dataset *see README for why this is done*

        trialno = np.max(saraharray[:,9]) # total number of trials for that day and mouse - need later for plotting
        bins = np.arange(0.5,20.5,1) # array of bins for location
        
        # Extract data for beaconed, non-beaconed, probe each
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        # gain modulation trials in this experiment replaced 1 out of every 2 probe trials, they were on the same track so must differentiate between probe and gain
        dailymouse_nb1 = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        dailymouse_p = np.delete(dailymouse_nb1, np.where(dailymouse_nb1[:, 9] % 2 == 0), 0)
        dailymouse_nb = np.delete(dailymouse_nb1, np.where(dailymouse_nb1[:, 9] % 2 != 0), 0)
        
        # get trial numbers
        trialids_b = np.unique(dailymouse_b[:, 9])
        trialids_nb = np.unique(dailymouse_nb[:, 9])
        trialids_p = np.unique(dailymouse_p[:, 9])
        
        #find stops per trial
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)
        
        # filter stops (removes stops 0.5 cm after a stop)
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)
        
        #get location and trial number of rewards
        reward_beac = extractrewards(dailymouse_b)
        reward_nbeac = extractrewards(dailymouse_nb)
        
        # real and shuffled stop average
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
        
        if mcount ==0:
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
            ax.plot(reward_beac[:,0],reward_beac[:,2], '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
            ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
            ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
            ax.set_xlim(0,20)
            ax.set_ylim(0,trialno+0.5)
            adjust_spines(ax, ['left','bottom']) # removes top and right spines
            makelegend(fig,ax)
            ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
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
            ax.plot(reward_nbeac[:,0],reward_nbeac[:,2], '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
            ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
            ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
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
            ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
            ax.axvspan(5, 7, facecolor='orange', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
            ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
            ax.axvspan(20-3, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
            ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
            ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
            ax.plot(stopsdata_p[:,0],stopsdata_p[:,2], 'o',color = 'Black', markersize = 4.5, label = 'Non - beaconed') #plot becaoned trials
            ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =22)
            ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =22)
            ax.set_xlim(0,20)
            ax.set_ylim(0,trialno+0.5)
            adjust_spines(ax, ['left','bottom']) # removes top and right spines
            ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
            ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
            ax.set_xticklabels(['', '', ''])
            ax.set_yticklabels(['', '', ''])
            
            ax1 = fig.add_subplot(3,3,4)
            ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
            ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
            ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
            ax1.axvline(0, linewidth = 6, color = 'black')
            ax1.axhline(0, linewidth = 6, color = 'black')
            ax1.plot(bins*5, srbin_mean_b, color = 'Black',linewidth=1) #plot becaoned trials              
            ax1.fill_between(bins*5,srbin_mean_b-srbin_std_b,srbin_mean_b+srbin_std_b, facecolor = 'Black', alpha = 0.3)
            ax1.plot(bins*5, shuffled_mean_b, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
            ax1.fill_between(bins*5,shuffled_mean_b-shuffled_std_b,shuffled_mean_b+shuffled_std_b, facecolor = 'DodgerBlue', alpha = 0.3)
            ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
            ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
            ax1.set_xlim(0, 101)
            ax1.set_ylim(0,0.65)
            adjust_spines(ax1, ['left','bottom'])
            ax1.set_ylabel('Stops (cm/trial)', fontsize=18, labelpad = 23)
            ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
            ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
            ax1.set_xticklabels(['', '', ''])
                
            #avg stops histogram - non beaconed
            ax1 = fig.add_subplot(3,3,5)
            ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
            ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
            ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
            ax1.axvline(0, linewidth = 6, color = 'black')
            ax1.axhline(0, linewidth = 6, color = 'black')
            ax1.plot(bins*5,srbin_mean_nb, color = 'Black',linewidth=1, label = 'Real') #plot becaoned trials
            ax1.fill_between(bins*5,srbin_mean_nb-srbin_std_nb,srbin_mean_nb+srbin_std_nb, facecolor = 'Black', alpha = 0.3)
            ax1.plot(bins*5, shuffled_mean_nb, color = 'DodgerBlue',linewidth=1, label = 'Shuffled') #plot becaoned trials
            ax1.fill_between(bins*5,shuffled_mean_nb-shuffled_std_nb,shuffled_mean_nb+shuffled_std_nb, facecolor = 'DodgerBlue', alpha = 0.3)
            ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
            ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
            adjust_spines(ax1, ['left','bottom'])
            makelegend2(fig,ax1) # makes legend 
            ax1.set_xlim(0, 101)
            ax1.set_ylim(0,0.65)
            ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
            ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
            ax1.set_xticklabels(['', '', ''])
            ax1.set_yticklabels(['', '', ''])
                
            ax1 = fig.add_subplot(3,3,6)
            ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
            ax1.axvspan(25, 35, facecolor='b', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
            ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
            ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
            ax1.axvline(0, linewidth = 6, color = 'black')
            ax1.axhline(0, linewidth = 6, color = 'black')
            ax1.plot(bins*5,srbin_mean_p,color = 'Black',linewidth=1) #plot becaoned trials
            ax1.fill_between(bins*5,srbin_mean_p-srbin_std_p,srbin_mean_p+srbin_std_p, facecolor = 'Black', alpha = 0.3)
            ax1.plot(bins*5, shuffled_mean_p, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
            ax1.fill_between(bins*5,shuffled_mean_p-shuffled_std_p,shuffled_mean_p+shuffled_std_p, facecolor = 'DodgerBlue', alpha = 0.3)
            ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
            ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
            adjust_spines(ax1, ['left','bottom'])
            ax1.set_xlim(0, 101)
            ax1.set_ylim(0,0.65)
            ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
            ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
            ax1.set_xticklabels(['', '', ''])
            ax1.set_yticklabels(['', '', ''])

            ax1 = fig.add_subplot(3,3,7)
            ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
            ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
            ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
            ax1.axvline(0, linewidth = 6, color = 'black')
            ax1.axhline(0, linewidth = 6, color = 'black')
            ax1.plot(bins*5,speed_beaconed,'-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
            ax1.fill_between(bins*5,speed_beaconed-sd_speed_beaconed,speed_beaconed+sd_speed_beaconed, facecolor = 'Black', alpha = 0.3)
            ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
            ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
            ax1.set_xlim(0, 101)
            ax1.set_ylim(0,120)
            adjust_spines(ax1, ['left','bottom'])
            ax1.set_ylabel('Speed (cm/s)', fontsize=18, labelpad = 20)
            ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
            ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
            ax1.set_xticklabels(['0', '100', '200'])
            ax1.set_xlabel('                                                      Location (cm/VU)', fontsize=18, labelpad = 20)

            #avg stops histogram - non beaconed
            ax1 = fig.add_subplot(3,3,8)
            ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone                
            ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
            ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
            ax1.axvline(0, linewidth = 6, color = 'black')
            ax1.axhline(0, linewidth = 6, color = 'black')
            ax1.plot(bins*5,speed_nbeaconed, '-',markersize = 2,color = 'Black',linewidth = 1) #plot becaoned trials
            ax1.fill_between(bins*5,speed_nbeaconed-sd_speed_nbeaconed,speed_nbeaconed+sd_speed_nbeaconed, facecolor = 'Black', alpha = 0.3)
            ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
            ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
            adjust_spines(ax1, ['left','bottom'])
            ax1.set_xlim(0, 101)
            ax1.set_ylim(0,120)
            ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
            ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
            ax1.set_xticklabels(['0', '100', '200'])
            ax1.set_yticklabels(['', '', ''])
                
            ax1 = fig.add_subplot(3,3,9)
            ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
            ax1.axvspan(25, 35, facecolor='b', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
            ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
            ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
            ax1.axvline(0, linewidth = 6, color = 'black')
            ax1.axhline(0, linewidth = 6, color = 'black')
            ax1.plot(bins*5,speed_probe, '-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
            ax1.fill_between(bins*5,speed_probe-sd_speed_probe,speed_probe+sd_speed_probe, facecolor = 'Black', alpha = 0.3)                
            ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
            ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
            adjust_spines(ax1, ['left','bottom'])
            ax1.set_xlim(0, 101)
            ax1.set_ylim(0,120)
            ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
            ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
            ax1.set_xticklabels(['0', '100', '200'])
            ax1.set_xlabel('Location (VU)', fontsize=18, labelpad = 20)
            ax1.set_yticklabels(['', '', ''])
                
                
            plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
            fig.savefig('Plots/Figure2/ExampleData/Example' + 'Data' + '_' + str(mouse) + '_' + str(day) + '_2.png',  dpi = 200)
            plt.close()
