# -*- coding: utf-8 -*-
"""

# Calculates Z-scores for each bin of the track
- Location bins are 10 cm
- Z-scores calculated for each mouse in last two training weeks then averaged over mice

"""


# import packages and functions
from Functions_Core_0100 import extractstops,filterstops,create_srdata, makebinarray, speed_per_trial, makelegend, makelegend2, makelegend3, makelegend4,shuffle_analysis_pertrial3, z_score1, adjust_spines, makelegend2,readhdfdata,maketrialarray
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform
from math import floor


# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task12_0600.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(31,35.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,8.1)]# choose specific day/s

bins = np.arange(0,19+1e-6,1) # track bins
nbins = np.arange(0,38+1e-6,0.5) # track bins for gain modulated track

# arrays for storing data
firststopstorebeac = np.zeros((len(days), len(mice), 20));firststopstorenbeac = np.zeros((len(days), len(mice), 20));firststopstoreprobe = np.zeros((len(days), len(mice), 20))
firststopstorebeac[:,:,:] = np.nan;firststopstorenbeac[:,:,:] = np.nan; firststopstoreprobe[:,:,:] = np.nan

# loop thorugh mice and days to get data
for mcount,mouse in enumerate(mice):
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')# get raw datafile for mouse and day
        except KeyError:
            print ('Error, no file')
            continue
        # get stops and trial arrays
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)
        trialno = np.max(saraharray[:,9]) # total number of trials for that day and mouse

        # Extract data for beaconed, non-beaconed, probe each
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        # gain modulation trials in this experiment replaced 1 out of every 2 probe trials, they were on the same track so must differentiate between probe and gain
        dailymouse_nb1 = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        dailymouse_nb = np.delete(dailymouse_nb1, np.where(dailymouse_nb1[:, 9] % 2 == 0), 0)
        dailymouse_p = np.delete(dailymouse_nb1, np.where(dailymouse_nb1[:, 9] % 2 != 0), 0)
        
        #extract stops
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)
        
        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)
        
        # Shuffle stops data & store data
        if mcount == 0 or mcount == 3 or mcount == 5 or mcount == 6 or mcount == 7:
            if stopsdata_b.size >0 :
                trialids_b = np.unique(stopsdata_b[:, 2])# get unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b )# get real and shuffled stops
                zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate z-score
                firststopstorebeac[dcount,mcount,:] = zscore_b# store data
            if stopsdata_nb.size >0 :
                trialids_nb = np.unique(stopsdata_nb[:, 2])# get unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb )# get real and shuffled stops
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate z-score
                firststopstorenbeac[dcount, mcount,:] = zscore_nb# store data
            if stopsdata_p.size >0 :
                trialids_p = np.unique(stopsdata_p[:, 2])# get unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p )# get real and shuffled stops
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate z-score
                firststopstoreprobe[dcount, mcount,:] = zscore_p# store data
        print('##...', mcount,day, '...##')
    mcount +=1




# average over days for each mouse

con_beac = np.nanmean(np.nanmean(firststopstorebeac, axis = 0), axis = 0)
con_nbeac = np.nanmean(np.nanmean(firststopstorenbeac, axis =0), axis = 0)
con_probe = np.nanmean(np.nanmean(firststopstoreprobe, axis = 0), axis = 0)
sd_con_beac = np.nanstd(np.nanmean(firststopstorebeac, axis = 0), axis = 0)/math.sqrt(5)
sd_con_nbeac = np.nanstd(np.nanmean(firststopstorenbeac, axis =0), axis = 0)/math.sqrt(5)
sd_con_probe = np.nanstd(np.nanmean(firststopstoreprobe, axis = 0), axis = 0)/math.sqrt(5)


"""

# PLOT GRAPHS

bins = np.arange(0.5,20.5,1)

fig = plt.figure(figsize = (15,3.3))
ax = fig.add_subplot(1,3,1) #stops per trial
#ax.set_title('Beaconed trials', fontsize = 18,verticalalignment = 'bottom', style = 'italic')
ax.axvspan(8.8, 8.8+2.4, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-5, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_beac,color = 'Black',label = 'Beaconed') #plot becaoned trials
ax.fill_between(bins,con_beac-sd_con_beac,con_beac+sd_con_beac, facecolor = 'Black', alpha = 0.3)
#ax.plot(bins,con_beac_s,color = 'DodgerBlue',label = 'Beaconed') #plot becaoned trials
#ax.fill_between(bins,con_beac_s-sd_con_beac_s,con_beac_s+sd_con_beac_s, facecolor = 'DodgerBlue', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
#ax.set_ylim(0,0.92)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=5) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
ax.set_ylabel('Stops (cm)', fontsize=16, labelpad = 18)
ax.set_xlabel('                                                      Location (cm/VU)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2) #stops per trial
#ax.set_title('Probe trials', fontsize = 18, style = 'italic',verticalalignment = 'bottom')
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-5, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_nbeac,color = 'Black') #plot becaoned trials
ax.fill_between(bins,con_nbeac-sd_con_nbeac,con_nbeac+sd_con_nbeac, facecolor = 'Black', alpha = 0.3)
#ax.plot(bins,con_nbeac_s,color = 'DodgerBlue') #plot becaoned trials
#ax.fill_between(bins,con_nbeac_s-sd_con_nbeac_s,con_nbeac_s+sd_con_nbeac_s, facecolor = 'DodgerBlue', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
#ax.set_ylim(0,0.7)
adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=5) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
#ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1,3,3) #stops per trial
#ax.set_title('Gain Mod trials', fontsize = 18, style = 'italic',verticalalignment = 'bottom')
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 6, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(8.8, 12.8, facecolor='orange', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(34, 40, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-6, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_nbeac,color = 'blue', linewidth = 2) #plot becaoned trials
ax.fill_between(bins,con_nbeac-sd_con_nbeac,con_nbeac+sd_con_nbeac, facecolor = 'blue', alpha = 0.3)


ax.plot(bins*2,con_probe,color = 'red', label = 'Beaconed', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*2,con_probe-sd_con_probe,con_probe+sd_con_probe, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =22)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =22)
ax.set_xlim(0,40)
ax.set_ylim(-6,6)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=5) # set number of ticks on y axis
#ax.set_yticklabels(['', '', ''])
ax.set_xticklabels(['0', '100', '200'])
ax.set_xlabel('Location (VU)', fontsize=16, labelpad = 18)
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)

fig.savefig('Plots/Figure2/Task13_ZscoreHist_Gain_0100'+'.png', dpi =200) # path to save file
plt.close()

"""

bins = np.arange(0.5,20.5,1)

fig = plt.figure(figsize = (12,3.3))
ax = fig.add_subplot(1,3,1) #stops per trial
#ax.set_title('Beaconed trials', fontsize = 18,verticalalignment = 'bottom', style = 'italic')
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-5, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_beac,color = 'Black',label = 'Beaconed') #plot becaoned trials
ax.fill_between(bins,con_beac-sd_con_beac,con_beac+sd_con_beac, facecolor = 'Black', alpha = 0.3)
#ax.plot(bins,con_beac_s,color = 'DodgerBlue',label = 'Beaconed') #plot becaoned trials
#ax.fill_between(bins,con_beac_s-sd_con_beac_s,con_beac_s+sd_con_beac_s, facecolor = 'DodgerBlue', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
#ax.set_ylim(0,0.92)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=5) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
ax.set_ylabel('Stops (cm)', fontsize=16, labelpad = 18)
ax.set_xlabel('                                                      Location (cm/VU)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2) #stops per trial
#ax.set_title('Probe trials', fontsize = 18, style = 'italic',verticalalignment = 'bottom')
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-5, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_nbeac,color = 'Black') #plot becaoned trials
ax.fill_between(bins,con_nbeac-sd_con_nbeac,con_nbeac+sd_con_nbeac, facecolor = 'Black', alpha = 0.3)
#ax.plot(bins,con_nbeac_s,color = 'DodgerBlue') #plot becaoned trials
#ax.fill_between(bins,con_nbeac_s-sd_con_nbeac_s,con_nbeac_s+sd_con_nbeac_s, facecolor = 'DodgerBlue', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
#ax.set_ylim(0,0.7)
adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=5) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
#ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1,3,3) #stops per trial
#ax.set_title('Gain Mod trials', fontsize = 18, style = 'italic',verticalalignment = 'bottom')
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(4.8, 6.8, facecolor='orange', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(17, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-6, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_nbeac,color = 'blue', linewidth = 2) #plot becaoned trials
ax.fill_between(bins,con_nbeac-sd_con_nbeac,con_nbeac+sd_con_nbeac, facecolor = 'blue', alpha = 0.3)
ax.plot(bins,con_probe,color = 'red', label = 'Beaconed', linewidth = 2) #plot becaoned trials
ax.fill_between(bins,con_probe-sd_con_probe,con_probe+sd_con_probe, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =22)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =22)
ax.set_xlim(0,20)
ax.set_ylim(-6,6)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=5) # set number of ticks on y axis
#ax.set_yticklabels(['', '', ''])
ax.set_xticklabels(['0', '100', '200'])
ax.set_xlabel('Location (VU)', fontsize=16, labelpad = 18)
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)

fig.savefig('Plots/Figure2/Task13_ZscoreHist_Gain_0100'+'.png', dpi =200) # path to save file
plt.close() 




# save data for R
con_beac =np.nanmean(firststopstorebeac, axis = 0)
con_nbeac = np.nanmean(firststopstorenbeac, axis =0)
con_probe = np.nanmean(firststopstoreprobe, axis = 0)

x = np.vstack((con_beac[3,:],con_beac[5,:],con_beac[6,:],con_beac[1,:],con_beac[0,:]))
x1 = np.vstack((con_nbeac[3,:],con_nbeac[5,:],con_nbeac[6,:],con_nbeac[1,:],con_nbeac[0,:]))
x2 = np.vstack((con_probe[3,:],con_probe[5,:],con_probe[6,:],con_probe[1,:],con_probe[0,:]))

mice = np.array([1,2,3,4,5]); mouse = np.hstack((mice, mice, mice))
trialb = np.array([1,1,1,1,1]); trialnb = np.array([2,2,2,2,2]); trialp = np.array([3,3,3,3,3]); trials = np.hstack((trialb, trialnb, trialp))
x = np.vstack((x,x1,x2))
data = np.vstack((mouse, trials)); data=np.transpose(data)
data = np.hstack((data,x))


np.savetxt('Data_Output/Figure2/Figure2_C_0100.csv', data,fmt = '%i,%i,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f', delimiter = '\t', header = 'Mouse, Trial, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20')



