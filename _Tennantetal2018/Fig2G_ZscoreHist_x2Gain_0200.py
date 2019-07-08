# -*- coding: utf-8 -*-
"""

### Calculates Z-scores for each bin of the track 
- Location bins are 10 cm
- Z-scores calculated for each mouse in last two training weeks then averaged over mice


"""

# import packages and functions
from Functions_Core_0100 import extractstops,filterstops, create_srdata, makebinarray, speed_per_trial, makelegend, makelegend2, makelegend3, makelegend4, shuffle_analysis_pertrial3, z_score1, shuffle_analysis_pertrial_tracks, adjust_spines, makelegend2,readhdfdata,maketrialarray
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform
from math import floor


# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task18_0100.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(15,46.1)]
mice = ['M' + str(int(x)) for x in [6.1]]# specific day/s

# arrays for storing data
s2_con_firststopstorebeac = np.zeros((len(days), len(mice), 20));s2_con_firststopstorenbeac = np.zeros((len(days), len(mice), 20));s2_con_firststopstoreprobe = np.zeros((len(days), len(mice), 26))
s2_con_firststopstorebeac[:,:,:] = np.nan;s2_con_firststopstorenbeac[:,:,:] = np.nan; s2_con_firststopstoreprobe[:,:,:] = np.nan

# loop thorugh mice and days to get data
for mcount,mouse in enumerate(mice):
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')# get raw datafile for mouse and day
        except KeyError:
            print ('Error, no file')
            continue
        # get stops and trial arrays
        tracklength = np.max(saraharray[:,1]) # tracklength
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)

        if tracklength > 22 and tracklength < 28: # if gain modulated
            tracklength = 26

        # split data according to trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != -10), 0)#
        dailymouse = np.delete(saraharray, np.where(saraharray[:, 8] == 0), 0)#
        dailymouse_nb = np.delete(dailymouse, np.where(dailymouse[:, 8] == -10), 0)#
        
        #extract stops
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)
        
        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)
        
        # Shuffle stops data & store data
        if tracklength == 26:
            trialids_b = np.unique(stopsdata_b[:, 2]) # get unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b ) # get real and shuffled stops
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate z-score
            s2_con_firststopstorebeac[dcount,mcount,:] = zscore_b # store data
            if stopsdata_nb.size >0 :
                trialids_nb = np.unique(stopsdata_nb[:, 2])# get unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb )# get real and shuffled stops
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate z-score
                s2_con_firststopstorenbeac[dcount, mcount,:] = zscore_nb# store data
            if stopsdata_p.size >0 :
                trialids_p = np.unique(stopsdata_p[:, 2])# get unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_p, trialids_p, tracklength )# get real and shuffled stops
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate z-score
                s2_con_firststopstoreprobe[dcount, mcount,:] = zscore_p# store data
        print('##...', mcount,day, '...##')
    mcount +=1


# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task19_0100.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(15,46.1)]
mice = ['M' + str(int(x)) for x in [2,7,13]]# specific day/s

# arrays for storing data
s2_con_12_firststopstorebeac = np.zeros((len(days), len(mice), 20));s2_con_12_firststopstorenbeac= np.zeros((len(days), len(mice), 20));s2_con_12_firststopstoreprobe= np.zeros((len(days), len(mice), 26))
s2_con_12_firststopstorebeac[:,:,:] = np.nan;s2_con_12_firststopstorenbeac[:,:,:] = np.nan;s2_con_12_firststopstoreprobe[:,:,:] = np.nan

# loop thorugh mice and days to get data
for mcount,mouse in enumerate(mice):
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        # get stops and trial arrays
        tracklength = np.max(saraharray[:,1])
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)

        if tracklength > 22 and tracklength < 28:
            tracklength = 26

        # Extract data for beaconed, non-beaconed, probe each
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) #
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != -10), 0)#
        dailymouse = np.delete(saraharray, np.where(saraharray[:, 8] == 0), 0)#
        dailymouse_nb = np.delete(dailymouse, np.where(dailymouse[:, 8] == -10), 0)#
        
        #extract stops
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)
        
        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)
        
        # shuffle data and store in arrays
        if tracklength == 26 :
            if stopsdata_b.size >0:
                trialids_b = np.unique(stopsdata_b[:, 2])# get unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b )# get real and shuffled stops
                zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate z-score
                s2_con_12_firststopstorebeac[dcount,mcount,:] = zscore_b# store data
            if stopsdata_nb.size >0 :
                trialids_nb = np.unique(stopsdata_nb[:, 2])# get unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb )# get real and shuffled stops
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate z-score
                s2_con_12_firststopstorenbeac[dcount, mcount,:] = zscore_nb# store data
            if stopsdata_p.size >0 :
                trialids_p = np.unique(stopsdata_p[:, 2])# get unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_p, trialids_p,tracklength )# get real and shuffled stops
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate z-score
                s2_con_12_firststopstoreprobe[dcount, mcount,:] = zscore_p# store data
        print('##...', mcount,day, '...##')
    mcount +=1




# stack experiments then average over days for each mouse
con_b = np.nanmean(np.vstack((np.nanmean(s2_con_firststopstorebeac[:,:,:], axis=0),np.nanmean(s2_con_12_firststopstorebeac[:,:,:], axis=0))), axis = 0)
con_nb = np.nanmean(np.vstack((np.nanmean(s2_con_firststopstorenbeac[:,:,:], axis=0),np.nanmean(s2_con_12_firststopstorenbeac[:,:,:], axis=0))), axis = 0)
con_p = np.nanmean(np.vstack((np.nanmean(s2_con_firststopstoreprobe[:,:,:], axis=0),np.nanmean(s2_con_12_firststopstoreprobe[:,:,:], axis=0))), axis = 0)
sdcon_b = np.nanstd(np.vstack((np.nanmean(s2_con_firststopstorebeac[:,:,:], axis=0),np.nanmean(s2_con_12_firststopstorebeac[:,:,:], axis=0))), axis = 0)/math.sqrt(6)
sdcon_nb = np.nanstd(np.vstack((np.nanmean(s2_con_firststopstorenbeac[:,:,:], axis=0),np.nanmean(s2_con_12_firststopstorenbeac[:,:,:], axis=0))), axis = 0)/math.sqrt(6)
sdcon_p = np.nanstd(np.vstack((np.nanmean(s2_con_firststopstoreprobe[:,:,:], axis=0),np.nanmean(s2_con_12_firststopstoreprobe[:,:,:], axis=0))), axis = 0)/math.sqrt(6)



# PLOT GRAPHS


bins = np.arange(0.5,19.5+1e-6,1) # track bins

fig = plt.figure(figsize = (15,3.3))
ax = fig.add_subplot(1,3,1)
ax.axvspan(44, 44+12, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.plot(bins*5,con_b,color = 'red',label = 'AAV-fl-GFP', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,con_b-sdcon_b,con_b+sdcon_b, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,100)
ax.set_ylim(-10,10)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
ax = plt.ylabel('Location (cm)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2) #stops per trial
ax.axvspan(44, 44+12, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.plot(bins*5,con_nb,color = 'red', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,con_nb-sdcon_nb,con_nb+sdcon_nb, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,100)
ax.set_ylim(-10,10)
adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
ax = plt.xlabel('Location (cm)', fontsize=16, labelpad = 18)

bins = np.arange(0.5,25.5+1e-6,1) # track bins
print(bins.shape, con_p.shape)
ax = fig.add_subplot(1,3,3) #stops per trial
ax.axvspan(75, 75+12, facecolor='orange', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(44, 44+12, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(130-15, 130, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins*5,con_p,color = 'red', label = 'Beaconed', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,con_p-sdcon_p,con_p+sdcon_p, facecolor = 'red', alpha = 0.3)

bins = np.arange(0.5,19.5+1e-6,1) # track bins
ax.plot(bins*5,con_nb,color = 'blue', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,con_nb-sdcon_nb,con_nb+sdcon_nb, facecolor = 'blue', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =22)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =22)
ax.set_xlim(0,130)
ax.set_ylim(-6,6)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Figure2/Task18_ZscoreHist_Gain_0200'+'.png', dpi =200) # path to save file
plt.close()




# save data for R

con_beac = np.nanmean(np.hstack((s2_con_firststopstorebeac[:,:,:],s2_con_12_firststopstorebeac[:,:,:])), axis = 0)
con_nbeac = np.nanmean(np.hstack((s2_con_firststopstorenbeac[:,:,:],s2_con_12_firststopstorenbeac[:,:,:])), axis =0)
con_probe = np.nanmean(np.hstack((s2_con_firststopstoreprobe[:,:,:],s2_con_12_firststopstoreprobe[:,:,:])), axis = 0)
sd_con_beac = np.nanstd(np.hstack((s2_con_firststopstorebeac[:,:,:],s2_con_12_firststopstorebeac[:,:,:])), axis = 0)/math.sqrt(6)
sd_con_nbeac = np.nanstd(np.hstack((s2_con_firststopstorenbeac[:,:,:],s2_con_12_firststopstorenbeac[:,:,:])), axis =0)/math.sqrt(6)
sd_con_probe = np.nanstd(np.hstack((s2_con_firststopstoreprobe[:,:,:],s2_con_12_firststopstoreprobe[:,:,:])), axis = 0)/math.sqrt(6)


x=np.zeros((6)); x[:]=np.nan
con_beac = np.vstack((np.hstack((con_beac[0,:], x)),np.hstack((con_beac[3,:], x)),np.hstack((con_beac[1,:], x)),np.hstack((con_beac[2,:], x))))
con_nbeac = np.vstack((np.hstack((con_nbeac[0,:], x)),np.hstack((con_nbeac[3,:], x)),np.hstack((con_nbeac[1,:], x)),np.hstack((con_nbeac[2,:], x))))
con_probe = np.vstack((con_probe[0,:],con_probe[3,:],con_probe[1,:],con_probe[2,:]))
x = np.vstack((con_beac, con_nbeac, con_probe))
mouse = np.array((1,2,3,4)); mouse = np.hstack((mouse,mouse,mouse))
trial = np.array([1,1,1,1,3,2,2,2,4,4,4,4])
data = np.vstack((mouse, trial)); data=np.transpose(data)
data = np.hstack((data,x))


np.savetxt('Data_Output/Figure2/Figure2_G_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Mouse, Trial, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,12,22,23,24,25,26')



