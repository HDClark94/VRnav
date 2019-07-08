# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:15:18 2014
Updated : 28/06/2014 to incorporate new stops function - includes stops in consecutive bins, 
	also, trial number is taken directly from raw data in HDF5 file (stored in column 9 in raw_data)
Edited: 10/7/15: show first stops for pre and post learning

@author: Sarah Tennant


### Calculates the speed an animal runs along the track
- location bins are 10 cm 


# Experiment details #

Task 13:
Experimental: 1,2,3,5
Control: 4,6,7,8,9
Days: 1-30

Task 12:
Experimental: 1,2,3,4,5
Control: 6,7,8
Days: 1-30

"""

# import packages and functions
from Functions_CoreFunctions_0100 import adjust_spines, makelegend2,readhdfdata
from Functions_Core_0100 import maketrialarray
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform

#Load data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task13_0300.h5' # raw data files

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,30.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,9.1)]

bins = np.arange(0.5,20.5,1) # track bins

# Specify track parameters
REAL_LENGTH = 200 #track length in cm
HDF_LENGTH = 20 #track length in VU
SCALE = HDF_LENGTH/REAL_LENGTH
BINNR = 20 #location bins
SHUFFLE_N = 1000
STOP_THRESHOLD = 0.7 #stop threshold

# Arrays for storing data (output)
s2_con_firststopstorebeac = np.zeros((len(days), len(mice), len(bins)));s2_con_firststopstorenbeac = np.zeros((len(days), len(mice), len(bins)));s2_con_firststopstoreprobe = np.zeros((len(days), len(mice), len(bins)))
s2_con_firststopstorebeac[:,:,:] = np.nan;s2_con_firststopstorenbeac[:,:,:] = np.nan; s2_con_firststopstoreprobe[:,:,:] = np.nan

# ------------------------------------------------------------------------------ #

# Functions specific to this code

# find values in each bin
def makebinarray(tarray, bins):

    interval = 1
    binarray = np.zeros((tarray.shape[0], 1))

    # Collect data for each bin
    for bcount,b in enumerate(bins):
        binmin = tarray[:,1]>=b # lowest value in bin
        binmax = tarray[:,1]<b+interval # highest value in bin
        arraylogical = np.logical_and(binmin,binmax) #get all rows that satisfy being within bin
        binarray[arraylogical, 0] = b #assign each row in tarray its respective bin
        
    return binarray

#function to calculate speed vs location for each trial in session
def speed_per_trial(bins,saraharray, trarray):
    
    stopsarraybeacon = np.zeros(((bins.shape[0]), (trarray.shape[0]))) # rows == same number of bins
    stopsarraybeacon[:,:] = np.nan # fill array with nan's otherwise 0's pull the mean down
     
    for tcount,trial in enumerate(trarray):# loop through each trial
        speedarray = saraharray[saraharray[:,9] ==trial,:] # get data only for each trial
        binarray = makebinarray(speedarray, bins)  # allocate each raw data row to a bin
        for bcount, b in enumerate(bins): #iterate round bins
            barray = speedarray[binarray[:,0] == b,2] # get data for each bin only
            speedmean = np.nanmean(barray, axis= 0) # average speed in that bin
            stopsarraybeacon[bcount,tcount] = speedmean # store speed
            bcount+=1
    return stopsarraybeacon



# ------------------------------------------------------------------------------ #

#GET AND STORE STOPS DATA 

for dcount,day in enumerate(days):
    for mcount,mouse in enumerate(mice):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data') # get raw datafile for mouse and day
        except KeyError: # if data file doesnt exist...
            print ('Error, no file')
            continue
        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)
        
        trialarray = saraharray[:,9] # makes an array of trial number per row in saraharray
        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]+0.1),1)
        
        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)

        # get trial numbers
        trialids_b = np.unique(dailymouse_b[:, 9])
        trialids_nb = np.unique(dailymouse_nb[:, 9])
        trialids_p = np.unique(dailymouse_p[:, 9])
        
        # get speed for each trial in session
        stops_b = speed_per_trial(bins,dailymouse_b, trialids_b)
        stops_nb = speed_per_trial(bins,dailymouse_nb, trialids_nb)
        stops_p = speed_per_trial(bins,dailymouse_p, trialids_p)
        
        #average speed over trials: average for session
        beac = np.nanmean(stops_b,axis=1)        
        nbeac = np.nanmean(stops_nb,axis=1)        
        probe = np.nanmean(stops_p,axis=1)        

        print('##...', mcount,day, '...##')
        
        # store data
        if mcount == 3 or mcount == 5 or mcount == 6 or mcount == 7 or mcount == 8:
            s2_con_firststopstorebeac[dcount,mcount,:] = beac
            if stops_nb.size >0 :     
                s2_con_firststopstorenbeac[dcount, mcount,:] = nbeac
            if stops_p.size >0:
                s2_con_firststopstoreprobe[dcount, mcount,:] = probe

        mcount +=1        



#Load data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task12_0600.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,30.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,8.1)]# choose specific day/s

# Stores
s2_con_12_firststopstorebeac = np.zeros((len(days), len(mice), len(bins)));s2_con_12_firststopstorenbeac = np.zeros((len(days), len(mice), len(bins)));s2_con_12_firststopstoreprobe = np.zeros((len(days), len(mice), len(bins)))
s2_con_12_firststopstorebeac[:,:,:] = np.nan;s2_con_12_firststopstorenbeac[:,:,:] = np.nan;s2_con_12_firststopstoreprobe[:,:,:] = np.nan

# LOOP DAYS AND MICE
for dcount,day in enumerate(days):
    for mcount,mouse in enumerate(mice):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)

        trialarray = saraharray[:,9] # makes an array of trial number per row in saraharray
        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]+0.1),1)
        
        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)

        # get trial number
        trialids_b = np.unique(dailymouse_b[:, 9])
        trialids_nb = np.unique(dailymouse_nb[:, 9])
        trialids_p = np.unique(dailymouse_p[:, 9])
        
        # get speed per bin
        stops_b = speed_per_trial(bins,dailymouse_b, trarray)
        stops_nb = speed_per_trial(bins,dailymouse_nb, trarray)
        stops_p = speed_per_trial(bins,dailymouse_p, trarray)
        
        # average speed over trials
        beac = np.nanmean(stops_b,axis=1)        
        nbeac = np.nanmean(stops_nb,axis=1)        
        probe = np.nanmean(stops_p,axis=1)        

        print('##...', mcount,day, '...##')
        
        #Store data
        if mcount == 5 or mcount == 6 or mcount == 7:
            s2_con_12_firststopstorebeac[dcount,mcount,:] = beac
            if stops_nb.size >0 :     
                s2_con_12_firststopstorenbeac[dcount, mcount,:] = nbeac
            if stops_p.size >0:
                s2_con_12_firststopstoreprobe[dcount, mcount,:] = probe

        mcount +=1        


# Average over days for all mice

con_beac1_w1 = np.nanmean(np.nanmean(np.hstack((s2_con_firststopstorebeac[0:5,:,:],s2_con_12_firststopstorebeac[0:5,:,:])), axis = 0), axis = 0)
con_nbeac1_w1 = np.nanmean(np.nanmean(np.hstack((s2_con_firststopstorenbeac[0:5,:,:],s2_con_12_firststopstorenbeac[0:5,:,:])), axis =0), axis = 0)
con_probe1_w1 = np.nanmean(np.nanmean(np.hstack((s2_con_firststopstoreprobe[0:5,:,:],s2_con_12_firststopstoreprobe[0:5,:,:])), axis = 0), axis = 0)
sd_con_beac1_w1 = np.nanstd(np.nanmean(np.hstack((s2_con_firststopstorebeac[0:5,:,:],s2_con_12_firststopstorebeac[0:5,:,:])), axis = 0), axis = 0)/math.sqrt(8)
sd_con_nbeac1_w1 = np.nanstd(np.nanmean(np.hstack((s2_con_firststopstorenbeac[0:5,:,:],s2_con_12_firststopstorenbeac[0:5,:,:])), axis =0), axis = 0)/math.sqrt(8)
sd_con_probe1_w1 = np.nanstd(np.nanmean(np.hstack((s2_con_firststopstoreprobe[0:5,:,:],s2_con_12_firststopstoreprobe[0:5,:,:])), axis = 0), axis = 0)/math.sqrt(8)

con_beac1_w4 = np.nanmean(np.nanmean(np.hstack((s2_con_firststopstorebeac[20:30,:,:],s2_con_12_firststopstorebeac[20:30,:,:])), axis = 0), axis = 0)
con_nbeac1_w4 = np.nanmean(np.nanmean(np.hstack((s2_con_firststopstorenbeac[20:30,:,:],s2_con_12_firststopstorenbeac[20:30,:,:])), axis =0), axis = 0)
con_probe1_w4 = np.nanmean(np.nanmean(np.hstack((s2_con_firststopstoreprobe[20:30,:,:],s2_con_12_firststopstoreprobe[20:30,:,:])), axis = 0), axis = 0)
sd_con_beac1_w4 = np.nanstd(np.nanmean(np.hstack((s2_con_firststopstorebeac[20:30,:,:],s2_con_12_firststopstorebeac[20:30,:,:])), axis = 0), axis = 0)/math.sqrt(8)
sd_con_nbeac1_w4 = np.nanstd(np.nanmean(np.hstack((s2_con_firststopstorenbeac[20:30,:,:],s2_con_12_firststopstorenbeac[20:30,:,:])), axis =0), axis = 0)/math.sqrt(8)
sd_con_probe1_w4 = np.nanstd(np.nanmean(np.hstack((s2_con_firststopstoreprobe[20:30,:,:],s2_con_12_firststopstoreprobe[20:30,:,:])), axis = 0), axis = 0)/math.sqrt(8)



# PLOT GRAPHS

bins = np.arange(0.5,20.5,1)

fig = plt.figure(figsize = (12,3))
ax = fig.add_subplot(1,3,1) #stops per trial
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_beac1_w1,color = 'blue',label = 'Beaconed') #plot becaoned trials
ax.fill_between(bins,con_beac1_w1-sd_con_beac1_w1,con_beac1_w1+sd_con_beac1_w1, facecolor = 'blue', alpha = 0.3)
ax.plot(bins,con_beac1_w4,color = 'red',label = 'Beaconed') #plot becaoned trials
ax.fill_between(bins,con_beac1_w4-sd_con_beac1_w4,con_beac1_w4+sd_con_beac1_w4, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
ax.set_ylabel('Avg stops / bin', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2) #stops per trial
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_nbeac1_w1,color = 'blue') #plot becaoned trials
ax.fill_between(bins,con_nbeac1_w1-sd_con_nbeac1_w1,con_nbeac1_w1+sd_con_nbeac1_w1, facecolor = 'blue', alpha = 0.3)
ax.plot(bins,con_nbeac1_w4,color = 'red') #plot becaoned trials
ax.fill_between(bins,con_nbeac1_w4-sd_con_nbeac1_w4,con_nbeac1_w4+sd_con_nbeac1_w4, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0)
adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1,3,3) #stops per trial
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_probe1_w1,color = 'blue', label = 'Beaconed') #plot becaoned trials
ax.fill_between(bins,con_probe1_w1-sd_con_probe1_w1,con_probe1_w1+sd_con_probe1_w1, facecolor = 'blue', alpha = 0.3)
ax.plot(bins,con_probe1_w4,color = 'red', label = 'Beaconed') #plot becaoned trials
ax.fill_between(bins,con_probe1_w4-sd_con_probe1_w4,con_probe1_w4+sd_con_probe1_w4, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_yticklabels(['', '', ''])
ax.set_xticklabels(['0', '100', '200'])


plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
#plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Supplemental1/Task13_AvgSpeed_Histogram' + '_0100.png',  dpi = 200)
plt.close()






bins = np.arange(0.5,20.5,1)

fig = plt.figure(figsize = (9,3))
ax = fig.add_subplot(1,3,1) #stops per trial
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_beac1_w1,color = 'Black',label = 'Beaconed') #plot becaoned trials
ax.fill_between(bins,con_beac1_w1-sd_con_beac1_w1,con_beac1_w1+sd_con_beac1_w1, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
ax.set_ylabel('Avg stops / bin', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2) #stops per trial
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_nbeac1_w1,color = 'Black') #plot becaoned trials
ax.fill_between(bins,con_nbeac1_w1-sd_con_nbeac1_w1,con_nbeac1_w1+sd_con_nbeac1_w1, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0)
adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1,3,3) #stops per trial
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_yticklabels(['', '', ''])
ax.set_xticklabels(['0', '100', '200'])


plt.subplots_adjust(hspace = 1, wspace = 0.4,  bottom = 0.35, left = 0.12, right = 0.97, top = .8)
fig.savefig('Plots/Supplemental1/Task13_AvgSpeed_Histogram_week1' + '_0100.png',  dpi = 200)
plt.close()



bins = np.arange(0.5,20.5,1)

fig = plt.figure(figsize = (9,3))
ax = fig.add_subplot(1,3,1) #stops per trial
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_beac1_w4,color = 'Black',label = 'Beaconed') #plot becaoned trials
ax.fill_between(bins,con_beac1_w4-sd_con_beac1_w4,con_beac1_w4+sd_con_beac1_w4, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
ax.set_ylabel('Avg stops / bin', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2) #stops per trial
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_nbeac1_w4,color = 'Black') #plot becaoned trials
ax.fill_between(bins,con_nbeac1_w4-sd_con_nbeac1_w4,con_nbeac1_w4+sd_con_nbeac1_w4, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0)
adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1,3,3) #stops per trial
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_probe1_w4,color = 'Black', label = 'Beaconed') #plot becaoned trials
ax.fill_between(bins,con_probe1_w4-sd_con_probe1_w4,con_probe1_w4+sd_con_probe1_w4, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_yticklabels(['', '', ''])
ax.set_xticklabels(['0', '100', '200'])

plt.subplots_adjust(hspace = 1, wspace = 0.4,  bottom = 0.35, left = 0.12, right = 0.97, top = .8)
fig.savefig('Plots/Supplemental1/Task13_AvgSpeed_Histogram_week4' + '_0100.png',  dpi = 200)
plt.close()


#-----------------------------------------------------------------------------------------------------------#



# SAVE DATA

x = np.vstack((np.nanmean(s2_con_firststopstorebeac[20:25,3,:],axis=0),np.nanmean(s2_con_firststopstorebeac[20:25,5,:],axis=0),np.nanmean(s2_con_firststopstorebeac[20:25,6,:],axis=0),np.nanmean(s2_con_firststopstorebeac[20:25,7,:],axis=0),np.nanmean(s2_con_firststopstorebeac[20:25,8,:],axis=0),np.nanmean(s2_con_12_firststopstorebeac[20:25,5,:],axis=0),np.nanmean(s2_con_12_firststopstorebeac[20:25,6,:],axis=0),np.nanmean(s2_con_12_firststopstorebeac[20:25,7,:],axis=0)))

x1 = np.vstack((np.nanmean(s2_con_firststopstorenbeac[20:25,3,:],axis=0),np.nanmean(s2_con_firststopstorenbeac[20:25,5,:],axis=0),np.nanmean(s2_con_firststopstorenbeac[20:25,6,:],axis=0),np.nanmean(s2_con_firststopstorenbeac[20:25,7,:],axis=0),np.nanmean(s2_con_firststopstorenbeac[20:25,8,:],axis=0),np.nanmean(s2_con_12_firststopstorenbeac[20:25,5,:],axis=0),np.nanmean(s2_con_12_firststopstorenbeac[20:25,6,:],axis=0),np.nanmean(s2_con_12_firststopstorenbeac[20:25,7,:],axis=0)))

x2 = np.vstack((np.nanmean(s2_con_firststopstoreprobe[20:25,3,:],axis=0),np.nanmean(s2_con_firststopstoreprobe[20:25,5,:],axis=0),np.nanmean(s2_con_firststopstoreprobe[20:25,6,:],axis=0),np.nanmean(s2_con_firststopstoreprobe[20:25,7,:],axis=0),np.nanmean(s2_con_firststopstoreprobe[20:25,8,:],axis=0),np.nanmean(s2_con_12_firststopstoreprobe[20:25,5,:],axis=0),np.nanmean(s2_con_12_firststopstoreprobe[20:25,6,:],axis=0),np.nanmean(s2_con_12_firststopstoreprobe[20:25,7,:],axis=0)))

x = np.vstack((x,x1,x2))


mice = np.array([1,2,3,4,5,6,7,8]); mouse = np.hstack((mice, mice, mice))
#print('mice', mouse.shape)
trialb = np.array([1,1,1,1,1,1,1,1]); trialnb = np.array([2,2,2,2,2,2,2,2]); trialp = np.array([3,3,3,3,3,3,3,3]); trials = np.hstack((trialb, trialnb, trialp))
#print('trials', trials.shape)
data = np.vstack((mouse, trials)); data=np.transpose(data)
#print('x--', x.shape, data.shape)
data = np.hstack((data,x))
#print(data.shape)

np.savetxt('Data_Output/Supplemental1/FigureS1_B_Week5_0100.csv', data,fmt = '%i,%i,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f', delimiter = '\t', header = 'Mouse, Trial, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20')




x = np.vstack((np.nanmean(s2_con_firststopstorebeac[0:5,3,:],axis=0),np.nanmean(s2_con_firststopstorebeac[0:5,5,:],axis=0),np.nanmean(s2_con_firststopstorebeac[0:5,6,:],axis=0),np.nanmean(s2_con_firststopstorebeac[0:5,7,:],axis=0),np.nanmean(s2_con_firststopstorebeac[0:5,8,:],axis=0),np.nanmean(s2_con_12_firststopstorebeac[0:5,5,:],axis=0),np.nanmean(s2_con_12_firststopstorebeac[0:5,6,:],axis=0),np.nanmean(s2_con_12_firststopstorebeac[0:5,7,:],axis=0)))

x1 = np.vstack((np.nanmean(s2_con_firststopstorenbeac[0:5,3,:],axis=0),np.nanmean(s2_con_firststopstorenbeac[0:5,5,:],axis=0),np.nanmean(s2_con_firststopstorenbeac[0:5,6,:],axis=0),np.nanmean(s2_con_firststopstorenbeac[0:5,7,:],axis=0),np.nanmean(s2_con_firststopstorenbeac[0:5,8,:],axis=0),np.nanmean(s2_con_12_firststopstorenbeac[0:5,5,:],axis=0),np.nanmean(s2_con_12_firststopstorenbeac[0:5,6,:],axis=0),np.nanmean(s2_con_12_firststopstorenbeac[0:5,7,:],axis=0)))

x2 = np.vstack((np.nanmean(s2_con_firststopstoreprobe[0:5,3,:],axis=0),np.nanmean(s2_con_firststopstoreprobe[0:5,5,:],axis=0),np.nanmean(s2_con_firststopstoreprobe[0:5,6,:],axis=0),np.nanmean(s2_con_firststopstoreprobe[0:5,7,:],axis=0),np.nanmean(s2_con_firststopstoreprobe[0:5,8,:],axis=0),np.nanmean(s2_con_12_firststopstoreprobe[0:5,5,:],axis=0),np.nanmean(s2_con_12_firststopstoreprobe[0:5,6,:],axis=0),np.nanmean(s2_con_12_firststopstoreprobe[0:5,7,:],axis=0)))

x = np.vstack((x,x1,x2))
data = np.vstack((mouse, trials)); data=np.transpose(data)
data = np.hstack((data,x))

np.savetxt('Data_Output/Supplemental1/FigureS1_B_Week1_0100.csv', data,fmt = '%i,%i,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f', delimiter = '\t', header = 'Mouse, Trial, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20')

