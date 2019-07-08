# -*- coding: utf-8 -*-
"""

# Calculates the proportion of times an animal stops first in each location bin of the track per training session

For each of the days and mice specified, location along the track is split into 10 cm bins and the proportion of trials in which animals stop in each location bin is calculated. This is then averaged over days for each animal, then over animals. The average proportion is then plotted against track location.

"""

# Import packages and functions
from Functions_Core_0100 import extractstops,filterstops, create_srdata, makebinarray, extractrewards, FirstStops, readhdfdata, adjust_spines, maketrialarray, shuffle_analysis_pertrial3
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform

# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task13_0300.h5' # raw data files

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,22.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,9.1)]

# Arrays for storing data (output)
firststopstorebeac = np.zeros((len(days), len(mice), 20,2));firststopstorenbeac = np.zeros((len(days), len(mice), 20,2));firststopstoreprobe = np.zeros((len(days), len(mice), 20,2))
firststopstorebeac[:,:,:,:] = np.nan;firststopstorenbeac[:,:,:,:] = np.nan; firststopstoreprobe[:,:,:,:] = np.nan

# For each day and mouse, pull raw data, calculate first stops and store data
for dcount,day in enumerate(days):
    for mcount,mouse in enumerate(mice):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        print('##...', mcount,day, '...##')
        
        # make array of trial number for each row in dataset
        trialarray = maketrialarray(saraharray) # write array of trial per row in datafile
        saraharray[:,9] = trialarray[:,0] # replace trial column in dataset *see README for why this is done*
        
        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        
        # get stops
        stops_b = extractstops(dailymouse_b)
        stops_nb = extractstops(dailymouse_nb)
        stops_p= extractstops(dailymouse_p)
        
        #filterstops
        stops_b = filterstops(stops_b)
        stops_nb = filterstops(stops_nb)
        stops_p= filterstops(stops_p)
        
        # get first stop for each trial
        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]+0.1),1)# array of trial numbers
        beac=[];nbeac=[];probe=[] # make empty arrays to store data
        trialids_b = np.unique(stops_b[:, 2]) # find unique trial numbers
        stops_f_b = FirstStops( trarray,stops_b ) # get locations of first stop for each trial
        stops_f_b = create_srdata( stops_f_b, trialids_b ) # bin first stop data
        beac = np.nanmean(stops_f_b, axis = 0) # average times mouse stops first in each bin
        if stops_nb.size >0 :
            trialids_nb = np.unique(stops_nb[:, 2])
            stops_f_nb = FirstStops( trarray,stops_nb )# get locations of first stop for each trial
            stops_f_nb = create_srdata( stops_f_nb, trialids_nb )# bin first stop data
            nbeac = np.nanmean(stops_f_nb, axis = 0)# average times mouse stops first in each bin
        if stops_p.size >0 :
            trialids_p = np.unique(stops_p[:, 2])
            stops_f_p = FirstStops( trarray,stops_p )# get locations of first stop for each trial
            stops_f_p = create_srdata( stops_f_p, trialids_p )# bin first stop data
            probe = np.nanmean(stops_f_p, axis = 0)# average times mouse stops first in each bin

        # store data
        if mcount == 3 or mcount == 5 or mcount == 6 or mcount == 7 or mcount == 8:
            firststopstorebeac[dcount,mcount,:,0] = beac # store first stop data
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stops_b, trialids_b ) # get average stops per location bin
            firststopstorebeac[dcount,mcount,:,1] = srbin_mean # store stops data
            if stops_nb.size >0 :     
                firststopstorenbeac[dcount, mcount,:,0] = nbeac# store first stop data
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stops_nb, trialids_nb )# get average stops per location bin
                firststopstorenbeac[dcount,mcount,:,1] = srbin_mean# store stops data
            if stops_p.size >0:
                firststopstoreprobe[dcount, mcount,:,0] = probe# store first stop data
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stops_p, trialids_p )# get average stops per location bin
                firststopstoreprobe[dcount,mcount,:,1] = srbin_mean# store stops data
        mcount +=1        



# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task12_0600.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,22.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,9.1)]

# Arrays for storing data (output)
firststopstorebeac2 = np.zeros((len(days), len(mice), 20,2));firststopstorenbeac2= np.zeros((len(days), len(mice), 20,2));firststopstoreprobe2= np.zeros((len(days), len(mice), 20,2))
firststopstorebeac2[:,:,:,:] = np.nan;firststopstorenbeac2[:,:,:,:] = np.nan;firststopstoreprobe2[:,:,:,:] = np.nan

# For each day and mouse, pull raw data, calculate first stops and store data
for dcount,day in enumerate(days):
    for mcount,mouse in enumerate(mice):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        print('##...', mcount,day, '...##')

        # make array of trial number for each row in dataset
        trialarray = maketrialarray(saraharray) # write array of trial per row in datafile
        saraharray[:,9] = trialarray[:,0] # replace trial column in dataset *see README for why this is done*
        
        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        
        # get stops
        stops_b = extractstops(dailymouse_b)
        stops_nb = extractstops(dailymouse_nb)
        stops_p= extractstops(dailymouse_p)
        
        #filterstops
        stops_b = filterstops(stops_b)
        stops_nb = filterstops(stops_nb)
        stops_p= filterstops(stops_p)

        # get first stop for each trial
        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]+0.1),1)# array of trial numbers
        beac=[];nbeac=[];probe=[] # make empty arrays to store data
        trialids_b = np.unique(stops_b[:, 2]) # find unique trial numbers
        stops_f_b = FirstStops( trarray,stops_b ) # get locations of first stop for each trial
        stops_f_b = create_srdata( stops_f_b, trialids_b ) # bin first stop data
        beac = np.nanmean(stops_f_b, axis = 0)# average times mouse stops first in each bin
        if stops_nb.size >0 :
            trialids_nb = np.unique(stops_nb[:, 2]) # find unique trial numbers
            stops_f_nb = FirstStops( trarray,stops_nb )# get locations of first stop for each trial
            stops_f_nb = create_srdata( stops_f_nb, trialids_nb )# bin first stop data
            nbeac = np.nanmean(stops_f_nb, axis = 0)# average times mouse stops first in each bin
        if stops_p.size >0 :
            trialids_p = np.unique(stops_p[:, 2]) # find unique trial numbers
            stops_f_p = FirstStops( trarray,stops_p ) # get locations of first stop for each trial
            stops_f_p = create_srdata( stops_f_p, trialids_p )# bin first stop data
            probe = np.nanmean(stops_f_p, axis = 0)# average times mouse stops first in each bin

        # Get average stops per location bin & store data
        if mcount == 7 or mcount == 5 or mcount == 6:
            firststopstorebeac2[dcount,mcount,:,0] = beac # store first stop data
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stops_b, trialids_b ) # get average stops per location bin
            firststopstorebeac2[dcount,mcount,:,1] = srbin_mean # store stops data
            if stops_nb.size >0 :     
                firststopstorenbeac2[dcount, mcount,:,0] = nbeac # store first stop data
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stops_nb, trialids_nb ) # get average stops per location bin
                firststopstorenbeac2[dcount,mcount,:,1] = srbin_mean # store stops data
            if stops_p.size >0:
                firststopstoreprobe2[dcount, mcount,:,0] = probe # store first stop data
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stops_p, trialids_p ) # get average stops per location bin
                firststopstoreprobe2[dcount,mcount,:,1] = srbin_mean # store stops data
        mcount +=1        



# output dataset config: [days, mice, bins, firststop[0]/averagestop[1]]

# Average over days for all mice

# week 1 (first stop)
con_beac_w1 = np.nanmean(np.nanmean(np.hstack((firststopstorebeac[0:5,:,:,0],firststopstorebeac2[0:5,:,:,0])), axis = 0), axis = 0)
con_nbeac_w1 = np.nanmean(np.nanmean(np.hstack((firststopstorenbeac[0:5,:,:,0],firststopstorenbeac2[0:5,:,:,0])), axis =0), axis = 0)
con_probe_w1 = np.nanmean(np.nanmean(np.hstack((firststopstoreprobe[0:5,:,:,0],firststopstoreprobe2[0:5,:,:,0])), axis = 0), axis = 0)
sd_con_beac_w1 = np.nanstd(np.nanmean(np.hstack((firststopstorebeac[0:5,:,:,0],firststopstorebeac2[0:5,:,:,0])), axis = 0), axis = 0)/math.sqrt(8)
sd_con_nbeac_w1 = np.nanstd(np.nanmean(np.hstack((firststopstorenbeac[0:5,:,:,0],firststopstorenbeac2[0:5,:,:,0])), axis = 0), axis = 0)/math.sqrt(8)
sd_con_probe_w1 = np.nanstd(np.nanmean(np.hstack((firststopstoreprobe[0:5,:,:,0],firststopstoreprobe2[0:5,:,:,0])), axis = 0), axis = 0)/math.sqrt(8)
# week 1 (all stops)
con_beac1_w1 = np.nanmean(np.nanmean(np.hstack((firststopstorebeac[0:5,:,:,1],firststopstorebeac2[0:5,:,:,1])), axis = 0), axis = 0)
con_nbeac1_w1 = np.nanmean(np.nanmean(np.hstack((firststopstorenbeac[0:5,:,:,1],firststopstorenbeac2[0:5,:,:,1])), axis =0), axis = 0)
con_probe1_w1 = np.nanmean(np.nanmean(np.hstack((firststopstoreprobe[0:5,:,:,1],firststopstoreprobe2[0:5,:,:,1])), axis = 0), axis = 0)
sd_con_beac1_w1 = np.nanstd(np.nanmean(np.hstack((firststopstorebeac[0:5,:,:,1],firststopstorebeac2[0:5,:,:,1])), axis = 0), axis = 0)/math.sqrt(8)
sd_con_nbeac1_w1 = np.nanstd(np.nanmean(np.hstack((firststopstorenbeac[0:5,:,:,1],firststopstorenbeac2[0:5,:,:,1])), axis =0), axis = 0)/math.sqrt(8)
sd_con_probe1_w1 = np.nanstd(np.nanmean(np.hstack((firststopstoreprobe[0:5,:,:,1],firststopstoreprobe2[0:5,:,:,1])), axis = 0), axis = 0)/math.sqrt(8)

# week 4 (first stop)
con_beac_w4 = np.nanmean(np.nanmean(np.hstack((firststopstorebeac[18:22,:,:,0],firststopstorebeac2[18:22,:,:,0])), axis = 0), axis = 0)
con_nbeac_w4 = np.nanmean(np.nanmean(np.hstack((firststopstorenbeac[18:22,:,:,0],firststopstorenbeac2[18:22,:,:,0])), axis =0), axis = 0)
con_probe_w4 = np.nanmean(np.nanmean(np.hstack((firststopstoreprobe[18:22,:,:,0],firststopstoreprobe2[18:22,:,:,0])), axis = 0), axis = 0)
sd_con_beac_w4 = np.nanstd(np.nanmean(np.hstack((firststopstorebeac[18:22,:,:,0],firststopstorebeac2[18:22,:,:,0])), axis = 0), axis = 0)/math.sqrt(8)
sd_con_nbeac_w4 = np.nanstd(np.nanmean(np.hstack((firststopstorenbeac[18:22,:,:,0],firststopstorenbeac2[18:22,:,:,0])), axis =0), axis = 0)/math.sqrt(8)
sd_con_probe_w4 = np.nanstd(np.nanmean(np.hstack((firststopstoreprobe[18:22,:,:,0],firststopstoreprobe2[18:22,:,:,0])), axis = 0), axis = 0)/math.sqrt(8)
# week 4 (all stops)
con_beac1_w4 = np.nanmean(np.nanmean(np.hstack((firststopstorebeac[18:22,:,:,1],firststopstorebeac2[18:22,:,:,1])), axis = 0), axis = 0)
con_nbeac1_w4 = np.nanmean(np.nanmean(np.hstack((firststopstorenbeac[18:22,:,:,1],firststopstorenbeac2[18:22,:,:,1])), axis =0), axis = 0)
con_probe1_w4 = np.nanmean(np.nanmean(np.hstack((firststopstoreprobe[18:22,:,:,1],firststopstoreprobe2[18:22,:,:,1])), axis = 0), axis = 0)
sd_con_beac1_w4 = np.nanstd(np.nanmean(np.hstack((firststopstorebeac[18:22,:,:,1],firststopstorebeac2[18:22,:,:,1])), axis = 0), axis = 0)/math.sqrt(8)
sd_con_nbeac1_w4 = np.nanstd(np.nanmean(np.hstack((firststopstorenbeac[18:22,:,:,1],firststopstorenbeac2[18:22,:,:,1])), axis =0), axis = 0)/math.sqrt(8)
sd_con_probe1_w4 = np.nanstd(np.nanmean(np.hstack((firststopstoreprobe[18:22,:,:,1],firststopstoreprobe2[18:22,:,:,1])), axis = 0), axis = 0)/math.sqrt(8)




# PLOT GRAPHS

bins = np.arange(0.5,20.5,1)

# first stop histogram
fig = plt.figure(figsize = (12,3))
ax = fig.add_subplot(1,3,1) #stops per trial
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_beac_w1,color = 'blue',label = 'Beaconed', linewidth = 2) #plot becaoned trials
ax.fill_between(bins,con_beac_w1-sd_con_beac_w1,con_beac_w1+sd_con_beac_w1, facecolor = 'blue', alpha = 0.3)
ax.plot(bins,con_beac_w4,color = 'red',label = 'Beaconed', linewidth = 2) #plot becaoned trials
ax.fill_between(bins,con_beac_w4-sd_con_beac_w4,con_beac_w4+sd_con_beac_w4, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0,0.45)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
ax.set_ylabel('Avg stops / bin', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2) #stops per trial
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_nbeac_w1,color = 'blue', linewidth = 2) #plot becaoned trials
ax.fill_between(bins,con_nbeac_w1-sd_con_nbeac_w1,con_nbeac_w1+sd_con_nbeac_w1, facecolor = 'blue', alpha = 0.3)
ax.plot(bins,con_nbeac_w4,color = 'red', linewidth = 2) #plot becaoned trials
ax.fill_between(bins,con_nbeac_w4-sd_con_nbeac_w4,con_nbeac_w4+sd_con_nbeac_w4, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0,0.45)
adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1,3,3) #stops per trial
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_probe_w1,color = 'blue', label = 'Beaconed', linewidth = 2) #plot becaoned trials
ax.fill_between(bins,con_probe_w1-sd_con_probe_w1,con_probe_w1+sd_con_probe_w1, facecolor = 'blue', alpha = 0.3)
ax.plot(bins,con_probe_w4,color = 'red', label = 'Beaconed', linewidth = 2) #plot becaoned trials
ax.fill_between(bins,con_probe_w4-sd_con_probe_w4,con_probe_w4+sd_con_probe_w4, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0,0.45)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_yticklabels(['', '', ''])
ax.set_xticklabels(['0', '100', '200'])

plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)

fig.savefig('Plots/Figure1/Task13_FirstStop_Histogram' + '_0100.png',  dpi = 200)
plt.close()





# SAVE DATA


# first stops
x = np.vstack((np.nanmean(firststopstorebeac[18:22,3,:,0],axis=0),np.nanmean(firststopstorebeac[18:22,5,:,0],axis=0),np.nanmean(firststopstorebeac[18:22,6,:,0],axis=0),np.nanmean(firststopstorebeac[18:22,7,:,0],axis=0),np.nanmean(firststopstorebeac[18:22,8,:,0],axis=0),np.nanmean(firststopstorebeac2[18:22,5,:,0],axis=0),np.nanmean(firststopstorebeac2[18:22,6,:,0],axis=0),np.nanmean(firststopstorebeac2[18:22,7,:,0],axis=0)))
x1 = np.vstack((np.nanmean(firststopstorenbeac[18:22,3,:,0],axis=0),np.nanmean(firststopstorenbeac[18:22,5,:,0],axis=0),np.nanmean(firststopstorenbeac[18:22,6,:,0],axis=0),np.nanmean(firststopstorenbeac[18:22,7,:,0],axis=0),np.nanmean(firststopstorenbeac[18:22,8,:,0],axis=0),np.nanmean(firststopstorenbeac2[18:22,5,:,0],axis=0),np.nanmean(firststopstorenbeac2[18:22,6,:,0],axis=0),np.nanmean(firststopstorenbeac2[18:22,7,:,0],axis=0)))
x2 = np.vstack((np.nanmean(firststopstoreprobe[18:22,3,:,0],axis=0),np.nanmean(firststopstoreprobe[18:22,5,:,0],axis=0),np.nanmean(firststopstoreprobe[18:22,6,:,0],axis=0),np.nanmean(firststopstoreprobe[18:22,7,:,0],axis=0),np.nanmean(firststopstoreprobe[18:22,8,:,0],axis=0),np.nanmean(firststopstoreprobe2[18:22,5,:,0],axis=0),np.nanmean(firststopstoreprobe2[18:22,6,:,0],axis=0),np.nanmean(firststopstoreprobe2[18:22,7,:,0],axis=0)))

mice = np.array([1,2,3,4,5,6,7,8]); mouse = np.hstack((mice, mice, mice))
trialb = np.array([1,1,1,1,1,1,1,1]); trialnb = np.array([2,2,2,2,2,2,2,2]); trialp = np.array([3,3,3,3,3,3,3,3]); trials = np.hstack((trialb, trialnb, trialp))
x = np.vstack((x,x1,x2))
data = np.vstack((mouse, trials)); data=np.transpose(data)
data = np.hstack((data,x))

np.savetxt('Data_Output/Figure1/Figure1_D_Week5_0100.csv', data,fmt = '%i,%i,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f', delimiter = '\t', header = 'Mouse, Trial, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20')


x = np.vstack((np.nanmean(firststopstorebeac[0:5,3,:,0],axis=0),np.nanmean(firststopstorebeac[0:5,5,:,0],axis=0),np.nanmean(firststopstorebeac[0:5,6,:,0],axis=0),np.nanmean(firststopstorebeac[0:5,7,:,0],axis=0),np.nanmean(firststopstorebeac[0:5,8,:,0],axis=0),np.nanmean(firststopstorebeac2[0:5,5,:,0],axis=0),np.nanmean(firststopstorebeac2[0:5,6,:,0],axis=0),np.nanmean(firststopstorebeac2[0:5,7,:,0],axis=0)))
x1 = np.vstack((np.nanmean(firststopstorenbeac[0:5,3,:,0],axis=0),np.nanmean(firststopstorenbeac[0:5,5,:,0],axis=0),np.nanmean(firststopstorenbeac[0:5,6,:,0],axis=0),np.nanmean(firststopstorenbeac[0:5,7,:,0],axis=0),np.nanmean(firststopstorenbeac[0:5,8,:,0],axis=0),np.nanmean(firststopstorenbeac2[0:5,5,:,0],axis=0),np.nanmean(firststopstorenbeac2[0:5,6,:,0],axis=0),np.nanmean(firststopstorenbeac2[0:5,7,:,0],axis=0)))
x2 = np.vstack((np.nanmean(firststopstoreprobe[0:5,3,:,0],axis=0),np.nanmean(firststopstoreprobe[0:5,5,:,0],axis=0),np.nanmean(firststopstoreprobe[0:5,6,:,0],axis=0),np.nanmean(firststopstoreprobe[0:5,7,:,0],axis=0),np.nanmean(firststopstoreprobe[0:5,8,:,0],axis=0),np.nanmean(firststopstoreprobe2[0:5,5,:,0],axis=0),np.nanmean(firststopstoreprobe2[0:5,6,:,0],axis=0),np.nanmean(firststopstoreprobe2[0:5,7,:,0],axis=0)))

mice = np.array([1,2,3,4,5,6,7,8]); mouse = np.hstack((mice, mice, mice))
trialb = np.array([1,1,1,1,1,1,1,1]); trialnb = np.array([2,2,2,2,2,2,2,2]); trialp = np.array([3,3,3,3,3,3,3,3]); trials = np.hstack((trialb, trialnb, trialp))
x = np.vstack((x,x1,x2))
data = np.vstack((mouse, trials)); data=np.transpose(data)
data = np.hstack((data,x))

np.savetxt('Data_Output/Figure1/Figure1_D_Week1_0100.csv', data,fmt = '%i,%i,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f', delimiter = '\t', header = 'Mouse, Trial, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20')




           