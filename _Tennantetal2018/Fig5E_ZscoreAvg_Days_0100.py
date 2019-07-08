# -*- coding: utf-8 -*-
"""

# Calculates Z-scores for each bin of the track
- Location bins are 10 cm
- Z-scores calculated for each mouse in last two training weeks then averaged over mice
- Compares high, low TeLC and GFP


"""

# import packages and functions
from Functions_Core_0100 import extractstops,filterstops, create_srdata, makebinarray, speed_per_trial, makelegend, makelegend2, makelegend3, makelegend4, shuffle_analysis_pertrial3, z_score1, adjust_spines, makelegend2,readhdfdata,maketrialarray
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform
from math import floor
import random
import matplotlib.gridspec as gridspec

# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task15_0100.h5' # raw data files

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,19.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,11.1)]

# empty arrays for storing data
s2_con_firststopstorebeac = np.zeros((len(days), len(mice),2));s2_con_firststopstorenbeac = np.zeros((len(days), len(mice),2));s2_con_firststopstoreprobe = np.zeros((len(days), len(mice),2))
s2_con_firststopstorebeac[:,:,:] = np.nan;s2_con_firststopstorenbeac[:,:,:] = np.nan; s2_con_firststopstoreprobe[:,:,:] = np.nan
s2_con_firststopstorebeac_s = np.zeros((len(days), len(mice),2));s2_con_firststopstorenbeac_s = np.zeros((len(days), len(mice),2));s2_con_firststopstoreprobe_s = np.zeros((len(days), len(mice),2))
s2_con_firststopstorebeac_s[:,:,:] = np.nan;s2_con_firststopstorenbeac_s[:,:,:] = np.nan; s2_con_firststopstoreprobe_s[:,:,:] = np.nan

s2_tetl_firststopstorebeac = np.zeros((len(days), len(mice),2));s2_tetl_firststopstorenbeac = np.zeros((len(days), len(mice),2));s2_tetl_firststopstoreprobe = np.zeros((len(days), len(mice),2))
s2_tetl_firststopstorebeac[:,:,:] = np.nan;s2_tetl_firststopstorenbeac[:,:,:] = np.nan; s2_tetl_firststopstoreprobe[:,:,:] = np.nan
s2_tetl_firststopstorebeac_s = np.zeros((len(days), len(mice),2));s2_tetl_firststopstorenbeac_s = np.zeros((len(days), len(mice),2));s2_tetl_firststopstoreprobe_s = np.zeros((len(days), len(mice),2))
s2_tetl_firststopstorebeac_s[:,:,:] = np.nan;s2_tetl_firststopstorenbeac_s[:,:,:] = np.nan; s2_tetl_firststopstoreprobe_s[:,:,:] = np.nan

s2_teth_firststopstorebeac = np.zeros((len(days), len(mice),2));s2_teth_firststopstorenbeac = np.zeros((len(days), len(mice),2));s2_teth_firststopstoreprobe = np.zeros((len(days), len(mice),2))
s2_teth_firststopstorebeac[:,:,:] = np.nan;s2_teth_firststopstorenbeac[:,:,:] = np.nan; s2_teth_firststopstoreprobe[:,:,:] = np.nan
s2_teth_firststopstorebeac_s = np.zeros((len(days), len(mice),2));s2_teth_firststopstorenbeac_s = np.zeros((len(days), len(mice),2));s2_teth_firststopstoreprobe_s = np.zeros((len(days), len(mice),2))
s2_teth_firststopstorebeac_s[:,:,:] = np.nan;s2_teth_firststopstorenbeac_s[:,:,:] = np.nan; s2_teth_firststopstoreprobe_s[:,:,:] = np.nan


#loop days and mice to collect data
for mcount,mouse in enumerate(mice):
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)
        
        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)# delete all data not on non beaconed tracks
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)# delete all data not on probe tracks
        
        #extract stops
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_p = extractstops(dailymouse_p)
        
        # filter stops
        stops_b = filterstops(stopsdata_b)
        stops_p = filterstops(stopsdata_p)
        
        if stops_b.size>0:
            trialids_b = np.unique(stops_b[:, 2]) # find trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std= shuffle_analysis_pertrial3(stops_b,trialids_b) # calculate real and shuffled stops along the track
            shuff_beac = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate zscore for each bin of the track
            start_bb_b = shuff_beac[3]; start_rz_b = shuff_beac[9] # black box bin minus the reward zone bin
            end_bb_b = shuff_beac[17]; end_rz_b = shuff_beac[11] # black box bin minus the reward zone bin
            start_score_b = start_rz_b-start_bb_b # black box bin minus the reward zone bin
            end_score_b = end_rz_b-end_bb_b # black box bin minus the reward zone bin
        if stops_p.size >0:
            trialids_p = np.unique(stops_p[:, 2]) # find trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std= shuffle_analysis_pertrial3(stops_p,trialids_p) # calculate real and shuffled stops along the track
            shuff_probe = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate zscore for each bin of the track
            start_bb_p = shuff_probe[3]; start_rz_p = shuff_probe[9] # black box bin minus the reward zone bin
            end_bb_p = shuff_probe[17]; end_rz_p = shuff_probe[11] # black box bin minus the reward zone bin
            start_score_p = start_rz_b-start_bb_p # black box bin minus the reward zone bin
            end_score_p = end_rz_p-end_bb_p # black box bin minus the reward zone bin
        
        # store data
        if mcount == 0 or mcount == 2 or mcount == 3 or mcount == 9: # if control mouse
            s2_con_firststopstorebeac[dcount,mcount,0] = start_score_b
            s2_con_firststopstorebeac[dcount,mcount,1] = end_score_b
            if stops_p.size >0 :
                s2_con_firststopstoreprobe[dcount,mcount,0] = start_score_p
                s2_con_firststopstoreprobe[dcount,mcount,1] = end_score_p
            if stops_p.size >0 :
                s2_con_firststopstoreprobe[dcount,mcount,0] = start_score_p
                s2_con_firststopstoreprobe[dcount,mcount,1] = end_score_p
        if mcount == 1 or mcount == 5 or mcount == 6 or mcount == 8: # if high telc mouse
            s2_teth_firststopstorebeac[dcount,mcount,0] = start_score_b
            s2_teth_firststopstorebeac[dcount,mcount,1] = end_score_b
            if stops_p.size >0 :
                s2_teth_firststopstoreprobe[dcount, mcount,0] = start_score_p
                s2_teth_firststopstoreprobe[dcount, mcount,1] = end_score_p
        if mcount == 4 or mcount == 7 or mcount == 10: # if low telc mouse
            s2_tetl_firststopstorebeac[dcount,mcount,0] = start_score_b
            s2_tetl_firststopstorebeac[dcount,mcount,1] = end_score_b
            if stops_p.size >0 :
                s2_tetl_firststopstoreprobe[dcount,mcount,0] = start_score_p
                s2_tetl_firststopstoreprobe[dcount,mcount,1] = end_score_p
        dcount+=1
    mcount +=1



# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task15_b_0300.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,19.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,5.1)]# choose specific day/s

# Stores
s2_con_12_firststopstorebeac = np.zeros((len(days), len(mice),2));s2_con_12_firststopstorenbeac= np.zeros((len(days), len(mice),2));s2_con_12_firststopstoreprobe= np.zeros((len(days), len(mice),2))
s2_con_12_firststopstorebeac[:,:,:] = np.nan;s2_con_12_firststopstorenbeac[:,:,:] = np.nan;s2_con_12_firststopstoreprobe[:,:,:] = np.nan

s2_con_12_firststopstorebeac_s = np.zeros((len(days), len(mice),2));s2_con_12_firststopstorenbeac_s= np.zeros((len(days), len(mice),2));s2_con_12_firststopstoreprobe_s= np.zeros((len(days), len(mice),2))
s2_con_12_firststopstorebeac_s[:,:,:] = np.nan;s2_con_12_firststopstorenbeac_s[:,:,:] = np.nan;s2_con_12_firststopstoreprobe_s[:,:,:] = np.nan

s2_tetl_12_firststopstorebeac = np.zeros((len(days), len(mice),2));s2_tetl_12_firststopstorenbeac= np.zeros((len(days), len(mice),2));s2_tetl_12_firststopstoreprobe= np.zeros((len(days), len(mice),2))
s2_tetl_12_firststopstorebeac[:,:,:] = np.nan;s2_tetl_12_firststopstorenbeac[:,:,:] = np.nan;s2_tetl_12_firststopstoreprobe[:,:,:] = np.nan

s2_tetl_12_firststopstorebeac_s = np.zeros((len(days), len(mice),2));s2_tetl_12_firststopstorenbeac_s= np.zeros((len(days), len(mice),2));s2_tetl_12_firststopstoreprobe_s= np.zeros((len(days), len(mice),2))
s2_tetl_12_firststopstorebeac_s[:,:,:] = np.nan;s2_tetl_12_firststopstorenbeac_s[:,:,:] = np.nan;s2_tetl_12_firststopstoreprobe_s[:,:,:] = np.nan

s2_teth_12_firststopstorebeac = np.zeros((len(days), len(mice),2));s2_teth_12_firststopstorenbeac= np.zeros((len(days), len(mice),2));s2_teth_12_firststopstoreprobe= np.zeros((len(days), len(mice),2))
s2_teth_12_firststopstorebeac[:,:,:] = np.nan;s2_teth_12_firststopstorenbeac[:,:,:] = np.nan;s2_teth_12_firststopstoreprobe[:,:,:] = np.nan

s2_teth_12_firststopstorebeac_s = np.zeros((len(days), len(mice),2));s2_teth_12_firststopstorenbeac_s= np.zeros((len(days), len(mice),2));s2_teth_12_firststopstoreprobe_s= np.zeros((len(days), len(mice),2))
s2_teth_12_firststopstorebeac_s[:,:,:] = np.nan;s2_teth_12_firststopstorenbeac_s[:,:,:] = np.nan;s2_teth_12_firststopstoreprobe_s[:,:,:] = np.nan

for mcount,mouse in enumerate(mice):
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)
        
        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        
        #extract stops
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)
        
        # filter stops
        stops_b = filterstops(stopsdata_b)
        stops_nb = filterstops(stopsdata_nb)
        stops_p = filterstops(stopsdata_p)
        
        # Shuffle stops data & get zscores
        if stops_b.size>0:
            trialids_b = np.unique(stops_b[:, 2]) # find trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std= shuffle_analysis_pertrial3(stops_b,trialids_b) # calculate real and shuffled stops along the track
            shuff_beac = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate zscore for each bin of the track
            start_bb_b = shuff_beac[3]; start_rz_b = shuff_beac[9] # black box bin minus the reward zone bin
            end_bb_b = shuff_beac[17]; end_rz_b = shuff_beac[11] # black box bin minus the reward zone bin
            start_score_b = start_rz_b-start_bb_b # black box bin minus the reward zone bin
            end_score_b = end_rz_b-end_bb_b # black box bin minus the reward zone bin
        if stops_p.size >0:
            trialids_p = np.unique(stops_p[:, 2]) # find trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std= shuffle_analysis_pertrial3(stops_p,trialids_p) # calculate real and shuffled stops along the track
            shuff_probe = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate zscore for each bin of the track
            start_bb_p = shuff_probe[3]; start_rz_p = shuff_probe[9] # black box bin minus the reward zone bin
            end_bb_p = shuff_probe[17]; end_rz_p = shuff_probe[11] # black box bin minus the reward zone bin
            start_score_p = start_rz_b-start_bb_p # black box bin minus the reward zone bin
            end_score_p = end_rz_p-end_bb_p # black box bin minus the reward zone bin
        
        #store data
        if mcount == 3 or mcount == 4:
            s2_con_12_firststopstorebeac[dcount,mcount,0] = start_score_b
            s2_con_12_firststopstorebeac[dcount,mcount,1] = end_score_b
            if stops_p.size >0 :
                s2_con_12_firststopstoreprobe[dcount,mcount,0] = start_score_p
                s2_con_12_firststopstoreprobe[dcount,mcount,1] = end_score_p
        if mcount == 0 or mcount == 1 or mcount == 2:
            s2_tetl_12_firststopstorebeac[dcount,mcount,0] = start_score_b
            s2_tetl_12_firststopstorebeac[dcount,mcount,1] = end_score_b
            if stops_p.size >0 :
                s2_tetl_12_firststopstoreprobe[dcount,mcount,0] = start_score_p
                s2_tetl_12_firststopstoreprobe[dcount,mcount,1] = end_score_p
        dcount+=1
    mcount +=1




# average over days for each mouse

start_con_beac = np.nanmean(np.hstack((s2_con_firststopstorebeac[:,:,0],s2_con_12_firststopstorebeac[:,:,0])), axis = 1)
start_con_probe = np.nanmean(np.hstack((s2_con_firststopstoreprobe[:,:,0],s2_con_12_firststopstoreprobe[:,:,0])), axis = 1)
start_teth_beac = np.nanmean(np.hstack((s2_teth_firststopstorebeac[:,:,0],s2_teth_12_firststopstorebeac[:,:,0])), axis = 1)
start_teth_probe = np.nanmean(np.hstack((s2_teth_firststopstoreprobe[:,:,0],s2_teth_12_firststopstoreprobe[:,:,0])), axis = 1)
start_tetl_beac = np.nanmean(np.hstack((s2_tetl_firststopstorebeac[:,:,0],s2_tetl_12_firststopstorebeac[:,:,0])), axis = 1)
start_tetl_probe = np.nanmean(np.hstack((s2_tetl_firststopstoreprobe[:,:,0],s2_tetl_12_firststopstoreprobe[:,:,0])), axis = 1)
sdstart_con_beac = np.nanstd(np.hstack((s2_con_firststopstorebeac[:,:,0],s2_con_12_firststopstorebeac[:,:,0])), axis = 1)/math.sqrt(6)
sdstart_con_probe = np.nanstd(np.hstack((s2_con_firststopstoreprobe[:,:,0],s2_con_12_firststopstoreprobe[:,:,0])), axis = 1)/math.sqrt(6)
sdstart_teth_beac = np.nanstd(np.hstack((s2_teth_firststopstorebeac[:,:,0],s2_teth_12_firststopstorebeac[:,:,0])), axis = 1)/math.sqrt(4)
sdstart_teth_probe = np.nanstd(np.hstack((s2_teth_firststopstoreprobe[:,:,0],s2_teth_12_firststopstoreprobe[:,:,0])), axis = 1)/math.sqrt(4)
sdstart_tetl_beac = np.nanstd(np.hstack((s2_tetl_firststopstorebeac[:,:,0],s2_tetl_12_firststopstorebeac[:,:,0])), axis = 1)/math.sqrt(6)
sdstart_tetl_probe = np.nanstd(np.hstack((s2_tetl_firststopstoreprobe[:,:,0],s2_tetl_12_firststopstoreprobe[:,:,0])), axis = 1)/math.sqrt(6)

end_con_beac = np.nanmean(np.hstack((s2_con_firststopstorebeac[:,:,1],s2_con_12_firststopstorebeac[:,:,1])), axis = 1)
end_con_probe = np.nanmean(np.hstack((s2_con_firststopstoreprobe[:,:,1],s2_con_12_firststopstoreprobe[:,:,1])), axis = 1)
end_teth_beac = np.nanmean(np.hstack((s2_teth_firststopstorebeac[:,:,1],s2_teth_12_firststopstorebeac[:,:,1])), axis = 1)
end_teth_probe = np.nanmean(np.hstack((s2_teth_firststopstoreprobe[:,:,1],s2_teth_12_firststopstoreprobe[:,:,1])), axis = 1)
end_tetl_beac = np.nanmean(np.hstack((s2_tetl_firststopstorebeac[:,:,1],s2_tetl_12_firststopstorebeac[:,:,1])), axis = 1)
end_tetl_probe = np.nanmean(np.hstack((s2_tetl_firststopstoreprobe[:,:,1],s2_tetl_12_firststopstoreprobe[:,:,1])), axis = 1)
sdend_con_beac = np.nanstd(np.hstack((s2_con_firststopstorebeac[:,:,1],s2_con_12_firststopstorebeac[:,:,1])), axis = 1)/math.sqrt(6)
sdend_con_probe = np.nanstd(np.hstack((s2_con_firststopstoreprobe[:,:,1],s2_con_12_firststopstoreprobe[:,:,1])), axis = 1)/math.sqrt(6)
sdend_teth_beac = np.nanstd(np.hstack((s2_teth_firststopstorebeac[:,:,1],s2_teth_12_firststopstorebeac[:,:,1])), axis = 1)/math.sqrt(4)
sdend_teth_probe = np.nanstd(np.hstack((s2_teth_firststopstoreprobe[:,:,1],s2_teth_12_firststopstoreprobe[:,:,1])), axis = 1)/math.sqrt(4)
sdend_tetl_beac = np.nanstd(np.hstack((s2_tetl_firststopstorebeac[:,:,1],s2_tetl_12_firststopstorebeac[:,:,1])), axis = 1)/math.sqrt(6)
sdend_tetl_probe = np.nanstd(np.hstack((s2_tetl_firststopstoreprobe[:,:,1],s2_tetl_12_firststopstoreprobe[:,:,1])), axis = 1)/math.sqrt(6)




# PLOT GRAPHS

bins = np.arange(0.5,18.5+1e-6,1) # track bins

fig = plt.figure(figsize = (12,3))
ax = fig.add_subplot(1,2,1)
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.plot(bins,start_con_beac,color = 'Black',label = 'AAV-fl-GFP', linewidth = 2, markersize = 4) #plot becaoned trials
ax.errorbar(bins,start_con_beac,sdstart_con_beac, fmt = 'o', color = 'black', capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.plot(bins,start_tetl_beac,color = 'blue',label = 'AAV-fl-GFP', linewidth = 2, markersize = 4) #plot becaoned trials
ax.errorbar(bins,start_tetl_beac,sdstart_tetl_beac, fmt = 'o', color = 'blue', capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.plot(bins,start_teth_beac,color = 'red',label = 'AAV-fl-GFP', linewidth = 2, markersize = 4) #plot becaoned trials
ax.errorbar(bins,start_teth_beac,sdstart_teth_beac, fmt = 'o', color = 'red', capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,19)
#ax.set_ylim(-10,20)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax = plt.ylabel('Training day)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,2,2) #stops per trial
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,start_con_probe,color = 'Black', label = 'Beaconed', linewidth = 2, markersize = 4) #plot becaoned trials
ax.errorbar(bins,start_con_probe,sdstart_con_probe, fmt = 'o', color = 'black',  capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.plot(bins,start_tetl_probe,color = 'blue',label = 'AAV-fl-GFP', linewidth = 2, markersize = 4) #plot becaoned trials
ax.errorbar(bins,start_tetl_probe,sdstart_tetl_probe, fmt = 'o', color = 'blue', capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.plot(bins,start_teth_probe,color = 'red',label = 'AAV-fl-GFP', linewidth = 2, markersize = 4) #plot becaoned trials
ax.errorbar(bins,start_teth_probe,sdstart_teth_probe, fmt = 'o', color = 'red', capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,22)
#ax.set_ylim(-10,19)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax = plt.xlabel('Training day', fontsize=16, labelpad = 18)

plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)

fig.savefig('Plots/Figure5/Task15_ZscoreDays_L2-L1_0100'+'.png', dpi =200) # path to save file
plt.close()




bins = np.arange(0.5,18.5+1e-6,1) # track bins
print(bins.shape)
n_groups = np.arange(3)
bar_width = 0.5


fig = plt.figure(figsize = (4,6))
ax = fig.add_subplot(1,1,1)
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.plot(bins,start_con_beac,'o',color = 'Black',label = 'AAV-fl-GFP', linewidth = 2, markersize = 8) #plot becaoned trials
ax.errorbar(bins,start_con_beac,sdstart_con_beac, fmt = 'o', color = 'black', capsize = 2, capthick = 1, markersize = 8, elinewidth = 1.5, markeredgecolor = 'black')
ax.plot(bins,start_tetl_beac,'o',color = 'blue',label = 'AAV-fl-GFP', linewidth = 2, markersize = 8) #plot becaoned trials
ax.errorbar(bins,start_tetl_beac,sdstart_tetl_beac, fmt = 'o', color = 'blue', capsize = 2, capthick = 1, markersize = 8, elinewidth = 1.5, markeredgecolor = 'blue')
ax.plot(bins,start_teth_beac,'o',color = 'red',label = 'AAV-fl-GFP', linewidth = 2, markersize = 8) #plot becaoned trials
ax.errorbar(bins,start_teth_beac,sdstart_teth_beac, fmt = 'o', color = 'red', capsize = 2, capthick = 1, markersize = 8, elinewidth = 1.5, markeredgecolor = 'red')
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 5, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =8)
ax.set_xlim(0,20)
ax.set_ylim(-10,25)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'y', nbins=8) # set number of ticks on y axis
ax = plt.ylabel('Training day)', fontsize=16, labelpad = 18)
#plt.xticks(n_groups + bar_width, ('','','',''))
plt.locator_params(axis = 'x', nbins  = 8)
plt.subplots_adjust(hspace = 1, wspace = .7,  bottom = 0.25, left = 0.1, right = 0.9, top = .9)

fig.savefig('Plots/Figure5/Task15_ZscoreDays_L2-L1_0200'+'.png', dpi =200) # path to save file
plt.close()




bins = np.arange(0.5,18.5+1e-6,1) # track bins

fig = plt.figure(figsize = (12,3))
ax = fig.add_subplot(1,2,1)

ax = fig.add_subplot(1,2,1)
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.plot(bins,end_con_beac,color = 'Black',label = 'AAV-fl-GFP', linewidth = 2, markersize = 4) #plot becaoned trials
ax.errorbar(bins,end_con_beac,sdend_con_beac, fmt = 'o', color = 'black', capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.plot(bins,end_tetl_beac,color = 'blue',label = 'AAV-fl-GFP', linewidth = 2, markersize = 4) #plot becaoned trials
ax.errorbar(bins,end_tetl_beac,sdend_tetl_beac, fmt = 'o', color = 'blue', capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.plot(bins,end_teth_beac,color = 'red',label = 'AAV-fl-GFP', linewidth = 2, markersize = 4) #plot becaoned trials
ax.errorbar(bins,end_teth_beac,sdend_teth_beac, fmt = 'o', color = 'red', capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,19)
#ax.set_ylim(-10,20)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax = plt.ylabel('Training day)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,2,2) #stops per trial
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,end_con_probe,color = 'Black', label = 'Beaconed', linewidth = 2, markersize = 4) #plot becaoned trials
ax.errorbar(bins,end_con_probe,sdend_con_probe, fmt = 'o', color = 'black',  capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.plot(bins,end_tetl_probe,color = 'blue',label = 'AAV-fl-GFP', linewidth = 2, markersize = 4) #plot becaoned trials
ax.errorbar(bins,end_tetl_probe,sdend_tetl_probe, fmt = 'o', color = 'blue', capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
#ax.plot(bins,end_teth_probe,color = 'red',label = 'AAV-fl-GFP', linewidth = 2, markersize = 4) #plot becaoned trials
#ax.errorbar(bins,end_teth_probe,sdend_teth_probe, fmt = 'o', color = 'red', capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,19)
#ax.set_ylim(-10,20)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax = plt.xlabel('Training day', fontsize=16, labelpad = 18)
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)

fig.savefig('Plots/Figure5/Task15_ZscoreDays_L4-L3_0100'+'.png', dpi =200) # path to save file
plt.close()

"""
start_con_beac = np.hstack((s2_con_firststopstorebeac[:,:,0],s2_con_12_firststopstorebeac[:,:,0]))
start_con_probe = np.hstack((s2_con_firststopstoreprobe[:,:,0],s2_con_12_firststopstoreprobe[:,:,0]))
start_teth_beac = np.hstack((s2_teth_firststopstorebeac[:,:,0],s2_teth_12_firststopstorebeac[:,:,0]))
start_teth_probe = np.hstack((s2_teth_firststopstoreprobe[:,:,0],s2_teth_12_firststopstoreprobe[:,:,0]))
start_tetl_beac = np.hstack((s2_tetl_firststopstorebeac[:,:,0],s2_tetl_12_firststopstorebeac[:,:,0]))
start_tetl_probe = np.hstack((s2_tetl_firststopstoreprobe[:,:,0],s2_tetl_12_firststopstoreprobe[:,:,0]))

end_con_beac = np.hstack((s2_con_firststopstorebeac[:,:,1],s2_con_12_firststopstorebeac[:,:,1]))
end_con_probe = np.hstack((s2_con_firststopstoreprobe[:,:,1],s2_con_12_firststopstoreprobe[:,:,1]))
end_teth_beac = np.hstack((s2_teth_firststopstorebeac[:,:,1],s2_teth_12_firststopstorebeac[:,:,1]))
end_teth_probe = np.hstack((s2_teth_firststopstoreprobe[:,:,1],s2_teth_12_firststopstoreprobe[:,:,1]))
end_tetl_beac = np.hstack((s2_tetl_firststopstorebeac[:,:,1],s2_tetl_12_firststopstorebeac[:,:,1]))
end_tetl_probe = np.hstack((s2_tetl_firststopstoreprobe[:,:,1],s2_tetl_12_firststopstoreprobe[:,:,1]))
"""


start_con_beac= np.hstack((s2_con_firststopstorebeac[:,0,0],s2_con_firststopstorebeac[:,2,0], s2_con_firststopstorebeac[:,3,0],s2_con_firststopstorebeac[:,9,0],s2_con_12_firststopstorebeac[:,3,0],s2_con_12_firststopstorebeac[:,4,0]))
start_teth_beac = np.hstack((s2_teth_firststopstorebeac[:,1,0],s2_teth_firststopstorebeac[:,5,0],s2_teth_firststopstorebeac[:,6,0],s2_teth_firststopstorebeac[:,8,0]))
start_tetl_beac = np.hstack((s2_tetl_firststopstorebeac[:,4,0],s2_tetl_firststopstorebeac[:,7,0],s2_tetl_firststopstorebeac[:,10,0],s2_tetl_12_firststopstorebeac[:,0,0],s2_tetl_12_firststopstorebeac[:,1,0],s2_tetl_12_firststopstorebeac[:,2,0]))
start_con_probe= np.hstack((s2_con_firststopstoreprobe[:,2,0],s2_con_firststopstoreprobe[:,2,0], s2_con_firststopstoreprobe[:,3,0],s2_con_firststopstoreprobe[:,9,0],s2_con_12_firststopstoreprobe[:,3,0],s2_con_12_firststopstoreprobe[:,4,0]))
start_teth_probe = np.hstack((s2_teth_firststopstoreprobe[:,1,0],s2_teth_firststopstoreprobe[:,5,0],s2_teth_firststopstoreprobe[:,6,0],s2_teth_firststopstoreprobe[:,8,0]))
start_tetl_probe = np.hstack((s2_tetl_firststopstoreprobe[:,4,0],s2_tetl_firststopstoreprobe[:,7,0],s2_tetl_firststopstoreprobe[:,10,0],s2_tetl_12_firststopstoreprobe[:,0,0],s2_tetl_12_firststopstoreprobe[:,1,0],s2_tetl_12_firststopstoreprobe[:,2,0]))




# SAVE DATA FOR R

ltelc = ("lTeLC","lTeLC","lTeLC","lTeLC","lTeLC","lTeLC")
htelc = ("hTeLC","hTeLC","hTeLC","hTeLC")
gfp = ("GFP","GFP","GFP" ,"GFP" ,"GFP" ,"GFP")
genotype = np.hstack((np.repeat(htelc, 19),np.repeat(ltelc, 19),np.repeat(gfp, 19)))
print('genotype',genotype.shape)

d = np.arange(1,19.1,1)
day = np.hstack((d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d))
print('day',day.shape)

m1 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]); m2 = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]);m3 = np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]); m4 = np.array([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]);m5 = np.array([5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]); m6 = np.array([6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]); m7 = np.array([7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]); m8 = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
m9 = np.array([9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]); m10 = np.array([10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]);m11 = np.array([11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11]); m12 = np.array([12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12]);m13 = np.array([13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13]); m14 = np.array([14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14]); m15 = np.array([15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]); m16 = np.array([16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16])


mice = np.hstack((m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16))
data_b=np.hstack((start_teth_beac,start_tetl_beac,start_con_beac))
data_p=np.hstack((start_teth_probe,start_tetl_probe,start_con_probe))


data = np.vstack((genotype, mice, day, data_b,data_p )); data = np.transpose(data)
np.savetxt('Data_Output/Figure5/Figure5_zscoredays_L2-L1_0100.csv', data,fmt =  '%s', delimiter = ',', header = 'Genotype, Mouse, Day, Beaconed, Probe')



start_con_beac= np.hstack((s2_con_firststopstorebeac[:,0,1],s2_con_firststopstorebeac[:,2,1], s2_con_firststopstorebeac[:,3,1],s2_con_firststopstorebeac[:,9,1],s2_con_12_firststopstorebeac[:,3,1],s2_con_12_firststopstorebeac[:,4,1]))
start_teth_beac = np.hstack((s2_teth_firststopstorebeac[:,1,1],s2_teth_firststopstorebeac[:,5,1],s2_teth_firststopstorebeac[:,6,1],s2_teth_firststopstorebeac[:,8,1]))
start_tetl_beac = np.hstack((s2_tetl_firststopstorebeac[:,4,1],s2_tetl_firststopstorebeac[:,7,1],s2_tetl_firststopstorebeac[:,10,1],s2_tetl_12_firststopstorebeac[:,0,1],s2_tetl_12_firststopstorebeac[:,1,1],s2_tetl_12_firststopstorebeac[:,2,1]))
start_con_probe= np.hstack((s2_con_firststopstoreprobe[:,2,1],s2_con_firststopstoreprobe[:,2,1], s2_con_firststopstoreprobe[:,3,1],s2_con_firststopstoreprobe[:,9,1],s2_con_12_firststopstoreprobe[:,3,1],s2_con_12_firststopstoreprobe[:,4,1]))
start_teth_probe = np.hstack((s2_teth_firststopstoreprobe[:,1,1],s2_teth_firststopstoreprobe[:,5,1],s2_teth_firststopstoreprobe[:,6,1],s2_teth_firststopstoreprobe[:,8,1]))
start_tetl_probe = np.hstack((s2_tetl_firststopstoreprobe[:,4,1],s2_tetl_firststopstoreprobe[:,7,1],s2_tetl_firststopstoreprobe[:,10,1],s2_tetl_12_firststopstoreprobe[:,0,1],s2_tetl_12_firststopstoreprobe[:,1,1],s2_tetl_12_firststopstoreprobe[:,2,1]))



data_b=np.hstack((start_teth_beac,start_tetl_beac,start_con_beac))
data_p=np.hstack((start_teth_probe,start_tetl_probe,start_con_probe))


data = np.vstack((genotype, mice, day, data_b,data_p )); data = np.transpose(data)
print(data.shape)
np.savetxt('Data_Output/Figure5/Figure5_zscoredays_L3-L4_0100.csv', data,fmt =  '%s', delimiter = ',', header = 'Genotype, Mouse, Day, Beaconed, Probe')




