# -*- coding: utf-8 -*-
"""
    
    
# Calculates ratio between stops in the visual reward zone and motor reward zone
    
    
"""

# import packages and functions
from Functions_Core_0100 import extractstops,filterstops,create_srdata, makebinarray, speed_per_trial, makelegend, makelegend2, makelegend3, makelegend4, shuffle_analysis_pertrial3, z_score1, adjust_spines, makelegend2,readhdfdata,maketrialarray
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform
from math import floor
import statsmodels.stats.multitest as smm


# ------------------------------------------------------------------------------ #
# Define functions specific to this code

def shuffle_analysis(stopsdata, trialids):
    # Calculate stop rate for each section of the track
    srbin = create_srdata( stopsdata, trialids )                        # Array(BINNR, trialnum)
    srbin_mean = np.mean(srbin, axis=0)                                 # Array(BINNR)
    srbin_std = np.std(srbin, axis=0)                                 # Array(BINNR)
    
    return srbin_mean, srbin_std


def ratio_stop(stops):
    motor = np.sum(stops[5:7])
    visual = np.sum(stops[9:11])
    ratios = visual/(motor+visual)
    return ratios


# ------------------------------------------------------------------------------ #


# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task12_0600.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(31,35.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,8.1)]# choose specific day/s

# Arrays for storing data (output)
ratios_b = np.zeros((len(mice), len(days)));ratios_b[:,:] = np.nan
ratios_nb = np.zeros((len(mice), len(days)));ratios_nb[:,:] = np.nan
ratios_p = np.zeros((len(mice), len(days)));ratios_p[:,:] = np.nan
sdratios_b = np.zeros((len(mice), len(days)));ratios_b[:,:] = np.nan
sdratios_nb = np.zeros((len(mice), len(days)));ratios_nb[:,:] = np.nan
sdratios_p = np.zeros((len(mice), len(days)));ratios_p[:,:] = np.nan

# for each day and mouse specified, calculate stop ratio and store
for dcount,day in enumerate(days):
    for mcount,mouse in enumerate(mice):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        # make array of trial number for each row in dataset
        trialarray = maketrialarray(saraharray) # write array of trial per row in datafile
        saraharray[:,9] = trialarray[:,0] # replace trial column in dataset *see README for why this is done*

        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        # gain modulation trials in this experiment replaced 1 out of every 2 probe trials, they were on the same track so must differentiate between probe and gain
        dailymouse_nb1 = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        dailymouse_nb = np.delete(dailymouse_nb1, np.where(dailymouse_nb1[:, 9] % 2 == 0), 0)
        dailymouse_p = np.delete(dailymouse_nb1, np.where(dailymouse_nb1[:, 9] % 2 != 0), 0)
        
        # get stops
        stops_b = extractstops(dailymouse_b)
        stops_nb = extractstops(dailymouse_nb)
        stops_p= extractstops(dailymouse_p)
        
        # filter stops
        stopsdata_b = filterstops(stops_b)
        stopsdata_nb = filterstops(stops_nb)
        stopsdata_p = filterstops(stops_p)
        
        # get trials
        trialids_b = np.unique(stopsdata_b[:, 2])
        trialids_nb = np.unique(stopsdata_nb[:, 2])
        trialids_p = np.unique(stopsdata_p[:, 2])
        
        beac=[];nbeac=[];probe=[]
        # get first stop for each trial
        if mcount == 0 or mcount == 2 or mcount == 5 or mcount == 6 or mcount == 7:
            stops_f_b,x = shuffle_analysis( stopsdata_b, trialids_b) # get real stops
            ratio = ratio_stop(stops_f_b) # calculate ratio
            ratios_b[mcount,dcount] = ratio # store data
            
            if stops_nb.size >0 :
                stops_f_nb,x = shuffle_analysis( stopsdata_nb, trialids_nb)# get real stops
                ratio = ratio_stop(stops_f_nb)# calculate ratio
                ratios_nb[mcount,dcount] = ratio # store data
            
            if stops_p.size >0 :
                stops_f_p,x = shuffle_analysis( stopsdata_p, trialids_p)# get real stops
                ratio = ratio_stop(stops_f_p)# calculate ratio
                ratios_p[mcount,dcount] = ratio # store data
    
        print('##...', mcount,day, '...##')
        mcount +=1



# average days then mice
ratios1_b = np.nanmean(np.nanmean(ratios_b, axis=0),axis=0)
ratios1_nb = np.nanmean(np.nanmean(ratios_nb, axis=0),axis=0)
ratios1_p = np.nanmean(np.nanmean(ratios_p, axis=0),axis=0)
sdratios1_b = stats.nanstd(np.nanmean(ratios_b, axis=0),axis=0)
sdratios1_nb = stats.nanstd(np.nanmean(ratios_nb, axis=0),axis=0)
sdratios1_p = stats.nanstd(np.nanmean(ratios_p, axis=0),axis=0)


ratios1_b1 = np.nanmean(ratios_b, axis=1)
ratios1_nb1 = np.nanmean(ratios_nb, axis=1)
ratios1_p1 = np.nanmean(ratios_p, axis=1)



## PLOT MEANS

mice1 = np.hstack((ratios1_nb,ratios1_p))
mice1sd = np.hstack((sdratios1_nb,sdratios1_p))
index = np.hstack((1,2))

n_groups = np.arange(3)
bar_width = 0.5
width = 0.4
z = np.arange(0,3,1)
X = n_groups+width/2

fig = plt.figure(figsize = (5,7))
ax = fig.add_subplot(111)
ax.plot(index,mice1, 'o', color = 'k')
ax.errorbar(index,mice1,mice1sd, fmt = 'o', color = 'k', capsize = 11, markersize = 16, elinewidth =5, capthick = 3)
ax.plot(np.hstack((1,1,1,1,1,1,1,1)),ratios1_nb1, 'o', color = 'k', alpha = 0.4, markersize = 13)
ax.plot(np.hstack((2,2,2,2,2,2,2,2)),ratios1_p1, 'o', color = 'k', alpha = 0.4, markersize = 13)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 14, width = 3, labelsize =22)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 14, width = 3, labelsize =22)
ax.set_ylabel('Ratio', fontsize=22, labelpad = 20)
plt.locator_params(axis = 'x', nbins  = 2)
plt.locator_params(axis = 'y', nbins  = 4)
ax.axhline(linewidth=6, color="black")
ax.axvline(0.5,linewidth=6, color="black")
ax.set_ylim(0,1)
plt.locator_params(axis = 'y', nbins  = 3)
plt.xticks(n_groups + bar_width, ('','','',''))
plt.locator_params(axis = 'x', nbins  = 3)
ax.set_xlim(0.5,2.5)

plt.subplots_adjust(hspace = 1, wspace = .5,  bottom = 0.25, left = 0.15, right = 0.9, top = .9)

fig.savefig('Plots/Figure2/Task13_StopRatio_Average_0100' +' .png', dpi = 200)
plt.close()





# Store data for R

ratios1_nb1 = ratios1_nb1[~np.isnan(ratios1_nb1)]
ratios1_p1 = ratios1_p1[~np.isnan(ratios1_p1)]
trial = np.array([3,3,3,3,3,4,4,4,4,4])
mouse =np.array([1,2,3,4,5,1,2,3,4,5])
tracks = np.hstack((ratios1_nb1,ratios1_p1))
data = np.vstack((tracks,trial,mouse)); data = np.transpose(data)

np.savetxt('Data_Output/Figure2/Figure2_D_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Ratio,Trial, Mouse')


