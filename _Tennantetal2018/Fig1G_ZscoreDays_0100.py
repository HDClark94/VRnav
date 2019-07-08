# -*- coding: utf-8 -*-
"""

@author: Sarah Tennant


### Calculates the difference between Z-scores in the black box and reward zone for each session - plots over days


"""

# Import packages and functions
from Functions_Core_0100 import extractstops,filterstops, create_srdata, makebinarray, shuffle_analysis_pertrial3, z_score1, adjust_spines, makelegend2,readhdfdata,maketrialarray
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform
from math import floor

# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task13_0300.h5' # raw data files

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,22.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,9.1)]

# Arrays for storing data (output)
firststopstorebeac = np.zeros((len(days), len(mice)));firststopstorenbeac = np.zeros((len(days), len(mice)));firststopstoreprobe = np.zeros((len(days), len(mice)))
firststopstorebeac[:,:] = np.nan;firststopstorenbeac[:,:] = np.nan; firststopstoreprobe[:,:] = np.nan
firststopstorebeac_s = np.zeros((len(days), len(mice)));firststopstorenbeac_s = np.zeros((len(days), len(mice)));firststopstoreprobe_s = np.zeros((len(days), len(mice)))
firststopstorebeac_s[:,:] = np.nan;firststopstorenbeac_s[:,:] = np.nan; firststopstoreprobe_s[:,:] = np.nan


# For each day and mouse, pull raw data, calculate zscore and store data
for mcount,mouse in enumerate(mice):
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')# get raw datafile for mouse and day
        except KeyError:
            print ('Error, no file')
            continue

        # make array of trial number for each row in dataset
        trialarray = maketrialarray(saraharray) # write array of trial per row in datafile
        saraharray[:,9] = trialarray[:,0] # replace trial column in dataset *see README for why this is done*
        
        # get stops and trial arrays
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)# delete all data not on non beaconed tracks
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)# delete all data not on

        # get stops
        stops_b = extractstops(dailymouse_b)
        stops_nb = extractstops(dailymouse_nb)
        stops_p= extractstops(dailymouse_p)
        
        # filter stops
        stops_b = filterstops(stops_b)
        stops_nb = filterstops(stops_nb)
        stops_p = filterstops(stops_p)
        
        # Shuffle stops data & get zscores
        if mcount == 3 or mcount == 5 or mcount == 6 or mcount == 7 or  mcount == 8:
            if stops_b.size>0:
                trialids_b = np.unique(stops_b[:, 2]) # make array of unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stops_b,trialids_b) # get average real stops & shuffled stops per lcoation bin
                shuff_beac = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std ) # calculate z-scores
                bb = shuff_beac[3]; rz = shuff_beac[9] # black box - reward zone zscore
                score = rz-bb
                firststopstorebeac[dcount,mcount] = score # store data
            if stops_nb.size >0:
                trialids_nb = np.unique(stops_nb[:, 2]) # make array of unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stops_nb,trialids_nb) # get average real stops & shuffled stops per lcoation bin
                shuff_nbeac = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std ) # calculate z-scores
                bb = shuff_nbeac[3]; rz = shuff_nbeac[9] # black box - reward zone zscore
                score = rz-bb
                firststopstorenbeac[dcount,mcount] = score # store data
            if stops_p.size >0:
                trialids_p = np.unique(stops_p[:, 2]) # make array of unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stops_p,trialids_p) # get average real stops & shuffled stops per lcoation bin
                shuff_probe = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std ) # calculate z-scores
                bb = shuff_probe[3]; rz = shuff_probe[9] # black box - reward zone zscore
                score = rz-bb
                firststopstoreprobe[dcount,mcount] = score # store data
        dcount+=1
    mcount +=1



# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task12_0600.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,22.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,8.1)]# choose specific day/s

# Arrays for storing data (output)
firststopstorebeac2 = np.zeros((len(days), len(mice)));firststopstorenbeac2= np.zeros((len(days), len(mice)));firststopstoreprobe2= np.zeros((len(days), len(mice)))
firststopstorebeac2[:,:] = np.nan;firststopstorenbeac2[:,:] = np.nan;firststopstoreprobe2[:,:] = np.nan
firststopstorebeac2_s = np.zeros((len(days), len(mice)));firststopstorenbeac2_s= np.zeros((len(days), len(mice)));firststopstoreprobe2_s= np.zeros((len(days), len(mice)))
firststopstorebeac2_s[:,:] = np.nan;firststopstorenbeac2_s[:,:] = np.nan;firststopstoreprobe2_s[:,:] = np.nan

# For each day and mouse, pull raw data, calculate zscore and store data
for mcount,mouse in enumerate(mice):
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        # make array of trial number for each row in dataset
        trialarray = maketrialarray(saraharray) # write array of trial per row in datafile
        saraharray[:,9] = trialarray[:,0] # replace trial column in dataset *see README for why this is done*
        
        # get stops and trial arrays
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)# delete all data not on non beaconed tracks
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)# delete all data not on

        # extract stops
        stops_b = extractstops(dailymouse_b)
        stops_nb = extractstops(dailymouse_nb)
        stops_p= extractstops(dailymouse_p)
        
        # filter stops
        stops_b = filterstops(stops_b)
        stops_nb = filterstops(stops_nb)
        stops_p = filterstops(stops_p)
        
        # Shuffle stops data & get zscores
        if mcount == 5 or mcount == 6 or mcount == 7: # if control mouse, save data
            if stops_b.size >0:
                trialids_b = np.unique(stops_b[:, 2]) # make array of unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stops_b,trialids_b)  # get average real stops & shuffled stops per lcoation bin
                shuff_beac = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std ) # calculate z-scores
                bb = shuff_beac[3]; rz = shuff_beac[9] # black box - reward zone zscore
                score = rz-bb
                firststopstorebeac2[dcount,mcount] = score # store data
            if stops_nb.size >0:
                trialids_nb = np.unique(stops_nb[:, 2]) # make array of unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stops_nb,trialids_nb)  # get average real stops & shuffled stops per lcoation bin
                shuff_nbeac = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std ) # calculate z-scores
                bb = shuff_nbeac[3]; rz = shuff_nbeac[9] # black box - reward zone zscore
                score = rz-bb
                firststopstorenbeac2[dcount,mcount] = score # store data
            if stops_p.size >0:
                trialids_p = np.unique(stops_p[:, 2]) # make array of unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3(stops_p,trialids_p)  # get average real stops & shuffled stops per lcoation bin
                shuff_probe = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std ) # calculate z-scores
                bb = shuff_probe[3]; rz = shuff_probe[9] # black box - reward zone zscore
                score = rz-bb
                firststopstoreprobe2[dcount,mcount] = score # store data
        dcount+=1
    mcount +=1




# stack experiments then average over days for each mouse
con_b = np.nanmean(np.hstack((firststopstorebeac[:,:],firststopstorebeac2[:,:])), axis = 1)
con_nb = np.nanmean(np.hstack((firststopstorenbeac[:,:],firststopstorenbeac2[:,:])), axis =1)
con_p = np.nanmean(np.hstack((firststopstoreprobe[:,:],firststopstoreprobe2[:,:])), axis = 1)
sd_con_b = np.nanstd(np.hstack((firststopstorebeac[:,:],firststopstorebeac2[:,:])), axis = 1)/math.sqrt(6)
sd_con_nb = np.nanstd(np.hstack((firststopstorenbeac[:,:],firststopstorenbeac2[:,:])), axis =1)/math.sqrt(6)
sd_con_p = np.nanstd(np.hstack((firststopstoreprobe[:,:],firststopstoreprobe2[:,:])), axis = 1)/math.sqrt(6)


# PLOT GRAPHS

bins = np.arange(0.5,21.5+1e-6,1) # track bins

fig = plt.figure(figsize = (12,3))
ax = fig.add_subplot(1,3,1)
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.plot(bins,con_b,'o',color = 'Black',label = 'AAV-fl-GFP', linewidth = 2, markersize = 6, markeredgecolor = 'black') #plot becaoned trials
ax.errorbar(bins,con_b,sd_con_b, fmt = 'o', color = 'black', capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,22)
ax.set_ylim(-10,20)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax = plt.ylabel('Training day)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2) #stops per trial
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.plot(bins,con_nb,'o', color = 'Black', linewidth = 2, markersize = 6, markeredgecolor = 'black') #plot becaoned trials
ax.errorbar(bins,con_nb,sd_con_nb, fmt = 'o', color = 'black',  capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,22)
ax.set_ylim(-10,20)
adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax = plt.xlabel('Training day', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,3) #stops per trial
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,con_p, 'o', color = 'Black',label = 'Beaconed', linewidth = 2, markersize = 6, markeredgecolor = 'black') #plot becaoned trials
ax.errorbar(bins,con_p,sd_con_p, fmt = 'o', color = 'black',  capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,22)
ax.set_ylim(-10,20)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax = plt.xlabel('Training day', fontsize=16, labelpad = 18)
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)

fig.savefig('Plots/Figure1/Task13_ZscoreDays_0200'+'.png', dpi =200) # path to save file
plt.close() 






# STORE DATA FOR R

x = np.hstack((firststopstorebeac[:,3],firststopstorebeac[:,5],firststopstorebeac[:,6],firststopstorebeac[:,7],firststopstorebeac[:,8],firststopstorebeac2[:,5],firststopstorebeac2[:,6],firststopstorebeac2[:,7]))
xsd = np.hstack((firststopstorebeac[:,3],firststopstorebeac[:,5],firststopstorebeac[:,6],firststopstorebeac[:,7],firststopstorebeac[:,8],firststopstorebeac2[:,5],firststopstorebeac2[:,6],firststopstorebeac2[:,7]))

x1 = np.hstack((firststopstorenbeac[:,3],firststopstorenbeac[:,5],firststopstorenbeac[:,6],firststopstorenbeac[:,7],firststopstorenbeac[:,8],firststopstorenbeac2[:,5],firststopstorenbeac2[:,6],firststopstorenbeac2[:,7]))
x1sd = np.hstack((firststopstorenbeac[:,3],firststopstorenbeac[:,5],firststopstorenbeac[:,6],firststopstorenbeac[:,7],firststopstorenbeac[:,8],firststopstorenbeac2[:,5],firststopstorenbeac2[:,6],firststopstorenbeac2[:,7]))

x2 = np.hstack((firststopstoreprobe[:,3],firststopstoreprobe[:,5],firststopstoreprobe[:,6],firststopstoreprobe[:,7],firststopstoreprobe[:,8],firststopstoreprobe2[:,5],firststopstoreprobe2[:,6],firststopstoreprobe2[:,7]))
x2sd = np.hstack((firststopstoreprobe[:,3],firststopstoreprobe[:,5],firststopstoreprobe[:,6],firststopstoreprobe[:,7],firststopstoreprobe[:,8],firststopstoreprobe2[:,5],firststopstoreprobe2[:,6],firststopstoreprobe2[:,7]))


days = np.arange(1,22.1,1)
days = np.hstack((np.hstack((days,days,days,days,days,days,days,days)), np.hstack((days,days,days,days,days,days,days,days)), np.hstack((days,days,days,days,days,days,days,days))))
m1 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]); m2 = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]);m3 = np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]); m4 = np.array([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]);m5 = np.array([5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]); m6 = np.array([6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]); m7 = np.array([7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]); m8 = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
mice = np.hstack(([np.hstack((m1,m2,m3,m4,m5,m6,m7,m8)), np.hstack((m1,m2,m3,m4,m5,m6,m7,m8)), np.hstack((m1,m2,m3,m4,m5,m6,m7,m8))]))

b = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
nb = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
p = np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3])

trialtype = np.hstack((np.hstack((b,b,b,b,b,b,b,b)), np.hstack((nb,nb,nb,nb,nb,nb,nb,nb)), np.hstack((p,p,p,p,p,p,p,p))))
firststop = np.hstack((x,x1,x2))
data = np.vstack((mice, days, trialtype, firststop)); data = np.transpose(data)

np.savetxt('Data_Output/Figure1/Figure1_G_0100.csv', data,fmt =  '%i,%i,%i,%10.3f', delimiter = ',', header = 'Mouse, Day, Trialtype, Location (cm)')


