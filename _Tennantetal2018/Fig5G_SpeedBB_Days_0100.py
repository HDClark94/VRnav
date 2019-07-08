# -*- coding: utf-8 -*-
"""
    Created on Mon Jan 27 14:15:18 2014
    Updated : 28/06/2014 to incorporate new stops function - includes stops in consecutive bins,
    also, trial number is taken directly from raw data in HDF5 file (stored in column 9 in raw_data)
    Edited: 10/7/15: show first stops for pre and post learning
    
    @author: Sarah Tennant
    
    
    ### Calculates the speed an animal runs along the track
    - location bins are 10 cm
    
    
    # Experiment details
    
    # Task 15
    - GFP : 1,3,4,10
    - Low TeLC : 5,8,10
    - High TeLC : 2,6,7,9
    Days: 1-22
    
    # Task 15b
    - GFP : 4,5
    - Low TeLC : 1,2,3
    Days: 1-18
    
    """

# import packages and functions
from Functions_Core_0100 import makelegend,FirstStopTime, SplitTrials, SplitTrials2, maketrialarray, shuffle_analysis_pertrial3, z_score1, lowhighstops, filterstops, create_srdata, timetorz, extractstops, timetostop, StopFrequency,adjust_spines, makelegend2,readhdfdata, maketrialarray, makebinarray
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform
import matplotlib.gridspec as gridspec


# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task15_0100.h5' # raw data files

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,19.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,11.1)]

bins = np.arange(0.5,19.5,1) # location bins

# parameters
REAL_LENGTH = 200
HDF_LENGTH = 20
SCALE = HDF_LENGTH/REAL_LENGTH
BINNR = 20
SHUFFLE_N = 1000
STOP_THRESHOLD = 0.7

# arrays for storing data (output)
control_b = np.zeros((len(mice), len(days)))
control_nb = np.zeros((len(mice), len(days)))
control_p = np.zeros((len(mice), len(days)))
tetl_b = np.zeros((len(mice), len(days)))
tetl_nb = np.zeros((len(mice), len(days)))
tetl_p = np.zeros((len(mice), len(days)))
teth_b = np.zeros((len(mice), len(days)))
teth_nb = np.zeros((len(mice), len(days)))
teth_p = np.zeros((len(mice), len(days)))

control_b[:,:] = np.nan
control_nb[:,:] = np.nan
control_p[:,:] = np.nan
tetl_b[:,:] = np.nan
tetl_nb[:,:] = np.nan
tetl_p[:,:] = np.nan
teth_b[:,:] = np.nan
teth_nb[:,:] = np.nan
teth_p[:,:] = np.nan

# ------------------------------------------------------------------------------ #

# function specific to this code

#function for speed per trial
def speed_per_trial(bins,saraharray, trarray):
    
    stopsarraybeacon = np.zeros(((bins.shape[0]), (trarray.shape[0]))) # rows == same number of bins
    stopsarraybeacon[:,:] = np.nan
    
    for tcount,trial in enumerate(trarray):# loops through each trial
        speedarray = saraharray[saraharray[:,9] ==trial,:] # get data only for each trial
        binarray = makebinarray(speedarray, bins)  # allocate each raw data row to a bi
        for bcount, b in enumerate(bins): #iterate round bins
            barray = speedarray[binarray[:,0] == b,2] # get data for each bin only
            speedmean = np.nanmean(barray, axis= 0)
            stopsarraybeacon[bcount,tcount] = speedmean
            bcount+=1
    return stopsarraybeacon



# ------------------------------------------------------------------------------ #

#GET AND STORE STOPS DATA

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
        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]+0.1),1)
        
        # split data by trial type
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        
        # get trial number
        trialids_b = np.unique(dailymouse_b[:, 9])
        trialids_nb = np.unique(dailymouse_nb[:, 9])
        trialids_p = np.unique(dailymouse_p[:, 9])

        # get stops
        stops_b = speed_per_trial(bins,dailymouse_b, trialids_b)
        stops_nb = speed_per_trial(bins,dailymouse_nb, trialids_nb)
        stops_p = speed_per_trial(bins,dailymouse_p, trialids_p)
        
        #average
        beac = np.nanmean(stops_b,axis=1)
        nbeac = np.nanmean(stops_nb,axis=1)
        probe = np.nanmean(stops_p,axis=1)
        
        print('##...', mcount,day, '...##')
        # store data
        if mcount ==0 or mcount == 2 or mcount == 3 or mcount == 9:
            speeddiff = np.nanmean(beac[17:20])
            control_b[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            control_nb[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(probe[17:20])
            control_p[mcount,dcount] = speeddiff
        if mcount == 1 or mcount == 5 or mcount == 8 or mcount == 6: # high dorsal
            speeddiff = np.nanmean(beac[17:20])
            teth_b[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            teth_nb[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(probe[17:20])
            teth_p[mcount,dcount] = speeddiff
        if mcount == 7 or mcount == 10 or mcount == 4:
            speeddiff = np.nanmean(beac[17:20])
            tetl_b[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            tetl_nb[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(probe[17:20])
            tetl_p[mcount,dcount] = speeddiff
        
        mcount +=1



# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task15_b_0300.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,19.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,5.1)]# choose specific day/s

# Arrays to store data (output)
control_b1 = np.zeros((len(mice), len(days)))
control_nb1 = np.zeros((len(mice), len(days)))
control_p1 = np.zeros((len(mice), len(days)))
tetl_b1 = np.zeros((len(mice), len(days)))
tetl_nb1 = np.zeros((len(mice), len(days)))
tetl_p1 = np.zeros((len(mice), len(days)))
teth_b1 = np.zeros((len(mice), len(days)))
teth_nb1 = np.zeros((len(mice), len(days)))
teth_p1 = np.zeros((len(mice), len(days)))

control_b1[:,:] = np.nan
control_nb1[:,:] = np.nan
control_p1[:,:] = np.nan
tetl_b1[:,:] = np.nan
tetl_nb1[:,:] = np.nan
tetl_p1[:,:] = np.nan
teth_b1[:,:] = np.nan
teth_nb1[:,:] = np.nan
teth_p1[:,:] = np.nan

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
        stops_b = speed_per_trial(bins,dailymouse_b, trarray)
        stops_nb = speed_per_trial(bins,dailymouse_nb, trarray)
        stops_p = speed_per_trial(bins,dailymouse_p, trarray)
        
        #average
        beac = np.nanmean(stops_b,axis=1)
        nbeac = np.nanmean(stops_nb,axis=1)
        probe = np.nanmean(stops_p,axis=1)
        
        print('##...', mcount,day, '...##')
        # store data
        if mcount == 3 or mcount == 4:
            speeddiff = np.nanmean(beac[17:20])
            control_b1[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            control_nb1[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(probe[17:20])
            control_p1[mcount,dcount] = speeddiff
        if mcount == 0 or mcount == 1 or mcount == 2:
            if mcount == 0:
                print(np.nanmean(beac[17:20]), 'Speed')
            speeddiff = np.nanmean(beac[17:20])
            tetl_b1[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            tetl_nb1[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(probe[17:20])
            tetl_p1[mcount,dcount] = speeddiff
        
        mcount +=1


# average over days for each mouse
start_con_beac = np.nanmean(np.vstack((control_b[:,:],control_b1[:,:])), axis = 0)
start_con_probe = np.nanmean(np.vstack((control_p[:,:],control_p1[:,:])), axis = 0)

start_teth_beac = np.nanmean(np.vstack((teth_b[:,:],teth_b1[:,:])), axis = 0)
start_teth_probe = np.nanmean(np.vstack((teth_p[:,:],teth_p1[:,:])), axis = 0)

start_tetl_beac = np.nanmean(np.vstack((tetl_b[:,:],tetl_b1[:,:])), axis = 0)
start_tetl_probe = np.nanmean(np.vstack((tetl_p[:,:],tetl_p1[:,:])), axis = 0)

sdstart_con_beac = np.nanstd(np.vstack((control_b[:,:],control_b1[:,:])), axis = 0)/math.sqrt(6)
sdstart_con_probe = np.nanstd(np.vstack((control_p[:,:],control_p1[:,:])), axis = 0)/math.sqrt(6)

sdstart_teth_beac = np.nanstd(np.vstack((teth_b[:,:],teth_b1[:,:])), axis = 0)/math.sqrt(6)
sdstart_teth_probe = np.nanstd(np.vstack((teth_p[:,:],teth_p1[:,:])), axis = 0)/math.sqrt(6)

sdstart_tetl_beac = np.nanstd(np.vstack((tetl_b[:,:],tetl_b1[:,:])), axis = 0)/math.sqrt(4)
sdstart_tetl_probe = np.nanstd(np.vstack((tetl_p[:,:],tetl_p1[:,:])), axis = 0)/math.sqrt(4)



# PLOT GRAPHS

bins = np.arange(0.5,18.5+1e-6,1) # track bins

fig = plt.figure(figsize = (10,3))
ax = fig.add_subplot(1,2,1)
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
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
ax.set_ylim(0)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax = plt.ylabel('Training day)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,2,2) #stops per trial
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins,start_con_probe,color = 'Black', label = 'Beaconed', linewidth = 2, markersize = 4) #plot becaoned trials
ax.errorbar(bins,start_con_probe,sdstart_con_probe, fmt = 'o', color = 'black',  capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.plot(bins,start_tetl_probe,color = 'blue',label = 'AAV-fl-GFP', linewidth = 2, markersize = 4) #plot becaoned trials
ax.errorbar(bins,start_tetl_probe,sdstart_tetl_probe, fmt = 'o', color = 'blue', capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.plot(bins,start_teth_probe,color = 'red',label = 'AAV-fl-GFP', linewidth = 2, markersize = 4) #plot becaoned trials
ax.errorbar(bins,start_teth_probe,sdstart_teth_probe, fmt = 'o', color = 'red', capsize = 2, capthick = 1, markersize = 4, elinewidth = 1.5)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,22)
ax.set_ylim(0)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=6) # set number of ticks on y axis
ax = plt.xlabel('Training day', fontsize=16, labelpad = 18)
fig.savefig('Plots/Figure5/Task15_SpeedBB_Days_0100'+'.png', dpi =200) # path to save file
plt.close()


n_groups = np.arange(3)
bar_width = 0.5
bins = np.arange(0.5,18.5+1e-6,1) # track bins

fig = plt.figure(figsize = (4,6))
ax = fig.add_subplot(1,1,1)
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
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
ax.set_ylim(0)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'y', nbins=8) # set number of ticks on y axis
ax = plt.ylabel('Training day)', fontsize=16, labelpad = 18)
plt.locator_params(axis = 'x', nbins  = 8)
plt.subplots_adjust(hspace = 1, wspace = .7,  bottom = 0.25, left = 0.1, right = 0.9, top = .9)

fig.savefig('Plots/Figure5/Task15_SpeedBB_Days_0200'+'.png', dpi =200) # path to save file
plt.close()


control_b = np.swapaxes(control_b, 1,0)
teth_b = np.swapaxes(teth_b, 1,0)
tetl_b = np.swapaxes(tetl_b, 1,0)

control_p = np.swapaxes(control_p, 1,0)
teth_p = np.swapaxes(teth_p, 1,0)
tetl_p = np.swapaxes(tetl_p, 1,0)

control_b1 = np.swapaxes(control_b1, 1,0)
teth_b1 = np.swapaxes(teth_b1, 1,0)
tetl_b1 = np.swapaxes(tetl_b1, 1,0)

control_p1 = np.swapaxes(control_p1, 1,0)
teth_p1 = np.swapaxes(teth_p1, 1,0)
tetl_p1 = np.swapaxes(tetl_p1, 1,0)


start_con_beac= np.hstack((control_b[:,0],control_b[:,2], control_b[:,3],control_b[:,9],control_b1[:,3],control_b1[:,4]))
start_teth_beac = np.hstack((teth_b[:,1],teth_b[:,5],teth_b[:,6],teth_b[:,8]))
start_tetl_beac = np.hstack((tetl_b[:,4],tetl_b[:,7],tetl_b[:,10],tetl_b[:,0],tetl_b1[:,1],tetl_b1[:,2]))

start_con_probe= np.hstack((control_p[:,0],control_p[:,2], control_p[:,3],control_p[:,9],control_p1[:,3],control_p1[:,4]))
start_teth_probe = np.hstack((teth_p[:,1],teth_p[:,5],teth_p[:,6],teth_p[:,8]))
start_tetl_probe = np.hstack((tetl_p[:,4],tetl_p[:,7],tetl_p[:,10],tetl_p[:,0],tetl_p1[:,1],tetl_p1[:,2]))



# SAVE DATA FOR R

ltelc = ("lTeLC","lTeLC","lTeLC","lTeLC","lTeLC","lTeLC")
htelc = ("hTeLC","hTeLC","hTeLC","hTeLC")
gfp = ("GFP","GFP","GFP" ,"GFP" ,"GFP" ,"GFP")
genotype = np.hstack((np.repeat(htelc, 19),np.repeat(ltelc, 19),np.repeat(gfp, 19)))
d = np.arange(1,19.1,1)
day = np.hstack((d,d,d,d,d,d,d,d,d,d,d,d,d,d,d,d))


m1 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]); m2 = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]);m3 = np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]); m4 = np.array([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]);m5 = np.array([5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]); m6 = np.array([6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]); m7 = np.array([7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]); m8 = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8])
m9 = np.array([9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]); m10 = np.array([10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]);m11 = np.array([11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11]); m12 = np.array([12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12]);m13 = np.array([13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13]); m14 = np.array([14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14]); m15 = np.array([15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]); m16 = np.array([16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16])


mice = np.hstack((m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16))

data_b=np.hstack((start_teth_beac,start_tetl_beac,start_con_beac))
data_p=np.hstack((start_teth_probe,start_tetl_probe,start_con_probe))
data = np.vstack((genotype, mice, day, data_b,data_p )); data = np.transpose(data)
np.savetxt('Data_Output/Figure5/Figure5_SpeedBB_Days_0100.csv', data,fmt =  '%s', delimiter = ',', header = 'Genotype, Mouse, Day, Beaconed, Probe')










con_beac= np.hstack((np.nanmean(control_b[15:19,0]),np.nanmean(control_b[15:19,2]), np.nanmean(control_b[15:19,3]),np.nanmean(control_b[15:19,9]),np.nanmean(control_b1[15:19,3]),np.nanmean(control_b1[15:19,4])))
teth_beac = np.hstack((np.nanmean(teth_b[15:19,1]),np.nanmean(teth_b[15:19,5]),np.nanmean(teth_b[15:19,6]),np.nanmean(teth_b[15:19,8])))
tetl_beac = np.hstack((np.nanmean(tetl_b[15:19,4]),np.nanmean(tetl_b[15:19,7]),np.nanmean(tetl_b[15:19,10]),np.nanmean(tetl_b[15:19,0]),np.nanmean(tetl_b1[15:19,1]),np.nanmean(tetl_b1[15:19,2])))

con_probe= np.hstack((np.nanmean(control_p[15:19,0]),np.nanmean(control_p[15:19,2]), np.nanmean(control_p[15:19,3]),np.nanmean(control_p[15:19,9]),np.nanmean(control_p1[15:19,3]),np.nanmean(control_p1[15:19,4])))
teth_probe = np.hstack((np.nanmean(teth_p[15:19,1]),np.nanmean(teth_p[15:19,5]),np.nanmean(teth_p[15:19,6]),np.nanmean(teth_p[15:19,8])))
tetl_probe = np.hstack((np.nanmean(tetl_p[15:19,4]),np.nanmean(tetl_p[15:19,7]),np.nanmean(tetl_p[15:19,10]),np.nanmean(tetl_p[15:19,0]),np.nanmean(tetl_p1[15:19,1]),np.nanmean(tetl_p1[15:19,2])))



teth_probe = np.zeros((4)); teth_probe[:] = np.nan
mice_b = np.hstack((con_beac,tetl_beac,teth_beac))
mice_p = np.hstack((con_probe,tetl_probe, teth_probe))
genotype = np.array(("GFP","GFP","GFP" ,"GFP" ,"GFP" ,"GFP" ,"lTeLC","lTeLC","lTeLC","lTeLC","lTeLC","lTeLC","hTeLC","hTeLC","hTeLC","hTeLC"))

data = np.vstack((genotype, mice_b, mice_p)); data=np.transpose(data)

np.savetxt('Data_Output/Figure5/Figure5_SpeedBBAvg_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Genotype,Beaconed,Probe')



