# -*- coding: utf-8 -*-
"""

@author: Sarah Tennant


### Calculates Z-scores for each bin of the track
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


# ----------------------------------------------------------------------------------------------------- #

### get zscores for first week

# ----------------------------------------------------------------------------------------------------- #

# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task15_0100.h5' # raw data files

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,5.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,11.1)]

# empty arrays for storing data
control_beac = np.zeros((len(days),len(mice), 20));control_nbeac = np.zeros((len(days),len(mice), 20));control_probe = np.zeros((len(days),len(mice), 20))
control_beac[:,:,:] = np.nan;control_nbeac[:,:,:] = np.nan; control_probe[:,:,:] = np.nan
tetl_beac = np.zeros((len(days),len(mice), 20));tetl_nbeac = np.zeros((len(days),len(mice), 20));tetl_probe = np.zeros((len(days),len(mice), 20))
tetl_beac[:,:,:] = np.nan;tetl_nbeac[:,:,:] = np.nan; tetl_probe[:,:,:] = np.nan
teth_beac = np.zeros((len(days),len(mice), 20));teth_nbeac = np.zeros((len(days),len(mice), 20));teth_probe = np.zeros((len(days),len(mice), 20))
teth_beac[:,:,:] = np.nan;teth_nbeac[:,:,:] = np.nan; teth_probe[:,:,:] = np.nan

# loop thorugh mice and days to get data
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
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)# delete all data not on probe
        
        #extract stops
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)
        
        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)
        
        #get trials
        trialids_b = np.unique(stopsdata_b[:, 2])
        if stopsdata_nb.size>0:
            trialids_nb = np.unique(stopsdata_nb[:, 2])
        if stopsdata_p.size>0:
            trialids_p = np.unique(stopsdata_p[:, 2])
                
        # Shuffle stops data & get zscores
        if mcount ==0 or mcount == 2 or mcount == 3 or mcount == 9: # if control mouse
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b ) # get average real and shuffled stops per location bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate zscore
            control_beac[dcount,mcount,:] = zscore_b
            if stopsdata_nb.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb ) # get average real and shuffled stops per location bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                control_nbeac[dcount, mcount,:] = zscore_nb
            if stopsdata_p.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p ) # get average real and shuffled stops per location bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                control_probe[dcount, mcount,:] = zscore_p
        
        if mcount == 1 or mcount == 5 or mcount == 6 or mcount == 8: # if high telc mouse
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b )# get average real and shuffled stops per location bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
            teth_beac[dcount,mcount,:] = zscore_b
            if stopsdata_nb.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb )# get average real and shuffled stops per location bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                teth_nbeac[dcount, mcount,:] = zscore_nb
            if stopsdata_p.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p )# get average real and shuffled stops per location bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                teth_probe[dcount, mcount,:] = zscore_p
                    
        if mcount == 4 or mcount == 7 or mcount == 10: # if low telc mouse
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b )# get average real and shuffled stops per location bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
            tetl_beac[dcount,mcount,:] = zscore_b
            if stopsdata_nb.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb )# get average real and shuffled stops per location bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                tetl_nbeac[dcount, mcount,:] = zscore_nb
            if stopsdata_p.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p )# get average real and shuffled stops per location bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                tetl_probe[dcount, mcount,:] = zscore_p
    mcount +=1


# COLLECT DATA FOR TASK 12

# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task15_b_0300.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,5.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,5.1)]# choose specific day/s

# Data stores
control_beac2 = np.zeros((len(days),len(mice), 20));control_nbeac2 = np.zeros((len(days),len(mice), 20));control_probe2 = np.zeros((len(days),len(mice), 20))
control_beac2[:,:,:] = np.nan;control_nbeac2[:,:,:] = np.nan; control_probe2[:,:,:] = np.nan

tetl_beac2 = np.zeros((len(days),len(mice), 20));tetl_nbeac2 = np.zeros((len(days),len(mice), 20));tetl_probe2 = np.zeros((len(days),len(mice), 20))
tetl_beac2[:,:,:] = np.nan;tetl_nbeac2[:,:,:] = np.nan; tetl_probe2[:,:,:] = np.nan

teth_beac2 = np.zeros((len(days),len(mice), 20));teth_nbeac2 = np.zeros((len(days),len(mice), 20));teth_probe2 = np.zeros((len(days),len(mice), 20))
teth_beac2[:,:,:] = np.nan;teth_nbeac2[:,:,:] = np.nan; teth_probe2[:,:,:] = np.nan

# loop thorugh mice and days to get data
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
        
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)# delete all data not on non beaconed tracks
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)# delete all data not on probe
        
        #extract stops
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)
        
        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)
        
        #get trials
        trialids_b = np.unique(stopsdata_b[:, 2])
        if stopsdata_nb.size>0:
            trialids_nb = np.unique(stopsdata_nb[:, 2])
        if stopsdata_p.size>0:
            trialids_p = np.unique(stopsdata_p[:, 2])
                
        # Shuffle stops data & get zscores
        if mcount == 3 or mcount == 4:
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b )# get average real and shuffled stops per location bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
            control_beac2[dcount,mcount,:] = zscore_b
            if stopsdata_nb.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb )# get average real and shuffled stops per location bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                control_nbeac2[dcount, mcount,:] = zscore_nb
            if stopsdata_p.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p )# get average real and shuffled stops per location bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                control_probe2[dcount, mcount,:] = zscore_p
        
        if mcount == 0 or mcount == 1 or mcount == 2:
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b )# get average real and shuffled stops per location bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
            tetl_beac2[dcount,mcount,:] = zscore_b
            if stopsdata_nb.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb )# get average real and shuffled stops per location bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                tetl_nbeac2[dcount, mcount,:] = zscore_nb
            if stopsdata_p.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p )# get average real and shuffled stops per location bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                tetl_probe2[dcount, mcount,:] = zscore_p

    mcount +=1


# get zscores for the outbound and homebound zones of track

reward_con_beaconed1 = np.nanmean(np.nanmean(np.hstack((control_beac[:,:,11],control_beac2[:,:,11])), axis = 0), axis = 0)
reward_con_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((control_nbeac[:,:,11],control_nbeac2[:,:,11])), axis =0), axis = 0)
reward_con_probe1 = np.nanmean(np.nanmean(np.hstack((control_probe[:,:,11],control_probe2[:,:,11])), axis = 0), axis = 0)

reward_tetl_beaconed1 = np.nanmean(np.nanmean(np.hstack((tetl_beac[:,:,11],tetl_beac2[:,:,11])), axis = 0), axis = 0)
reward_tetl_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((tetl_nbeac[:,:,11],tetl_nbeac2[:,:,11])), axis =0), axis = 0)
reward_tetl_probe1 = np.nanmean(np.nanmean(np.hstack((tetl_probe[:,:,11],tetl_probe2[:,:,11])), axis = 0), axis = 0)

reward_teth_beaconed1 = np.nanmean(np.nanmean(np.hstack((teth_beac[:,:,11],teth_beac2[:,:,11])), axis = 0), axis = 0)
reward_teth_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((teth_nbeac[:,:,11],teth_nbeac2[:,:,11])), axis =0), axis = 0)
reward_teth_probe1 = np.nanmean(np.nanmean(np.hstack((teth_probe[:,:,11],teth_probe2[:,:,11])), axis = 0), axis = 0)

bb_con_beaconed1 = np.nanmean(np.nanmean(np.hstack((control_beac[:,:,17],control_beac2[:,:,17])), axis = 0), axis = 0)
bb_con_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((control_nbeac[:,:,17],control_nbeac2[:,:,17])), axis =0), axis = 0)
bb_con_probe1 = np.nanmean(np.nanmean(np.hstack((control_probe[:,:,17],control_probe2[:,:,17])), axis = 0), axis = 0)

bb_tetl_beaconed1 = np.nanmean(np.nanmean(np.hstack((tetl_beac[:,:,17],tetl_beac2[:,:,17])), axis = 0), axis = 0)
bb_tetl_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((tetl_nbeac[:,:,17],tetl_nbeac2[:,:,17])), axis =0), axis = 0)
bb_tetl_probe1 = np.nanmean(np.nanmean(np.hstack((tetl_probe[:,:,17],tetl_probe2[:,:,17])), axis = 0), axis = 0)

bb_teth_beaconed1 = np.nanmean(np.nanmean(np.hstack((teth_beac[:,:,17],teth_beac2[:,:,17])), axis = 0), axis = 0)
bb_teth_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((teth_nbeac[:,:,17],teth_nbeac2[:,:,17])), axis =0), axis = 0)
bb_teth_probe1 = np.nanmean(np.nanmean(np.hstack((teth_probe[:,:,17],teth_probe2[:,:,17])), axis = 0), axis = 0)

# find difference between homebound and outbound z scores
difference_con_beac =reward_con_beaconed1 - bb_con_beaconed1
difference_tetl_beac =reward_tetl_beaconed1 - bb_tetl_beaconed1
difference_teth_beac =reward_teth_beaconed1 - bb_teth_beaconed1

difference_con_probe =reward_con_probe1 - bb_con_probe1
difference_tetl_probe =reward_tetl_probe1 - bb_tetl_probe1
difference_teth_probe =reward_teth_probe1 - bb_teth_probe1




# average over days for all mice

con_beaconed1 = np.nanmean(np.nanmean(np.hstack((control_beac[:,:,:],control_beac2[:,:,:])), axis = 0), axis = 0)
con_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((control_nbeac[:,:,:],control_nbeac2[:,:,:])), axis =0), axis = 0)
con_probes1 = np.nanmean(np.nanmean(np.hstack((control_probe[:,:,:],control_probe2[:,:,:])), axis = 0), axis = 0)
sd_con_beaconed1 = np.nanstd(np.nanmean(np.hstack((control_beac[:,:,:],control_beac2[:,:,:])), axis = 0), axis = 0)/math.sqrt(6)
sd_con_nbeaconed1 = np.nanstd(np.nanmean(np.hstack((control_nbeac[:,:,:],control_nbeac2[:,:,:])), axis =0), axis = 0)/math.sqrt(6)
sd_con_probes1 = np.nanstd(np.nanmean(np.hstack((control_probe[:,:,:],control_probe2[:,:,:])), axis = 0), axis = 0)/math.sqrt(6)

tetl_beaconed1 = np.nanmean(np.nanmean(np.hstack((tetl_beac[:,:,:],tetl_beac2[:,:,:])), axis = 0), axis = 0)
tetl_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((tetl_nbeac[:,:,:],tetl_nbeac2[:,:,:])), axis =0), axis = 0)
tetl_probes1 = np.nanmean(np.nanmean(np.hstack((tetl_probe[:,:,:],tetl_probe2[:,:,:])), axis = 0), axis = 0)
sd_tetl_beaconed1 = np.nanstd(np.nanmean(np.hstack((tetl_beac[:,:,:],tetl_beac2[:,:,:])), axis = 0), axis = 0)/math.sqrt(6)
sd_tetl_nbeaconed1 = np.nanstd(np.nanmean(np.hstack((tetl_nbeac[:,:,:],tetl_nbeac2[:,:,:])), axis =0), axis = 0)/math.sqrt(6)
sd_tetl_probes1 = np.nanstd(np.nanmean(np.hstack((tetl_probe[:,:,:],tetl_probe2[:,:,:])), axis = 0), axis = 0)/math.sqrt(6)

teth_beaconed1 = np.nanmean(np.nanmean(np.hstack((teth_beac[:,:,:],teth_beac2[:,:,:])), axis = 0), axis = 0)
teth_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((teth_nbeac[:,:,:],teth_nbeac2[:,:,:])), axis =0), axis = 0)
teth_probes1 = np.nanmean(np.nanmean(np.hstack((teth_probe[:,:,:],teth_probe2[:,:,:])), axis = 0), axis = 0)
sd_teth_beaconed1 = np.nanstd(np.nanmean(np.hstack((teth_beac[:,:,:],teth_beac2[:,:,:])), axis = 0), axis = 0)/math.sqrt(4)
sd_teth_nbeaconed1 = np.nanstd(np.nanmean(np.hstack((teth_nbeac[:,:,:],teth_nbeac2[:,:,:])), axis =0), axis = 0)/math.sqrt(4)
sd_teth_probes1 = np.nanstd(np.nanmean(np.hstack((teth_probe[:,:,:],teth_probe2[:,:,:])), axis = 0), axis = 0)/math.sqrt(4)


# average and stack two experiments
con_beac = np.vstack((np.nanmean(control_beac,axis=0),np.nanmean(control_beac2,axis=0)))
con_nbeac = np.vstack((np.nanmean(control_nbeac,axis=0),np.nanmean(control_nbeac2,axis=0)))
con_probe = np.vstack((np.nanmean(control_probe,axis=0),np.nanmean(control_probe2,axis=0)))
con_beac = np.vstack((con_beac[0,:],con_beac[2,:],con_beac[3,:],con_beac[9,:],con_beac[14,:],con_beac[15,:]))
con_nbeac = np.vstack((con_nbeac[0,:],con_nbeac[2,:],con_nbeac[3,:],con_nbeac[9,:],con_nbeac[14,:],con_nbeac[15,:]))
con_probe = np.vstack((con_probe[0,:],con_probe[2,:],con_probe[3,:],con_probe[9,:],con_probe[14,:],con_probe[15,:]))

teth_beac = np.vstack((np.nanmean(teth_beac,axis=0),np.nanmean(teth_beac2,axis=0)))
teth_nbeac = np.vstack((np.nanmean(teth_nbeac,axis=0),np.nanmean(teth_nbeac2,axis=0)))
teth_probe = np.vstack((np.nanmean(teth_probe,axis=0),np.nanmean(teth_probe2,axis=0)))
teth_beac = np.vstack((teth_beac[1,:],teth_beac[5,:],teth_beac[6,:],teth_beac[8,:]))
teth_nbeac = np.vstack((teth_nbeac[1,:],teth_nbeac[5,:],teth_nbeac[6,:],teth_nbeac[8,:]))
teth_probe = np.vstack((teth_probe[1,:],teth_probe[5,:],teth_probe[6,:],teth_probe[8,:]))

tetl_beac = np.vstack((np.nanmean(tetl_beac,axis=0),np.nanmean(tetl_beac2,axis=0)))
tetl_nbeac = np.vstack((np.nanmean(tetl_nbeac,axis=0),np.nanmean(tetl_nbeac2,axis=0)))
tetl_probe = np.vstack((np.nanmean(tetl_probe,axis=0),np.nanmean(tetl_probe2,axis=0)))
tetl_beac = np.vstack((tetl_beac[4,:], tetl_beac[7,:], tetl_beac[10,:], tetl_beac[11,:],tetl_beac[12,:],tetl_beac[13,:]))
tetl_nbeac = np.vstack((tetl_nbeac[4,:], tetl_nbeac[7,:], tetl_nbeac[10,:], tetl_nbeac[11,:],tetl_nbeac[12,:],tetl_nbeac[13,:]))
tetl_probe = np.vstack((tetl_probe[4,:], tetl_probe[7,:], tetl_probe[10,:], tetl_probe[11,:],tetl_probe[12,:],tetl_probe[13,:]))

"""
# save data for R

x_lTeLC = np.vstack((tetl_beac, tetl_nbeac, tetl_probe))
x_hTeLC = np.vstack((teth_beac, teth_nbeac, teth_probe))
x_GFP = np.vstack((con_beac, con_nbeac, con_probe))
x = np.vstack((x_lTeLC,x_hTeLC,x_GFP))

ltelc = ("lTeLC","lTeLC","lTeLC","lTeLC","lTeLC","lTeLC")
htelc = ("hTeLC","hTeLC","hTeLC","hTeLC")
gfp = ("GFP","GFP","GFP" ,"GFP" ,"GFP" ,"GFP")
genotype = np.hstack((ltelc, ltelc, ltelc, htelc, htelc, htelc, gfp, gfp, gfp))

ltelc_m = (5,8,11,12,13,14)
htelc_m = (2,6,7,9)
gfp_m = (1,3,4 ,10 ,15 ,16)
mouse = np.hstack((ltelc_m, ltelc_m, ltelc_m, htelc_m, htelc_m, htelc_m, gfp_m, gfp_m, gfp_m))
trial1 = np.array([1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]); trial2 = np.array([1,1,1,1,2,2,2,2,3,3,3,3]);trial = np.hstack((trial1, trial2, trial1))

data = np.vstack((mouse,genotype, trial)); data=np.transpose(data)
data = np.hstack((data,x))

np.savetxt('Data_Output/Figure5/Figure5_C&D_Week1_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Mouse,Virus, Trial, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20')

"""

# ----------------------------------------------------------------------------------------------------- #

### get zscores for final week

# ----------------------------------------------------------------------------------------------------- #


# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task15_0100.h5' # raw data files

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(15,19.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,11.1)]

#store data
control_beac = np.zeros((len(days),len(mice), 20));control_nbeac = np.zeros((len(days),len(mice), 20));control_probe = np.zeros((len(days),len(mice), 20))
control_beac[:,:,:] = np.nan;control_nbeac[:,:,:] = np.nan; control_probe[:,:,:] = np.nan

tetl_beac = np.zeros((len(days),len(mice), 20));tetl_nbeac = np.zeros((len(days),len(mice), 20));tetl_probe = np.zeros((len(days),len(mice), 20))
tetl_beac[:,:,:] = np.nan;tetl_nbeac[:,:,:] = np.nan; tetl_probe[:,:,:] = np.nan

teth_beac = np.zeros((len(days),len(mice), 20));teth_nbeac = np.zeros((len(days),len(mice), 20));teth_probe = np.zeros((len(days),len(mice), 20))
teth_beac[:,:,:] = np.nan;teth_nbeac[:,:,:] = np.nan; teth_probe[:,:,:] = np.nan

#loop to get data
for mcount,mouse in enumerate(mice):
    print(mcount)
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
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)
        
        #get trials
        trialids_b = np.unique(stopsdata_b[:, 2])
        if stopsdata_nb.size>0:
            trialids_nb = np.unique(stopsdata_nb[:, 2])
        if stopsdata_p.size>0:
            trialids_p = np.unique(stopsdata_p[:, 2])
                
        # Shuffle stops data & get zscores
        if mcount == 2 or mcount == 3 or mcount == 9: # if control mouse
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b )# get average real and shuffled stops per location bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
            control_beac[dcount,mcount,:] = zscore_b
            if stopsdata_nb.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb )# get average real and shuffled stops per location bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                control_nbeac[dcount, mcount,:] = zscore_nb
            if stopsdata_p.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p )# get average real and shuffled stops per location bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                control_probe[dcount, mcount,:] = zscore_p
        if mcount == 0 and dcount <2: # if control mouse # see methods about why these days are discounted
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b )# get average real and shuffled stops per location bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
            control_beac[dcount,mcount,:] = zscore_b
            if stopsdata_nb.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb )# get average real and shuffled stops per location bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                control_nbeac[dcount, mcount,:] = zscore_nb
            if stopsdata_p.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p )# get average real and shuffled stops per location bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                control_probe[dcount, mcount,:] = zscore_p
        
        if mcount == 1 or mcount == 5 or mcount == 6 or mcount == 8: # if high telc mouse
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b )# get average real and shuffled stops per location bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
            teth_beac[dcount,mcount,:] = zscore_b
            if stopsdata_nb.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb )# get average real and shuffled stops per location bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                teth_nbeac[dcount, mcount,:] = zscore_nb
            if stopsdata_p.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p )# get average real and shuffled stops per location bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                teth_probe[dcount, mcount,:] = zscore_p
                    
        if mcount == 4 or mcount == 7 or mcount == 10: # if low telc mouse
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b )# get average real and shuffled stops per location bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
            tetl_beac[dcount,mcount,:] = zscore_b
            if stopsdata_nb.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb )# get average real and shuffled stops per location bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                tetl_nbeac[dcount, mcount,:] = zscore_nb
            if stopsdata_p.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p )# get average real and shuffled stops per location bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                tetl_probe[dcount, mcount,:] = zscore_p
    mcount +=1



# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task15_b_0300.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(15,19.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,5.1)]# choose specific day/s

# Data stores
control_beac2 = np.zeros((len(days),len(mice), 20));control_nbeac2 = np.zeros((len(days),len(mice), 20));control_probe2 = np.zeros((len(days),len(mice), 20))
control_beac2[:,:,:] = np.nan;control_nbeac2[:,:,:] = np.nan; control_probe2[:,:,:] = np.nan

tetl_beac2 = np.zeros((len(days),len(mice), 20));tetl_nbeac2 = np.zeros((len(days),len(mice), 20));tetl_probe2 = np.zeros((len(days),len(mice), 20))
tetl_beac2[:,:,:] = np.nan;tetl_nbeac2[:,:,:] = np.nan; tetl_probe2[:,:,:] = np.nan

teth_beac2 = np.zeros((len(days),len(mice), 20));teth_nbeac2 = np.zeros((len(days),len(mice), 20));teth_probe2 = np.zeros((len(days),len(mice), 20))
teth_beac2[:,:,:] = np.nan;teth_nbeac2[:,:,:] = np.nan; teth_probe2[:,:,:] = np.nan

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
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)
        
        #get trials
        trialids_b = np.unique(stopsdata_b[:, 2])
        if stopsdata_nb.size>0:
            trialids_nb = np.unique(stopsdata_nb[:, 2])
        if stopsdata_p.size>0:
            trialids_p = np.unique(stopsdata_p[:, 2])
                
        # Shuffle stops data & get zscores
        if mcount == 3 or mcount == 4:
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b )# get average real and shuffled stops per location bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
            control_beac2[dcount,mcount,:] = zscore_b
            if stopsdata_nb.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb )# get average real and shuffled stops per location bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                control_nbeac2[dcount, mcount,:] = zscore_nb
            if stopsdata_p.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p )# get average real and shuffled stops per location bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                control_probe2[dcount, mcount,:] = zscore_p
        if mcount == 0 or mcount == 1 or mcount == 2:
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b )# get average real and shuffled stops per location bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
            tetl_beac2[dcount,mcount,:] = zscore_b
            if stopsdata_nb.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb )# get average real and shuffled stops per location bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                tetl_nbeac2[dcount, mcount,:] = zscore_nb
            if stopsdata_p.size >0 :
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p )# get average real and shuffled stops per location bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)# calculate zscore
                tetl_probe2[dcount, mcount,:] = zscore_p

    mcount +=1


# ----------------------------------------------------------------------------------------------------- #














# get zscores for the outbound and homebound zones of track
reward_con_beaconed1 = np.nanmean(np.nanmean(np.hstack((control_beac[:,:,11],control_beac2[:,:,11])), axis = 0), axis = 0)
reward_con_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((control_nbeac[:,:,11],control_nbeac2[:,:,11])), axis =0), axis = 0)
reward_con_probe1 = np.nanmean(np.nanmean(np.hstack((control_probe[:,:,11],control_probe2[:,:,11])), axis = 0), axis = 0)

reward_tetl_beaconed1 = np.nanmean(np.nanmean(np.hstack((tetl_beac[:,:,11],tetl_beac2[:,:,11])), axis = 0), axis = 0)
reward_tetl_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((tetl_nbeac[:,:,11],tetl_nbeac2[:,:,11])), axis =0), axis = 0)
reward_tetl_probe1 = np.nanmean(np.nanmean(np.hstack((tetl_probe[:,:,11],tetl_probe2[:,:,11])), axis = 0), axis = 0)

reward_teth_beaconed1 = np.nanmean(np.nanmean(np.hstack((teth_beac[:,:,11],teth_beac2[:,:,11])), axis = 0), axis = 0)
reward_teth_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((teth_nbeac[:,:,11],teth_nbeac2[:,:,11])), axis =0), axis = 0)
reward_teth_probe1 = np.nanmean(np.nanmean(np.hstack((teth_probe[:,:,11],teth_probe2[:,:,11])), axis = 0), axis = 0)

bb_con_beaconed1 = np.nanmean(np.nanmean(np.hstack((control_beac[:,:,17],control_beac2[:,:,17])), axis = 0), axis = 0)
bb_con_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((control_nbeac[:,:,17],control_nbeac2[:,:,17])), axis =0), axis = 0)
bb_con_probe1 = np.nanmean(np.nanmean(np.hstack((control_probe[:,:,17],control_probe2[:,:,17])), axis = 0), axis = 0)

bb_tetl_beaconed1 = np.nanmean(np.nanmean(np.hstack((tetl_beac[:,:,17],tetl_beac2[:,:,17])), axis = 0), axis = 0)
bb_tetl_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((tetl_nbeac[:,:,17],tetl_nbeac2[:,:,17])), axis =0), axis = 0)
bb_tetl_probe1 = np.nanmean(np.nanmean(np.hstack((tetl_probe[:,:,17],tetl_probe2[:,:,17])), axis = 0), axis = 0)

bb_teth_beaconed1 = np.nanmean(np.nanmean(np.hstack((teth_beac[:,:,17],teth_beac2[:,:,17])), axis = 0), axis = 0)
bb_teth_nbeaconed1 = np.nanmean(np.nanmean(np.hstack((teth_nbeac[:,:,17],teth_nbeac2[:,:,17])), axis =0), axis = 0)
bb_teth_probe1 = np.nanmean(np.nanmean(np.hstack((teth_probe[:,:,17],teth_probe2[:,:,17])), axis = 0), axis = 0)


# find difference between outbound and homebound zscores
difference_con_beac1 =reward_con_beaconed1 - bb_con_beaconed1
difference_tetl_beac1 =reward_tetl_beaconed1 - bb_tetl_beaconed1
difference_teth_beac1 =reward_teth_beaconed1 - bb_teth_beaconed1

difference_con_probe1 =reward_con_probe1 - bb_con_probe1
difference_tetl_probe1 =reward_tetl_probe1 - bb_tetl_probe1
difference_teth_probe1 =reward_teth_probe1 - bb_teth_probe1



# find difference between week 1 and 4
diff_con_beac = difference_con_beac1 - difference_con_beac
diff_tetl_beac = difference_tetl_beac1 - difference_tetl_beac
diff_teth_beac = difference_teth_beac1 - difference_teth_beac

diff_con_probe = difference_con_probe1 - difference_con_probe
diff_tetl_probe = difference_tetl_probe1 - difference_tetl_probe
diff_teth_probe = difference_teth_probe1 - difference_teth_probe




# plot graphs

fig = plt.figure(figsize=(4,6))
#gs = gridspec.GridSpec(1, 7)
ax = fig.add_subplot(1,1,1)
ax.plot(1,diff_con_beac, 'o', color = 'k')
#ax.errorbar(1,diff_con_beac,consd, fmt = 'o', color = 'k', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(2,diff_tetl_beac, 'o', color = 'blue')
#ax.errorbar(2,tetl,tetlsd, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(3,diff_teth_beac, 'o', color = 'red')

adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =15)
#ax.set_ylabel('Dist (cm)', fontsize=32, labelpad = 20)
plt.locator_params(axis = 'x', nbins  = 2)
plt.locator_params(axis = 'y', nbins  = 4)
ax.axhline(-5,linewidth=3, color="black")
ax.axvline(0.5,linewidth=3, color="black")
#ax.axvline(3.5,linewidth=3, color="black")
ax.set_ylim(-5,20)
ax.set_xlim(0.5,3.5)
plt.locator_params(axis = 'y', nbins  = 5)
ax.axhspan(88,100, linewidth = 0,facecolor='LimeGreen', alpha=0.2, hatch = '/') # bold line on the x axis
ax.axhline(30, linewidth = 1,color='Black', ls = '--') # bold line on the x axis
plt.locator_params(axis = 'x', nbins  = 3)
plt.subplots_adjust(hspace = 1, wspace = .7,  bottom = 0.25, left = 0.1, right = 0.9, top = .9)

plt.savefig('Plots/Figure5/Task15_ShuffledMeans_homebound_test' +' .png', dpi = 200)
plt.close()



# average over days for all mice

con_beaconed = np.nanmean(np.nanmean(np.hstack((control_beac[:,:,:],control_beac2[:,:,:])), axis = 0), axis = 0)
con_nbeaconed = np.nanmean(np.nanmean(np.hstack((control_nbeac[:,:,:],control_nbeac2[:,:,:])), axis =0), axis = 0)
con_probes = np.nanmean(np.nanmean(np.hstack((control_probe[:,:,:],control_probe2[:,:,:])), axis = 0), axis = 0)
sd_con_beaconed = np.nanstd(np.nanmean(np.hstack((control_beac[:,:,:],control_beac2[:,:,:])), axis = 0), axis = 0)/math.sqrt(6)
sd_con_nbeaconed = np.nanstd(np.nanmean(np.hstack((control_nbeac[:,:,:],control_nbeac2[:,:,:])), axis =0), axis = 0)/math.sqrt(6)
sd_con_probes = np.nanstd(np.nanmean(np.hstack((control_probe[:,:,:],control_probe2[:,:,:])), axis = 0), axis = 0)/math.sqrt(6)

tetl_beaconed = np.nanmean(np.nanmean(np.hstack((tetl_beac[:,:,:],tetl_beac2[:,:,:])), axis = 0), axis = 0)
tetl_nbeaconed = np.nanmean(np.nanmean(np.hstack((tetl_nbeac[:,:,:],tetl_nbeac2[:,:,:])), axis =0), axis = 0)
tetl_probes = np.nanmean(np.nanmean(np.hstack((tetl_probe[:,:,:],tetl_probe2[:,:,:])), axis = 0), axis = 0)
sd_tetl_beaconed = np.nanstd(np.nanmean(np.hstack((tetl_beac[:,:,:],tetl_beac2[:,:,:])), axis = 0), axis = 0)/math.sqrt(6)
sd_tetl_nbeaconed = np.nanstd(np.nanmean(np.hstack((tetl_nbeac[:,:,:],tetl_nbeac2[:,:,:])), axis =0), axis = 0)/math.sqrt(6)
sd_tetl_probes = np.nanstd(np.nanmean(np.hstack((tetl_probe[:,:,:],tetl_probe2[:,:,:])), axis = 0), axis = 0)/math.sqrt(6)

teth_beaconed = np.nanmean(np.nanmean(np.hstack((teth_beac[:,:,:],teth_beac2[:,:,:])), axis = 0), axis = 0)
teth_nbeaconed = np.nanmean(np.nanmean(np.hstack((teth_nbeac[:,:,:],teth_nbeac2[:,:,:])), axis =0), axis = 0)
teth_probes = np.nanmean(np.nanmean(np.hstack((teth_probe[:,:,:],teth_probe2[:,:,:])), axis = 0), axis = 0)
sd_teth_beaconed = np.nanstd(np.nanmean(np.hstack((teth_beac[:,:,:],teth_beac2[:,:,:])), axis = 0), axis = 0)/math.sqrt(4)
sd_teth_nbeaconed = np.nanstd(np.nanmean(np.hstack((teth_nbeac[:,:,:],teth_nbeac2[:,:,:])), axis =0), axis = 0)/math.sqrt(4)
sd_teth_probes = np.nanstd(np.nanmean(np.hstack((teth_probe[:,:,:],teth_probe2[:,:,:])), axis = 0), axis = 0)/math.sqrt(4)




# PLOT GRAPH


n_groups = np.arange(25)
bins = np.arange(0.5,19.5+1e-6,1) # track bins

fig = plt.figure(figsize = (12,3))
ax = fig.add_subplot(1,3,1)
ax.axvspan(44, 44+12, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.plot(bins*5,con_beaconed,color = 'Black',label = 'AAV-fl-GFP', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,con_beaconed-sd_con_beaconed,con_beaconed+sd_con_beaconed, facecolor = 'Black', alpha = 0.2)
ax.plot(bins*5,tetl_beaconed,color = 'blue',label = 'AVV-fl-TeLC\nHigh', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,tetl_beaconed-sd_tetl_beaconed,tetl_beaconed+sd_tetl_beaconed, facecolor = 'blue', alpha = 0.2)
ax.plot(bins*5,teth_beaconed,color = 'red',label = 'AVV-fl-TeLC\nLow', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,teth_beaconed-sd_teth_beaconed,teth_beaconed+sd_teth_beaconed, facecolor = 'red', alpha = 0.2)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,100)
ax.set_ylim(-10)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])

ax = fig.add_subplot(1,3,2) #stops per trial
ax.axvspan(44, 44+12, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins*5,con_probes,color = 'Black', label = 'Beaconed', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,con_probes-sd_con_probes,con_probes+sd_con_probes, facecolor = 'Black', alpha = 0.2)
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.plot(bins*5,tetl_probes,color = 'blue', label = 'Beaconed', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,tetl_probes-sd_tetl_probes,tetl_probes+sd_tetl_probes, facecolor = 'blue', alpha = 0.2)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,100)
ax.set_ylim(-10)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Figure5/Task15_ZscoreHist_histoverlay_0100'+'.png', dpi =200)
plt.close()






n_groups = np.arange(25)
bins = np.arange(0.5,19.5+1e-6,1) # track bins

fig = plt.figure(figsize = (12,3))
ax = fig.add_subplot(1,3,1)
ax.axvspan(44, 44+12, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.plot(bins*5,con_beaconed1,'--',color = 'black',label = 'AAV-fl-GFP', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,con_beaconed1-sd_con_beaconed1,con_beaconed1+sd_con_beaconed1, facecolor = 'black', alpha = 0.2)
ax.plot(bins*5,con_beaconed,color = 'black',label = 'AAV-fl-GFP', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,con_beaconed-sd_con_beaconed,con_beaconed+sd_con_beaconed, facecolor = 'black', alpha = 0.2)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,100)
ax.set_ylim(-10, 14)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['', '', ''])

ax = fig.add_subplot(1,3,2) #stops per trial
ax.axvspan(44, 44+12, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins*5,tetl_beaconed1,'--',color = 'blue',label = 'AVV-fl-TeLC\nHigh', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,tetl_beaconed1-sd_tetl_beaconed1,tetl_beaconed1+sd_tetl_beaconed1, facecolor = 'blue', alpha = 0.2)
ax.plot(bins*5,tetl_beaconed,color = 'blue',label = 'AVV-fl-TeLC\nHigh', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,tetl_beaconed-sd_tetl_beaconed,tetl_beaconed+sd_tetl_beaconed, facecolor = 'blue', alpha = 0.2)
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,100)
ax.set_ylim(-10, 14)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['', '', ''])

ax = fig.add_subplot(1,3,3)
ax.axvspan(44, 44+12, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.plot(bins*5,teth_beaconed1,'--',color = 'red',label = 'AVV-fl-TeLC\nLow', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,teth_beaconed1-sd_teth_beaconed1,teth_beaconed1+sd_teth_beaconed1, facecolor = 'red', alpha = 0.2)
ax.plot(bins*5,teth_beaconed,color = 'red',label = 'AVV-fl-TeLC\nLow', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,teth_beaconed-sd_teth_beaconed,teth_beaconed+sd_teth_beaconed, facecolor = 'red', alpha = 0.2)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,100)
ax.set_ylim(-10,14)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['', '', ''])

plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)

fig.savefig('Plots/Figure5/Task15_ZscoreHist_histoverlayW1&4_0100'+'.png', dpi =200)
plt.close()






# average over days for all mice
con_beac = np.vstack((np.nanmean(control_beac,axis=0),np.nanmean(control_beac2,axis=0)))
con_nbeac = np.vstack((np.nanmean(control_nbeac,axis=0),np.nanmean(control_nbeac2,axis=0)))
con_probe = np.vstack((np.nanmean(control_probe,axis=0),np.nanmean(control_probe2,axis=0)))
con_beac = np.vstack((con_beac[0,:],con_beac[2,:],con_beac[3,:],con_beac[9,:],con_beac[14,:],con_beac[15,:]))
con_nbeac = np.vstack((con_nbeac[0,:],con_nbeac[2,:],con_nbeac[3,:],con_nbeac[9,:],con_nbeac[14,:],con_nbeac[15,:]))
con_probe = np.vstack((con_probe[0,:],con_probe[2,:],con_probe[3,:],con_probe[9,:],con_probe[14,:],con_probe[15,:]))

teth_beac = np.vstack((np.nanmean(teth_beac,axis=0),np.nanmean(teth_beac2,axis=0)))
teth_nbeac = np.vstack((np.nanmean(teth_nbeac,axis=0),np.nanmean(teth_nbeac2,axis=0)))
teth_probe = np.vstack((np.nanmean(teth_probe,axis=0),np.nanmean(teth_probe2,axis=0)))
teth_beac = np.vstack((teth_beac[1,:],teth_beac[5,:],teth_beac[6,:],teth_beac[8,:]))
teth_nbeac = np.vstack((teth_nbeac[1,:],teth_nbeac[5,:],teth_nbeac[6,:],teth_nbeac[8,:]))
teth_probe = np.vstack((teth_probe[1,:],teth_probe[5,:],teth_probe[6,:],teth_probe[8,:]))

tetl_beac = np.vstack((np.nanmean(tetl_beac,axis=0),np.nanmean(tetl_beac2,axis=0)))
tetl_nbeac = np.vstack((np.nanmean(tetl_nbeac,axis=0),np.nanmean(tetl_nbeac2,axis=0)))
tetl_probe = np.vstack((np.nanmean(tetl_probe,axis=0),np.nanmean(tetl_probe2,axis=0)))
tetl_beac = np.vstack((tetl_beac[4,:], tetl_beac[7,:], tetl_beac[10,:], tetl_beac[11,:],tetl_beac[12,:],tetl_beac[13,:]))
tetl_nbeac = np.vstack((tetl_nbeac[4,:], tetl_nbeac[7,:], tetl_nbeac[10,:], tetl_nbeac[11,:],tetl_nbeac[12,:],tetl_nbeac[13,:]))
tetl_probe = np.vstack((tetl_probe[4,:], tetl_probe[7,:], tetl_probe[10,:], tetl_probe[11,:],tetl_probe[12,:],tetl_probe[13,:]))


"""
# get data for R

x_lTeLC = np.vstack((tetl_beac, tetl_nbeac, tetl_probe))
x_hTeLC = np.vstack((teth_beac, teth_nbeac, teth_probe))
x_GFP = np.vstack((con_beac, con_nbeac, con_probe))
x = np.vstack((x_lTeLC,x_hTeLC,x_GFP))

ltelc = ("lTeLC","lTeLC","lTeLC","lTeLC","lTeLC","lTeLC")
htelc = ("hTeLC","hTeLC","hTeLC","hTeLC")
gfp = ("GFP","GFP","GFP" ,"GFP" ,"GFP" ,"GFP")
genotype = np.hstack((ltelc, ltelc, ltelc, htelc, htelc, htelc, gfp, gfp, gfp))

ltelc_m = (5,8,11,12,13,14)
htelc_m = (2,6,7,9)
gfp_m = (1,3,4 ,10 ,15 ,16)
mouse = np.hstack((ltelc_m, ltelc_m, ltelc_m, htelc_m, htelc_m, htelc_m, gfp_m, gfp_m, gfp_m))
trial1 = np.array([1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]); trial2 = np.array([1,1,1,1,2,2,2,2,3,3,3,3]);trial = np.hstack((trial1, trial2, trial1))

data = np.vstack((mouse,genotype, trial)); data=np.transpose(data)
data = np.hstack((data,x))

np.savetxt('Data_Output/Figure5/Figure4_C&D_Week4_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Mouse,Virus, Trial, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20')
"""


