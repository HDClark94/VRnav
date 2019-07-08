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

# ------------------------------------------------------------------------------ #

marray = np.loadtxt('SummaryData/SummaryData_T13.txt', dtype = 'S10',delimiter = '\t')
filename = 'Data_Input/Behaviour_DataFiles/Task15_0100.h5' # raw data files
days = ['Day' + str(int(x)) for x in np.arange(15,19.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,11.1)]

bins = np.arange(0.5,19.5,1) # track bins
#daybins = np.arange(0,19.1, 1) # days

REAL_LENGTH = 200
HDF_LENGTH = 20
SCALE = HDF_LENGTH/REAL_LENGTH
BINNR = 20
SHUFFLE_N = 1000
STOP_THRESHOLD = 0.7

# ARRAYS FOR STORING DATA FOR ALL MICE ON ALL DAYS
#experiment 13

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

# function to get location of stops per trial

#function for speed per trial
def speed_per_trial(bins,saraharray, trarray):
    
    stopsarraybeacon = np.zeros(((bins.shape[0]), (trarray.shape[0]))) # rows == same number of bins
    stopsarraybeacon[:,:] = np.nan
    
    for tcount,trial in enumerate(trarray):# loops through each trial
        speedarray = saraharray[saraharray[:,9] ==trial,:] # get data only for each trial
        #print(speedarray.shape,'speed')
        binarray = makebinarray(speedarray, bins)  # allocate each raw data row to a bi
        #print(binarray.shape, 'binarray')
        for bcount, b in enumerate(bins): #iterate round bins
            barray = speedarray[binarray[:,0] == b,2] # get data for each bin only
            #print(barray, 'barray')
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
        dayb = day.encode('UTF-8')#""""
        mouseb = mouse.encode('UTF-8') #required for importing string from marray in python3
        
        #marraybym = marray[marray[:,0]==dayb,:] # get descriptive data for session
        trialarray = saraharray[:,9] # makes an array of trial number per row in saraharray
        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]+0.1),1)
        # split data by trial type
        trialarray = maketrialarray(saraharray)
        saraharray[:,9] = trialarray[:,0]
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        
        #print(dailymouse_nb.shape)
        # get trial number
        trialids_b = np.unique(dailymouse_b[:, 9])
        trialids_nb = np.unique(dailymouse_nb[:, 9])
        trialids_p = np.unique(dailymouse_p[:, 9])
        #print(trialids_b,trialids_nb,trialids_p)
        # get stops
        stops_b = speed_per_trial(bins,dailymouse_b, trialids_b)
        stops_nb = speed_per_trial(bins,dailymouse_nb, trialids_nb)
        stops_p = speed_per_trial(bins,dailymouse_p, trialids_p)
        #print('STOPS',stops_nb)
        beac = np.nanmean(stops_b,axis=1)
        nbeac = np.nanmean(stops_nb,axis=1)
        probe = np.nanmean(stops_p,axis=1)
        
        print('##...', mcount,day, '...##')
        # store data
        if mcount == 2 or mcount == 3 or mcount == 9:
            speeddiff = np.nanmean(beac[17:20])
            control_b[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            control_nb[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            control_p[mcount,dcount] = speeddiff
        if mcount == 0 and dcount <2:
            speeddiff = beac[11] - beac[17]
            control_b[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            control_nb[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            control_p[mcount,dcount] = speeddiff
        if mcount == 1 or mcount == 5 or mcount == 8 or mcount == 6: # high dorsal
            speeddiff = np.nanmean(beac[17:20])
            teth_b[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            teth_nb[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            teth_p[mcount,dcount] = speeddiff
        if mcount == 7 or mcount == 10 or mcount == 4:
            speeddiff = np.nanmean(beac[17:20])
            tetl_b[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            tetl_nb[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            tetl_p[mcount,dcount] = speeddiff
        
        mcount +=1




marray = np.loadtxt('SummaryData/SummaryData_T15B.txt', dtype = 'S10',delimiter = '\t')
filename = 'Data_Input/Behaviour_DataFiles/Task15_b_0300.h5'
days = ['Day' + str(int(x)) for x in np.arange(15,19.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,8.1)]# choose specific day/s
# Stores
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
        dayb = day.encode('UTF-8')#""""
        mouseb = mouse.encode('UTF-8') #required for importing string from marray in python3
        marraybym = marray[marray[:,0]==dayb,:] # get descriptive data for session
        trialarray = saraharray[:,9] # makes an array of trial number per row in saraharray
        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]+0.1),1)
        tt = float(marraybym[marraybym[:,1]==mouseb,5][0]) #
        
        trialarray = maketrialarray(saraharray)
        saraharray[:,9] = trialarray[:,0]
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
        
        beac = np.nanmean(stops_b,axis=1)
        nbeac = np.nanmean(stops_nb,axis=1)
        probe = np.nanmean(stops_p,axis=1)
        
        print('##...', mcount,day, '...##')
        #STORE RATIOS
        if mcount == 3 or mcount == 4:
            speeddiff = np.nanmean(beac[17:20])
            control_b1[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            control_nb1[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            control_p1[mcount,dcount] = speeddiff
        if mcount == 0 or mcount == 1 or mcount == 2:
            speeddiff = np.nanmean(beac[17:20])
            tetl_b1[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            tetl_nb1[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            tetl_p1[mcount,dcount] = speeddiff
        
        mcount +=1


#------------------------------------------------------------------------------------#

"""
    # average over days for all mice
    con_beac1 = np.nanmean(np.vstack((control_b,control_b1)),axis=1)
    con_probe1 = np.nanmean(np.vstack((control_p,control_p1)),axis=1)
    sd_con_beac1 = np.nanstd(np.vstack((control_b,control_b1)), axis = 1)/math.sqrt(6)
    sd_con_probe1 = np.nanstd(np.vstack((control_p,control_p1)), axis = 1)/math.sqrt(6)
    
    tetl_beac1 = np.nanmean(np.vstack((tetl_b,tetl_b1)),axis=1)
    tetl_probe1 = np.nanmean(np.vstack((tetl_p,tetl_p1)),axis=1)
    sd_tetl_beac1 = np.nanstd(np.vstack((tetl_b,tetl_b1)), axis = 1)/math.sqrt(6)
    sd_tetl_probe1 = np.nanstd(np.vstack((tetl_p,tetl_p1)), axis = 1)/math.sqrt(6)
    
    teth_beac1 = np.nanmean(teth_b,axis=1)
    teth_probe1 = np.nanmean(teth_p,axis=1)
    sd_teth_beac1 = np.nanstd(teth_b, axis = 1)/math.sqrt(4)
    sd_teth_probe1 = np.nanstd(teth_p, axis = 1)/math.sqrt(4)
    """

# average over days for all mice
con_beac_week4 = np.vstack((control_b,control_b1))
con_probe_week4 = np.vstack((control_p,control_p1))

tetl_beac_week4 = np.vstack((tetl_b,tetl_b1))
tetl_probe_week4 = np.vstack((tetl_p,tetl_p1))

teth_beac_week4 = teth_b
teth_probe_week4 = teth_p




# ---------------------------------------------------------------------------------------- #




## WEEK 1

marray = np.loadtxt('SummaryData/SummaryData_T13.txt', dtype = 'S10',delimiter = '\t')
filename = 'DataFiles/Task15_0100.h5' # raw data files
days = ['Day' + str(int(x)) for x in np.arange(1,5.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,11.1)]

bins = np.arange(0.5,19.5,1) # track bins
#daybins = np.arange(0,19.1, 1) # days

REAL_LENGTH = 200
HDF_LENGTH = 20
SCALE = HDF_LENGTH/REAL_LENGTH
BINNR = 20
SHUFFLE_N = 1000
STOP_THRESHOLD = 0.7

# ARRAYS FOR STORING DATA FOR ALL MICE ON ALL DAYS
#experiment 13

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


#GET AND STORE STOPS DATA

for dcount,day in enumerate(days):
    for mcount,mouse in enumerate(mice):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        dayb = day.encode('UTF-8')#""""
        mouseb = mouse.encode('UTF-8') #required for importing string from marray in python3
        
        #marraybym = marray[marray[:,0]==dayb,:] # get descriptive data for session
        trialarray = saraharray[:,9] # makes an array of trial number per row in saraharray
        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]+0.1),1)
        # split data by trial type
        trialarray = maketrialarray(saraharray)
        saraharray[:,9] = trialarray[:,0]
        dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0)
        dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)
        dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)
        
        #print(dailymouse_nb.shape)
        # get trial number
        trialids_b = np.unique(dailymouse_b[:, 9])
        trialids_nb = np.unique(dailymouse_nb[:, 9])
        trialids_p = np.unique(dailymouse_p[:, 9])
        #print(trialids_b,trialids_nb,trialids_p)
        # get stops
        stops_b = speed_per_trial(bins,dailymouse_b, trialids_b)
        stops_nb = speed_per_trial(bins,dailymouse_nb, trialids_nb)
        stops_p = speed_per_trial(bins,dailymouse_p, trialids_p)
        #print('STOPS',stops_nb)
        beac = np.nanmean(stops_b,axis=1)
        nbeac = np.nanmean(stops_nb,axis=1)
        probe = np.nanmean(stops_p,axis=1)
        
        print('##...', mcount,day, '...##')
        # store data
        if mcount == 2 or mcount == 3 or mcount == 9:
            speeddiff = np.nanmean(beac[17:20])
            control_b[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            control_nb[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            control_p[mcount,dcount] = speeddiff
        if mcount == 0 and dcount <2:
            speeddiff = np.nanmean(beac[17:20])
            control_b[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            control_nb[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            control_p[mcount,dcount] = speeddiff
        if mcount == 1 or mcount == 5 or mcount == 8 or mcount == 6: # high dorsal
            speeddiff = np.nanmean(beac[17:20])
            teth_b[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            teth_nb[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            teth_p[mcount,dcount] = speeddiff
        if mcount == 7 or mcount == 10 or mcount == 4:
            speeddiff = np.nanmean(beac[17:20])
            tetl_b[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            tetl_nb[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            tetl_p[mcount,dcount] = speeddiff
        
        mcount +=1




marray = np.loadtxt('SummaryData/SummaryData_T15B.txt', dtype = 'S10',delimiter = '\t')
filename = 'DataFiles/Task15_b_0300.h5'
days = ['Day' + str(int(x)) for x in np.arange(1,5.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,8.1)]# choose specific day/s
# Stores
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
        dayb = day.encode('UTF-8')#""""
        mouseb = mouse.encode('UTF-8') #required for importing string from marray in python3
        marraybym = marray[marray[:,0]==dayb,:] # get descriptive data for session
        trialarray = saraharray[:,9] # makes an array of trial number per row in saraharray
        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]+0.1),1)
        tt = float(marraybym[marraybym[:,1]==mouseb,5][0]) #
        
        trialarray = maketrialarray(saraharray)
        saraharray[:,9] = trialarray[:,0]
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
        
        beac = np.nanmean(stops_b,axis=1)
        nbeac = np.nanmean(stops_nb,axis=1)
        probe = np.nanmean(stops_p,axis=1)
        
        print('##...', mcount,day, '...##')
        #STORE RATIOS
        if mcount == 3 or mcount == 4:
            speeddiff = np.nanmean(beac[17:20])
            control_b1[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            control_nb1[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            control_p1[mcount,dcount] = speeddiff
        if mcount == 0 or mcount == 1 or mcount == 2:
            speeddiff = np.nanmean(beac[17:20])
            tetl_b1[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            tetl_nb1[mcount,dcount] = speeddiff
            speeddiff = np.nanmean(beac[17:20])
            tetl_p1[mcount,dcount] = speeddiff
        
        mcount +=1


#------------------------------------------------------------------------------------#


# average over days for all mice
con_beac_week1 = np.vstack((control_b,control_b1))
con_probe_week1 = np.vstack((control_p,control_p1))

tetl_beac_week1 = np.vstack((tetl_b,tetl_b1))
tetl_probe_week1 = np.vstack((tetl_p,tetl_p1))

teth_beac_week1 = teth_b
teth_probe_week1 = teth_p



print(con_beac_week1.shape,'con_beac_week1')
print(con_beac_week4.shape,'con_beac_week4')


con1 = con_beac_week4 - con_beac_week1
tetl1 = tetl_beac_week4 - tetl_beac_week1
teth1 = teth_beac_week4 - teth_beac_week1



print(con1.shape, 'con1', 'after difference')

#average over days to get score for each mouse
con1 = np.nanmean(con1, axis=1)
tetl1 =np.nanmean(tetl1, axis=1)
teth1 =np.nanmean(teth1, axis=1)
print(con1.shape, 'con1', 'should be mice')
con1 = con1[~np.isnan(con1)]
tetl1 = tetl1[~np.isnan(tetl1)]
teth1 = teth1[~np.isnan(teth1)]


#average over mice for average scores
con = np.nanmean(con1, axis=0)
tetl =np.nanmean(tetl1, axis=0)
teth =np.nanmean(teth1, axis=0)
consd = np.nanstd(con1, axis=0)/math.sqrt(6)
tetlsd =np.nanstd(tetl1, axis=0)/math.sqrt(6)
tethsd =np.nanstd(teth1, axis=0)/math.sqrt(4)
print(con.shape, 'con1', 'average')
n_groups = np.arange(3)
bar_width = 0.5
n_groups = np.arange(3)
bar_width = 0.5
print(con.shape)
fig = plt.figure(figsize=(4,6))
#gs = gridspec.GridSpec(1, 7)
ax = fig.add_subplot(1,1,1)
ax.plot(1,con, 'o', color = 'k')
ax.errorbar(1,con,consd, fmt = 'o', color = 'k', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(2,tetl, 'o', color = 'blue')
ax.errorbar(2,tetl,tetlsd, fmt = 'o', color = 'blue', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(3,teth, 'o', color = 'red')
ax.errorbar(3,teth,tethsd, fmt = 'o', color = 'red', capsize = 8, markersize = 14, elinewidth =4, capthick = 3)
ax.plot(np.hstack((1,1,1,1,1,1)),con1, 'o', color = 'k', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((2,2,2,2,2,2)),tetl1, 'o', color = 'blue', alpha = 0.5, markersize = 10)
ax.plot(np.hstack((3,3,3,3,)),teth1, 'o', color = 'red', alpha = 0.5, markersize = 10)

adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =32)
ax.tick_params(axis='y', pad = 10, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 3, labelsize =15)
ax.set_ylabel('Dist (cm)', fontsize=32, labelpad = 20)
plt.locator_params(axis = 'x', nbins  = 2)
plt.locator_params(axis = 'y', nbins  = 4)
ax.axhline(-40,linewidth=3, color="black")
ax.axvline(0.5,linewidth=3, color="black")
#ax.axvline(3.5,linewidth=3, color="black")
ax.set_ylim(-40)
ax.set_xlim(0.5,3.5)
plt.locator_params(axis = 'y', nbins  = 5)
#ax.axhspan(88,100, linewidth = 0,facecolor='LimeGreen', alpha=0.2, hatch = '/') # bold line on the x axis
#ax.axhline(30, linewidth = 1,color='Black', ls = '--') # bold line on the x axis
plt.xticks(n_groups + bar_width, ('','','',''))
plt.locator_params(axis = 'x', nbins  = 3)
plt.subplots_adjust(hspace = 1, wspace = .7,  bottom = 0.25, left = 0.1, right = 0.9, top = .9)

plt.savefig('Chapter2/Task15_SpeedMeans_BB_0200' +' .png', dpi = 200)
plt.close()





mice_b = np.hstack((con1,tetl1,teth1))
genotype = np.array(("GFP","GFP","GFP" ,"GFP" ,"GFP" ,"GFP" ,"lTeLC","lTeLC","lTeLC","lTeLC","lTeLC","lTeLC","hTeLC","hTeLC","hTeLC","hTeLC"))
data = np.vstack((genotype, mice_b)); data=np.transpose(data)

np.savetxt('Manuscript/Data/Figure4_SpeedDiff_BB_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Genotype fluorescence,Beaconed')






