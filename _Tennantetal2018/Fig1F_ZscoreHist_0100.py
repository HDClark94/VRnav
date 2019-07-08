# -*- coding: utf-8 -*-
"""

### Calculates Z-scores for each location bin along the track
- Location bins are 10 cm
- Z-scores calculated for each mouse in last training week then averaged over mice


"""

# import packages and functions
from Functions_Core_0100 import extractstops, filterstops, create_srdata, makebinarray, shuffle_analysis_pertrial3, z_score1, adjust_spines,readhdfdata,maketrialarray
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import math
from scipy.stats import uniform

#--------------------------------------------------------------------------------------------------------------#

# First half of script gets data for first training week

#--------------------------------------------------------------------------------------------------------------#

# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task13_0300.h5' # raw data files

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,5.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,9.1)]
bins = np.arange(0,19+1e-6,1) # size of location bins

# empty arrays for storing data
firststopstorebeac = np.zeros((len(days), len(mice), 20));firststopstorenbeac = np.zeros((len(days), len(mice), 20));firststopstoreprobe = np.zeros((len(days), len(mice), 20))
firststopstorebeac[:,:,:] = np.nan;firststopstorenbeac[:,:,:] = np.nan; firststopstoreprobe[:,:,:] = np.nan

#loop days and mice to collect data
for mcount,mouse in enumerate(mice):
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')# get raw datafile for mouse and day
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
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)
        
        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)
        
        # Shuffle stops data & store data
        if mcount == 3 or mcount == 5 or mcount == 6 or mcount == 7 or  mcount == 8: # mice to analyse
            trialids_b = np.unique(stopsdata_b[:, 2]) # make array of unique trial numbers
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b ) # get average real stops & shuffled stops per lcoation bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate z-scores
            firststopstorebeac[dcount,mcount,:] = zscore_b # store zscores
            if stopsdata_nb.size >0 : # if there is non-beaconed data
                trialids_nb = np.unique(stopsdata_nb[:, 2])# make array of unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb ) # get average real stops & shuffled stops per lcoation bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate z-scores
                firststopstorenbeac[dcount, mcount,:] = zscore_nb # store zscores
            if stopsdata_p.size >0 : # if there is probe data
                trialids_p = np.unique(stopsdata_p[:, 2])# make array of unique trial numbers
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p ) # get average real stops & shuffled stops per lcoation bin

                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate z-scores
                firststopstoreprobe[dcount, mcount,:] = zscore_p # store zscores
        print('##...', mcount,day, '...##')
    mcount +=1


# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task12_0600.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,5.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,8.1)]# choose specific day/s

# empty arrays for storing data
firststopstorebeac2 = np.zeros((len(days), len(mice), 20));firststopstorenbeac2= np.zeros((len(days), len(mice), 20));firststopstoreprobe2= np.zeros((len(days), len(mice), 20))
firststopstorebeac2[:,:,:] = np.nan;firststopstorenbeac2[:,:,:] = np.nan;firststopstoreprobe2[:,:,:] = np.nan

#loop days and mice to collect data
for mcount,mouse in enumerate(mice):
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
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
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)
        
        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)
        
        # shuffle data and store in arrays
        if mcount == 5 or mcount == 6 or mcount == 7: # if control mouse, save data
            trialids_b = np.unique(stopsdata_b[:, 2]) # get array of trial numbers for beaconed
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b ) # get average real stops & shuffled stops per lcoation bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate z-scores
            firststopstorebeac2[dcount,mcount,:] = zscore_b # store zscores
            if stopsdata_nb.size >0 :
                trialids_nb = np.unique(stopsdata_nb[:, 2]) # get array of trial numbers for non-beaconed
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb ) # get average real stops & shuffled stops per lcoation bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate z-scores
                firststopstorenbeac2[dcount, mcount,:] = zscore_nb # store zscores
            if stopsdata_p.size >0 :
                trialids_p = np.unique(stopsdata_p[:, 2]) # get array of trial numbers for probe
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p ) # get average real stops & shuffled stops per lcoation bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate z-scores
                firststopstoreprobe2[dcount, mcount,:] = zscore_p # store zscores
        print('##...', mcount,day, '...##')
    mcount +=1



# AVERAGE DATA FOR PLOTS
# stack experiments then average over days then mice
con_b1 = np.nanmean(np.nanmean(np.hstack((firststopstorebeac[:,:,:],firststopstorebeac2[:,:,:])), axis = 0), axis = 0)
con_nb1 = np.nanmean(np.nanmean(np.hstack((firststopstorenbeac[:,:,:],firststopstorenbeac2[:,:,:])), axis =0), axis = 0)
con_p1 = np.nanmean(np.nanmean(np.hstack((firststopstoreprobe[:,:,:],firststopstoreprobe2[:,:,:])), axis = 0), axis = 0)
sdcon_b1 = np.nanstd(np.nanmean(np.hstack((firststopstorebeac[:,:,:],firststopstorebeac2[:,:,:])), axis = 0), axis = 0)/math.sqrt(8)
sdcon_nb1 = np.nanstd(np.nanmean(np.hstack((firststopstorenbeac[:,:,:],firststopstorenbeac2[:,:,:])), axis =0), axis = 0)/math.sqrt(8)
sdcon_p1 = np.nanstd(np.nanmean(np.hstack((firststopstoreprobe[:,:,:],firststopstoreprobe2[:,:,:])), axis = 0), axis = 0)/math.sqrt(8)


# WRITE DATA TO .CSV FOR R
# stack experiments then average over days
con_beac1 = np.nanmean(np.hstack((firststopstorebeac[:,:,:],firststopstorebeac2[:,:,:])), axis = 0)
con_nbeac1 = np.nanmean(np.hstack((firststopstorenbeac[:,:,:],firststopstorenbeac2[:,:,:])), axis =0)
con_probe1 = np.nanmean(np.hstack((firststopstoreprobe[:,:,:],firststopstoreprobe2[:,:,:])), axis = 0)
sdcon_beac1 = np.nanstd(np.hstack((firststopstorebeac[:,:,:],firststopstorebeac2[:,:,:])), axis = 0)/math.sqrt(8)
sdcon_nbeac1 = np.nanstd(np.hstack((firststopstorenbeac[:,:,:],firststopstorenbeac2[:,:,:])), axis =0)/math.sqrt(8)
sdcon_probe1 = np.nanstd(np.hstack((firststopstoreprobe[:,:,:],firststopstoreprobe2[:,:,:])), axis = 0)/math.sqrt(8)

x = np.vstack((con_beac1[3,:],con_beac1[5,:],con_beac1[6,:],con_beac1[7,:],con_beac1[8,:],con_beac1[14,:],con_beac1[15,:],con_beac1[16,:])) # stack mice
x1 = np.vstack((con_nbeac1[3,:],con_nbeac1[5,:],con_nbeac1[6,:],con_nbeac1[7,:],con_nbeac1[8,:],con_nbeac1[14,:],con_nbeac1[15,:],con_nbeac1[16,:]))
x2 = np.vstack((con_probe1[3,:],con_probe1[5,:],con_probe1[6,:],con_probe1[7,:],con_probe1[8,:],con_probe1[14,:],con_probe1[15,:],con_probe1[16,:],))

mice = np.array([1,2,3,4,5,6,7,8]); mouse = np.hstack((mice, mice, mice)) # array of mouse number
trialb = np.array([1,1,1,1,1,1,1,1]); trialnb = np.array([2,2,2,2,2,2,2,2]); trialp = np.array([3,3,3,3,3,3,3,3]); trials = np.hstack((trialb, trialnb, trialp)) # array for trial type
x = np.vstack((x,x1,x2)) # stack beaconed, nonbeaconed and probe
data = np.vstack((mouse, trials)); data=np.transpose(data) # stack mouse & trial arrays
data = np.hstack((data,x))# stack data and mouse, trial arrays

np.savetxt('Data_Output/Figure1/Figure1_F_Week1_0100.csv', data,fmt = '%i,%i,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f', delimiter = '\t', header = 'Mouse, Trial, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20') # save data


# WRITE DATA TO .CSV FOR R
# stack experiments then average over days
con_beac11 = np.nanmean(np.hstack((firststopstorebeac[:,:,:],firststopstorebeac2[:,:,:])), axis = 0)
con_nbeac11 = np.nanmean(np.hstack((firststopstorenbeac[:,:,:],firststopstorenbeac2[:,:,:])), axis =0)
con_probe11 = np.nanmean(np.hstack((firststopstoreprobe[:,:,:],firststopstoreprobe2[:,:,:])), axis = 0)
sdcon_beac11 = np.nanstd(np.hstack((firststopstorebeac[:,:,:],firststopstorebeac2[:,:,:])), axis = 0)/math.sqrt(8)
sdcon_nbeac11 = np.nanstd(np.hstack((firststopstorenbeac[:,:,:],firststopstorenbeac2[:,:,:])), axis =0)/math.sqrt(8)
sdcon_probe11 = np.nanstd(np.hstack((firststopstoreprobe[:,:,:],firststopstoreprobe2[:,:,:])), axis = 0)/math.sqrt(8)


#--------------------------------------------------------------------------------------------------------------#

# Next half of script gets data for last training week


#--------------------------------------------------------------------------------------------------------------#

# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task13_0300.h5' # raw data files

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(15,18.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,9.1)]

# empty arrays for storing data
firststopstorebeac = np.zeros((len(days), len(mice), 20));firststopstorenbeac = np.zeros((len(days), len(mice), 20));firststopstoreprobe = np.zeros((len(days), len(mice), 20)); firststopstorebeac[:,:,:] = np.nan;firststopstorenbeac[:,:,:] = np.nan; firststopstoreprobe[:,:,:] = np.nan

# loop days and mice to collect data
for mcount,mouse in enumerate(mice):
    for dcount,day in enumerate(days):
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')# get raw datafile for mouse and day
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
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)
        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)
        # Shuffle stops data & get zscores
        if mcount == 3 or mcount == 5 or mcount == 6 or mcount == 7 or  mcount == 8:
            trialids_b = np.unique(stopsdata_b[:, 2]) # get array of unique trial numbers for beaconed
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b ) # get average real stops & shuffled stops per lcoation bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate z-scores
            firststopstorebeac[dcount,mcount,:] = zscore_b # store zscores
            if stopsdata_nb.size >0 :
                trialids_nb = np.unique(stopsdata_nb[:, 2]) # get array of unique trial numbers for non-beaconed
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb ) # get average real stops & shuffled stops per lcoation bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate z-scores
                firststopstorenbeac[dcount, mcount,:] = zscore_nb # store zscores
            if stopsdata_p.size >0 :
                trialids_p = np.unique(stopsdata_p[:, 2]) # get array of unique trial numbers for probe
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p ) # get average real stops & shuffled stops per lcoation bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate z-scores
                firststopstoreprobe[dcount, mcount,:] = zscore_p # store zscores
    print('##...', mcount,day, '...##')
    mcount +=1


# Load raw data: specify the HDF5 file to read data from
filename = 'DataFiles/Task12_0600.h5'

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(15,18.1)]
mice = ['M' + str(int(x)) for x in np.arange(1,8.1)]

# empty arrays for storing data
firststopstorebeac2 = np.zeros((len(days), len(mice), 20));firststopstorenbeac2= np.zeros((len(days), len(mice), 20));firststopstoreprobe2= np.zeros((len(days), len(mice), 20))
firststopstorebeac2[:,:,:] = np.nan;firststopstorenbeac2[:,:,:] = np.nan;firststopstoreprobe2[:,:,:] = np.nan

# loop days and mice to collect data
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
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)
        
        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)

        # Shuffle stops data & get zscores
        if mcount == 5 or mcount == 6 or mcount == 7: # if control mouse, save data
            trialids_b = np.unique(stopsdata_b[:, 2])
            srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_b, trialids_b ) # get average real stops & shuffled stops per lcoation bin
            zscore_b = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate z-scores
            firststopstorebeac2[dcount,mcount,:] = zscore_b # store zscores
            if stopsdata_nb.size >0 :
                trialids_nb = np.unique(stopsdata_nb[:, 2])
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_nb, trialids_nb ) # get average real stops & shuffled stops per lcoation bin
                zscore_nb = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate z-scores
                firststopstorenbeac2[dcount, mcount,:] = zscore_nb # store zscores
            if stopsdata_p.size >0 :
                trialids_p = np.unique(stopsdata_p[:, 2])
                srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial3( stopsdata_p, trialids_p ) # get average real stops & shuffled stops per lcoation bin
                zscore_p = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate z-scores
                firststopstoreprobe2[dcount, mcount,:] = zscore_p # store zscores
    print('##...', mcount,day, '...##')
    mcount +=1


# AVERAGE DATA FOR PLOTS
# stack experiments then average over days then mice
con_b = np.nanmean(np.nanmean(np.hstack((firststopstorebeac[:,:,:],firststopstorebeac2[:,:,:])), axis = 0), axis = 0)
con_nb = np.nanmean(np.nanmean(np.hstack((firststopstorenbeac[:,:,:],firststopstorenbeac2[:,:,:])), axis =0), axis = 0)
con_p = np.nanmean(np.nanmean(np.hstack((firststopstoreprobe[:,:,:],firststopstoreprobe2[:,:,:])), axis = 0), axis = 0)
sdcon_b = np.nanstd(np.nanmean(np.hstack((firststopstorebeac[:,:,:],firststopstorebeac2[:,:,:])), axis = 0), axis = 0)/math.sqrt(6)
sdcon_nb = np.nanstd(np.nanmean(np.hstack((firststopstorenbeac[:,:,:],firststopstorenbeac2[:,:,:])), axis =0), axis = 0)/math.sqrt(6)
sdcon_p = np.nanstd(np.nanmean(np.hstack((firststopstoreprobe[:,:,:],firststopstoreprobe2[:,:,:])), axis = 0), axis = 0)/math.sqrt(6)


# WRITE DATA TO .CSV FOR R
# stack experiments then average over days
con_beac = np.nanmean(np.hstack((firststopstorebeac[:,:,:],firststopstorebeac2[:,:,:])), axis = 0)
con_nbeac = np.nanmean(np.hstack((firststopstorenbeac[:,:,:],firststopstorenbeac2[:,:,:])), axis =0)
con_probe = np.nanmean(np.hstack((firststopstoreprobe[:,:,:],firststopstoreprobe2[:,:,:])), axis = 0)
sd_con_beac = np.nanstd(np.hstack((firststopstorebeac[:,:,:],firststopstorebeac2[:,:,:])), axis = 0)/math.sqrt(6)
sd_con_nbeac = np.nanstd(np.hstack((firststopstorenbeac[:,:,:],firststopstorenbeac2[:,:,:])), axis =0)/math.sqrt(6)
sd_con_probe = np.nanstd(np.hstack((firststopstoreprobe[:,:,:],firststopstoreprobe2[:,:,:])), axis = 0)/math.sqrt(6)


# stack experiments then average over days
con_beac22 = np.nanmean(np.hstack((firststopstorebeac[:,:,:],firststopstorebeac2[:,:,:])), axis = 0)
con_nbeac22 = np.nanmean(np.hstack((firststopstorenbeac[:,:,:],firststopstorenbeac2[:,:,:])), axis =0)
con_probe22 = np.nanmean(np.hstack((firststopstoreprobe[:,:,:],firststopstoreprobe2[:,:,:])), axis = 0)
sd_con_beac22 = np.nanstd(np.hstack((firststopstorebeac[:,:,:],firststopstorebeac2[:,:,:])), axis = 0)/math.sqrt(6)
sd_con_nbeac22 = np.nanstd(np.hstack((firststopstorenbeac[:,:,:],firststopstorenbeac2[:,:,:])), axis =0)/math.sqrt(6)
sd_con_probe22 = np.nanstd(np.hstack((firststopstoreprobe[:,:,:],firststopstoreprobe2[:,:,:])), axis = 0)/math.sqrt(6)

x = np.vstack((con_beac[3,:],con_beac[5,:],con_beac[6,:],con_beac[7,:],con_beac[8,:],con_beac[14,:],con_beac[15,:],con_beac[16,:]))
x1 = np.vstack((con_nbeac[3,:],con_nbeac[5,:],con_nbeac[6,:],con_nbeac[7,:],con_nbeac[8,:],con_nbeac[14,:],con_nbeac[15,:],con_nbeac[16,:]))
x2 = np.vstack((con_probe[3,:],con_probe[5,:],con_probe[6,:],con_probe[7,:],con_probe[8,:],con_probe[14,:],con_probe[15,:],con_probe[16,:],))

mice = np.array([1,2,3,4,5,6,7,8]); mouse = np.hstack((mice, mice, mice))
trialb = np.array([1,1,1,1,1,1,1,1]); trialnb = np.array([2,2,2,2,2,2,2,2]); trialp = np.array([3,3,3,3,3,3,3,3]); trials = np.hstack((trialb, trialnb, trialp))
x = np.vstack((x,x1,x2))
data = np.vstack((mouse, trials)); data=np.transpose(data)
data = np.hstack((data,x))

np.savetxt('Data_Output/Figure1/Figure1_F_Week4_0100.csv', data,fmt = '%i,%i,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f,%10.3f', delimiter = '\t', header = 'Mouse, Trial, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20')





# PLOT GRAPHS

bins = np.arange(0.5,19.5+1e-6,1) # track bins

fig = plt.figure(figsize = (12,3))
ax = fig.add_subplot(1,3,1)
ax.axvspan(44, 44+12, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.plot(bins*5,con_b,color = 'red',label = 'AAV-fl-GFP', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,con_b-sdcon_b,con_b+sdcon_b, facecolor = 'red', alpha = 0.3)
ax.plot(bins*5,con_b1,'',color = 'blue',label = 'AAV-fl-GFP', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,con_b1-sdcon_b1,con_b1+sdcon_b1, facecolor = 'blue', alpha = 0.3)
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
ax.plot(bins*5,con_nb1,color = 'blue', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,con_nb1-sdcon_nb1,con_nb1+sdcon_nb1, facecolor = 'blue', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,100)
ax.set_ylim(-10,10)
adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
ax = plt.xlabel('Location (cm)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,3) #stops per trial
ax.axvspan(44, 44+12, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins*5,con_p,color = 'red', label = 'Beaconed', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,con_p-sdcon_p,con_p+sdcon_p, facecolor = 'red', alpha = 0.3)
ax.plot(bins*5,con_p1,color = 'blue', label = 'Beaconed', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,con_p1-sdcon_p1,con_p1+sdcon_p1, facecolor = 'blue', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,100)
ax.set_ylim(-10,10)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Figure1/Task13_ZscoreHist_0100'+'.png', dpi =200) # path to save file
plt.close() 





# PLOT GRAPHS

bins = np.arange(0.5,19.5+1e-6,1) # track bins

fig = plt.figure(figsize = (12,3))
ax = fig.add_subplot(1,3,1)
ax.axvspan(44, 44+12, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 15, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(100-15, 100, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-10, linewidth = 3, color = 'black') # bold line on the x axis
ax.axhline(0, linewidth = 1,ls='--', color = 'black') # bold line on the x axis
ax.plot(bins*5,con_beac22[5,:],color = 'red',label = 'AAV-fl-GFP', linewidth = 2) #plot becaoned trials
ax.fill_between(bins*5,con_beac22[5,:]-sd_con_beac22[5,:],con_beac22[5,:]+sd_con_beac22[5,:], facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.set_xlim(0,100)
ax.set_ylim(-10,11)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
ax.set_xticklabels(['0', '100', '200'])
ax = plt.ylabel('Location (cm)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2) #stops per trial

ax = fig.add_subplot(1,3,3) #stops per trial
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Figure1/Task13_ZscoreHist_M5_0100'+'.png', dpi =200) # path to save file
plt.close()




