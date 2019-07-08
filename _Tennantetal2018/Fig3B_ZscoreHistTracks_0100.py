
"""


# Calculates Zscores for each location bin along the track for increasing track lengths


"""


# import packages and functions
from Functions_Core_0100 import extractstops,filterstops, create_srdata, makebinarray, speed_per_trial, makelegend, makelegend2, makelegend3, makelegend4, shuffle_analysis_pertrial3, z_score1, shuffle_analysis_pertrial_tracks, adjust_spines, makelegend2,readhdfdata,maketrialarray
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
from scipy.stats import uniform
import random

# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task18_0100.h5'
# Load raw data: txt file that specifies days to analyse with average first stop closest to reward zone in beaconed trials
array = np.loadtxt('Data_Input/Behaviour_SummaryData/Task18_FirstDays.txt',delimiter = '\t')

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,20.1)]
mice = ['M' + str(int(x)) for x in [1,5.1]]# specific day/s

# empty arrays for storing data
track1_b = np.zeros((len(mice), len(days), 20));track1_b[:,:,:] = np.nan
track1_nb = np.zeros((len(mice), len(days), 20));track1_nb[:,:,:] = np.nan
track1_p = np.zeros((len(mice), len(days), 20));track1_p[:,:,:] = np.nan

track2_b = np.zeros((len(mice), len(days), 23));track2_b[:,:,:] = np.nan
track2_nb = np.zeros((len(mice), len(days), 23));track2_nb[:,:,:] = np.nan
track2_p = np.zeros((len(mice), len(days), 23));track2_p[:,:,:] = np.nan

track3_b = np.zeros((len(mice), len(days), 29));track3_b[:,:,:] = np.nan
track3_nb = np.zeros((len(mice), len(days), 29));track3_nb[:,:,:] = np.nan
track3_p = np.zeros((len(mice), len(days), 29));track3_p[:,:,:] = np.nan

track4_b = np.zeros((len(mice), len(days), 33));track4_b[:,:,:] = np.nan
track4_nb = np.zeros((len(mice), len(days), 33));track4_nb[:,:,:] = np.nan
track4_p = np.zeros((len(mice), len(days), 33));track4_p[:,:,:] = np.nan

track5_b = np.zeros((len(mice), len(days), 43));track5_b[:,:,:] = np.nan
track5_nb = np.zeros((len(mice), len(days), 43));track5_nb[:,:,:] = np.nan
track5_p = np.zeros((len(mice), len(days), 43));track5_p[:,:,:] = np.nan

track6_b = np.zeros((len(mice), len(days), 59));track6_b[:,:,:] = np.nan
track6_nb = np.zeros((len(mice), len(days), 59));track6_nb[:,:,:] = np.nan
track6_p = np.zeros((len(mice), len(days), 59));track6_p[:,:,:] = np.nan



# loop thorugh mice and days to get data
for dcount,day in enumerate(days): #load mouse and day
    for mcount,mouse in enumerate(mice):
        print ('Processing...',day,mouse)
        #load HDF5 data set for that day and mouse
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        trial = np.max(saraharray[:,9]) # max trial numebr
        dayss = array[dcount,mcount] # days to analyse
        
        #define track length parameters
        rewardzonestart = saraharray[1,11] # start of reward zone
        rewardzoneend = saraharray[1,12] #end of reward zone
        tracklength = np.max(saraharray[:,1]) # tracklength on HDF5 file
        binmin = 0;binmax = tracklength; interval = tracklength/20 # make location bins
        bins = np.arange(binmin,binmax+1e-6,interval)# make location bins
        
        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)

        # define track length
        if rewardzonestart == 8.8:
            tracklength = 20
            col = 0
        if rewardzonestart == 11.8:
            tracklength = 23
            col = 1
        if rewardzonestart == 16.3:
            tracklength = 29
            col = 2
        if rewardzonestart == 23.05:
            tracklength = 33
            col = 3
        if rewardzonestart == 33.1:
            tracklength = 43
            col = 4
        if rewardzonestart == 48.1:
            tracklength = 59
            col = 5

        # Extract data for beaconed, non-beaconed, probe
        if tracklength == 20:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)# delete all data not on probe tracks
        if tracklength == 23:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 30), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 40), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 50), 0)# delete all data not on probe tracks
        if tracklength == 29:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 60), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 70), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 80), 0)# delete all data not on probe tracks
        if tracklength == 33:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 90), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 100), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 110), 0)# delete all data not on probe tracks
        if tracklength == 43:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 120), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 130), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 140), 0)# delete all data not on probe tracks
        if tracklength == 59:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 150), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 160), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 170), 0)# delete all data not on probe tracks

        #extract stops
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)

        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)

        # get array of trial numbers
        trialids_b = np.unique(stopsdata_b[:, 2])
        if stopsdata_nb.size >0:
            trialids_nb = np.unique(stopsdata_nb[:, 2])
        if stopsdata_p.size >0:
            trialids_p = np.unique(stopsdata_p[:, 2])

        # get mean stops per bin for real and shuffled data
        try:
            if dayss == 1:
                if col == 0:
                    if dailymouse_b.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_b, trialids_b , tracklength) # get real and shuffled data
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std) # calculate zscores
                        track1_b[mcount,dcount,:] =zscore # store data
                    if dailymouse_nb.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_nb, trialids_nb , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track1_nb[mcount,dcount,:] =zscore
                    if dailymouse_p.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_p, trialids_p , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track1_p[mcount,dcount,:] =zscore
                if col == 1:
                    if dailymouse_b.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_b, trialids_b , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track2_b[mcount,dcount,:] =zscore
                    if dailymouse_nb.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_nb, trialids_nb , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track2_nb[mcount,dcount,:] =zscore
                    if dailymouse_p.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_p, trialids_p , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track2_p[mcount,dcount,:] =zscore
                if col == 2:
                    if dailymouse_b.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_b, trialids_b , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track3_b[mcount,dcount,:] =zscore
                    if dailymouse_nb.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_nb, trialids_nb , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track3_nb[mcount,dcount,:] =zscore
                    if dailymouse_p.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_p, trialids_p , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track3_p[mcount,dcount,:] =zscore
                if col == 3:
                    if dailymouse_b.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_b, trialids_b , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track4_b[mcount,dcount,:] =zscore
                    if dailymouse_nb.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_nb, trialids_nb , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track4_nb[mcount,dcount,:] =zscore
                    if dailymouse_p.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_p, trialids_p , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track4_p[mcount,dcount,:] =zscore
                if col == 4:
                    if dailymouse_b.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_b, trialids_b , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track5_b[mcount,dcount,:] =zscore
                    if dailymouse_nb.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_nb, trialids_nb , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track5_nb[mcount,dcount,:] =zscore
                    if dailymouse_p.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_p, trialids_p , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track5_p[mcount,dcount,:] =zscore
                if col == 5:
                    if dailymouse_b.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_b, trialids_b , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track6_b[mcount,dcount,:] =zscore
                    if dailymouse_nb.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_nb, trialids_nb , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track6_nb[mcount,dcount,:] =zscore
                    if dailymouse_p.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_p, trialids_p , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track6_p[mcount,dcount,:] =zscore
        except IndexError:
            print('Error')





# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task19_0100.h5'
# Load raw data: txt file that specifies days to analyse with average first stop closest to reward zone in beaconed trials
array = np.loadtxt('Data_Input/Behaviour_SummaryData/Task19_FirstDays.txt',delimiter = '\t')

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,46.1)]
mice = ['M' + str(int(x)) for x in [3,6,7,8,9]]# specific day/s

# arrays for storing data (output)
track1_b1 = np.zeros((len(mice), len(days), 20));track1_b[:,:,:] = np.nan
track1_nb1 = np.zeros((len(mice), len(days), 20));track1_nb[:,:,:] = np.nan
track1_p1 = np.zeros((len(mice), len(days), 20));track1_p[:,:,:] = np.nan

track2_b1 = np.zeros((len(mice), len(days), 23));track2_b[:,:,:] = np.nan
track2_nb1 = np.zeros((len(mice), len(days), 23));track2_nb[:,:,:] = np.nan
track2_p1 = np.zeros((len(mice), len(days), 23));track2_p[:,:,:] = np.nan

track3_b1 = np.zeros((len(mice), len(days), 29));track3_b[:,:,:] = np.nan
track3_nb1 = np.zeros((len(mice), len(days), 29));track3_nb[:,:,:] = np.nan
track3_p1 = np.zeros((len(mice), len(days), 29));track3_p[:,:,:] = np.nan

track4_b1 = np.zeros((len(mice), len(days), 33));track4_b[:,:,:] = np.nan
track4_nb1 = np.zeros((len(mice), len(days), 33));track4_nb[:,:,:] = np.nan
track4_p1 = np.zeros((len(mice), len(days), 33));track4_p[:,:,:] = np.nan

track5_b1 = np.zeros((len(mice), len(days), 43));track5_b[:,:,:] = np.nan
track5_nb1 = np.zeros((len(mice), len(days), 43));track5_nb[:,:,:] = np.nan
track5_p1 = np.zeros((len(mice), len(days), 43));track5_p[:,:,:] = np.nan

track6_b1 = np.zeros((len(mice), len(days), 59));track6_b[:,:,:] = np.nan
track6_nb1 = np.zeros((len(mice), len(days), 59));track6_nb[:,:,:] = np.nan
track6_p1 = np.zeros((len(mice), len(days), 59));track6_p[:,:,:] = np.nan

# loop thorugh mice and days to get data
for dcount,day in enumerate(days): #load mouse and day
    for mcount,mouse in enumerate(mice):
        print ('Processing...',day,mouse)
        #load HDF5 data set for that day and mouse
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        trial = np.max(saraharray[:,9]) # max trial numebr
        dayss = array[dcount,mcount] # days to analyse
        
        #define track length parameters
        rewardzonestart = saraharray[1,11] # start of reward zone
        rewardzoneend = saraharray[1,12] #end of reward zone
        tracklength = np.max(saraharray[:,1]) # tracklength on HDF5 file
        binmin = 0;binmax = tracklength; interval = tracklength/20 # make location bins
        bins = np.arange(binmin,binmax+1e-6,interval)# make location bins
        
        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)

        #define track length parameters
        if rewardzonestart == 8.8:
            tracklength = 20
            col = 0
        if rewardzonestart == 11.8:
            tracklength = 23
            col = 1
        if rewardzonestart == 16.3:
            tracklength = 29
            col = 2
        if rewardzonestart == 23.05:
            tracklength = 33
            col = 3
        if rewardzonestart == 33.1:
            tracklength = 43
            col = 4
        if rewardzonestart == 48.1:
            tracklength = 59
            col = 5

        # Extract data for beaconed, non-beaconed, probe
        if tracklength == 20:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)# delete all data not on probe tracks
        if tracklength == 23:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 30), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 40), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 50), 0)# delete all data not on probe tracks
        if tracklength == 29:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 60), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 70), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 80), 0)# delete all data not on probe tracks
        if tracklength == 33:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 90), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 100), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 110), 0)# delete all data not on probe tracks
        if tracklength == 43:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 120), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 130), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 140), 0)# delete all data not on probe tracks
        if tracklength == 59:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 150), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 160), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 170), 0)# delete all data not on probe tracks

        #extract stops
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)

        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)

        # get array of trial numbers
        trialids_b = np.unique(stopsdata_b[:, 2])
        if stopsdata_nb.size >0:
            trialids_nb = np.unique(stopsdata_nb[:, 2])
        if stopsdata_p.size >0:
            trialids_p = np.unique(stopsdata_p[:, 2])

        # get mean stops per bin for real and shuffled data
        try:
            if dayss == 1:
                if col == 0:
                    if dailymouse_b.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_b, trialids_b , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track1_b1[mcount,dcount,:] =zscore
                    if dailymouse_nb.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_nb, trialids_nb , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track1_nb1[mcount,dcount,:] =zscore
                    if dailymouse_p.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_p, trialids_p , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track1_p1[mcount,dcount,:] =zscore
                if col == 1:
                    if dailymouse_b.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_b, trialids_b , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track2_b1[mcount,dcount,:] =zscore
                    if dailymouse_nb.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_nb, trialids_nb , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track2_nb1[mcount,dcount,:] =zscore
                    if dailymouse_p.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_p, trialids_p , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track2_p1[mcount,dcount,:] =zscore
                if col == 2:
                    if dailymouse_b.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_b, trialids_b , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track3_b1[mcount,dcount,:] =zscore
                    if dailymouse_nb.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_nb, trialids_nb , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track3_nb1[mcount,dcount,:] =zscore
                    if dailymouse_p.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_p, trialids_p , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track3_p1[mcount,dcount,:] =zscore
                if col == 3:
                    if dailymouse_b.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_b, trialids_b , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track4_b1[mcount,dcount,:] =zscore
                    if dailymouse_nb.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_nb, trialids_nb , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track4_nb1[mcount,dcount,:] =zscore
                    if dailymouse_p.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_p, trialids_p , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track4_p1[mcount,dcount,:] =zscore
                if col == 4:
                    if dailymouse_b.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_b, trialids_b , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track5_b1[mcount,dcount,:] =zscore
                    if dailymouse_nb.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_nb, trialids_nb , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track5_nb1[mcount,dcount,:] =zscore
                    if dailymouse_p.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_p, trialids_p , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track5_p1[mcount,dcount,:] =zscore
                if col == 5:
                    if dailymouse_b.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_b, trialids_b , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track6_b1[mcount,dcount,:] =zscore
                    if dailymouse_nb.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_nb, trialids_nb , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track6_nb1[mcount,dcount,:] =zscore
                    if dailymouse_p.size > 0:
                        srbin_mean, srbin_std, shuffled_mean, shuffled_std = shuffle_analysis_pertrial_tracks( stopsdata_p, trialids_p , tracklength)
                        zscore = z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std)
                        track6_p1[mcount,dcount,:] =zscore
        except IndexError:
            print('Error')



# stack experiments then average

tracks1_b = np.nanmean(np.vstack((np.nanmean(track1_b, axis=1),np.nanmean(track1_b1, axis=1))),axis=0)
tracks1_nb = np.nanmean(np.vstack((np.nanmean(track1_nb, axis=1),np.nanmean(track1_nb1, axis=1))),axis=0)
tracks1_p = np.nanmean(np.vstack((np.nanmean(track1_p, axis=1),np.nanmean(track1_p1, axis=1))),axis=0)

sdtracks1_b = np.nanstd(np.vstack((np.nanmean(track1_b, axis=1),np.nanmean(track1_b1, axis=1))),axis=0)
sdtracks1_nb = np.nanstd(np.vstack((np.nanmean(track1_nb, axis=1),np.nanmean(track1_nb1, axis=1))),axis=0)
sdtracks1_p = np.nanstd(np.vstack((np.nanmean(track1_p, axis=1),np.nanmean(track1_p1, axis=1))),axis=0)

tracks2_b = np.nanmean(np.vstack((np.nanmean(track2_b, axis=1),np.nanmean(track2_b1, axis=1))),axis=0)
tracks2_nb = np.nanmean(np.vstack((np.nanmean(track2_nb, axis=1),np.nanmean(track2_nb1, axis=1))),axis=0)
tracks2_p = np.nanmean(np.vstack((np.nanmean(track2_p, axis=1),np.nanmean(track2_p1, axis=1))),axis=0)

sdtracks2_b = np.nanstd(np.vstack((np.nanmean(track2_b, axis=1),np.nanmean(track2_b1, axis=1))),axis=0)
sdtracks2_nb = np.nanstd(np.vstack((np.nanmean(track2_nb, axis=1),np.nanmean(track2_nb1, axis=1))),axis=0)
sdtracks2_p = np.nanstd(np.vstack((np.nanmean(track2_p, axis=1),np.nanmean(track2_p1, axis=1))),axis=0)

tracks3_b = np.nanmean(np.vstack((np.nanmean(track3_b, axis=1),np.nanmean(track3_b1, axis=1))),axis=0)
tracks3_nb = np.nanmean(np.vstack((np.nanmean(track3_nb, axis=1),np.nanmean(track3_nb1, axis=1))),axis=0)
tracks3_p = np.nanmean(np.vstack((np.nanmean(track3_p, axis=1),np.nanmean(track3_p1, axis=1))),axis=0)

sdtracks3_b = np.nanstd(np.vstack((np.nanmean(track3_b, axis=1),np.nanmean(track3_b1, axis=1))),axis=0)
sdtracks3_nb = np.nanstd(np.vstack((np.nanmean(track3_nb, axis=1),np.nanmean(track3_nb1, axis=1))),axis=0)
sdtracks3_p = np.nanstd(np.vstack((np.nanmean(track3_p, axis=1),np.nanmean(track3_p1, axis=1))),axis=0)

tracks4_b = np.nanmean(np.vstack((np.nanmean(track4_b, axis=1),np.nanmean(track4_b1, axis=1))),axis=0)
tracks4_nb = np.nanmean(np.vstack((np.nanmean(track4_nb, axis=1),np.nanmean(track4_nb1, axis=1))),axis=0)
tracks4_p = np.nanmean(np.vstack((np.nanmean(track4_p, axis=1),np.nanmean(track4_p1, axis=1))),axis=0)

sdtracks4_b = np.nanstd(np.vstack((np.nanmean(track4_b, axis=1),np.nanmean(track4_b1, axis=1))),axis=0)
sdtracks4_nb = np.nanstd(np.vstack((np.nanmean(track4_nb, axis=1),np.nanmean(track4_nb1, axis=1))),axis=0)
sdtracks4_p = np.nanstd(np.vstack((np.nanmean(track4_p, axis=1),np.nanmean(track4_p1, axis=1))),axis=0)

tracks5_b = np.nanmean(np.vstack((np.nanmean(track5_b, axis=1),np.nanmean(track5_b1, axis=1))),axis=0)
tracks5_nb = np.nanmean(np.vstack((np.nanmean(track5_nb, axis=1),np.nanmean(track5_nb1, axis=1))),axis=0)
tracks5_p = np.nanmean(np.vstack((np.nanmean(track5_p, axis=1),np.nanmean(track5_p1, axis=1))),axis=0)

sdtracks5_b = np.nanstd(np.vstack((np.nanmean(track5_b, axis=1),np.nanmean(track5_b1, axis=1))),axis=0)
sdtracks5_nb = np.nanstd(np.vstack((np.nanmean(track5_nb, axis=1),np.nanmean(track5_nb1, axis=1))),axis=0)
sdtracks5_p = np.nanstd(np.vstack((np.nanmean(track5_p, axis=1),np.nanmean(track5_p1, axis=1))),axis=0)

tracks6_b = np.nanmean(np.vstack((np.nanmean(track6_b, axis=1),np.nanmean(track6_b1, axis=1))),axis=0)
tracks6_nb = np.nanmean(np.vstack((np.nanmean(track6_nb, axis=1),np.nanmean(track6_nb1, axis=1))),axis=0)
tracks6_p = np.nanmean(np.vstack((np.nanmean(track6_p, axis=1),np.nanmean(track6_p1, axis=1))),axis=0)

sdtracks6_b = np.nanstd(np.vstack((np.nanmean(track6_b, axis=1),np.nanmean(track6_b1, axis=1))),axis=0)
sdtracks6_nb = np.nanstd(np.vstack((np.nanmean(track6_nb, axis=1),np.nanmean(track6_nb1, axis=1))),axis=0)
sdtracks6_p = np.nanstd(np.vstack((np.nanmean(track6_p, axis=1),np.nanmean(track6_p1, axis=1))),axis=0)


# plot data

bins1 = np.arange(0.5,20.5,1)

fig = plt.figure(figsize = (12,3))  # make figure, this shape (width, height)
ax = fig.add_subplot(1,3,1)
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks1_b,color = 'Black',label = 'Track 1') #plot becaoned trials
ax.fill_between(bins1,tracks1_b-sdtracks1_b,tracks1_b+sdtracks1_b, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(0,1)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_ylabel('Stops (cm)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2)
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks1_nb,color = 'Black',label = 'Track 1') #plot becaoned trials
ax.fill_between(bins1,tracks1_nb-sdtracks1_nb,tracks1_nb+sdtracks1_nb, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
#ax.set_ylim(0,1)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1,3,3)
ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(17, 20, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-0.25, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks1_b,color = 'blue',label = 'Track 1') #plot becaoned trials
ax.fill_between(bins1,tracks1_b-sdtracks1_b,tracks1_b+sdtracks1_b, facecolor = 'blue', alpha = 0.3)
ax.plot(bins1,tracks1_p,color = 'red',label = 'Track 1') #plot becaoned trials
ax.fill_between(bins1,tracks1_p-sdtracks1_p,tracks1_p+sdtracks1_p, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
ax.set_ylim(-0.25, 0.65)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=6) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Figure3/Task18_StopHist_Tracks1' + '_0200.png',  dpi = 200)
plt.close()



bins1 = np.arange(0.5,23.5,1)
fig = plt.figure(figsize = (13.8,3))  # make figure, this shape (width, height)
ax = fig.add_subplot(1,3,1)
ax.axvspan(10.261, 10.261+1.91, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 2.6, facecolor='k', alpha=0.2, hatch = '/', linewidth =0) # black box
ax.axvspan(20-2.6, 20, facecolor='k', alpha=0.2, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks2_b,color = 'Black',label = 'Track 2') #plot becaoned trials
ax.fill_between(bins1,tracks2_b-sdtracks2_b,tracks2_b+sdtracks2_b, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_ylabel('Stops (cm)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2)
ax.axvspan(10.261, 10.261+1.91, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 2.6, facecolor='k', alpha=0.2, hatch = '/', linewidth =0) # black box
ax.axvspan(20-2.6, 20, facecolor='k', alpha=0.2, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks2_nb,color = 'Black',label = 'Track 2') #plot becaoned trials
ax.fill_between(bins1,tracks2_nb-sdtracks2_nb,tracks2_nb+sdtracks2_nb, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1,3,3)
ax.axvspan(10.261, 10.261+1.91, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 2.6, facecolor='k', alpha=0.2, hatch = '/', linewidth =0) # black box
ax.axvspan(20-2.6, 20, facecolor='k', alpha=0.2, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.plot(bins1,tracks2_b,color = 'blue',label = 'Track 2') #plot becaoned trials
ax.fill_between(bins1,tracks2_b-sdtracks2_b,tracks2_b+sdtracks2_b, facecolor = 'blue', alpha = 0.3)
ax.plot(bins1,tracks2_p,color = 'red',label = 'Track 2') #plot becaoned trials
ax.fill_between(bins1,tracks2_p-sdtracks2_p,tracks2_p+sdtracks2_p, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,23)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Figure3/Task18_StopHist_Tracks2' + '_0200.png',  dpi = 200)
plt.close()




bins1 = np.arange(0.5,29.5,1)
fig = plt.figure(figsize = (16.5,3))  # make figure, this shape (width, height)
ax = fig.add_subplot(1,3,1)
ax.axvspan(11.241, 11.241+1.517, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 2.1, facecolor='k', alpha=0.2, hatch = '/', linewidth =0) # black box
ax.axvspan(20-2.1, 20, facecolor='k', alpha=0.2, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks3_b,color = 'Black',label = 'Track 3') #plot becaoned trials
ax.fill_between(bins1,tracks3_b-sdtracks3_b,tracks3_b+sdtracks3_b, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_ylabel('Stops (cm)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2)
ax.axvspan(11.241, 11.241+1.517, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 2.1, facecolor='k', alpha=0.2, hatch = '/', linewidth =0) # black box
ax.axvspan(20-2.1, 20, facecolor='k', alpha=0.2, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks3_nb,color = 'Black',label = 'Track 3') #plot becaoned trials
ax.fill_between(bins1,tracks3_nb-sdtracks3_nb,tracks3_nb+sdtracks3_nb, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
#ax.set_ylim(0,1)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1,3,3)
ax.axvspan(16.3, 16.3+2.2, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(29-3, 29, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-0.25, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks3_b,color = 'blue',label = 'Track 4') #plot becaoned trials
ax.fill_between(bins1,tracks3_b-sdtracks3_b,tracks3_b+sdtracks3_b, facecolor = 'blue', alpha = 0.3)
ax.plot(bins1,tracks3_p,color = 'red',label = 'Track 4') #plot becaoned trials
ax.fill_between(bins1,tracks3_p-sdtracks3_p,tracks3_p+sdtracks3_p, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,29)
ax.set_ylim(-0.25,0.65)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=6) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Figure3/Task18_StopHist_Tracks3' + '_0200.png',  dpi = 200)
plt.close()




bins1 = np.arange(0.5,33.5,1)
fig = plt.figure(figsize = (20.52,3))  # make figure, this shape (width, height)
ax = fig.add_subplot(1,3,1)
ax.axvspan(13.89, 13.89+1.325, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 1.81, facecolor='k', alpha=0.2, hatch = '/', linewidth =0) # black box
ax.axvspan(20-1.81, 20, facecolor='k', alpha=0.2, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks4_b,color = 'Black',label = 'Track 3') #plot becaoned trials
ax.fill_between(bins1,tracks4_b-sdtracks4_b,tracks4_b+sdtracks4_b, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_ylabel('Stops (cm)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2)
ax.axvspan(13.89, 13.89+1.325, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 1.81, facecolor='k', alpha=0.2, hatch = '/', linewidth =0) # black box
ax.axvspan(20-1.81, 20, facecolor='k', alpha=0.2, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks4_nb,color = 'Black',label = 'Track 3') #plot becaoned trials
ax.fill_between(bins1,tracks4_nb-sdtracks4_nb,tracks4_nb+sdtracks4_nb, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1,3,3)
ax.axvspan(23.05, 23.05+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(33-3, 33, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-0.5, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks4_b,color = 'blue',label = 'Track 4') #plot becaoned trials
ax.fill_between(bins1,tracks4_b-sdtracks4_b,tracks4_b+sdtracks4_b, facecolor = 'blue', alpha = 0.3)
ax.plot(bins1,tracks4_p,color = 'red',label = 'Track 4') #plot becaoned trials
ax.fill_between(bins1,tracks4_p-sdtracks4_p,tracks4_p+sdtracks4_p, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,33)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Figure3/Task18_StopHist_Tracks4' + '_0200.png',  dpi = 200)
plt.close()


bins1 = np.arange(0.5,43.5,1)
fig = plt.figure(figsize = (27.48,3))  # make figure, this shape (width, height)
ax = fig.add_subplot(1,3,1)
ax.axvspan(15.4, 15.4+1.023, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 1.4, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(20-1.4, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks5_b,color = 'Black',label = 'Track 5') #plot becaoned trials
ax.fill_between(bins1,tracks5_b-sdtracks5_b,tracks5_b+sdtracks5_b, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_ylabel('Stops (cm)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2)
ax.axvspan(15.4, 15.4+1.023, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 1.4, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(20-1.4, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks5_nb,color = 'Black',label = 'Track 5') #plot becaoned trials
ax.fill_between(bins1,tracks5_nb-sdtracks5_nb,tracks5_nb+sdtracks5_nb, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1,3,3)
ax.axvspan(15.4, 15.4+1.023, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 1.4, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(20-1.4, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.plot(bins1,tracks5_b,color = 'blue',label = 'Track 5') #plot becaoned trials
ax.fill_between(bins1,tracks5_b-sdtracks5_b,tracks5_b+sdtracks5_b, facecolor = 'blue', alpha = 0.3)
ax.plot(bins1,tracks5_p,color = 'red',label = 'Track 5') #plot becaoned trials
ax.fill_between(bins1,tracks5_p-sdtracks5_p,tracks5_p+sdtracks5_p, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,43)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
fig.savefig('Plots/Figure3/Task18_StopHist_Tracks5' + '_0200.png',  dpi = 200)
plt.close()



bins1 = np.arange(0.5,59.5,1)
fig = plt.figure(figsize = (35.34,3))  # make figure, this shape (width, height)
ax = fig.add_subplot(1,3,1)
ax.axvspan(16.3, 16.3+0.736, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 1, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(20-1, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks6_b,color = 'Black',label = 'Track 5') #plot becaoned trials
ax.fill_between(bins1,tracks6_b-sdtracks6_b,tracks6_b+sdtracks6_b, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
#ax.set_ylim(0,1)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_ylabel('Stops (cm)', fontsize=16, labelpad = 18)

ax = fig.add_subplot(1,3,2)
ax.axvspan(16.3, 16.3+0.736, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 1, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
ax.axvspan(20-1, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(0, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks6_nb,color = 'Black',label = 'Track 5') #plot becaoned trials
ax.fill_between(bins1,tracks6_nb-sdtracks6_nb,tracks6_nb+sdtracks6_nb, facecolor = 'Black', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,20)
#ax.set_ylim(0,1)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=2) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])
ax.set_yticklabels(['', '', ''])

ax = fig.add_subplot(1,3,3)
ax.axvspan(48.1, 48.1+2.2, facecolor='g', alpha=0.25, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
ax.axvspan(0, 3, facecolor='k', alpha=0.15, hatch = '/', linewidth =0) # black box
ax.axvspan(59-3, 59, facecolor='k', alpha=0.15, hatch = '/', linewidth =0)# black box
ax.axvline(0, linewidth = 3, color = 'black') # bold line on the y axis
ax.axhline(-0.25, linewidth = 3, color = 'black') # bold line on the x axis
ax.plot(bins1,tracks6_b,color = 'blue',label = 'Track 6') #plot becaoned trials
ax.fill_between(bins1,tracks6_b-sdtracks6_b,tracks6_b+sdtracks6_b, facecolor = 'blue', alpha = 0.3)
ax.plot(bins1,tracks6_p,color = 'red',label = 'Track 6') #plot becaoned trials
ax.fill_between(bins1,tracks6_p-sdtracks6_p,tracks6_p+sdtracks6_p, facecolor = 'red', alpha = 0.3)
ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7,labelsize =16)
ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 2, length = 7, labelsize =16)
ax.set_xlim(0,59)
ax.set_ylim(-0.25,0.65)
adjust_spines(ax, ['left','bottom']) # removes top and right spines
ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
ax.locator_params(axis = 'y', nbins=6) # set number of ticks on y axis
ax.set_xticklabels(['', '', '', '', ''])

plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)

fig.savefig('Plots/Figure3/Task18_StopHist_Tracks6' + '_0200.png',  dpi = 200)
plt.close()




# save data for R

tracks1_b = np.vstack((np.nanmean(track1_b, axis=1),np.nanmean(track1_b1, axis=1)))
tracks1_nb = np.vstack((np.nanmean(track1_nb, axis=1),np.nanmean(track1_nb1, axis=1)))
tracks1_p = np.vstack((np.nanmean(track1_p, axis=1),np.nanmean(track1_p1, axis=1)))
      
tracks3_b = np.vstack((np.nanmean(track3_b, axis=1),np.nanmean(track3_b1, axis=1)))
tracks3_nb = np.vstack((np.nanmean(track3_nb, axis=1),np.nanmean(track3_nb1, axis=1)))
tracks3_p = np.vstack((np.nanmean(track3_p, axis=1),np.nanmean(track3_p1, axis=1)))
      
tracks6_b = np.vstack((np.nanmean(track6_b, axis=1),np.nanmean(track6_b1, axis=1)))
tracks6_nb = np.vstack((np.nanmean(track6_nb, axis=1),np.nanmean(track6_nb1, axis=1)))
tracks6_p = np.vstack((np.nanmean(track6_p, axis=1),np.nanmean(track6_p1, axis=1)))
      
mouse = np.array((1,2,3,4,5,6,7))
print(mouse.shape,tracks1_b.shape)
data = np.vstack((mouse,tracks1_b))
np.savetxt('Data_Output/Figure3/Figure3_B_Track1_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Mouse, Trial, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20')




