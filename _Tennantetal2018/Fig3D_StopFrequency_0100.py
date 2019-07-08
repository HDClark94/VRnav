
"""


# Finds the location along the track animals stop most frequently in


"""



# import packages and functions
from Functions_Core_0100 import extractstops,filterstops, create_srdata, makebinarray, speed_per_trial, makelegend, makelegend2, makelegend3, makelegend4, shuffle_analysis_pertrial3, z_score1, shuffle_analysis_pertrial_tracks, adjust_spines, makelegend2,readhdfdata,maketrialarray
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
from scipy.stats import uniform

# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task18_0100.h5'
# Load raw data: txt file that specifies days to analyse with average first stop closest to reward zone in beaconed trials
array = np.loadtxt('Data_Input/Behaviour_SummaryData/Task18_FirstDays.txt',delimiter = '\t')


# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,20.1)]
#mice = ['M' + str(int(x)) for x in np.arange(1,7.1)]
mice = ['M' + str(int(x)) for x in [1,5.1]]# specific day/s
#days = ['Day' + str(int(x)) for x in [1]]# specific day/s

print ('Loading data from...' + filename)
track = 200 # specify track size in cm

# arrays for storing data (output)
firststop_b = np.zeros((len(mice), 6, 2));firststop_b[:,:,:] = np.nan
firststop_nb = np.zeros((len(mice), 6, 2));firststop_nb[:,:,:] = np.nan
firststop_p = np.zeros((len(mice), 6, 2));firststop_p[:,:,:] = np.nan


### --------------------------------------------------------------------------------------------------------------------- #

# funtions specific to this code


def shuffle_analysis(stopsdata, trialids, tracklength):
    # Calculate stop rate for each section of the track
    srbin = create_srdata( stopsdata, trialids, tracklength )                        # Array(BINNR, trialnum)
    srbin_mean = np.sum(srbin, axis=0)                                 # Array(BINNR)
    srbin_std = np.std(srbin, axis=0)                                 # Array(BINNR)
    
    return srbin_mean, srbin_std


def create_srdata( stops, trialids,tracklength ):

    BINNR= tracklength
    posrange = np.linspace(0, tracklength, num=BINNR+1)
    trialrange = trialids
    trialrange = np.append(trialrange, trialrange[-1]+1)  # Add end of range
    values = np.array([[trialrange[0], trialrange[-1]],
                       [posrange[0], posrange[-1]]])
        
    H, bins, ranges = np.histogram2d(stops[:,2], stops[:,0], bins=(trialrange, posrange), range=values)
    H[np.where(H[::]>1)] = 1
                       
    return H

def StopFrequency(data):
    print(data.shape)
    m = np.max(data[3:])
    print('Max!',m)
    for bcount, b in enumerate(data):
        if data[bcount] == m and bcount >3:
            loc = bcount
            print(loc)
            break
    return loc

### --------------------------------------------------------------------------------------------------------------------- #


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
        
        # define track parameters
        trial = np.max(saraharray[:,9])
        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]+0.1),1) # array of trial number
        dayss = array[dcount,mcount] # day to analyse (1/0 - yes/no)
        rewardzonestart = saraharray[1,11] # find reward zone start
        rewardzoneend = saraharray[1,12] # find reward zone end
        tracklength = np.max(saraharray[:,1]) # find HDF5 track length

        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)

        #define track length
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
            tracklength = 33.2
            col = 3
        if rewardzonestart == 33.1:
            tracklength = 43
            col = 4
        if rewardzonestart == 48.1:
            tracklength = 59.1
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
        if tracklength == 33.2:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 90), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 100), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 110), 0)# delete all data not on probe tracks
        if tracklength == 43:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 120), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 130), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 140), 0)# delete all data not on probe tracks
        if tracklength == 59.1:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 150), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 160), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 170), 0)# delete all data not on probe tracks

        # get array of trial numbers
        trialids_b = np.unique(dailymouse_b[:, 9])
        trialids_nb = np.unique(dailymouse_nb[:, 9])
        trialids_p = np.unique(dailymouse_p[:, 9])
        #extract stops
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)
        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)
        # get mean stops per bin for real and shuffled data
        try:
            if dayss == 1:
                if col == 0:
                    if dailymouse_b.size > 0:
                        stops_f_b, srbin_std = shuffle_analysis( stopsdata_b, trialids_b ,tracklength) # get average stops vs location
                        loc = StopFrequency(stops_f_b) # find location most frequently stopped in
                        firststop_b[mcount,col,0] = loc*10 # * 10 to go from HDF5 to cm
                        firststop_b[mcount,col,1] = rewardzonestart*10
                    if dailymouse_nb.size > 0:
                        stops_f_nb, srbin_std = shuffle_analysis( stopsdata_nb, trialids_nb ,tracklength)
                        loc = StopFrequency(stops_f_nb)
                        firststop_nb[mcount,col,0] = loc*10
                        firststop_nb[mcount,col,1] = rewardzonestart*10
                    if dailymouse_p.size > 0:
                        stops_f_p, srbin_std = shuffle_analysis( stopsdata_p, trialids_p ,tracklength)
                        loc = StopFrequency(stops_f_p)
                        firststop_p[mcount,col,0] = loc*10
                        firststop_p[mcount,col,1] = rewardzonestart*10
                if col == 1:
                    if dailymouse_b.size > 0:
                        stops_f_b, srbin_std = shuffle_analysis( stopsdata_b, trialids_b ,tracklength )
                        loc = StopFrequency(stops_f_b)
                        firststop_b[mcount,col,0] = loc*10
                        firststop_b[mcount,col,1] = rewardzonestart*10
                    if dailymouse_nb.size > 0:
                        stops_f_nb, srbin_std = shuffle_analysis( stopsdata_nb, trialids_nb ,tracklength)
                        loc = StopFrequency(stops_f_nb)
                        firststop_nb[mcount,col,0] = loc*10
                        firststop_nb[mcount,col,1] = rewardzonestart*10
                    if dailymouse_p.size > 0:
                        stops_f_p, srbin_std = shuffle_analysis( stopsdata_p, trialids_p ,tracklength )
                        loc = StopFrequency(stops_f_p)
                        firststop_p[mcount,col,0] = loc*10
                        firststop_p[mcount,col,1] = rewardzonestart*10
                if col == 2:
                    if dailymouse_b.size > 0:
                        stops_f_b, srbin_std = shuffle_analysis( stopsdata_b, trialids_b ,tracklength )
                        loc = StopFrequency(stops_f_b)
                        firststop_b[mcount,col,0] = loc*10
                        firststop_b[mcount,col,1] = rewardzonestart*10
                    if dailymouse_nb.size > 0:
                        stops_f_nb, srbin_std = shuffle_analysis( stopsdata_nb, trialids_nb ,tracklength )
                        loc = StopFrequency(stops_f_nb)
                        firststop_nb[mcount,col,0] = loc*10
                        firststop_nb[mcount,col,1] = rewardzonestart*10
                    if dailymouse_p.size > 0:
                        stops_f_p, srbin_std = shuffle_analysis( stopsdata_p, trialids_p ,tracklength )
                        loc = StopFrequency(stops_f_p)
                        firststop_p[mcount,col,0] = loc*10
                        firststop_p[mcount,col,1] = rewardzonestart*10
                if col == 3:
                    if dailymouse_b.size > 0:
                        stops_f_b, srbin_std = shuffle_analysis( stopsdata_b, trialids_b ,tracklength )
                        loc = StopFrequency(stops_f_b)
                        firststop_b[mcount,col,0] = loc*10
                        firststop_b[mcount,col,1] = rewardzonestart*10
                    if dailymouse_nb.size > 0:
                        stops_f_nb, srbin_std = shuffle_analysis( stopsdata_nb, trialids_nb ,tracklength )
                        loc = StopFrequency(stops_f_nb)
                        firststop_nb[mcount,col,0] = loc*10
                        firststop_nb[mcount,col,1] = rewardzonestart*10
                    if dailymouse_p.size > 0:
                        stops_f_p, srbin_std = shuffle_analysis( stopsdata_p, trialids_p ,tracklength )
                        loc = StopFrequency(stops_f_p)
                        firststop_p[mcount,col,0] = loc*10
                        firststop_p[mcount,col,1] = rewardzonestart*10
                if col == 4:
                    if dailymouse_b.size > 0:
                        stops_f_b, srbin_std = shuffle_analysis( stopsdata_b, trialids_b ,tracklength )
                        loc = StopFrequency(stops_f_b)
                        firststop_b[mcount,col,0] = loc*10
                        firststop_b[mcount,col,1] = rewardzonestart*10
                    if dailymouse_nb.size > 0:
                        stops_f_nb, srbin_std = shuffle_analysis( stopsdata_nb, trialids_nb ,tracklength )
                        loc = StopFrequency(stops_f_nb)
                        firststop_nb[mcount,col,0] = loc*10
                        firststop_nb[mcount,col,1] = rewardzonestart*10
                    if dailymouse_p.size > 0:
                        stops_f_p, srbin_std = shuffle_analysis( stopsdata_p, trialids_p ,tracklength )
                        loc = StopFrequency(stops_f_p)
                        firststop_p[mcount,col,0] = loc*10
                        firststop_p[mcount,col,1] = rewardzonestart*10
                if col == 5:
                    if dailymouse_b.size > 0:
                        stops_f_b, srbin_std = shuffle_analysis( stopsdata_b, trialids_b ,tracklength )
                        loc = StopFrequency(stops_f_b)
                        firststop_b[mcount,col,0] = loc*10
                        firststop_b[mcount,col,1] = rewardzonestart*10
                    if dailymouse_nb.size > 0:
                        stops_f_nb, srbin_std = shuffle_analysis( stopsdata_nb, trialids_nb ,tracklength )
                        loc = StopFrequency(stops_f_nb)
                        firststop_nb[mcount,col,0] = loc*10
                        firststop_nb[mcount,col,1] = rewardzonestart*10
                    if dailymouse_p.size > 0:
                        stops_f_p, srbin_std = shuffle_analysis( stopsdata_p, trialids_p ,tracklength )
                        loc = StopFrequency(stops_f_p)
                        firststop_p[mcount,col,0] = loc*10
                        firststop_p[mcount,col,1] = rewardzonestart*10
        except IndexError:
            print('Error')





# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task19_0100.h5'
# Load raw data: txt file that specifies days to analyse with average first stop closest to reward zone in beaconed trials
array = np.loadtxt('Data_Input/Behaviour_SummaryData/Task19_FirstDays.txt',delimiter = '\t')

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,46.1)]
mice = ['M' + str(int(x)) for x in [3,6,7,8,9]]# specific day/s

firststop1_b = np.zeros((len(mice), 6, 2));firststop1_b[:,:,:] = np.nan
firststop1_nb = np.zeros((len(mice), 6, 2));firststop1_nb[:,:,:] = np.nan
firststop1_p = np.zeros((len(mice), 6, 2));firststop1_p[:,:,:] = np.nan

for dcount,day in enumerate(days): #load mouse and day
    for mcount,mouse in enumerate(mice):
        print ('Processing...',day,mouse)
        #load HDF5 data set for that day and mouse
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        
        #define track length parameters
        rewardzonestart = saraharray[1,11] # start of reward zone
        rewardzoneend = saraharray[1,12] #end of reward zone
        tracklength = np.max(saraharray[:,1]) # tracklength on HDF5 file
        trial = np.max(saraharray[:,9]) # max trial numebr
        dayss = array[dcount,mcount] # days to analyse
        
        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)
        
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
            tracklength = 33.2
            col = 3
        if rewardzonestart == 33.1:
            tracklength = 43
            col = 4
        if rewardzonestart == 48.1:
            tracklength = 59.1
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
        if tracklength == 33.2:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 90), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 100), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 110), 0)# delete all data not on probe tracks
        if tracklength == 43:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 120), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 130), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 140), 0)# delete all data not on probe tracks
        if tracklength == 59.1:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 150), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 160), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 170), 0)# delete all data not on probe tracks

        # get array of trial numbers
        trialids_b = np.unique(dailymouse_b[:, 9])
        trialids_nb = np.unique(dailymouse_nb[:, 9])
        trialids_p = np.unique(dailymouse_p[:, 9])

        #extract stops
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)

        # filter stops
        stopsdata_b = filterstops(stopsdata_b)
        stopsdata_nb = filterstops(stopsdata_nb)
        stopsdata_p = filterstops(stopsdata_p)
        
        # get mean stops per bin for real and shuffled data
        try:
            if dayss == 1:
                if col == 0:
                    if dailymouse_b.size > 0:
                        stops_f_b, srbin_std = shuffle_analysis( stopsdata_b, trialids_b ,tracklength )
                        loc = StopFrequency(stops_f_b)
                        firststop1_b[mcount,col,0] = loc*10
                        firststop1_b[mcount,col,1] = rewardzonestart*10
                    #print(loc)
                    if dailymouse_nb.size > 0:
                        stops_f_nb, srbin_std = shuffle_analysis( stopsdata_nb, trialids_nb ,tracklength )
                        loc = StopFrequency(stops_f_nb)
                        firststop1_nb[mcount,col,0] = loc*10
                        firststop1_nb[mcount,col,1] = rewardzonestart*10
                    if dailymouse_p.size > 0:
                        stops_f_p, srbin_std = shuffle_analysis( stopsdata_p, trialids_p ,tracklength )
                        loc = StopFrequency(stops_f_p)
                        firststop1_p[mcount,col,0] = loc*10
                        firststop1_p[mcount,col,1] = rewardzonestart*10
                if col == 1:
                    if dailymouse_b.size > 0:
                        stops_f_b, srbin_std = shuffle_analysis( stopsdata_b, trialids_b ,tracklength )
                        loc = StopFrequency(stops_f_b)
                        firststop1_b[mcount,col,0] = loc*10
                        firststop1_b[mcount,col,1] = rewardzonestart*10
                    if dailymouse_nb.size > 0:
                        stops_f_nb, srbin_std = shuffle_analysis( stopsdata_nb, trialids_nb ,tracklength )
                        loc = StopFrequency(stops_f_nb)
                        firststop1_nb[mcount,col,0] = loc*10
                        firststop1_nb[mcount,col,1] = rewardzonestart*10
                    if dailymouse_p.size > 0:
                        stops_f_p, srbin_std = shuffle_analysis( stopsdata_p, trialids_p ,tracklength )
                        loc = StopFrequency(stops_f_p)
                        firststop1_p[mcount,col,0] = loc*10
                        firststop1_p[mcount,col,1] = rewardzonestart*10
                if col == 2:
                    if dailymouse_b.size > 0:
                        stops_f_b, srbin_std = shuffle_analysis( stopsdata_b, trialids_b ,tracklength )
                        loc = StopFrequency(stops_f_b)
                        firststop1_b[mcount,col,0] = loc*10
                        firststop1_b[mcount,col,1] = rewardzonestart*10
                    if dailymouse_nb.size > 0:
                        stops_f_nb, srbin_std = shuffle_analysis( stopsdata_nb, trialids_nb ,tracklength )
                        loc = StopFrequency(stops_f_nb)
                        firststop1_nb[mcount,col,0] = loc*10
                        firststop1_nb[mcount,col,1] = rewardzonestart*10
                    if dailymouse_p.size > 0:
                        stops_f_p, srbin_std = shuffle_analysis( stopsdata_p, trialids_p ,tracklength )
                        loc = StopFrequency(stops_f_p)
                        firststop1_p[mcount,col,0] = loc*10
                        firststop1_p[mcount,col,1] = rewardzonestart*10
                if col == 3:
                    if dailymouse_b.size > 0:
                        stops_f_b, srbin_std = shuffle_analysis( stopsdata_b, trialids_b ,tracklength )
                        loc = StopFrequency(stops_f_b)
                        firststop1_b[mcount,col,0] = loc*10
                        firststop1_b[mcount,col,1] = rewardzonestart*10
                    if dailymouse_nb.size > 0:
                        stops_f_nb, srbin_std = shuffle_analysis( stopsdata_nb, trialids_nb ,tracklength )
                        loc = StopFrequency(stops_f_nb)
                        firststop1_nb[mcount,col,0] = loc*10
                        firststop1_nb[mcount,col,1] = rewardzonestart*10
                    if dailymouse_p.size > 0:
                        stops_f_p, srbin_std = shuffle_analysis( stopsdata_p, trialids_p ,tracklength )
                        loc = StopFrequency(stops_f_p)
                        firststop1_p[mcount,col,0] = loc*10
                        firststop1_p[mcount,col,1] = rewardzonestart*10
                if col == 4:
                    if dailymouse_b.size > 0:
                        stops_f_b, srbin_std = shuffle_analysis( stopsdata_b, trialids_b ,tracklength )
                        loc = StopFrequency(stops_f_b)
                        firststop1_b[mcount,col,0] = loc*10
                        firststop1_b[mcount,col,1] = rewardzonestart*10
                    if dailymouse_nb.size > 0:
                        stops_f_nb, srbin_std = shuffle_analysis( stopsdata_nb, trialids_nb ,tracklength )
                        loc = StopFrequency(stops_f_nb)
                        firststop1_nb[mcount,col,0] = loc*10
                        firststop1_nb[mcount,col,1] = rewardzonestart*10
                    if dailymouse_p.size > 0:
                        stops_f_p, srbin_std = shuffle_analysis( stopsdata_p, trialids_p ,tracklength )
                        loc = StopFrequency(stops_f_p)
                        firststop1_p[mcount,col,0] = loc*10
                        firststop1_p[mcount,col,1] = rewardzonestart*10
                if col == 5:
                    if dailymouse_b.size > 0:
                        stops_f_b, srbin_std = shuffle_analysis( stopsdata_b, trialids_b ,tracklength )
                        loc = StopFrequency(stops_f_b)
                        firststop1_b[mcount,col,0] = loc*10
                        firststop1_b[mcount,col,1] = rewardzonestart*10
                    if dailymouse_nb.size > 0:
                        stops_f_nb, srbin_std = shuffle_analysis( stopsdata_nb, trialids_nb ,tracklength )
                        loc = StopFrequency(stops_f_nb)
                        firststop1_nb[mcount,col,0] = loc*10
                        firststop1_nb[mcount,col,1] = rewardzonestart*10
                    if dailymouse_p.size > 0:
                        stops_f_p, srbin_std = shuffle_analysis( stopsdata_p, trialids_p ,tracklength )
                        loc = StopFrequency(stops_f_p)
                        firststop1_p[mcount,col,0] = loc*10
                        firststop1_p[mcount,col,1] = rewardzonestart*10
        except IndexError:
            print('Error!!')


tracks = np.array((98,128,173,240.5,341,491)) # array of track lengths

#average across days on tracks
tracks_b = np.nanmean(np.vstack((firststop_b,firststop1_b)), axis=0)
tracks_nb = np.nanmean(np.vstack((firststop_nb, firststop1_nb)), axis=0)
tracks_p = np.nanmean(np.vstack((firststop_p, firststop1_p)), axis=0)
sdtracks_b = np.nanstd(np.vstack((firststop_b,firststop1_b)), axis=0)/math.sqrt(6)
sdtracks_nb = np.nanstd(np.vstack((firststop_nb, firststop1_nb)), axis=0)/math.sqrt(6)
sdtracks_p = np.nanstd(np.vstack((firststop_p, firststop1_p)), axis=0)/math.sqrt(6)


# plot graphs

n_groups = np.arange(7)
bar_width = 0.4
width = 0.4

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)

ax.plot(tracks,tracks,ls='-', color = 'Green', linewidth = 10, alpha = 0.2) # mark black box

ax.plot(firststop_b[0,:,1],firststop_b[0,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
ax.plot(firststop_p[0,:,1],firststop_p[0,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop_b[1,:,1],firststop_b[1,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
ax.plot(firststop_p[1,:,1],firststop_p[1,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop1_b[0,:,1],firststop1_b[0,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
ax.plot(firststop1_p[0,:,1],firststop1_p[0,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop1_b[1,:,1],firststop1_b[1,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
ax.plot(firststop1_p[1,:,1],firststop1_p[1,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop1_b[2,:,1],firststop1_b[2,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
ax.plot(firststop1_p[2,:,1],firststop1_p[2,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop1_b[3,:,1],firststop1_b[3,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
ax.plot(firststop1_p[3,:,1],firststop1_p[3,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop1_b[4,:,1],firststop1_b[4,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
ax.plot(firststop1_p[4,:,1],firststop1_p[4,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.errorbar(tracks_b[:,1],tracks_b[:,0], sdtracks_b[:,0], fmt = '^', color = 'blue', capsize = 15, markersize = 20,elinewidth = 3, capthick = 3)
ax.errorbar(tracks_p[:,1],tracks_p[:,0],sdtracks_p[:,0], fmt = 's', color = 'red', capsize = 15, markersize = 16,elinewidth = 3, capthick = 3)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =20)
ax.tick_params(axis='y', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =18)
ax.set_ylabel('Location (cm)', fontsize=20, labelpad = 18)
ax.axhline(linewidth=6, color="black")
ax.axvline(50,linewidth=6, color="black")
ax.set_ylim(0,700)
plt.locator_params(axis = 'y', nbins  = 4)
plt.locator_params(axis = 'x', nbins  = 6)
ax.set_xlim(50,500)
plt.subplots_adjust(hspace = 0.6, wspace = .5,  bottom = 0.3, left = 0.2, right = 0.8, top = .9)

fig.savefig('Plots/Figure3/Task18_StopFrequency_Tracks_log_0100' +' .png', dpi = 200)
plt.close()



# save data for R

tracks_b = np.vstack((firststop_b[:,:,0],firststop1_b[:,:,0]))
tracks_nb = np.vstack((firststop_nb[:,:,0], firststop1_nb[:,:,0]))
tracks_p = np.vstack((firststop_p[:,:,0], firststop1_p[:,:,0]))

tracks_b = tracks_b.ravel()
tracks_nb = tracks_nb.ravel()
tracks_p = tracks_p.ravel()

tracks = np.array((88,118,163,230,331,481))
tracks = np.hstack((tracks,tracks,tracks,tracks,tracks,tracks,tracks))
m1 = np.array([1,1,1,1,1,1]); m2 = np.array([2,2,2,2,2,2]);m3 = np.array([3,3,3,3,3,3]); m4 = np.array([4,4,4,4,4,4]);m5 = np.array([5,5,5,5,5,5]); m6 = np.array([6,6,6,6,6,6]); m7 = np.array([7,7,7,7,7,7])

mice = np.hstack((m1,m2,m3,m4,m5,m6,m7))
data = np.vstack((mice, tracks, tracks_b, tracks_nb, tracks_p)); data = np.transpose(data)

np.savetxt('Data_Output/Figure3/Figure3_C_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Mouse, Reward zone,Beaconed, Non-beaconed,Probe')






