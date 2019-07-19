# -*- coding: utf-8 -*-
"""
    
    @author: Sarah Tennant
    
    Includes all core functions which are imported into main code for analysis.
    
    """

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib import rcParams
from _Tennantetal2018.Functions_Params_0100 import STOP_THRESHOLD, DIST, HDF_LENGTH, BINNR, SHUFFLE_N
from scipy.stats import uniform
from scipy import stats
import random

# -------------------------------------------------------------------------------------------------------------- #

# FUNCTIONS BELOW ARE FOR EXTRACTING DATA

# FUNCTION FOR READING HDF5 FILES
def readhdfdata(filename,day,mouse,dataset):
    fopen = h5py.File(filename, 'r')
    datasetopen  = fopen[day + '/' + mouse  + '/' + dataset]
    openarray = datasetopen[:,:]
    fopen.close()
    return openarray



# FUNCTION TO EXTRACT STOPS FROM DATASET
#input: array[:,13] (columns: for raw data, see README.py file)
#output: array[:,3] (columns: location, time, trialno, reward)
#function: extracts row of data if speed drops below 0.7 cm/s.
def extractstops(stops):
    moving = False
    data = []
    for row in stops:
        if(row[2]<=STOP_THRESHOLD and moving): # if speed is below threshold
            moving = False
            data.append([float(row[1])+0.2, float(row[0]), int(row[9]), int(row[4])]) # location, (beaconed/non-beaconed/probe), trialid, reward(YES/NO)
        
        if(row[2]>STOP_THRESHOLD and not moving):
            moving = True
    return np.array(data)

#=============================================================ADDED FUNCTION TO HANDLE DIFFERENCE IN STOP DETECTION ======================================#
# FUNCTION TO EXTRACT STOPS FROM DATASET
# input: array[:,13] (columns: for raw data, see README.py file)
# output: array[:,3] (columns: location, time, trialno, reward)
# function: extracts row of data if speed drops below 0.7 cm/s.
def extractstops_HUMAN(stops):
    moving = False
    data = []
    for row in stops:
        if (np.round(row[2], decimals=5) == 0 and moving):  # if speed is below threshold
            moving = False
            data.append([float(row[1]), float(row[0]), int(row[9]),
                         int(row[4])])  # location, (beaconed/non-beaconed/probe), trialid, reward(YES/NO)

        if (np.round(row[2], decimals=5) > 0 and not moving):
            moving = True
    return np.array(data)

#================================================================================================================================================================#



# FUNCTION FOR EXTRACTING REWARDS FROM DATASET
#input: array[:,13]
#output: array[:,2] (columns: location, time, trialno)
#function: extracts row of data if reward was dispensed.
def extractrewards(stops):
    data = [] # list to put data into
    for row in stops: # for every row in dataarray
        if(row[4]==1): # if column 4 == 1 then a reward was dispensed
            data.append([float(row[1])+0.2, float(row[0]), float(row[9])]) # Append location, time and trialnumber
    return np.array(data)



# FUNCTION TO FILTER STOPS
#input: array[:,3] (columns: location, time, trialno, reward)
#output: array[:,4] (columns: location, time, trialno, reward, empty)
#function: remove stops that occur 1 cm after a stop
def filterstops(stops):
    if stops.size ==0: # if there is no data, return empty array
        return stops
    data = np.zeros((stops.shape[0],1)) # create array same size as data array
    for rowcount, row in enumerate(range(len(stops)-1)): # for every row of data / stop
        diff = float(stops[rowcount+1,0] - stops[rowcount,0]) # calculate the difference in location
        trialsame = float(stops[rowcount+1,2] - stops[rowcount,2]) # check if stops are in same trial
        if diff < 0.1 and trialsame == 0: # if the difference is below 1 cm and the stops are on the same trial
            data[rowcount+1,0] = 1 # add one to data array
    stops = np.hstack((stops, data)) # add data array to stops data
    stops = np.delete(stops, np.where(stops[:, 4] == 1), 0) # delete all rows where data array is 1
    return stops


# MAKE LOCATION BINS
def makebinarray(tarray, bins):
    interval = 1
    binarray = np.zeros((tarray.shape[0], 1))
    for bcount,b in enumerate(bins): # Collect data for each bin
        binmin = tarray[:,1]>=b # lowest value in bin
        binmax = tarray[:,1]<b+interval # highest value in bin
        arraylogical = np.logical_and(binmin,binmax) #get all rows that satisfy being within bin
        binarray[arraylogical, 0] = b #assign each row in tarray its respective bin
    return binarray


# MAKE ARRAY OF TRIAL NUMBER FOR EVERY ROW IN DATA
#input: array[:,13] (columns: for raw data, see README.py file)
#output: array[:,0] (columns: trialno)
#function: remove stops that occur 1 cm after a stop
def maketrialarray(saraharray):
    trialarray = np.zeros((saraharray.shape[0],1)) # make empty array same row number as datafile
    trialnumber = 1
    trialarray[0] = 1 #because the first row is always in the first trial
    for rowcount, row in enumerate(saraharray[:-1, :]): # for every row in data
        if saraharray[rowcount + 1, 1]-saraharray[rowcount,1]<-10: # if current location - last location is less than -10
            trialnumber+=1 # add one to trial number
        trialarray[rowcount + 1] = trialnumber # creates array for trial number in each row of saraharray
        rowcount+=1
    return trialarray


# -------------------------------------------------------------------------------------------------------------- #

# FUNCTIONS BELOW ARE FOR ANALYSIS


# GET LOCATION WHERE THE ANIMAL STOPS THE MOST
def StopFrequency(data):
    max = np.argmax(data) # find max number of stops
    return max


# FUNCTION REPLACES ABSOLUTE TIME WITH TIME IT TAKES ANIMAL TO REACH REWARD ZONE
def timetorz(data, trialids):
    datastore = np.zeros((0,13)) # empty array same size as dataarray
    trials = np.zeros((trialids.shape[0],2)) # array(trialnumber,2)
    
    for trialcount, trial in enumerate(trialids):# loops through each trial
        tarray = data[data[:,9] ==trial,:] # get data only for each trial
        for row in tarray: # for every row of data
            if row[1] <=3.2: # if row is below black box
                starttime = row[0] # get time just after leaving black box
            if row[1] < 8.8:
                rztime = float(row[0]) # get time just before leaving black box
            row+=1
        
        trialtime = rztime - starttime # time from black box to reward zone
        times = np.repeat(trialtime, tarray.shape[0]) # repeat trial time to same shape as trial data
        tarray[:,0] = times # replace time in tarray
        datastore =np.vstack((datastore,tarray)) # stack to empty array - after data from all trials added will end up with original dataset but time replaced
    return np.array(datastore)



# FUNCTION REPLACES ABSOLUTE TIME WITH TIME FROM END OF BLACK BOX
def timetostop( data , trialids):
    datastore = np.zeros((0,13)) # # empty array same size as dataarray
    for trialcount, trial in enumerate(trialids):# loops through each trial
        tarray = data[data[:,9] ==trial,:] # get data only for each trial
        starttime=0
        for rowcount,row in enumerate(tarray):
            if tarray[rowcount,1] <=3.2: # if row is below black box
                starttime = tarray[rowcount,0] # get time just before leaving black box
            timefrombb = tarray[rowcount,0] -starttime
            tarray[rowcount,0] = timefrombb
            row+=1
        datastore =np.vstack((datastore,tarray))
    return np.array(datastore)



# Input: array[:,4] (columns: location, time, trialno, reward, empty)
# Output: array[trialnumbers, locationbins]
# Function: Creates histogram of stops in bins
# BIN STOPS INTO 20, 10 CM LOCATION BINS
def create_srdata( stops, trialids ):
    if stops.size == 0:
        return np.zeros((BINNR,))
    
    # create histogram
    posrange = np.linspace(0, HDF_LENGTH, num=BINNR+1) # 0 VU to 20 VU split into 20
    trialrange = trialids
    trialrange = np.append(trialrange, trialrange[-1]+1)  # Add end of range
    values = np.array([[trialrange[0], trialrange[-1]],[posrange[0], posrange[-1]]])

    H, bins, ranges = np.histogram2d(stops[:,2], stops[:,0], bins=(trialrange, posrange), range=values)
    H[np.where(H[::]>1)] = 1

    return H


# Input: array[:,4] (columns: location, time, trialno, reward, empty)
# Output: array[trialnumbers, locationbins]
# Function: Creates histogram of stops in bins
# ABOVE BUT FOR DIFFERENT SIZED TRACKS
def create_srdata_tracks( stops, trialids , TRACKS):
    if stops.size == 0:
        return np.zeros((TRACKS,))
    
    # create histogram
    posrange = np.linspace(0, TRACKS, num=TRACKS+1)
    trialrange = trialids
    trialrange = np.append(trialrange, trialrange[-1]+1)  # Add end of range
    values = np.array([[trialrange[0], trialrange[-1]],[posrange[0], posrange[-1]]])

    H, bins, ranges = np.histogram2d(stops[:,2], stops[:,0], bins=(trialrange, posrange), range=values)
    H[np.where(H[::]>1)] = 1
    return H


# SHUFFLE STOPS
def shuffle_stops2( stops,n ):
    shuffled_stops = np.copy(stops) # this is required as otherwise the original dataset would be altered
    # create an array that contains the amount by which every stop will be shuffled
    rand_rotation = uniform.rvs(loc=0, scale=HDF_LENGTH, size=stops.shape[0])
    # add random value
    shuffled_stops[:,0] = rand_rotation
    shuffled_stops[:,2] = n
    
    return shuffled_stops



# ABOVE BUT FOR DIFFERENT SIZED TRACKS
def shuffle_stops_tracks( stops,n , TRACKS):
    shuffled_stops = np.copy(stops) # this is required as otherwise the original dataset would be altered
    # create an array that contains the amount by which every stop will be shuffled
    rand_rotation = uniform.rvs(loc=0, scale=TRACKS, size=stops.shape[0])
    # add random value
    shuffled_stops[:,0] = rand_rotation
    shuffled_stops[:,2] = n
    
    return shuffled_stops


# Input: array[:,4] (columns: location, time, trialno, reward, zeros), array[unique trialnumbers]
# Output: array[20], array[20], array[20], array[20]
# Function: creates shuffled stops datasets from real dataset
# CREATE SHUFFLED DATASETS
def shuffle_analysis_pertrial3(stopsdata, trialids):
    if stopsdata.size == 0:
        return np.zeros((BINNR, )), np.zeros((BINNR, )), np.zeros((BINNR, )), np.zeros((BINNR, ))
    SHUFFLE1 = 100
    # Calculate stop rate for each section of the track
    srbin = create_srdata( stopsdata, trialids )                        # Array(BINNR, trialnum)
    srbin_mean = np.mean(srbin, axis=0)                                 # Array(BINNR)
    srbin_std = stats.sem(srbin, axis=0)                                 # Array(BINNR)
    # Shuffling random 100 trials 1000 times
    shuffled_srbin_mean = np.zeros((SHUFFLE_N, BINNR))
    #print(stopsdata.shape,stopsdata.size )
    for i in range(SHUFFLE_N): # for i in 1000
        shuffledtrials = np.zeros((SHUFFLE1, 5))
        shuffleddata =np.zeros((SHUFFLE1, BINNR))
        for n in range(SHUFFLE1): # Create sample data with 100 trials
            trial = random.choice(trialids) # select random trial from real dataset
            data = stopsdata[stopsdata[:,2] ==trial,:] # get data only for each tria
            shuffledtrial = shuffle_stops2(data,n) # shuffle the locations of stops in the trial
            shuffledtrials = np.vstack((shuffledtrials,shuffledtrial)) # stack shuffled stops
        trialids2 = np.unique(shuffledtrials[:, 2]) # find unique trials in the data
        shuffled_srbin = create_srdata( shuffledtrials, trialids2 ) #
        shuffled_srbin_mean[i] = np.mean(shuffled_srbin, axis=0)        # Array(BINNR)
    # Mean of the mean stops in the shuffled data for each bin
    shuffled_mean = np.mean(shuffled_srbin_mean, axis=0)                # Array(BINNR)
    shuffled_std = np.std(shuffled_srbin_mean, axis=0)                  # Array(BINNR)
    return srbin_mean, srbin_std, shuffled_mean, shuffled_std



# Input: array[:,4] (columns: location, time, trialno, reward, zeros), array[unique trialnumbers], int[tracklength]
# Output: array[20], array[20], array[20], array[20]
# Function: creates shuffled stops datasets from real dataset
# ABOVE BUT FOR LONGER TRACKS
def shuffle_analysis_pertrial_tracks(stopsdata, trialids, TRACKS):
    SHUFFLE1 = 100
    # Calculate stop rate for each section of the track
    srbin = create_srdata_tracks( stopsdata, trialids, TRACKS )                        # Array(BINNR, trialnum)
    srbin_mean = np.mean(srbin, axis=0)                                 # Array(BINNR)
    srbin_std = stats.sem(srbin, axis=0)                                 # Array(BINNR)
    # Shuffling random 100 trials 1000 times
    shuffled_srbin_mean = np.zeros((SHUFFLE_N, TRACKS))
    for i in range(SHUFFLE_N): # for i in 1000
        shuffledtrials = np.zeros((SHUFFLE1, 5))
        shuffleddata =np.zeros((SHUFFLE1, TRACKS))
        for n in range(SHUFFLE1): # Create sample data with 100 trials
            trial = random.choice(trialids) # select random trial from real dataset
            data = stopsdata[stopsdata[:,2] ==trial,:] # get data only for each tria
            shuffledtrial = shuffle_stops_tracks(data,n, TRACKS) # shuffle the locations of stops in the trial
            shuffledtrials = np.vstack((shuffledtrials,shuffledtrial)) # stack shuffled stops
        trialids2 = np.unique(shuffledtrials[:, 2]) # find unique trials in the data
        shuffled_srbin = create_srdata_tracks( shuffledtrials, trialids2, TRACKS ) #
        #print(shuffled_srbin.shape)
        shuffled_srbin_mean[i] = np.mean(shuffled_srbin, axis=0)        # Array(BINNR)
    # Mean of the mean stops in the shuffled data for each bin
    shuffled_mean = np.mean(shuffled_srbin_mean, axis=0)                # Array(BINNR)
    shuffled_std = np.std(shuffled_srbin_mean, axis=0)                  # Array(BINNR)
    return srbin_mean, srbin_std, shuffled_mean, shuffled_std


# Input: array[20], array[20], array[20], array[20]
# Output: array[20]
# Function: Calcuates z-scores for real and shuffled data for each location bin
# CALCULATE ZSCORES FROM SHUFFLED AND REAL DATASETS
def z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std): # input from shuffle_analysis_pertrial3 output
    zscore = np.zeros((srbin_mean.shape)) # ARRAY[BINNR]
    for bcount, b in enumerate(srbin_mean):# go through each bin
        diff = srbin_mean[bcount] - shuffled_mean[bcount] # real stops - shuffled stops
        multi = diff / shuffled_std[bcount] # divided by shuffled SD
        zscore[bcount] = multi # allocate zscore to bin
    
    return zscore


# CALCULATE AVERAGE SPEED PER TRIAL
def speed_per_trial(bins,saraharray, trarray):
    
    stopsarraybeacon = np.zeros(((bins.shape[0]), (trarray.shape[0]))) # rows == same number of bins
    stopsarraybeacon[:,:] = np.nan
    
    for tcount,trial in enumerate(trarray):# loops through each trial
        tarray = saraharray[saraharray[:,9] ==trial,:] # get data only for each tria
        binarray = makebinarray(tarray, bins)  # allocate each raw data row to a bi
        for bcount, b in enumerate(bins): #iterate round bins
            barray = tarray[binarray[:,0] == b,:] # get data for each bin only
            speedmean = np.nanmean(barray[:,2])
            stopsarraybeacon[bcount,tcount] = speedmean
            bcount+=1
    return stopsarraybeacon



# CALCULATE FIRST STOPPING LOCATION PER TRIAL
def FirstStops( trarray,stops ):
    data = []
    for row in trarray: # for each trial
        tarray = stops[stops[:,2] ==row,:] # get data only for each trial
        for row in tarray: # go through row of data for specified trial
            if float(row[0]) > 3.2 and float(row[0]) <= 11.2: # if stop is between black boxes
                data.append([float(row[0]), int(row[1]), int(row[2])]) # append data
                break # break so only get first stop then goes onto next trial
    return np.array(data)



# CALCULATE TIME OF FIRST STOP
def FirstStopTime( trarray,stops ):
    data = []
    for row in trarray: # loop through trials
        tarray = stops[stops[:,3] ==row,:] # get data only for each trial
        for row in tarray: # each row of data
            if float(row[1]) > 3.2 and float(row[1]) <= 17.2: # if stop after blackbox and before end of reward zone
                data.append([float(row[5]), int(row[3]),float(row[1])]) # time, trial location
                break # break out of loop - this way only gets first stop for each trial
    return np.array(data)

# extract time output:
#data.append([float(row[0]),float(row[1]), int(row[8]), int(row[9]), float(starttime), float(trialtime), int(row[4])]) # take data



# CALCUALTE STOP RATIO FOR GAIN MODULATION TRIALS
def ratio_stop(stops):
    motor = np.sum(stops[15:17]) # stops in motor reward zone
    visual = np.sum(stops[9:11]) # stops in visual reward zone
    ratios = visual/(motor+visual) # calculate ratio
    return ratios



# SPLIT TRIALS ACCORDING TO TIME TO FIRST STOP
def SplitTrials(stops):
    #find upper and lower percentiles
    l = np.percentile(stops[:,0], 25) # upper
    u = np.percentile(stops[:,0], 75) # lower
    middle = np.percentile(stops[:,0], 50) # median
    low = []; high = []; lower = []; upper = []
    for row in stops: # loop through stops
        if float(row[0]) >= middle:
            if float(row[0]) <= u:
                high.append([float(row[0]), int(row[1]),float(row[2])]) # take data for stop
            elif float(row[0]) > u:
                upper.append([float(row[0]), int(row[1]),float(row[2])]) # take data for stop
        if float(row[0]) <= middle:
            if float(row[0]) > l:
                low.append([float(row[0]), int(row[1]),float(row[2])]) # take data for stop
            elif float(row[0]) <= l:
                lower.append([float(row[0]), int(row[1]),float(row[2])]) # take data for stop
    return np.array(high), np.array(low), np.array(upper), np.array(lower)


def SplitTrials2(stops, trialids_h, trialids_l, trialids_h2, trialids_l2):
    low = []
    high = []
    lower = []
    upper = []
    
    for row in trialids_h:
        tarray = stops[stops[:,3] ==row,:] # get data only for each trial
        for row in tarray:
            high.append([float(row[1]),int(row[2]), int(row[3]), int(row[4])]) # take data for stop
    for row in trialids_l:
        tarray = stops[stops[:,3] ==row,:] # get data only for each trial
        for row in tarray:
            low.append([float(row[1]),int(row[2]), int(row[3]), int(row[4])]) # take data for stop
    for row in trialids_h2:
        tarray = stops[stops[:,3] ==row,:] # get data only for each trial
        for row in tarray:
            upper.append([float(row[1]),int(row[2]), int(row[3]), int(row[4])]) # take data for stop
    for row in trialids_l2:
        tarray = stops[stops[:,3] ==row,:] # get data only for each trial
        for row in tarray:
            lower.append([float(row[1]),int(row[2]), int(row[3]), int(row[4])]) # take data for stop
    return np.array(high), np.array(low), np.array(upper), np.array(lower)




# split trials according to low and high speed
def lowhighstops(stopsdata_p, stops_f_p):
    if stops_f_p.size < 7:
        return np.zeros((BINNR,)), np.zeros((BINNR,)), np.zeros((0,3)), np.zeros((0,3)), np.zeros((0,1)), np.zeros((0,1)), np.zeros((BINNR,)), np.zeros((BINNR,)), np.zeros((0,3)), np.zeros((0,3)), np.zeros((0,1)), np.zeros((0,1))
    
    if stops_f_p.size >6:
        #trials = maketrialarray(stopsdata_p)
        #stopsdata_p[:,2] = trials
        high_p,low_p, upper_p,lower_p = SplitTrials(stops_f_p)
        #print('size',low_p.size, lower_p.size, high_p.size, upper_p.size)
        #average time and location for quartiles
        time_high =np.nanmean(high_p[:,0]);time_low =np.nanmean(low_p[:,0])
        time_upper =np.nanmean(upper_p[:,0]);time_lower =np.nanmean(lower_p[:,0])
        loc_high =np.nanmean(high_p[:,2]);loc_low =np.nanmean(low_p[:,2])
        loc_upper =np.nanmean(upper_p[:,2]);loc_lower =np.nanmean(lower_p[:,2])
        # trial numbers for fast and slow trials
        #print(stops_f_p.size,high_p,low_p, upper_p,lower_p)
        h_trials_p = np.unique(high_p[:,1])
        l_trials_p = np.unique(low_p[:,1])
        h2_trials_p = np.unique(upper_p[:,1])
        l2_trials_p = np.unique(lower_p[:,1])
        #print('trials',l_trials_p.size, l2_trials_p.size, h_trials_p.size, h2_trials_p.size)
        # split all stop data according to trial
        high_p, low_p, upper_p, lower_p = SplitTrials2(stopsdata_p, h_trials_p, l_trials_p, h2_trials_p, l2_trials_p)
        #print('size2',low_p.size, lower_p.size, high_p.size, upper_p.size)
        high_p = filterstops(high_p)
        low_p = filterstops(low_p)
        upper_p = filterstops(upper_p)
        lower_p = filterstops(lower_p)
        # stop histograms
        high_stops_p = create_srdata(high_p, h_trials_p)
        avg_high_stops_p = np.nanmean(high_stops_p, axis=0)
        low_stops_p = create_srdata(low_p, l_trials_p)
        avg_low_stops_p = np.nanmean(low_stops_p, axis=0)
        upper_stops_p = create_srdata(upper_p, h2_trials_p)
        avg_upper_stops_p = np.nanmean(upper_stops_p, axis=0)
        lower_stops_p = create_srdata(lower_p, l2_trials_p)
        avg_lower_stops_p = np.nanmean(lower_stops_p, axis=0)
    
    return avg_high_stops_p, avg_low_stops_p, high_p, low_p, h_trials_p, l_trials_p, avg_upper_stops_p, avg_lower_stops_p,upper_p, lower_p, h2_trials_p, l2_trials_p, time_high,time_low,time_upper,time_lower, loc_high,loc_low,loc_upper,loc_lower




#function for speed per trial
def speed_per_trial_gain(bins,saraharray, trarray):
    
    stopsarraybeacon = np.zeros(((bins.shape[0]), (trarray.shape[0]))) # rows == same number of bins
    stopsarraybeacon[:,:] = np.nan
    
    for tcount,trial in enumerate(trarray):# loops through each trial
        tarray = saraharray[saraharray[:,9] ==trial,:] # get data only for each tria
        binarray = makebinarray(tarray, bins)  # allocate each raw data row to a bi
        moved = 0 # same no of rows as barray, 1 column
        for rowcount, row in enumerate(tarray[1:,:]): # for every row in barray, all columns
            if tarray[rowcount,2] <= 0.9: #if speed is below threshold (0.9 cm per second)
                moved+=1 # add one to row in movearray for every row stopped in barray
            rowcount+=1
        if moved > 1:
            for bcount, b in enumerate(bins): #iterate round bins
                barray = tarray[binarray[:,0] == b,:] # get data for each bin only
                speedmean = np.nanmean(barray[:,2])
                stopsarraybeacon[bcount,tcount] = speedmean
                bcount+=1
    return stopsarraybeacon



def create_timebindata( stops, trialids ):
    BINNR = 100
    if stops.size == 0:
        return np.zeros((BINNR,))
    
    # create histogram
    posrange = np.linspace(0.01, 100.01, num=BINNR+1)
    trialrange = trialids
    trialrange = np.append(trialrange, trialrange[-1]+1)  # Add end of range
    values = np.array([[trialrange[0], trialrange[-1]],
                       [posrange[0], posrange[-1]]])

    H, bins, ranges = np.histogram2d(stops[:,2], stops[:,1], bins=(trialrange, posrange), range=values)
    
    return H


# -------------------------------------------------------------------------------------------------------------- #

# FUNCTIONS BELOW ARE FOR PLOTTING GRAPHS




def adjust_spines(ax,spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',0)) # outward by 10 points
        #spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])



#functions for legends - each for a diff location
def makelegend(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(1.02, 0.9), fontsize = "xx-large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(0.976, 0.9), fontsize = "large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend2(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(0.976, 0.6), fontsize = "large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend3(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(0.716, 0.9), fontsize = "large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend4(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(0.716, 0.6), fontsize = "large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)


#-------------------------------------------------------------------------------------------------------------- #

