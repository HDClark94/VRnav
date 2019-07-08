# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:45:30 2013
Edited on the 27/3/2014 for conversion to python3
Edited: 21st may 2014


@author: Sarah Tennant


"""

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib import rcParams
from Functions_Params_0100 import STOP_THRESHOLD, DIST, HDF_LENGTH, BINNR, SHUFFLE_N
from scipy.stats import uniform
from scipy import stats
import random
from Functions_CoreFunctions_0100 import adjust_spines

# -------------------------------------------------------------------------------------------------------------- #

# function to get stops of stops per trial
def extractstops(stops):
    moving = False
    data = []
    for row in stops:
        if(row[2]<=STOP_THRESHOLD and moving):
            moving = False
            # position, (beaconed/non-beaconed/probe), trialid
            data.append([float(row[1])+0.2, int(row[8]), int(row[9]), float(row[2])])
        
        if(row[2]>STOP_THRESHOLD and not moving):
            moving = True
    return np.array(data)

# -------------------------------------------------------------------------------------------------------------- #


# function to get location of stops per trial
def extracttime(saraharray):
    moving = False
    data = []
    for row in saraharray:
        if row[1] <=3.2: # if row is below black box
            starttime = row[0] # get time just before leaving black box
        if(row[2]<=STOP_THRESHOLD and moving): # if stopped
            moving = False
            stoptime = row[0] # time of stop
            trialtime = stoptime - starttime
            data.append([float(row[0]),float(row[1]), int(row[8]), int(row[9]), float(starttime), float(trialtime)]) # take data
        
        if(row[2]>STOP_THRESHOLD and not moving):
            moving = True
    return np.array(data)

# -------------------------------------------------------------------------------------------------------------- #

#Corrects error in trial number caused by error in writing to raw data file - only run non beaconed and probe data through this
def SortTrial( stops ):
    for row in stops: # go through every row of saraharray
        if(row[9] % 3 !=0): # if the trial number does not divide by 5 with no remainders
            row[9] = int(row[9])-1 # change the trial number to original trial number - 1

    return stops

def SortTrial_b( saraharray ):
    for row in saraharray: # go through every row of saraharray
        if(row[1] > 11): # if the trial number does not divide by 5 with no remainders
            row[9] = int(row[9])-1 # change the trial number to original trial number - 1
    return saraharray

# -------------------------------------------------------------------------------------------------------------- #

def filterstops(stops):
    if stops.size ==0:
        return stops

    data = np.zeros((stops.shape[0],1))
    for rowcount, row in enumerate(range(len(stops)-1)):
        diff = float(stops[rowcount+1,0] - stops[rowcount,0])
        trialsame = float(stops[rowcount+1,2] - stops[rowcount,2])
        if diff < 0.1 and trialsame == 0:
            data[rowcount+1,0] = 1
    stops = np.hstack((stops, data))
    stops = np.delete(stops, np.where(stops[:, 4] == 1), 0)

    return stops

# -------------------------------------------------------------------------------------------------------------- #

def maketrialarray(saraharray):
    trialarray = np.zeros((saraharray.shape[0],1))
    trialnumber = 1
    for rowcount, row in enumerate(saraharray[1:,:]):
        if saraharray[rowcount, 1]-saraharray[rowcount-1,1]<-15:
            trialnumber+=1
        trialarray[rowcount] = trialnumber # creates array for trial number in each row of saraharray
        #print trialcount, trialarray
        rowcount+=1
    return trialarray


# -------------------------------------------------------------------------------------------------------------- #

def create_srdata( stops, trialids ):
    if stops.size == 0:
        return np.zeros((BINNR,))
    
    # create histogram
    posrange = np.linspace(0, HDF_LENGTH, num=BINNR+1)
    trialrange = trialids
    trialrange = np.append(trialrange, trialrange[-1]+1)  # Add end of range
    values = np.array([[trialrange[0], trialrange[-1]],
                       [posrange[0], posrange[-1]]])

    H, bins, ranges = np.histogram2d(stops[:,2], stops[:,0], bins=(trialrange, posrange), range=values)
    H[np.where(H[::]>1)] = 1
    
    return H


def create_srdata_tracks( stops, trialids , TRACKS):
    if stops.size == 0:
        return np.zeros((TRACKS,))
    
    # create histogram
    posrange = np.linspace(0, TRACKS, num=TRACKS+1)
    trialrange = trialids
    trialrange = np.append(trialrange, trialrange[-1]+1)  # Add end of range
    values = np.array([[trialrange[0], trialrange[-1]],
                       [posrange[0], posrange[-1]]])

    H, bins, ranges = np.histogram2d(stops[:,2], stops[:,0], bins=(trialrange, posrange), range=values)
    H[np.where(H[::]>1)] = 1
    
    return H

# -------------------------------------------------------------------------------------------------------------- #

def shuffle_analysis(stopsdata,trialids):
    # Calculate stop rate for each section of the track
    srbin = create_srdata( stopsdata, trialids )                        # Array(BINNR, trialnum)
    # Shuffling data
    shuffled_srbin_mean = np.zeros((SHUFFLE_N, BINNR))
    #shuffled = np.zeros((20,4))
    for i in range(SHUFFLE_N):
        shuffled_stops = shuffle_stops(stopsdata)
        shuffled_srbin = create_srdata( shuffled_stops, trialids )      # Array(BINNR, trialnum)
        shuffled_srbin_mean[i] = np.mean(shuffled_srbin, axis=0)        # Array(BINNR)
    
    return srbin,shuffled_srbin_mean



# create shuffled and real datasets
def shuffle_analysisfinal(stopsdata, trialids):
    SHUFFLE1 = 100
    # Calculate stop rate for each section of the track
    srbin = create_srdata( stopsdata, trialids )                        # Array(BINNR, trialnum)
    # Shuffling random 100 trials 1000 times
    shuffled_srbin_mean = np.zeros((SHUFFLE_N, BINNR))
    for i in range(SHUFFLE_N): # for i in 1000
        shuffledtrials = np.zeros((SHUFFLE1, 5))
        shuffleddata =np.zeros((SHUFFLE1, BINNR))
        for n in range(SHUFFLE1): # Create sample data with 100 trials
            trial = random.choice(trialids) # select random trial from real dataset
            data = stopsdata[stopsdata[:,2] ==trial,:] # get data only for each tria
            shuffledtrial = shuffle_stops2(data,n) # shuffle the locations of stops in the trial
            shuffledtrials = np.vstack((shuffledtrials,shuffledtrial)) # stack shuffled stop
        trialids2 = np.unique(shuffledtrials[:, 2]) # find unique trials in the data
        shuffled_srbin = create_srdata( shuffledtrials, trialids2 ) #
        shuffled_srbin_mean[i] = np.mean(shuffled_srbin, axis=0)        # Array(BINNR)
    # Mean of the mean stops in the shuffled data for each bin

    return srbin, shuffled_srbin_mean

# -------------------------------------------------------------------------------------------------------------- #

# create shuffled and real datasets
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
    print(stopsdata.shape,stopsdata.size )
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

# create shuffled and real datasets
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

"""
# OLD FUNCTION
# create shuffled and real datasets
def shuffle_analysis_pertrial(stopsdata, trialids = np.arange(1,100,1)):
    #SHUFFLE_N = 1000
    # Calculate stop rate for each section of the track
    srbin = create_srdata( stopsdata, trialids )                        # Array(BINNR, trialnum)
    srbin_mean = np.mean(srbin, axis=0)                                 # Array(BINNR)
    srbin_std = stats.sem(srbin, axis=0)                               # Array(BINNR)
    # Shuffling data
    shuffled_srbin_mean = np.zeros((SHUFFLE_N, BINNR))
    #shuffled_srbin_sd = np.zeros((SHUFFLE_N, BINNR))
    for i in range(SHUFFLE_N): # for i in 1000
        shuffled_stops = shuffle_stops(stopsdata) # shuffle stops
        shuffled_srbin = create_srdata( shuffled_stops, trialids )      # Array(BINNR, trialnum)
        shuffled_srbin_mean[i] = np.mean(shuffled_srbin, axis=0)        # Array(BINNR)
        #shuffled_srbin_sd[i] = stats.sem(shuffled_srbin, axis=0)        # Array(BINNR)

    # Mean of the mean stops in the shuffled data for each bin
    shuffled_mean = np.mean(shuffled_srbin_mean, axis=0)                # Array(BINNR)
    shuffled_std = np.std(shuffled_srbin_mean, axis=0)                  # Array(BINNR)

    return srbin_mean, srbin_std, shuffled_mean, shuffled_std


"""

# -------------------------------------------------------------------------------------------------------------- #

"""
# OLD FUNCTION
def shuffle_stops1( stops,t ):
    shuffled_stops = np.copy(stops) # this is required as otherwise the original dataset would be altered
    # create an array that contains the amount by which every stop will be shuffled
    rand_rotation = uniform.rvs(loc=0, scale=HDF_LENGTH, size=stops.shape[0])
    #print('rand_rotation',rand_rotation)
    # loop through every row
    for i in range(stops.shape[0]):
        # retrieve stops from current trial and add random value
        shuffled_stops[i,0] = rand_rotation[i]
        shuffled_stops[i,0] %= HDF_LENGTH
        shuffled_stops[i,2] = t

    return shuffled_stops
"""

def shuffle_stops2( stops,n ):
    shuffled_stops = np.copy(stops) # this is required as otherwise the original dataset would be altered
    # create an array that contains the amount by which every stop will be shuffled
    rand_rotation = uniform.rvs(loc=0, scale=HDF_LENGTH, size=stops.shape[0])
    # add random value
    shuffled_stops[:,0] = rand_rotation
    shuffled_stops[:,2] = n
    
    return shuffled_stops

def shuffle_stops_tracks( stops,n , TRACKS):
    shuffled_stops = np.copy(stops) # this is required as otherwise the original dataset would be altered
    # create an array that contains the amount by which every stop will be shuffled
    rand_rotation = uniform.rvs(loc=0, scale=TRACKS, size=stops.shape[0])
    # add random value
    shuffled_stops[:,0] = rand_rotation
    shuffled_stops[:,2] = n
    
    return shuffled_stops

# -------------------------------------------------------------------------------------------------------------- #

"""
#OLD CODE
def shuffle_stops( stops ):
    shuffled_stops = np.copy(stops) # this is required as otherwise the original dataset would be altered
    highest_trial_id = shuffled_stops[-1,2]
    # create an array that contains the amount by which every trial will be shuffled
    rand_rotation = uniform.rvs(loc=0, scale=HDF_LENGTH, size=highest_trial_id)
    # loop through every trial
    for i in range(stops.shape[0]):
        # retrieve stops from current trial and add random value
        shuffled_stops[i,0] += rand_rotation[shuffled_stops[i,2]-1]
        shuffled_stops[i,0] %= HDF_LENGTH
    
    return shuffled_stops
"""
# -------------------------------------------------------------------------------------------------------------- #


# FUNCTION FOR REWARDS
def extractrewards(stops):
    data = []
    for row in stops:
        if(row[4]==1):
            data.append([float(row[1])+0.2, int(row[8]), int(row[9])])
                
    return np.array(data)


# -------------------------------------------------------------------------------------------------------------- #

# find values in each bin
def makebinarray(tarray, bins):
    
    interval = 1
    binarray = np.zeros((tarray.shape[0], 1))
    
    # Collect data for each bin
    for bcount,b in enumerate(bins):
        binmin = tarray[:,1]>=b # lowest value in bin
        binmax = tarray[:,1]<b+interval # highest value in bin
        arraylogical = np.logical_and(binmin,binmax) #get all rows that satisfy being within bin
        binarray[arraylogical, 0] = b #assign each row in tarray its respective bin
    
    return binarray

# -------------------------------------------------------------------------------------------------------------- #


#function for speed per trial
def speed_per_trial(bins,saraharray, trarray):
    
    stopsarraybeacon = np.zeros(((bins.shape[0]), (trarray.shape[0]))) # rows == same number of bins
    stopsarraybeacon[:,:] = np.nan
    
    for tcount,trial in enumerate(trarray):# loops through each trial
        tarray = saraharray[saraharray[:,9] ==trial,:] # get data only for each tria
        binarray = makebinarray(tarray, bins)  # allocate each raw data row to a bi
        moved = 0 # same no of rows as barray, 1 column
        for rowcount, row in enumerate(tarray[1:,:]): # for every row in barray, all columns
            if tarray[rowcount,2] <= 0.7 and tarray[rowcount,1] > 6 and tarray[rowcount,1] < 14: #if speed is below threshold (0.9 cm per second)
                moved+=1 # add one to row in movearray for every row stopped in barray
            rowcount+=1
        if moved > 1:
            for bcount, b in enumerate(bins): #iterate round bins
                barray = tarray[binarray[:,0] == b,:] # get data for each bin only
                speedmean = np.nanmean(barray[:,2])
                stopsarraybeacon[bcount,tcount] = speedmean
                bcount+=1
    return stopsarraybeacon

# -------------------------------------------------------------------------------------------------------------- #


def z_score(srbin_mean, shuffled_mean):
    zscore = np.zeros((srbin_mean.shape[1])) # ARRAY[BINNR]
    shuffled_std = np.std(shuffled_mean, axis=0) # sd of shuffled data
    for bcount, b in enumerate(srbin_mean[1]):# go through each bin
        diff = np.nanmean(srbin_mean[:,bcount]) - np.nanmean(shuffled_mean[:,bcount]) # real stops - shuffled stops
        multi = diff / shuffled_std[bcount] # divided by shuffled SD
        zscore[bcount] = multi # allocate zscore to bin
    
    return zscore


def z_score1(srbin_mean, srbin_std, shuffled_mean, shuffled_std):
    zscore = np.zeros((srbin_mean.shape)) # ARRAY[BINNR]
    for bcount, b in enumerate(srbin_mean):# go through each bin
        diff = srbin_mean[bcount] - shuffled_mean[bcount] # real stops - shuffled stops
        multi = diff / shuffled_std[bcount] # divided by shuffled SD
        zscore[bcount] = multi # allocate zscore to bin
    
    return zscore

# -------------------------------------------------------------------------------------------------------------- #


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
# calculate first stop location for each trial
def FirstStops( trarray,stops ):
    data = []
    for row in trarray: # for each trial
        tarray = stops[stops[:,2] ==row,:] # get data only for each trial
        for row in tarray: # go through row of data for specified trial
            if float(row[0]) > 3.2 and float(row[0]) <= 17: # if stop is between black boxes
                data.append([float(row[0]), int(row[1]), int(row[2])]) # append data
                break # break so only get first stop then goes onto next trial
    return np.array(data)

#-------------------------------------------------------------------------------------------------------------- #

# function to get first stop
def FirstStops_Avg( trarray,stops ):
    data = []
    for row in trarray: # loop through trials
        tarray = stops[stops[:,2] ==row,:] # get data only for each trial
        for row in tarray: # each row of data
            if float(row[0]) > 3.2 and float(row[0]) <= 11: # if stop after blackbox and before end of reward zone
                data.append([float(row[0]), int(row[1]), int(row[2])]) # take data for stop
                break # break out of loop - this way only gets first stop for each trial
    return data

#-------------------------------------------------------------------------------------------------------------- #

def SortTrial_t12( stops ):
    for row in stops:
        if(row[9] % 3 !=0):
            row[9] = int(row[9])-1

    return stops

#-------------------------------------------------------------------------------------------------------------- #

# function to get first stop
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

#-------------------------------------------------------------------------------------------------------------- #

def RewardZoneStops( trarray,stops ):
    data = []
    for row in trarray: # loop through trials
        tarray = stops[stops[:,3] ==row,:] # get data only for each trial
        for row in tarray: # each row of data
            if row[1] >8.8 and row[1] <=11.2:
                data.append([float(row[0])-float(row[4]), int(row[3]), float(row[1])]) # take data for stop
                break
    return np.array(data)

#-------------------------------------------------------------------------------------------------------------- #

def Rewards( trarray,stops ):
    data = []
    for row in trarray:
        tarray = stops[stops[:,9] ==row,:] # get data only for each trial
        for row in tarray: # each row of data
            if float(row[1])<=3.2: # start time
                start = float(row[0])
            if int(row[4]) == 1: #
                #print('reward')
                data.append([float(row[0])-start, int(row[9]), float(row[1])]) # take data for stop
    return np.array(data)

#-------------------------------------------------------------------------------------------------------------- #
"""
def SplitTrials(stops):
    mediantime = np.median(stops[:,0])
    low = []
    high = []
    for row in stops: # loop through stops
        if float(row[0]) >= mediantime:
            high.append([float(row[0]), int(row[1]),float(row[2])]) # take data for stop
        elif float(row[0]) < mediantime:
            low.append([float(row[0]), int(row[1]),float(row[2])]) # take data for stop
    return np.array(high), np.array(low)

"""

def SplitTrials(stops):
    #mediantime = np.median(stops[:,0])
    #find upper and lower percentiles
    l = np.percentile(stops[:,0], 25)
    u = np.percentile(stops[:,0], 75)
    middle = np.percentile(stops[:,0], 50)
    print('quartiles',middle,l,u)
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

"""
def SplitTrials2(stops, trialids_h, trialids_l):
    low = []
    high = []
    
    for row in trialids_h:
        tarray = stops[stops[:,3] ==row,:] # get data only for each trial
        for row in tarray:
            high.append([float(row[1]),int(row[2]), int(row[3]), int(row[4])]) # take data for stop
    for row in trialids_l:
        tarray = stops[stops[:,3] ==row,:] # get data only for each trial
        for row in tarray:
            low.append([float(row[1]),int(row[2]), int(row[3]), int(row[4])]) # take data for stop
    return np.array(high), np.array(low)
"""

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


#-------------------------------------------------------------------------------------------------------------- #


def AllStops( trarray,stops ):
    data = []
    for row in trarray: # loop through trials
        tarray = stops[stops[:,3] ==row,:] # get data only for each trial
        for row in tarray: # each row of data
            data.append([float(row[0])-float(row[4]), int(row[3])]) # take data for stop
            if int(row[5]) == 1:
                break
    return np.array(data)


#-------------------------------------------------------------------------------------------------------------- #


# split trials according to low and high speed
def lowhighstops(stopsdata_p, stops_f_p):
    if stops_f_p.size < 7:
        return np.zeros((BINNR,)), np.zeros((BINNR,)), np.zeros((0,3)), np.zeros((0,3)), np.zeros((0,1)), np.zeros((0,1)), np.zeros((BINNR,)), np.zeros((BINNR,)), np.zeros((0,3)), np.zeros((0,3)), np.zeros((0,1)), np.zeros((0,1))

    if stops_f_p.size >6:
        #trials = maketrialarray(stopsdata_p)
        #stopsdata_p[:,2] = trials
        high_p,low_p, upper_p,lower_p = SplitTrials(stops_f_p)
        print('size',low_p.size, lower_p.size, high_p.size, upper_p.size)
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
        print('trials',l_trials_p.size, l2_trials_p.size, h_trials_p.size, h2_trials_p.size)
        # split all stop data according to trial
        high_p, low_p, upper_p, lower_p = SplitTrials2(stopsdata_p, h_trials_p, l_trials_p, h2_trials_p, l2_trials_p)
        print('size2',low_p.size, lower_p.size, high_p.size, upper_p.size)
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


#-------------------------------------------------------------------------------------------------------------- #

