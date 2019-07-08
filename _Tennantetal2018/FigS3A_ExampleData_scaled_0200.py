
"""


# PLOTS FIGURE OF EXAMPLE TRAINING SESSION


"""


#IMPORT FUNCTIONS AND PACKAGES
from Functions_CoreFunctions_0100 import adjust_spines,readhdfdata
from Functions_Core_0100 import maketrialarray
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
from scipy.stats import uniform
from matplotlib import lines, gridspec

#LOAD DATA
#specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task18_lengths_0100.h5'

#specify mouse and days to analyse
days = ['Day' + str(int(x)) for x in np.arange(11,25.1)]
mice = ['M' + str(int(x)) for x in [1,5]]# specific day/s

print ('Loading data from...' + filename)
track = 200 # specify track size in cm

# SPECIFY TRACK PARAMETERS
REAL_LENGTH = 200 #track length in cm
HDF_LENGTH = 20 #track length in VU
SCALE = HDF_LENGTH/REAL_LENGTH
BINNR = 200 #location bins
SHUFFLE_N = 1000
STOP_THRESHOLD = 0.7 #stop threshold

RZ_LEN = 20*SCALE
OB_LEN = 60*SCALE
RZ_START = 90*SCALE
RZ_END = 110*SCALE
OB_START = 30*SCALE
OB_END = 90*SCALE
BB_BACK = 170*SCALE


### --------------------------------------------------------------------------------------------------------------------- #
# functions specific to this code - increasing lengths

#function to extract stops from raw data
def extractstops(saraharray):
    moving = False
    data = []
    for row in saraharray: # every row in raw data
        if(row[2]<=STOP_THRESHOLD and moving): # is speed below stop threshold
            moving = False
            data.append([float(row[1])+0.2, int(row[8]), int(row[9])]) # append position, (beaconed/non-beaconed/probe), trialid
                            
        if(row[2]>STOP_THRESHOLD and not moving):
            moving = True
    return np.array(data)


# function to shuffle stops
def shuffle_stops( stops ):
    # this is required as otherwise the original dataset would be altered
    shuffled_stops = np.copy(stops)

    highest_trial_id = shuffled_stops[-1,2]

    # create an array that contains the amount by which every trial will be shuffled
    rand_rotation = uniform.rvs(loc=0, scale=tracklength/10, size=highest_trial_id)
    # loop through every trial
    for i in range(stops.shape[0]):
        # retrieve stops from current trial and add random value
        shuffled_stops[i,0] += rand_rotation[shuffled_stops[i,2]-1]
        shuffled_stops[i,0] %= tracklength/10
        
    return shuffled_stops

# function to create histogram of stop locations vs trial
def create_srdata( stops, trialids,tracklength ):
    if stops.size == 0:
        return np.zeros((tracklength,))

    # create histogram
    posrange = np.linspace(0, tracklength/10, num=tracklength+1)
    trialrange = trialids
    trialrange = np.append(trialrange, trialrange[-1]+1)  # Add end of range
    values = np.array([[trialrange[0], trialrange[-1]],
                       [posrange[0], posrange[-1]]])

    H, bins, ranges = np.histogram2d(stops[:,2], stops[:,0], bins=(trialrange, posrange), range=values)
    H[np.where(H[::]>1)] = 1
    
    return H

# create shuffled and real datasets
def shuffle_analysis(stopsdata, trialids,tracklength):
    # Calculate stop rate for each section of the track
    srbin = create_srdata( stopsdata, trialids, tracklength )                        # Array(BINNR, trialnum)
    srbin_mean = np.mean(srbin, axis=0)                                 # Array(BINNR)
    srbin_std = stats.sem(srbin, axis=0)                               # Array(BINNR)
    # Shuffling data
    shuffled_srbin_mean = np.zeros((SHUFFLE_N, tracklength))
    for i in range(SHUFFLE_N):
        shuffled_stops = shuffle_stops(stopsdata) 
        shuffled_srbin = create_srdata( shuffled_stops, trialids, tracklength )      # Array(BINNR, trialnum)
        shuffled_srbin_mean[i] = np.mean(shuffled_srbin, axis=0)        # Array(BINNR)
    # Mean of the mean stops in the shuffled data for each bin
    shuffled_mean = np.mean(shuffled_srbin_mean, axis=0)                # Array(BINNR)
    shuffled_std = np.std(shuffled_srbin_mean, axis=0)                  # Array(BINNR)

    return srbin_mean, srbin_std, shuffled_mean, shuffled_std



# function to find location & trials rewards are dispensed
def reward_per_trial(trialarray,bins,saraharray,trarray):
    

    rewardtrial_nb = []
    rewardloc_nb = []
    
    for tcount,trial in enumerate(trarray):# loops through each trial
        tarray = saraharray[saraharray[:,9] ==trial,:] # get data only for each trial
        binarray = makebinarray(tarray, bins)  # allocate each raw data row to a bin
        for bcount, b in enumerate(bins): #iterate round bins
            barray = tarray[binarray[:,0] == b,:] # get data for each bin only
            binrewardarray = barray[:,4] # check if reward was released
            rowcount = 0
            for rowcount, row in enumerate(binrewardarray):
                if binrewardarray[rowcount] > 0: # get reward data
                    rewardtrial_nb = np.append(rewardtrial_nb,trial) # store trial number
                    rewardloc_nb = np.append(rewardloc_nb, ((b)+0.2)) # store location
                rowcount +=1

    return rewardtrial_nb, rewardloc_nb



# function to assign each row of raw data to a location bin
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


# function to calculate speed per trial
def speed_per_trial(bins,saraharray, trarray):
    
    stopsarraybeacon = np.zeros(((bins.shape[0]), (trarray.shape[0]))) # rows == same number of bins
    stopsarraybeacon[:,:] = np.nan
     
    for tcount,trial in enumerate(trarray):# loops through each trial
        tarray = saraharray[saraharray[:,9] ==trial,:] # get data only for each tria
        binarray = makebinarray(tarray, bins)  # allocate each raw data row to a bin
        #check if animal stopped during trial
        moved = 0
        for rowcount, row in enumerate(tarray[1:,:]): # loop every row in barray, all columns
            if tarray[rowcount,2] <= 0.9: #if speed is below threshold (0.9 cm per second)
                moved+=1 # add one to row in movearray for every row stopped in barray
            rowcount+=1
        # if the animal stops at all in the trial, calculate speed
        if moved > 1:
            for bcount, b in enumerate(bins): #iterate round bins
                barray = tarray[binarray[:,0] == b,:] # get data for each bin only
                speedmean = np.nanmean(barray[:,2]) # average speed in that bin
                stopsarraybeacon[bcount,tcount] = speedmean # put average speed in store
                bcount+=1
    return stopsarraybeacon


#functions for legends - each for a diff location
def makelegend(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(1.02, 0.9), fontsize = "xx-large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)
def makelegend2(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(1.02, 0.6), fontsize = "xx-large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend3(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(0.75, 0.9), fontsize = "xx-large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)
def makelegend4(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(0.75, 0.6), fontsize = "xx-large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)


### --------------------------------------------------------------------------------------------------------------------- #


# FOR EACH DAY AND MOUSE, PULL RAW DATA, CALCULATE STOPS PER TRIAL AND PLOT GRAPH
for dcount,day in enumerate(days): #load mouse and day
    for mcount,mouse in enumerate(mice):
        print ('Processing...',day,mouse)
        #load HDF5 data set for that day and mouse
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        # make array of trial number per row of data in dataset
        trialarray = maketrialarray(saraharray) # make array of trial number same size as saraharray
        print(trialarray.shape, saraharray[:,9].shape)
        saraharray[:,9] = trialarray[:,0] # replace trial number because of increment error (see README.py)
        
        length = np.max(saraharray[:,1])
        trialno = np.max(saraharray[:,9])
        #define track length parameters
        rewardzonestart = saraharray[1,11]
        rewardzoneend = saraharray[1,12]
        
        if rewardzonestart == 8.8:
            tracklength = 200
        elif rewardzonestart == 11.8:
            tracklength = 230
        elif rewardzonestart == 16.3:
            tracklength = 275
        elif rewardzonestart == 23.05:
            tracklength = 342
        elif rewardzonestart == 33.1:
            tracklength = 458
        elif rewardzonestart == 48.1:
            tracklength = 591
        
        if length > 24 and length <27.5:
                tracklength = 245

        trarray = np.arange(np.min(saraharray[:,9]),np.max(saraharray[:,9]),1) #array of trial number
        binmin = np.min(saraharray[:,1]);binmax = np.max(saraharray[:,1]);interval = 0.1 # i.e if track is 20, 0.2 interval gives 100 bins 
        bins = np.arange(0,tracklength/10,interval) # add 1e-6 so that last point included - array of bins for location
        sbins = np.arange(binmin,binmax+1e-6,0.1)
        # Extract data for beaconed, non-beaconed, probe
        
        if tracklength == 200:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 10), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 20), 0)# delete all data not on probe tracks
        if tracklength == 230:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 30), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 40), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 50), 0)# delete all data not on probe tracks
        if tracklength == 275:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 60), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 70), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 80), 0)# delete all data not on probe tracks
        if tracklength == 342:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 90), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 100), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 110), 0)# delete all data not on probe tracks
        if tracklength == 458:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 120), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 130), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 140), 0)# delete all data not on probe tracks
        if tracklength == 591:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 150), 0) # delete all data not on beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(saraharray[:, 8] != 160), 0)# delete all data not on non beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 170), 0)# delete all data not on probe tracks
        # For gain modulated trials
        if tracklength == 245:
            dailymouse_b = np.delete(saraharray, np.where(saraharray[:, 8] > 0), 0) # delete all data not on beaconed tracks
            dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != -10), 0)# delete all data not on probe tracks
            dailymouse = np.delete(saraharray, np.where(saraharray[:, 8] == 0), 0)# delete all data not on non beaconed tracks
            dailymouse_nb = np.delete(saraharray, np.where(dailymouse[:, 8] == -10), 0)# delete all data not on non beaconed tracks

        # get array of trial numbers
        trialids_b = np.unique(dailymouse_b[:, 9])
        trialids_nb = np.unique(dailymouse_nb[:, 9])
        trialids_p = np.unique(dailymouse_p[:, 9])
        
        #extract stops
        stopsdata_b = extractstops(dailymouse_b)
        stopsdata_nb = extractstops(dailymouse_nb)
        stopsdata_p = extractstops(dailymouse_p)
                
        # get mean stops per bin for real and shuffled data
        if stopsdata_b.size > 0:
            srbin_mean_b, srbin_std_b,shuffled_mean_b, shuffled_std_b = shuffle_analysis(stopsdata_b, trialids_b,tracklength)
        if stopsdata_nb.size > 0:
            srbin_mean_nb, srbin_std_nb, shuffled_mean_nb, shuffled_std_nb = shuffle_analysis(stopsdata_nb, trialids_nb,tracklength)
        if stopsdata_p.size > 0:
            srbin_mean_p, srbin_std_p, shuffled_mean_p, shuffled_std_p = shuffle_analysis(stopsdata_p, trialids_p,tracklength)
        
        #get location and trial number of rewards
        rewardtrial_beac,rewardloc_beac = reward_per_trial(trialarray,bins,dailymouse_b,trarray)
        rewardtrial_nbeac,rewardloc_nbeac = reward_per_trial(trialarray,bins,dailymouse_nb,trarray)
        rewardtrial_probe,rewardloc_probe = reward_per_trial(trialarray,bins,dailymouse_p,trarray)
        
        # calculate speed
        speed_beaconed = speed_per_trial(sbins,saraharray,trialids_b)
        speed_nbeaconed = speed_per_trial(sbins,saraharray,trialids_nb)
        speed_probe = speed_per_trial(sbins,saraharray,trialids_p)
        sd_speed_beaconed = np.nanstd(speed_beaconed,axis = 1)
        sd_speed_nbeaconed = np.nanstd(speed_nbeaconed,axis = 1)
        sd_speed_probe = np.nanstd(speed_probe,axis = 1)
        speed_beaconed = np.nanmean(speed_beaconed,axis = 1)
        speed_nbeaconed = np.nanmean(speed_nbeaconed,axis = 1)
        speed_probe = np.nanmean(speed_probe,axis = 1)

        # plot graphs:
        if tracklength == 200 and trialno > 4:
            if stopsdata_p.size > 0: # if theres probe trials, plot 3x3 subplots -> stops per trial, average stops, speed for beaconed, non-beaconed and probe trials
                # make figure       
                fig = plt.figure(figsize = (12,12))
                ax = fig.add_subplot(3,3,1) #stops per trial
                ax.set_title('Beaconed trials', fontsize = 22,verticalalignment = 'bottom', style = 'italic')
                ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(20-3, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_b[:,0],stopsdata_b[:,2], 'o', color = 'Black', markersize =4.5, label = 'Stop') #plot becaoned trials
                ax.plot(rewardloc_beac,rewardtrial_beac, '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
                #ax.plot(lickloc_beac,licktrial_beac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,20)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                makelegend(fig,ax) # make legend
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_ylabel('Trial number', fontsize=18, labelpad = 20)

                ax = fig.add_subplot(3,3,2) #stops per trial
                ax.set_title('Non-beaconed trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(20-3, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_nb[:,0],stopsdata_nb[:,2], 'o',color = 'Black', markersize = 4.5) #plot becaoned trials
                ax.plot(rewardloc_nbeac,rewardtrial_nbeac, '>', color = 'Red', markersize = 6) #plot becaoned trials
                #ax.plot(lickloc_nbeac,licktrial_nbeac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,20)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
                makelegend(fig,ax) # makes legend 
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])

                ax = fig.add_subplot(3,3,3) #stops per trial
                ax.set_title('Probe trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(20-3, 20, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_p[:,0],stopsdata_p[:,2], 'o',color = 'Black', markersize = 4.5, label = 'Non - beaconed') #plot becaoned trials
                #ax.plot(lickloc_probe,licktrial_probe, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,20)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=8) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax1 = fig.add_subplot(3,3,4)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5, srbin_mean_b, color = 'Black',linewidth=1) #plot becaoned trials              
                ax1.fill_between(bins*5,srbin_mean_b-srbin_std_b,srbin_mean_b+srbin_std_b, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_b, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_b-shuffled_std_b,shuffled_mean_b+shuffled_std_b, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,0.65)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Stops (cm/trial)', fontsize=18, labelpad = 23)
                ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                
                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,5)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_nb, color = 'Black',linewidth=1, label = 'Real') #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_nb-srbin_std_nb,srbin_mean_nb+srbin_std_nb, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_nb, color = 'DodgerBlue',linewidth=1, label = 'Shuffled') #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_nb-shuffled_std_nb,shuffled_mean_nb+shuffled_std_nb, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                makelegend2(fig,ax1) # makes legend 
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])
                
                ax1 = fig.add_subplot(3,3,6)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_p,color = 'Black',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_p-srbin_std_p,srbin_mean_p+srbin_std_p, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_p, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_p-shuffled_std_p,shuffled_mean_p+shuffled_std_p, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])

                ax1 = fig.add_subplot(3,3,7)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*5,speed_beaconed,'-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*5,speed_beaconed-sd_speed_beaconed,speed_beaconed+sd_speed_beaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,120)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Speed (cm/s)', fontsize=18, labelpad = 20)
                ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['0', '100', '200'])

                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,8)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*5,speed_nbeaconed, '-',markersize = 2,color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*5,speed_nbeaconed-sd_speed_nbeaconed,speed_nbeaconed+sd_speed_nbeaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,120)
                ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                ax1.set_xlabel('Location (cm)', fontsize=18, labelpad = 20)
                ax1.set_xticklabels(['0', '100', '200'])
                
                ax1 = fig.add_subplot(3,3,9)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*5,speed_probe, '-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*5,speed_probe-sd_speed_probe,speed_probe+sd_speed_probe, facecolor = 'Black', alpha = 0.3)                
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,120)
                ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                ax1.set_xticklabels(['0', '100', '200'])
                
                plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
                fig.savefig('Plots/Supplemental3/Example' + 'Data' + '_' + str(mouse) + '_' + str(day) + '.png',  dpi = 200)
                plt.close()

            else: # if there is not probe trials, plot 2x3 subplots -> stops per trial, average stops, speed for beaconed and non-beaconed trials
                fig = plt.figure(figsize = (12,12))
                ax = fig.add_subplot(3,3,1) #stops per trial
                ax.set_title('Beaconed trials', fontsize = 18,verticalalignment = 'bottom', style = 'italic')
                ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax.axvspan(20-3, 20, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_b[:,0],stopsdata_b[:,2], 'o', color = 'Black', markersize =4.5, label = 'Stop') #plot becaoned trials
                ax.plot(rewardloc_beac,rewardtrial_beac, '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
                #ax.plot(lickloc_beac,licktrial_beac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,20)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                makelegend3(fig,ax)
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_ylabel('Trial number', fontsize=18, labelpad = 20)

                ax = fig.add_subplot(3,3,2) #stops per trial
                ax.set_title('Non-beaconed trials', fontsize = 18, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax.axvspan(20-3, 20, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_nb[:,0],stopsdata_nb[:,2], 'o',color = 'Black', markersize = 4.5) #plot becaoned trials
                ax.plot(rewardloc_nbeac,rewardtrial_nbeac, '>', color = 'Red', markersize = 6) #plot becaoned trials
                #ax.plot(lickloc_nbeac,licktrial_nbeac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,20)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
                makelegend3(fig,ax) # makes legend 
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
              
                ax1 = fig.add_subplot(3,3,4)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5, srbin_mean_b, color = 'Black',linewidth=1) #plot becaoned trials              
                ax1.fill_between(bins*5,srbin_mean_b-srbin_std_b,srbin_mean_b+srbin_std_b, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_b, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_b-shuffled_std_b,shuffled_mean_b+shuffled_std_b, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,0.3)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Stops (cm/trial)', fontsize=18, labelpad = 20)
                ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                
                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,5)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_nb, color = 'Black',linewidth=1, label = 'Real') #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_nb-srbin_std_nb,srbin_mean_nb+srbin_std_nb, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_nb, color = 'DodgerBlue',linewidth=1, label = 'Shuffled') #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_nb-shuffled_std_nb,shuffled_mean_nb+shuffled_std_nb, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_yticklabels(['', '', ''])
                makelegend4(fig,ax1) # makes legend
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,0.3)
                ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['0', '100', '200'])
                ax1.set_yticklabels(['', '', ''])

                ax1 = fig.add_subplot(3,3,7)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*5,speed_beaconed,'-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*5,speed_beaconed-sd_speed_beaconed,speed_beaconed+sd_speed_beaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,50)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Speed (cm/s)', fontsize=18, labelpad = 20)
                ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['0', '100', '200'])

                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,8)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*5,speed_nbeaconed, '-',markersize = 2,color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*5,speed_nbeaconed-sd_speed_nbeaconed,speed_nbeaconed+sd_speed_nbeaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_yticklabels(['', '', ''])
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,50)
                ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['0', '100', '200'])
                ax1.set_xlabel('Location (cm)', fontsize=18, labelpad = 20)
                ax1.set_yticklabels(['', '', ''])
                
            
                
                plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
                fig.savefig('Plots/Supplemental3/Example' + 'Data_Scaled' + '_' + str(mouse) + '_' + str(day) + '.png',  dpi = 200)
                plt.close()
  
  
        if tracklength == 230:
            if stopsdata_p.size > 0: # if theres probe trials, plot 3x3 subplots -> stops per trial, average stops, speed for beaconed, non-beaconed and probe trials
                # make figure       
                fig = plt.figure(figsize = (13.8,12))
                ax = fig.add_subplot(3,3,1) #stops per trial
                ax.set_title('Beaconed trials', fontsize = 22,verticalalignment = 'bottom', style = 'italic')
                ax.axvspan(11.8, 11.8+2.2, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax.axvspan(23-3, 23, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_b[:,0],stopsdata_b[:,2], 'o', color = 'Black', markersize =4.5, label = 'Stop') #plot becaoned trials
                ax.plot(rewardloc_beac,rewardtrial_beac, '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
                #ax.plot(lickloc_beac,licktrial_beac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,23)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                makelegend(fig,ax) # make legend
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_ylabel('Trial number', fontsize=18, labelpad = 20)

                ax = fig.add_subplot(3,3,2) #stops per trial
                ax.set_title('Non-beaconed trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(11.8, 11.8+2.2, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax.axvspan(23-3, 23, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_nb[:,0],stopsdata_nb[:,2], 'o',color = 'Black', markersize = 4.5) #plot becaoned trials
                ax.plot(rewardloc_nbeac,rewardtrial_nbeac, '>', color = 'Red', markersize = 6) #plot becaoned trials
                #ax.plot(lickloc_nbeac,licktrial_nbeac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,23)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
                makelegend(fig,ax) # makes legend 
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])

                ax = fig.add_subplot(3,3,3) #stops per trial
                ax.set_title('Probe trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(11.8, 11.8+2.2, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax.axvspan(23-3, 23, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_p[:,0],stopsdata_p[:,2], 'o',color = 'Black', markersize = 4.5, label = 'Non - beaconed') #plot becaoned trials
                #ax.plot(lickloc_probe,licktrial_probe, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,23)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                ax.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax1 = fig.add_subplot(3,3,4)
                ax1.axvspan(59, 59+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(115-15, 115, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5, srbin_mean_b, color = 'Black',linewidth=1) #plot becaoned trials              
                ax1.fill_between(bins*5,srbin_mean_b-srbin_std_b,srbin_mean_b+srbin_std_b, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_b, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_b-shuffled_std_b,shuffled_mean_b+shuffled_std_b, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0,116)
                ax1.set_ylim(0,0.65)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Stops (cm/trial)', fontsize=18, labelpad = 23)
                ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                
                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,5)
                ax1.axvspan(59, 59+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(115-15, 115, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_nb, color = 'Black',linewidth=1, label = 'Real') #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_nb-srbin_std_nb,srbin_mean_nb+srbin_std_nb, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_nb, color = 'DodgerBlue',linewidth=1, label = 'Shuffled') #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_nb-shuffled_std_nb,shuffled_mean_nb+shuffled_std_nb, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                makelegend2(fig,ax1) # makes legend 
                ax1.set_xlim(0, 116)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])
                
                ax1 = fig.add_subplot(3,3,6)
                ax1.axvspan(59, 59+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(115-15, 115, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_p,color = 'Black',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_p-srbin_std_p,srbin_mean_p+srbin_std_p, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_p, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_p-shuffled_std_p,shuffled_mean_p+shuffled_std_p, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 116)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])

                ax1 = fig.add_subplot(3,3,7)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 30, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(230-30, 230, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_beaconed,'-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_beaconed-sd_speed_beaconed,speed_beaconed+sd_speed_beaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0, 230)
                ax1.set_ylim(0,100)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Speed (cm/s)', fontsize=18, labelpad = 20)
                ax1.locator_params(axis = 'x', nbins=3) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                #ax1.set_xticklabels(['0', '100', '200'])

                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,8)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(230-15, 230, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_nbeaconed, '-',markersize = 2,color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_nbeaconed-sd_speed_nbeaconed,speed_nbeaconed+sd_speed_nbeaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 230)
                ax1.set_ylim(0,100)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                ax1.set_xlabel('Location (cm)', fontsize=18, labelpad = 20)
                
                ax1 = fig.add_subplot(3,3,9)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 30, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(230-30, 230, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_probe, '-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_probe-sd_speed_probe,speed_probe+sd_speed_probe, facecolor = 'Black', alpha = 0.3)                
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 230)
                ax1.set_ylim(0,100)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                                
                plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
                fig.savefig('Plots/Supplemental3/Example' + 'Data_Scaled' + '_' + str(mouse) + '_' + str(day) + '.png',  dpi = 200)
                plt.close()

        if tracklength == 275:
            if stopsdata_p.size > 0: # if theres probe trials, plot 3x3 subplots -> stops per trial, average stops, speed for beaconed, non-beaconed and probe trials
                # make figure       
                fig = plt.figure(figsize = (16.5,12))
                ax = fig.add_subplot(3,3,1) #stops per trial
                ax.set_title('Beaconed trials', fontsize = 22,verticalalignment = 'bottom', style = 'italic')
                ax.axvspan(16.3, 16.3+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(27.5-3, 27.5, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_b[:,0],stopsdata_b[:,2], 'o', color = 'Black', markersize =4.5, label = 'Stop') #plot becaoned trials
                ax.plot(rewardloc_beac,rewardtrial_beac, '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
                #ax.plot(lickloc_beac,licktrial_beac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,27.5)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                makelegend(fig,ax) # make legend
                ax.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_ylabel('Trial number', fontsize=18, labelpad = 20)

                ax = fig.add_subplot(3,3,2) #stops per trial
                ax.set_title('Non-beaconed trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(16.3, 16.3+2.2, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax.axvspan(27.5-3, 27.5, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_nb[:,0],stopsdata_nb[:,2], 'o',color = 'Black', markersize = 4.5) #plot becaoned trials
                ax.plot(rewardloc_nbeac,rewardtrial_nbeac, '>', color = 'Red', markersize = 6) #plot becaoned trials
                #ax.plot(lickloc_nbeac,licktrial_nbeac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,27.5)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
                makelegend(fig,ax) # makes legend 
                ax.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])

                ax = fig.add_subplot(3,3,3) #stops per trial
                ax.set_title('Probe trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(16.3, 16.3+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(27.5-3, 27.5, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_p[:,0],stopsdata_p[:,2], 'o',color = 'Black', markersize = 4.5, label = 'Non - beaconed') #plot becaoned trials
                #ax.plot(lickloc_probe,licktrial_probe, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,27.5)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=8) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax1 = fig.add_subplot(3,3,4)
                ax1.axvspan(81.5, 81.5+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(137.5-15, 137.5, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5, srbin_mean_b, color = 'Black',linewidth=1) #plot becaoned trials              
                ax1.fill_between(bins*5,srbin_mean_b-srbin_std_b,srbin_mean_b+srbin_std_b, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_b, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_b-shuffled_std_b,shuffled_mean_b+shuffled_std_b, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0,137.5)
                ax1.set_ylim(0,0.65)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Stops (cm/trial)', fontsize=18, labelpad = 23)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                
                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,5)
                ax1.axvspan(81.5, 81.5+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(137.5-15, 137.5, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_nb, color = 'Black',linewidth=1, label = 'Real') #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_nb-srbin_std_nb,srbin_mean_nb+srbin_std_nb, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_nb, color = 'DodgerBlue',linewidth=1, label = 'Shuffled') #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_nb-shuffled_std_nb,shuffled_mean_nb+shuffled_std_nb, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                makelegend2(fig,ax1) # makes legend 
                ax1.set_xlim(0, 137.5)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])
                
                ax1 = fig.add_subplot(3,3,6)
                ax1.axvspan(81.5, 81.5+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(137.5-15, 137.5, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_p,color = 'Black',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_p-srbin_std_p,srbin_mean_p+srbin_std_p, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_p, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_p-shuffled_std_p,shuffled_mean_p+shuffled_std_p, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 137.5)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])

                ax1 = fig.add_subplot(3,3,7)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 30, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(137.5-30, 137.5, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_beaconed,'-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_beaconed-sd_speed_beaconed,speed_beaconed+sd_speed_beaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0, 275)
                ax1.set_ylim(0,100)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Speed (cm/s)', fontsize=18, labelpad = 20)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                #ax1.set_xticklabels(['0', '100', '200'])

                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,8)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 30, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(275-30, 275, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_nbeaconed, '-',markersize = 2,color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_nbeaconed-sd_speed_nbeaconed,speed_nbeaconed+sd_speed_nbeaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 275)
                ax1.set_ylim(0,100)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                ax1.set_xlabel('Location (cm)', fontsize=18, labelpad = 20)
                
                ax1 = fig.add_subplot(3,3,9)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 30, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(275-30, 275, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_probe, '-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_probe-sd_speed_probe,speed_probe+sd_speed_probe, facecolor = 'Black', alpha = 0.3)                
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 275)
                ax1.set_ylim(0,100)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                                
                plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
                fig.savefig('Plots/Supplemental3/Example' + 'Data_Scaled' + '_' + str(mouse) + '_' + str(day) + '.png',  dpi = 200)
                plt.close()

        if tracklength == 342:
            if stopsdata_p.size > 0: # if theres probe trials, plot 3x3 subplots -> stops per trial, average stops, speed           for beaconed, non-beaconed and probe trials
                # make figure
                fig = plt.figure(figsize = (20.52,12))
                ax = fig.add_subplot(3,3,1) #stops per trial
                ax.set_title('Beaconed trials', fontsize = 22,verticalalignment = 'bottom', style = 'italic')
                ax.axvspan(23, 23+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(32.45-3, 32.45, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_b[:,0],stopsdata_b[:,2], 'o', color = 'Black', markersize =4.5, label = 'Stop') #plot becaoned trials
                ax.plot(rewardloc_beac,rewardtrial_beac, '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
                #ax.plot(lickloc_beac,licktrial_beac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,32.45)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                makelegend(fig,ax) # make legend
                ax.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_ylabel('Trial number', fontsize=18, labelpad = 20)
                
                ax = fig.add_subplot(3,3,2) #stops per trial
                ax.set_title('Non-beaconed trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(23, 23+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(32.45-3, 32.45, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_nb[:,0],stopsdata_nb[:,2], 'o',color = 'Black', markersize = 4.5) #plot becaoned trials
                ax.plot(rewardloc_nbeac,rewardtrial_nbeac, '>', color = 'Red', markersize = 6) #plot becaoned trials
                #ax.plot(lickloc_nbeac,licktrial_nbeac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,32.45)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
                makelegend(fig,ax) # makes legend
                ax.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax = fig.add_subplot(3,3,3) #stops per trial
                ax.set_title('Probe trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(23, 23+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(32.45-3, 32.45, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_p[:,0],stopsdata_p[:,2], 'o',color = 'Black', markersize = 4.5, label = 'Non - beaconed') #plot becaoned trials
                #ax.plot(lickloc_probe,licktrial_probe, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,32.45)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                ax.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax1 = fig.add_subplot(3,3,4)
                ax1.axvspan(116.25, 116.25+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(162.25-15, 162.25, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5, srbin_mean_b, color = 'Black',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_b-srbin_std_b,srbin_mean_b+srbin_std_b, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_b, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_b-shuffled_std_b,shuffled_mean_b+shuffled_std_b, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0,162.25)
                ax1.set_ylim(0,0.65)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Stops (cm/trial)', fontsize=18, labelpad = 23)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                
                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,5)
                ax1.axvspan(116.25, 116.25+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(162.25-15, 162.25, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_nb, color = 'Black',linewidth=1, label = 'Real') #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_nb-srbin_std_nb,srbin_mean_nb+srbin_std_nb, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_nb, color = 'DodgerBlue',linewidth=1, label = 'Shuffled') #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_nb-shuffled_std_nb,shuffled_mean_nb+shuffled_std_nb, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                makelegend2(fig,ax1) # makes legend
                ax1.set_xlim(0, 162.25)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])
                
                ax1 = fig.add_subplot(3,3,6)
                ax1.axvspan(116.25, 116.25+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(162.25-15, 162.25, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_p,color = 'Black',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_p-srbin_std_p,srbin_mean_p+srbin_std_p, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_p, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_p-shuffled_std_p,shuffled_mean_p+shuffled_std_p, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 162.25)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])
                
                ax1 = fig.add_subplot(3,3,7)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 30, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(342-30, 342, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_beaconed,'-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_beaconed-sd_speed_beaconed,speed_beaconed+sd_speed_beaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0, 342)
                ax1.set_ylim(0,100)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Speed (cm/s)', fontsize=18, labelpad = 20)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                #ax1.set_xticklabels(['0', '100', '200'])
                
                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,8)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 30, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(342-30, 342, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_nbeaconed, '-',markersize = 2,color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_nbeaconed-sd_speed_nbeaconed,speed_nbeaconed+sd_speed_nbeaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 342)
                ax1.set_ylim(0,100)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                ax1.set_xlabel('Location (cm)', fontsize=18, labelpad = 20)
                
                ax1 = fig.add_subplot(3,3,9)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 30, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(342-30, 342, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_probe, '-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_probe-sd_speed_probe,speed_probe+sd_speed_probe, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 342)
                ax1.set_ylim(0,100)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                
                plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
                fig.savefig('Plots/Supplemental3/Example' + 'Data_Scaled' + '_' + str(mouse) + '_' + str(day) + '.png',  dpi = 200)
                plt.close()

        if tracklength == 458:
            if stopsdata_p.size > 0: # if theres probe trials, plot 3x3 subplots -> stops per trial, average stops, speed           for beaconed, non-beaconed and probe trials
                # make figure
                fig = plt.figure(figsize = (27.48,12))
                ax = fig.add_subplot(3,3,1) #stops per trial
                ax.set_title('Beaconed trials', fontsize = 22,verticalalignment = 'bottom', style = 'italic')
                ax.axvspan(33.1, 33.1+2.2, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax.axvspan(45.8-3, 45.8, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_b[:,0],stopsdata_b[:,2], 'o', color = 'Black', markersize =4.5, label = 'Stop') #plot becaoned trials
                ax.plot(rewardloc_beac,rewardtrial_beac, '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
                #ax.plot(lickloc_beac,licktrial_beac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,45.8)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                makelegend(fig,ax) # make legend
                ax.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_ylabel('Trial number', fontsize=18, labelpad = 20)
                
                ax = fig.add_subplot(3,3,2) #stops per trial
                ax.set_title('Non-beaconed trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(33.1, 33.1+2.2, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax.axvspan(45.8-3, 45.8, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_nb[:,0],stopsdata_nb[:,2], 'o',color = 'Black', markersize = 4.5) #plot becaoned trials
                ax.plot(rewardloc_nbeac,rewardtrial_nbeac, '>', color = 'Red', markersize = 6) #plot becaoned trials
                #ax.plot(lickloc_nbeac,licktrial_nbeac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,45.8)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
                makelegend(fig,ax) # makes legend
                ax.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax = fig.add_subplot(3,3,3) #stops per trial
                ax.set_title('Probe trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(33.1, 33.1+2.2, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax.axvspan(45.8-3, 45.8, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_p[:,0],stopsdata_p[:,2], 'o',color = 'Black', markersize = 4.5, label = 'Non - beaconed') #plot becaoned trials
                #ax.plot(lickloc_probe,licktrial_probe, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,45.8)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                ax.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax1 = fig.add_subplot(3,3,4)
                ax1.axvspan(165.5, 165.5+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(229-15, 229, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5, srbin_mean_b, color = 'Black',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_b-srbin_std_b,srbin_mean_b+srbin_std_b, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_b, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_b-shuffled_std_b,shuffled_mean_b+shuffled_std_b, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0,229)
                ax1.set_ylim(0,0.65)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Stops (cm/trial)', fontsize=18, labelpad = 23)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                
                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,5)
                ax1.axvspan(165.5, 165.5+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(229-15, 229, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_nb, color = 'Black',linewidth=1, label = 'Real') #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_nb-srbin_std_nb,srbin_mean_nb+srbin_std_nb, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_nb, color = 'DodgerBlue',linewidth=1, label = 'Shuffled') #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_nb-shuffled_std_nb,shuffled_mean_nb+shuffled_std_nb, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                makelegend2(fig,ax1) # makes legend
                ax1.set_xlim(0, 229)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])
                
                ax1 = fig.add_subplot(3,3,6)
                ax1.axvspan(165.5, 165.5+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(229-15, 229, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_p,color = 'Black',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_p-srbin_std_p,srbin_mean_p+srbin_std_p, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_p, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_p-shuffled_std_p,shuffled_mean_p+shuffled_std_p, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 229)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])
                
                ax1 = fig.add_subplot(3,3,7)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 30, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(458-30, 458, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_beaconed,'-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_beaconed-sd_speed_beaconed,speed_beaconed+sd_speed_beaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0, 458)
                ax1.set_ylim(0,100)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Speed (cm/s)', fontsize=18, labelpad = 20)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                #ax1.set_xticklabels(['0', '100', '200'])
                
                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,8)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 30, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(458-30, 458, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_nbeaconed, '-',markersize = 2,color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_nbeaconed-sd_speed_nbeaconed,speed_nbeaconed+sd_speed_nbeaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 458)
                ax1.set_ylim(0,100)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                ax1.set_xlabel('Location (cm)', fontsize=18, labelpad = 20)
                
                ax1 = fig.add_subplot(3,3,9)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 30, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(458-30, 458, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_probe, '-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_probe-sd_speed_probe,speed_probe+sd_speed_probe, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 458)
                ax1.set_ylim(0,100)
                ax1.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                
                plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
                fig.savefig('Plots/Supplemental3/Example' + 'Data_Scaled' + '_' + str(mouse) + '_' + str(day) + '.png',  dpi = 200)
                plt.close()

        if tracklength == 591:
            if stopsdata_p.size > 0: # if theres probe trials, plot 3x3 subplots -> stops per trial, average stops, speed           for beaconed, non-beaconed and probe trials
                # make figure
                fig = plt.figure(figsize = (35.34,12))
                ax = fig.add_subplot(3,3,1) #stops per trial
                ax.set_title('Beaconed trials', fontsize = 22,verticalalignment = 'bottom', style = 'italic')
                ax.axvspan(48.1, 48.1+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(59.1-3, 59.1, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_b[:,0],stopsdata_b[:,2], 'o', color = 'Black', markersize =4.5, label = 'Stop') #plot becaoned trials
                ax.plot(rewardloc_beac,rewardtrial_beac, '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
                #ax.plot(lickloc_beac,licktrial_beac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,59.1)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                makelegend(fig,ax) # make legend
                ax.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_ylabel('Trial number', fontsize=18, labelpad = 20)
                
                ax = fig.add_subplot(3,3,2) #stops per trial
                ax.set_title('Non-beaconed trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(48.1, 48.1+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(59.1-3, 59.1, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_nb[:,0],stopsdata_nb[:,2], 'o',color = 'Black', markersize = 4.5) #plot becaoned trials
                ax.plot(rewardloc_nbeac,rewardtrial_nbeac, '>', color = 'Red', markersize = 6) #plot becaoned trials
                #ax.plot(lickloc_nbeac,licktrial_nbeac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,59.1)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
                makelegend(fig,ax) # makes legend
                ax.locator_params(axis = 'x', nbins=5) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax = fig.add_subplot(3,3,3) #stops per trial
                ax.set_title('Probe trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(48.1, 48.1+2.2, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax.axvspan(59.1-3, 59.1, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_p[:,0],stopsdata_p[:,2], 'o',color = 'Black', markersize = 4.5, label = 'Non - beaconed') #plot becaoned trials
                #ax.plot(lickloc_probe,licktrial_probe, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,59.1)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                ax.locator_params(axis = 'x', nbins=6) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=8) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax1 = fig.add_subplot(3,3,4)
                ax1.axvspan(240.5, 240.5+12, facecolor='g', alpha=0.2, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.1, hatch = '/', linewidth =0) # black box
                ax1.axvspan(295.5-15, 295.5, facecolor='k', alpha=0.1, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5, srbin_mean_b, color = 'Black',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_b-srbin_std_b,srbin_mean_b+srbin_std_b, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_b, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_b-shuffled_std_b,shuffled_mean_b+shuffled_std_b, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0,295.5)
                ax1.set_ylim(0,0.65)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Stops (cm/trial)', fontsize=18, labelpad = 23)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                
                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,5)
                ax1.axvspan(240.5, 240.5+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(295.5-15, 295.5, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_nb, color = 'Black',linewidth=1, label = 'Real') #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_nb-srbin_std_nb,srbin_mean_nb+srbin_std_nb, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_nb, color = 'DodgerBlue',linewidth=1, label = 'Shuffled') #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_nb-shuffled_std_nb,shuffled_mean_nb+shuffled_std_nb, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                makelegend2(fig,ax1) # makes legend
                ax1.set_xlim(0, 295.5)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])
                
                ax1 = fig.add_subplot(3,3,6)
                ax1.axvspan(240.5, 240.5+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(295.5-15, 295.5, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_p,color = 'Black',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_p-srbin_std_p,srbin_mean_p+srbin_std_p, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_p, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_p-shuffled_std_p,shuffled_mean_p+shuffled_std_p, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 295.5)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])
                
                ax1 = fig.add_subplot(3,3,7)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 30, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(591-30, 591, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_beaconed,'-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_beaconed-sd_speed_beaconed,speed_beaconed+sd_speed_beaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0, 591)
                ax1.set_ylim(0,100)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Speed (cm/s)', fontsize=18, labelpad = 20)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                #ax1.set_xticklabels(['0', '100', '200'])
                
                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,8)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 30, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(591-30, 591, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_nbeaconed, '-',markersize = 2,color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_nbeaconed-sd_speed_nbeaconed,speed_nbeaconed+sd_speed_nbeaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 591)
                ax1.set_ylim(0,100)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                ax1.set_xlabel('Location (cm)', fontsize=18, labelpad = 20)
                
                ax1 = fig.add_subplot(3,3,9)
                ax1.axvspan(rewardzonestart*10,(rewardzonestart*10)+22, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 30, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(591-30, 591, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*10,speed_probe, '-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*10,speed_probe-sd_speed_probe,speed_probe+sd_speed_probe, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 591)
                ax1.set_ylim(0,100)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                
                plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
                fig.savefig('Plots/Supplemental3/Example' + 'Data_Scaled' + '_' + str(mouse) + '_' + str(day) + '.png',  dpi = 200)
                plt.close()
                
        if tracklength == 245:
            if stopsdata_p.size > 0: # if theres probe trials, plot 3x3 subplots -> stops per trial, average stops, speed for beaconed, non-beaconed and probe trials
                # make figure       
                fig = plt.figure(figsize = (12,12))
                ax = fig.add_subplot(3,3,1) #stops per trial
                ax.set_title('Beaconed trials', fontsize = 22,verticalalignment = 'bottom', style = 'italic')
                ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax.axvspan(20-3, 20, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_b[:,0],stopsdata_b[:,2], 'o', color = 'Black', markersize =4.5, label = 'Stop') #plot becaoned trials
                ax.plot(rewardloc_beac,rewardtrial_beac, '>', color = 'Red', markersize = 6, label = 'Reward') #plot becaoned trials
                #ax.plot(lickloc_beac,licktrial_beac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,20)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                makelegend(fig,ax) # make legend
                ax.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_ylabel('Trial number', fontsize=18, labelpad = 20)

                ax = fig.add_subplot(3,3,2) #stops per trial
                ax.set_title('Non-beaconed trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax.axvspan(20-3, 20, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_nb[:,0],stopsdata_nb[:,2], 'o',color = 'Black', markersize = 4.5) #plot becaoned trials
                ax.plot(rewardloc_nbeac,rewardtrial_nbeac, '>', color = 'Red', markersize = 6) #plot becaoned trials
                #ax.plot(lickloc_nbeac,licktrial_nbeac, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,20)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # re;moves top and right spines
                makelegend(fig,ax) # makes legend 
                ax.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])

                ax = fig.add_subplot(3,3,3) #stops per trial
                ax.set_title('Gain trials', fontsize = 22, style = 'italic',verticalalignment = 'bottom')
                ax.axvspan(8.8, 8.8+2.2, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(15, 15+2.2, facecolor='b', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax.axvspan(0, 3, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax.axvspan(26.5-3, 26.5, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax.axvline(0, linewidth = 6, color = 'black') # bold line on the y axis
                ax.axhline(0, linewidth = 6, color = 'black') # bold line on the x axis
                ax.plot(stopsdata_p[:,0],stopsdata_p[:,2], 'o',color = 'Black', markersize = 4.5, label = 'Non - beaconed') #plot becaoned trials
                #ax.plot(lickloc_probe,licktrial_probe, '|', color = 'Red', markersize = 5) #plot becaoned trials
                ax.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax.set_xlim(0,26.5)
                ax.set_ylim(0,trialno+0.5)
                adjust_spines(ax, ['left','bottom']) # removes top and right spines
                ax.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax.set_xticklabels(['', '', ''])
                ax.set_yticklabels(['', '', ''])
                
                ax1 = fig.add_subplot(3,3,4)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5, srbin_mean_b, color = 'Black',linewidth=1) #plot becaoned trials              
                ax1.fill_between(bins*5,srbin_mean_b-srbin_std_b,srbin_mean_b+srbin_std_b, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_b, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_b-shuffled_std_b,shuffled_mean_b+shuffled_std_b, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,0.65)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Stops (cm/trial)', fontsize=18, labelpad = 23)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                
                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,5)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_nb, color = 'Black',linewidth=1, label = 'Real') #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_nb-srbin_std_nb,srbin_mean_nb+srbin_std_nb, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_nb, color = 'DodgerBlue',linewidth=1, label = 'Shuffled') #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_nb-shuffled_std_nb,shuffled_mean_nb+shuffled_std_nb, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                makelegend2(fig,ax1) # makes legend 
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])
                
                ax1 = fig.add_subplot(3,3,6)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(75, 75+12, facecolor='b', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(132.5-15, 132.5, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(bins*5,srbin_mean_p,color = 'Black',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,srbin_mean_p-srbin_std_p,srbin_mean_p+srbin_std_p, facecolor = 'Black', alpha = 0.3)
                ax1.plot(bins*5, shuffled_mean_p, color = 'DodgerBlue',linewidth=1) #plot becaoned trials
                ax1.fill_between(bins*5,shuffled_mean_p-shuffled_std_p,shuffled_mean_p+shuffled_std_p, facecolor = 'DodgerBlue', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 132.5)
                ax1.set_ylim(0,0.65)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_xticklabels(['', '', ''])
                ax1.set_yticklabels(['', '', '',''])

                ax1 = fig.add_subplot(3,3,7)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*5,speed_beaconed,'-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*5,speed_beaconed-sd_speed_beaconed,speed_beaconed+sd_speed_beaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,120)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_ylabel('Speed (cm/s)', fontsize=18, labelpad = 20)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                #ax1.set_xticklabels(['0', '100', '200'])

                #avg stops histogram - non beaconed
                ax1 = fig.add_subplot(3,3,8)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(100-15, 100, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*5,speed_nbeaconed, '-',markersize = 2,color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*5,speed_nbeaconed-sd_speed_nbeaconed,speed_nbeaconed+sd_speed_nbeaconed, facecolor = 'Black', alpha = 0.3)
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 101)
                ax1.set_ylim(0,120)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                ax1.set_xlabel('Location (cm)', fontsize=18, labelpad = 20)
                
                ax1 = fig.add_subplot(3,3,9)
                ax1.axvspan(44, 44+12, facecolor='g', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(75, 75+12, facecolor='b', alpha=0.3, hatch = '/', linewidth =0) # green box spanning the rewardzone - to mark reward zone
                ax1.axvspan(0, 15, facecolor='k', alpha=0.3, hatch = '/', linewidth =0) # black box
                ax1.axvspan(132.5-15, 132.5, facecolor='k', alpha=0.3, hatch = '/', linewidth =0)# black box
                ax1.axvline(0, linewidth = 6, color = 'black')
                ax1.axhline(0, linewidth = 6, color = 'black')
                ax1.plot(sbins*5,speed_probe, '-',markersize = 2, color = 'Black',linewidth = 1) #plot becaoned trials
                ax1.fill_between(sbins*5,speed_probe-sd_speed_probe,speed_probe+sd_speed_probe, facecolor = 'Black', alpha = 0.3)                
                ax1.tick_params(axis='x', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8,labelsize =18)
                ax1.tick_params(axis='y', pad = 10, top='off', right = 'off', direction = 'out',width = 4, length = 8, labelsize =18)
                adjust_spines(ax1, ['left','bottom'])
                ax1.set_xlim(0, 132.5)
                ax1.set_ylim(0,120)
                ax1.locator_params(axis = 'x', nbins=4) # set number of ticks on x axis
                ax1.locator_params(axis = 'y', nbins=4) # set number of ticks on y axis
                ax1.set_yticklabels(['', '', ''])
                                
                plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.15, left = 0.07, right = 0.82, top = 0.92)
                fig.savefig('Plots/Supplemental3/Example' + 'Data_Scaled' + '_' + str(mouse) + '_' + str(day) + '.png',  dpi = 200)
                plt.close()



