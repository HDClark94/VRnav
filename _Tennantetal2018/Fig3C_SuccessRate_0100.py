# -*- coding: utf-8 -*-
"""

@author: Sarah Tennant

# calculates the % of rewards mice recieve on different track lengths

"""

#IMPORT FUNCTIONS AND PACKAGES
from Functions_CoreFunctions_0100 import adjust_spines,readhdfdata, maketrialarray
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
mice = ['M' + str(int(x)) for x in [1,5]]
tracks = [str(int(x)) for x in np.arange(1,7.1)] # number of tracks in total

# empty arrays to store data
firststop_b = np.zeros((len(mice), 6, 2));firststop_b[:,:,:] = np.nan
firststop_nb = np.zeros((len(mice), 6, 2));firststop_nb[:,:,:] = np.nan
firststop_p = np.zeros((len(mice), 6, 2));firststop_p[:,:,:] = np.nan

# loop thorugh mice and days to get data
for mcount,mouse in enumerate(mice):
    daycount = 0
    for dcount,day in enumerate(days): #load mouse and day
        print ('Processing...',day,mouse)
        #load HDF5 data set for that day and mouse
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        dayb = day.encode('UTF-8')#""""
        mouseb = mouse.encode('UTF-8') #required for importing string from marray in python3
        length = np.max(saraharray[:,1]) # max track length
        trial = np.max(saraharray[:,9]) # max number of trials
        dayss = array[dcount,mcount] # array which has days to analyse (day with highest beaconed first stop on that track)
        trialarray = saraharray[:,9] # makes an array of trial number per row in saraharray
        trialno = np.max(saraharray[:,9]) # total number of trials for that day and mouse
        rewardzonestart = saraharray[1,11] # start of reward zne
        rewardzoneend = saraharray[1,12] # end of reward zone
        trialno = np.max(saraharray[:,9]) # total number of trials for that day and mouse

        # define tracklength according to reward zone (so its exact each time)
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
        
        # calculate percent of rewards (rewards are stored in column 4 as 1/0 (yes/no) if a reward was dispensed. So sum these across trials, divide by trial number and x 100 gives percentage of reward per session)
        rewards_b = (np.sum(dailymouse_b[:,4])/trialids_b.size)*100
        rewards_nb = (np.sum(dailymouse_nb[:,4])/trialids_nb.size)*100
        rewards_p = (np.sum(dailymouse_p[:,4])/trialids_p.size)*100

        try:
            if dayss == 1: # if its a day to analyse, store data
                if dailymouse_b.size > 0:
                    firststop_b[mcount,col,0] = rewards_b
                    firststop_b[mcount,col,1] = rewardzonestart*10 # x 10 to convert virtual units to cm
                if dailymouse_nb.size > 0:
                    firststop_nb[mcount,col,0] = rewards_nb
                    firststop_nb[mcount,col,1] = rewardzonestart*10 # x 10 to convert virtual units to cm
                if dailymouse_p.size > 0:
                    firststop_p[mcount,col,0] = rewards_p
                    firststop_p[mcount,col,1] = rewardzonestart*10 # x 10 to convert virtual units to cm
                daycount +=1
        except IndexError:
            continue



# Load raw data: specify the HDF5 file to read data from
filename = 'Data_Input/Behaviour_DataFiles/Task19_0100.h5'
# Load raw data: txt file that specifies days to analyse with average first stop closest to reward zone in beaconed trials
array = np.loadtxt('Data_Input/Behaviour_SummaryData/Task19_FirstDays.txt',delimiter = '\t')

# specify mouse/mice and day/s to analyse
days = ['Day' + str(int(x)) for x in np.arange(1,46.1)]
mice = ['M' + str(int(x)) for x in [3,6,7,8,9]]# specific day/s

tracks = [str(int(x)) for x in np.arange(1,7.1)] # number of tracks

# empty arrays to store data
firststop_b_2 = np.zeros((len(mice), 6, 2));firststop_b_2[:,:,:] = np.nan
firststop_nb_2 = np.zeros((len(mice), 6, 2));firststop_nb_2[:,:,:] = np.nan
firststop_p_2 = np.zeros((len(mice), 6, 2));firststop_p_2[:,:,:] = np.nan

# loop thorugh mice and days to get data
for mcount,mouse in enumerate(mice):
    daycount = 0
    for dcount,day in enumerate(days): #load mouse and day
        print ('Processing...',day,mouse)
        #load HDF5 data set for that day and mouse
        try:
            saraharray = readhdfdata(filename,day,mouse,'raw_data')
        except KeyError:
            print ('Error, no file')
            continue
        
        #define track length parameters
        length = np.max(saraharray[:,1])
        trial = np.max(saraharray[:,9])
        dayss = array[dcount,mcount]
        rewardzonestart = saraharray[1,11]
        rewardzoneend = saraharray[1,12]
        trialno = np.max(saraharray[:,9]) # total number of trials for that day and mouse
        binmin = np.min(saraharray[:,1]);binmax = np.max(saraharray[:,1]);interval = 0.1 # i.e if track is 20, 0.2 interval gives 100 bins
        bins = np.arange(0,tracklength/10,interval) # add 1e-6 so that last point included - array of bins for location
                
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

        # calculate percent of rewards (rewards are stored in column 4 as 1/0 (yes/no) if a reward was dispensed. So sum these across trials, divide by trial number and x 100 gives percentage of reward per session)
        rewards_b = (np.sum(dailymouse_b[:,4])/trialids_b.size)*100
        rewards_nb = (np.sum(dailymouse_nb[:,4])/trialids_nb.size)*100
        rewards_p = (np.sum(dailymouse_p[:,4])/trialids_p.size)*100
        try:
            if dayss == 1:
                if dailymouse_b.size > 0:
                    firststop_b_2[mcount,col,0] = rewards_b
                    firststop_b_2[mcount,col,1] = rewardzonestart*10 # x 10 to convert virtual units to cm
                if dailymouse_nb.size > 0:
                    firststop_nb_2[mcount,col,0] = rewards_nb
                    firststop_nb_2[mcount,col,1] = rewardzonestart*10 # x 10 to convert virtual units to cm
                if dailymouse_p.size > 0:
                    firststop_p_2[mcount,col,0] = rewards_p
                    firststop_p_2[mcount,col,1] = rewardzonestart*10 # x 10 to convert virtual units to cm
                daycount +=1
        except IndexError:
            continue





# stack experiments then average
tracks_b = np.nanmean(np.vstack((firststop_b,firststop_b_2)), axis=0)
tracks_nb = np.nanmean(np.vstack((firststop_nb, firststop_nb_2)), axis=0)
tracks_p = np.nanmean(np.vstack((firststop_p, firststop_p_2)), axis=0)
sdtracks_b = np.nanstd(np.vstack((firststop_b,firststop_b_2)), axis=0)/math.sqrt(6)
sdtracks_nb = np.nanstd(np.vstack((firststop_nb, firststop_nb_2)), axis=0)/math.sqrt(6)
sdtracks_p = np.nanstd(np.vstack((firststop_p, firststop_p_2)), axis=0 )/math.sqrt(6)


# plot graphs
n_groups = np.arange(7)
bar_width = 0.4
width = 0.4

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)
ax.plot(firststop_b[0,:,1],firststop_b[0,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
#ax.plot(firststop_nb[0,:,1],firststop_nb[0,:,0], '-', color = 'blue',markersize = 10, alpha =0.5)
ax.plot(firststop_p[0,:,1],firststop_p[0,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop_b[1,:,1],firststop_b[1,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
#ax.plot(firststop_nb[1,:,1],firststop_nb[1,:,0], '-', color = 'blue',markersize = 10, alpha =0.5)
ax.plot(firststop_p[1,:,1],firststop_p[1,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop_b_2[0,:,1],firststop_b_2[0,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
#ax.plot(firststop_nb_2[0,:,1],firststop_nb_2[0,:,0], '-', color = 'blue',markersize = 10, alpha =0.5)
ax.plot(firststop_p_2[0,:,1],firststop_p_2[0,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop_b_2[1,:,1],firststop_b_2[1,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
#ax.plot(firststop_nb_2[1,:,1],firststop_nb_2[1,:,0], '-', color = 'blue',markersize = 10, alpha =0.5)
ax.plot(firststop_p_2[1,:,1],firststop_p_2[1,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop_b_2[2,:,1],firststop_b_2[2,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
#ax.plot(firststop_nb_2[2,:,1],firststop_nb_2[2,:,0], '-', color = 'blue',markersize = 10, alpha =0.5)
ax.plot(firststop_p_2[2,:,1],firststop_p_2[2,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop_b_2[3,:,1],firststop_b_2[3,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
#ax.plot(firststop_nb_2[3,:,1],firststop_nb_2[3,:,0], '-', color = 'blue',markersize = 10, alpha =0.5)
ax.plot(firststop_p_2[3,:,1],firststop_p_2[3,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.plot(firststop_b_2[4,:,1],firststop_b_2[4,:,0],'-', color = 'blue', markersize = 10, alpha =0.5)
#ax.plot(firststop_nb_2[4,:,1],firststop_nb_2[4,:,0], '-', color = 'blue',markersize = 10, alpha =0.5)
ax.plot(firststop_p_2[4,:,1],firststop_p_2[4,:,0], '-', color = 'red',markersize = 10, alpha =0.5)

ax.errorbar(tracks_b[:,1],tracks_b[:,0], sdtracks_b[:,0], fmt = '^', color = 'blue', capsize = 6, markersize = 20,elinewidth = 3, capthick = 3)
ax.errorbar(tracks_p[:,1],tracks_p[:,0],sdtracks_p[:,0], fmt = 's', color = 'red', capsize = 6, markersize = 15,elinewidth = 3, capthick = 3)
adjust_spines(ax, ['left','bottom'])
ax.tick_params(axis='x', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =20)
ax.tick_params(axis='y', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 8, width = 2, labelsize =18)
ax.set_ylabel('Rewards (%)', fontsize=20, labelpad = 18)
ax.axhline(-5,linewidth=3, color="black")
ax.axvline(50,linewidth=3, color="black")
ax.set_ylim(-5,110)
plt.locator_params(axis = 'y', nbins  = 4)
plt.locator_params(axis = 'x', nbins  = 6)

ax.set_xlim(50,500)

plt.subplots_adjust(hspace = 0.6, wspace = .5,  bottom = 0.3, left = 0.2, right = 0.8, top = .9)

fig.savefig('Plots/Figure3/Task18_SuccessRate_Tracks_log_0100' +' .png', dpi = 200)
plt.close()



# save data for R

tracks_b = np.vstack((firststop_b[:,:,0],firststop_b_2[:,:,0]))
tracks_nb = np.vstack((firststop_nb[:,:,0], firststop_nb_2[:,:,0]))
tracks_p = np.vstack((firststop_p[:,:,0], firststop_p_2[:,:,0]))

tracks_b = tracks_b.ravel()
tracks_nb = tracks_nb.ravel()
tracks_p = tracks_p.ravel()

tracks = np.array((88,118,163,230,331,481))
tracks = np.hstack((tracks,tracks,tracks,tracks,tracks,tracks,tracks))
m1 = np.array([1,1,1,1,1,1]); m2 = np.array([2,2,2,2,2,2]);m3 = np.array([3,3,3,3,3,3]); m4 = np.array([4,4,4,4,4,4]);m5 = np.array([5,5,5,5,5,5]); m6 = np.array([6,6,6,6,6,6]); m7 = np.array([7,7,7,7,7,7])
mice = np.hstack((m1,m2,m3,m4,m5,m6,m7))
data = np.vstack((mice, tracks, tracks_b, tracks_nb, tracks_p)); data = np.transpose(data)


np.savetxt('Data_Output/Figure3/Figure3_C_0100.csv', data,fmt = '%s', delimiter = ',', header = 'Mouse, Reward zone,Beaconed, Non-beaconed,Probe')




