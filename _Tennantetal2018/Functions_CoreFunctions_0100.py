# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:45:30 2013
Edited on the 27/3/2014 for conversion to python3
Edited: 21st may 2014


@author: Sarah Tennant


"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib import rcParams


# makes an array for bins - finds all logical values within an array for each bin 
# imports 'bins' from core script so function is flexible 

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


def setparameters():

    dark2_colors = brewer2mpl.get_map('Dark2', 'Qualitative', 7).mpl_colors
    
    rcParams['figure.figsize'] = (10, 6)
    rcParams['figure.dpi'] = 150
    rcParams['axes.color_cycle'] = dark2_colors
    rcParams['lines.linewidth'] = 2
    rcParams['axes.facecolor'] = 'white'
    rcParams['font.size'] = 14
    rcParams['patch.edgecolor'] = 'white'
    rcParams['patch.facecolor'] = dark2_colors[0]
    rcParams['font.family'] = 'StixGeneral'



def readhdfdata(filename,day,mouse,dataset):
    fopen = h5py.File(filename, 'r')
    datasetopen  = fopen[day + '/' + mouse  + '/' + dataset]
    openarray = datasetopen[:,:]
    fopen.close()
    return openarray
    
    
def writehdfdata(filename, folder, dataset, narray):
    fopen = h5py.File(filename, 'w')
    datasetx = fopen.create_dataset(folder + '/' + dataset + '_x', (narray.shape), dtype=narray.dtype)
    datasetx[:,] = narray
#    datasety = fopen.create_dataset(folder + '/' + dataset + '_y', (narray.shape), dtype=narray.dtype)
#    datasety[:,] = narray
    fopen.close()


def splitbyreward(saraharray):
    # Set up bins
    interval = 0.134
    
#    bins = np.arange(np.min(saraharray[:,1]),np.max(saraharray[:,1])+1e-6,interval) # add 1e-6 so that last point included
    bins = np.arange(0,20+1e-6,interval)
    
    trialarray = np.zeros((saraharray.shape[0],1))
    trialcount = 0
    for rowcount, row in enumerate(saraharray[1:,:]):
        #saraharray[rowcount, 1] = distance measures
        rowcount+=1
        if saraharray[rowcount, 1]-saraharray[rowcount-1,1]<-15:
            print (rowcount, 'Increase trial')
            trialcount+=1
        trialarray[rowcount] =trialcount
        
    #np.savetxt('trial.txt', trialarray,fmt = '%d')
    
    trialdict = {}
    for trial in np.arange(0,trialcount+1):
        arrayoi = saraharray[trialarray[:,0]==trial,:]
        trialdict[trial] = np.sum(arrayoi[:,3],axis=0)
         
        
    rewardarray  = np.array([trialdict[i] for i in trialarray[:,0]])
    
        
    # Collect mean data for each bin
    results = np.zeros((bins.shape[0],3))
    count = 0
    for bcount,b in enumerate(bins):
        arraymin = saraharray[:,1]>=b # lowest value in bin
        arraymax = saraharray[:,1]<b+interval # highest value in bin
        arraydist = np.logical_and(arraymin,arraymax) #get all rows that satisfy being within bin
    
        # Get first half of trial
        reward = rewardarray ==1 # first half
        arraylogicalt = np.logical_and(arraydist,reward)
        arrayoi = np.mean(saraharray[arraylogicalt,2],axis = 0) # average the speed (column 2) in these arrays
        count+=saraharray[arraylogicalt,2].shape[0]
        
        # Get second half of trial 
        fail = rewardarray ==0 # second half
        arraylogicalt2 = np.logical_and(arraydist,fail)
        arrayoit2 = np.mean(saraharray[arraylogicalt2,2],axis = 0) # average the speed (column 2)
        count+=saraharray[arraylogicalt2,2].shape[0]
        
        results[bcount,:] = np.array([b,arrayoi, arrayoit2]) #Points are plotted at the beginning of the bin
    print (count, saraharray.shape[0])    # these should be the same
    
    np.savetxt('behaviour_out_reward.txt',results,fmt = '%.5f',delimiter = '\t')
        
    return results, bins.shape[0]


def makelegend(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(0.95, 0.7), fontsize = "large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend1(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="center right", bbox_to_anchor=(1.01, 0.4), fontsize = "large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend2(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="center right", fontsize = "large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

def makelegend3(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="bottom right", bbox_to_anchor=(0.95, 0.7), fontsize = "large")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)

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
        
        

def makebinarray(tarray, bins):

    interval = 0.1
    #    bins = np.arange(np.min(saraharray[:,1]),np.max(saraharray[:,1])+1e-6,interval) # add 1e-6 so that last point included
    #    bins = np.arange(0,20+1e-6,interval) # add 1e-6 so that last point included
    binarray = np.zeros((tarray.shape[0], 1))

    # Collect mean data for each bin
    for bcount,b in enumerate(bins):
        binmin = tarray[:,1]>=b # lowest value in bin
        binmax = tarray[:,1]<b+interval # highest value in bin
        arraylogical = np.logical_and(binmin,binmax) #get all rows that satisfy being within bin
        binarray[arraylogical, 0] = b
        
    return binarray


def makebins(saraharray):
        binmin = np.min(saraharray[:,1])
        binmax = np.max(saraharray[:,1])
        interval = 0.1 #makes 200 bins which is lenght of track in cm
        bins = np.arange(binmin,binmax+1e-6,interval) # add 1e-6 so that last point included
    
        return bins

def maketrialarray(saraharray):
    trialarray = np.zeros((saraharray.shape[0],1))
    trialnumber = 1
    for rowcount, row in enumerate(saraharray[1:,:]):
        if saraharray[rowcount, 1]-saraharray[rowcount-1,1]<-15:
             trialnumber+=1
        trialarray[rowcount] = trialnumber # creates array for trial number in each row of saraharray
        #print trialcount, trialarray
        rowcount+=1
    return trialarray[:,0]



def round_down(num, divisor):
	return num - (num%divisor)





def makelegend_small(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="baseline right", bbox_to_anchor=(0.95, 0.9), fontsize = "x-small")
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)
        