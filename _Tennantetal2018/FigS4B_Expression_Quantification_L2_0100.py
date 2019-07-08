# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 19:17:55 2015

@author: s0826919

### MAKES HEAT MAPS OF EXPRESSION IN EACH MOUSE AND ALL MICE ###

"""

#IMPORT PACKAGES AND FUNCTIONS
import matplotlib.pyplot as plt
from Functions_CoreFunctions_0100 import adjust_spines
import numpy as np
import math
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import plotly.plotly as py
import plotly.graph_objs as go

from mpl_toolkits.mplot3d import Axes3D

#from scipy import stat

def makelegend3(fig,ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles,labels, loc="center right", bbox_to_anchor=(0.95, 0.7), fontsize = "large", markerscale = 0.5)
    for l in leg.get_lines():l.set_linewidth(2)
    frame  = leg.get_frame()
    frame.set_edgecolor('w')
    frame.set_alpha(0.2)
    
#LOAD DATA
mice = [str(int(x)) for x in np.arange(1,16.1)]

"""
expression = np.loadtxt('Data_Input/ExpressionQuantification/T15_FinalQuantification_0100.txt')

# PLOT ALL MICE

mice = [str(int(x)) for x in np.arange(1,16.1)]

for mcount,mouse in enumerate(mice):
    marraybym = expression[expression[:,0]==mcount+1,:] # get data for each mouse
    rh = marraybym[:,1:4]

    x = rh[:,0]
    y = rh[:,2]
    z = rh[:,1]

    y = np.flipud(y)
    z = np.flipud(z)

    X = np.arange(2.8,3.5,1)
    Y = [1.25,1.75]
    X = [3.05,3.152,3.254,3.357,3.4605,3.5635,3.666]

    fig = plt.figure(figsize = (5,2))
    ax = fig.add_subplot(111)
    
    plt.hist2d(x,y, weights = z, bins =[7,2], cmap='viridis')
    plt.colorbar()
    
    ax.tick_params(axis='x', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 6, width = 2, labelsize =10)
    ax.tick_params(axis='y', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 6, width = 2, labelsize =10)
    ax.axhline(linewidth=3, color="black")
    ax.axvline(linewidth=3, color="black")
    
    plt.locator_params(axis = 'y', nbins  = 5)
    plt.locator_params(axis = 'x', nbins  = 8)
    plt.xticks(X,('3.72','','','','','', '3'))
    plt.yticks(Y,('Ventral','Dorsal'))
    ax.set_xlim(3,3.72)
    ax.set_ylim(1,2)

    plt.subplots_adjust(hspace = 0.25, bottom = 0.23, left = 0.13, right = 0.92, top = 0.87)
    
    fig.savefig('Chapter2/Expression/' + str(mcount+1) + '_heatmap_quantification_RH_0100'+'.png', dpi =200) 
    plt.close() 

    lh = marraybym[:,4:]

    x = lh[:,0]
    y = lh[:,2]
    z = lh[:,1]
    y = np.flipud(y)
    z = np.flipud(z)


    X = np.arange(2.8,3.5,1)
    Y = [1.25,1.75]
    X = [3.05,3.152,3.254,3.357,3.4605,3.5635,3.666]
    
  
    fig = plt.figure(figsize = (5,2))
    ax = fig.add_subplot(111)
    
    plt.hist2d(x,y, weights = z, bins =[7,2], cmap='viridis')
    plt.colorbar()
    
    ax.tick_params(axis='x', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 6, width = 2, labelsize =10)
    ax.tick_params(axis='y', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 6, width = 2, labelsize =10)
    ax.axhline(linewidth=3, color="black")
    ax.axvline(linewidth=3, color="black")
    #ax.set_ylim(0,2)
    #ax.set_ylim(1,1.5)
    plt.locator_params(axis = 'y', nbins  = 5)
    plt.locator_params(axis = 'x', nbins  = 8)
    plt.xticks(X,('3.72','','','','','', '3'))
    plt.yticks(Y,('Ventral','Dorsal'))
    ax.set_xlim(3,3.72)
    ax.set_ylim(1,2)
    plt.subplots_adjust(hspace = 0.25, bottom = 0.23, left = 0.13, right = 0.92, top = 0.87)
        
    fig.savefig('Plots/Supplemental4/' + str(mcount+1) + '_heatmap_quantification_LH_0100'+'.png', dpi =200)
    plt.close() 



"""

# EACH MOUSE ON 1 GRAPH 



## PLOT EXAMPLE MOUSE
expression = np.loadtxt('Data_Input/ExpressionQuantification/T15_FinalQuantification_0200.txt')

rh = expression[:,1:4]

y = rh[:,0]
x = rh[:,2]
z = rh[:,1]

X = np.arange(2.8,3.5,1)
#Y = [2.8,2,2.7]


mouse_groups = np.arange(1,31,1.92)
m_groups = np.zeros((len(mouse_groups)))

for mcount, m in enumerate(mouse_groups):
    #print(x, xcount)
    xy = m/10
    m_groups[mcount] = (m - xy)
    
fig = plt.figure(figsize = (8,3))
ax = fig.add_subplot(111)

plt.hist2d(x,y, weights = z, bins =[32,7], cmap='viridis')
plt.colorbar()

ax.tick_params(axis='x', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 6, width = 2, labelsize =11)
ax.tick_params(axis='y', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 6, width = 2, labelsize =12)
ax.axhline(linewidth=3, color="black")
ax.axvline(linewidth=3, color="black")
#ax.set_xlim(1,33)
#ax.set_ylim(2.8,3.5)
ax.set_xlabel('Mouse', fontsize=12, labelpad = 12)
ax.set_ylabel('Medial-lateral position', fontsize=10, labelpad = 20)

plt.locator_params(axis = 'y', nbins  = 7)
plt.locator_params(axis = 'x', nbins  = 18)
#plt.yticks(Y,('2.8','','3','','3.2','','3.4','','3.6', '', '3.8'))
plt.xticks(mouse_groups+1,('GFP(1)','TeLC(2)','GFP(3)','GFP(4)','TeLC(5)','TeLC(6)','TeLC(7)','TeLC(8)','TeLC(9)','GFP(10)','TeLC(11)','TeLC(12)','TeLC(13)','TeLC(14)','GFP(15)','GFP(16)',))
#ax.set_xlim(2.8,3.8)
plt.xticks(rotation=70)
ax.set_xlim(1,32)

plt.figtext(0.125,0.89, 'D', fontsize = 12, fontstyle = 'italic')
#plt.figtext(0.166,0.89, 'M', fontsize = 6, fontstyle = 'italic')
plt.figtext(0.15,0.89, 'V', fontsize = 12, fontstyle = 'italic')

plt.subplots_adjust(hspace = 1.2, bottom = 0.28, left = 0.12, right = 1.02, top = 0.87)
fig.savefig('Plots/Supplemental4/Heatmap_quantification_RH_0100'+'.png', dpi =200)
plt.close() 




rh = expression[:,4:]

y = rh[:,0]
x = rh[:,2]
z = rh[:,1]

X = np.arange(2.8,3.5,1)
Y = [1.3,2,2.7]

fig = plt.figure(figsize = (8,3))
ax = fig.add_subplot(111)

plt.hist2d(x,y, weights = z, bins =[32,7], cmap='viridis')
plt.colorbar()

ax.tick_params(axis='x', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 6, width = 2, labelsize =11)
ax.tick_params(axis='y', pad = 7, which = 'both', top='off', right = 'off', direction = 'out', length = 6, width = 2, labelsize =12)
ax.axhline(linewidth=3, color="black")
ax.axvline(linewidth=3, color="black")

#ax.set_ylim(2.8,3.5)
ax.set_xlabel('Mouse', fontsize=12, labelpad = 20)
ax.set_ylabel('Medial-lateral position', fontsize=10, labelpad = 20)

plt.locator_params(axis = 'y', nbins  = 7)
plt.locator_params(axis = 'x', nbins  = 18)

#plt.yticks(Y,('2.8','','3','','3.2','','3.4','','3.6', '', '3.8'))
plt.xticks(mouse_groups+1,('GFP(1)','TeLC(2)','GFP(3)','GFP(4)','TeLC(5)','TeLC(6)','TeLC(7)','TeLC(8)','TeLC(9)','GFP(10)','TeLC(11)','TeLC(12)','TeLC(13)','TeLC(14)','GFP(15)','GFP(16)',))
ax.set_xlim(1,32)
plt.xticks(rotation=70)
#ax.set_xlim(2.8,3.8)
plt.figtext(0.125,0.89, 'D', fontsize = 12, fontstyle = 'italic')
#plt.figtext(0.166,0.89, 'M', fontsize = 6.5, fontstyle = 'italic', fontweight = 'bold')
plt.figtext(0.15,0.89, 'V', fontsize = 12, fontstyle = 'italic')

plt.subplots_adjust(hspace = 1.2, bottom = 0.28, left = 0.12, right = 1.02, top = 0.87)
fig.savefig('Plots/Supplemental4/Heatmap_quantification_LH_0100'+'.png', dpi =200)
plt.close() 





