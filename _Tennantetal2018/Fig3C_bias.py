#IMPORT FUNCTIONS AND PACKAGES
from _Tennantetal2018.Functions_CoreFunctions_0100 import adjust_spines,readhdfdata, maketrialarray
from _Tennantetal2018.Functions_Core_0100 import extractstops, filterstops
from Modelling.no_kalman_model import *
from Modelling.fit import *
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import math
from scipy.stats import uniform

def get_average_stops(daily_stops):
    average_stops=[]
    for trial in np.unique(daily_stops[:,2]):
        stops = daily_stops[daily_stops[:,2]==trial]
        average_stop = (np.mean(stops[:,0])*10)-30 # virtual units to cm
        average_stops.append(average_stop)

    return np.array(average_stops)

def analyse(save_path, trial_type, normalise=False):
    test_x = np.arange(0,500, 20)
    df = pd.DataFrame()

    # Load raw data: specify the HDF5 file to read data from
    filename = r'C:\Users\44756\Downloads\Task18_0100.h5'
    array = np.loadtxt(r'Z:\ActiveProjects\Harry\OculusVR\Data_Input\Behaviour_SummaryData\Task18_FirstDays.txt', delimiter = '\t')

    # specify mouse/mice and day/s to analyse
    days = ['Day' + str(int(x)) for x in np.arange(1,20.1)]
    mice = ['M' + str(int(x)) for x in [1,5]]

    # loop thorugh mice and days to get data
    for mcount,mouse in enumerate(mice):
        for dcount,day in enumerate(days): #load mouse and day
            print ('Processing...',day,mouse)

            try:
                saraharray = readhdfdata(filename,day,mouse,'raw_data')
            except KeyError:
                print ('Error, no file')
                continue
            dayb = day.encode('UTF-8')#""""
            mouseb = mouse.encode('UTF-8') #required for importing string from marray in python3
            length = np.max(saraharray[:,1]) # max track length
            trial = np.max(saraharray[:,9]) # max number of trials
            trialarray = saraharray[:,9] # makes an array of trial number per row in saraharray
            dayss = array[dcount,mcount] # array which has days to analyse (day with highest beaconed first stop on that track)
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

            #extract stops
            stopsdata_b = extractstops(dailymouse_b)
            stopsdata_nb = extractstops(dailymouse_nb)
            stopsdata_p = extractstops(dailymouse_p)

            # filter stops (removes stops 0.5 cm after a stop)
            stopsdata_b = filterstops(stopsdata_b)
            stopsdata_nb = filterstops(stopsdata_nb)
            stopsdata_p = filterstops(stopsdata_p)

            try:
                if dayss == 1: # if its a day to analyse, store data
                    if dailymouse_b.size > 0:
                        print("mouse running on ", (rewardzonestart*10)-30, " for ", len(np.unique(dailymouse_b[:,9])), " beaconed trials")

                        df_tmp = pd.DataFrame()
                        average_stops_b = get_average_stops(stopsdata_b)
                        df_tmp['Mouse_id'] = [mouse] *(len(np.unique(stopsdata_b[:,2])))
                        df_tmp['Integration length'] = [(rewardzonestart*10)-30] *(len(np.unique(stopsdata_b[:,2])))
                        df_tmp['Average Stop'] = average_stops_b
                        df_tmp['Trial_type'] = ['beaconed'] *(len(np.unique(stopsdata_b[:,2])))
                        df = pd.concat([df, df_tmp], ignore_index=True)
                    if dailymouse_nb.size > 0:
                        print("mouse running on ", (rewardzonestart*10)-30, " for ", len(np.unique(dailymouse_nb[:,9])), " non beaconed trials")
                        df_tmp = pd.DataFrame()
                        average_stops_nb = get_average_stops(stopsdata_nb)
                        df_tmp['Mouse_id'] = [mouse] *(len(np.unique(stopsdata_nb[:,2])))
                        df_tmp['Integration length'] = [(rewardzonestart*10)-30] *(len(np.unique(stopsdata_nb[:,2])))
                        df_tmp['Average Stop'] = average_stops_nb
                        df_tmp['Trial_type'] = ['non_beaconed'] *(len(np.unique(stopsdata_nb[:,2])))
                        df = pd.concat([df, df_tmp], ignore_index=True)
                    if dailymouse_p.size > 0:
                        print("mouse running on ", (rewardzonestart*10)-30, " for ", len(np.unique(dailymouse_p[:,9])), " probe trials")
                        df_tmp = pd.DataFrame()
                        average_stops_p = get_average_stops(stopsdata_p)
                        df_tmp['Mouse_id'] = [mouse] *(len(np.unique(stopsdata_p[:,2])))
                        df_tmp['Integration length'] = [(rewardzonestart*10)-30] *(len(np.unique(stopsdata_p[:,2])))
                        df_tmp['Average Stop'] = average_stops_p
                        df_tmp['Trial_type'] = ['probe'] *(len(np.unique(stopsdata_p[:,2])))
                        df = pd.concat([df, df_tmp], ignore_index=True)

            except IndexError:
                print("there was an index error ..............")
                continue

    # Load raw data: specify the HDF5 file to read data from
    filename = r'C:\Users\44756\Downloads\Task19_0100.h5'
    array = np.loadtxt(r'Z:\ActiveProjects\Harry\OculusVR\Data_Input\Behaviour_SummaryData\Task19_FirstDays.txt', delimiter = '\t')

    # specify mouse/mice and day/s to analyse
    days = ['Day' + str(int(x)) for x in np.arange(1,46.1)]
    mice = ['M' + str(int(x)) for x in [3,6,7,8,9]]# specific day/s

    # loop thorugh mice and days to get data
    for mcount,mouse in enumerate(mice):
        for dcount,day in enumerate(days): #load mouse and day
            print ('Processing...',day,mouse)

            try:
                saraharray = readhdfdata(filename,day,mouse,'raw_data')
            except KeyError:
                print ('Error, no file')
                continue

            #define track length parameters
            length = np.max(saraharray[:,1])
            trial = np.max(saraharray[:,9])
            rewardzonestart = saraharray[1,11]
            rewardzoneend = saraharray[1,12]
            dayss = array[dcount,mcount] # array which has days to analyse (day with highest beaconed first stop on that track)
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
                dailymouse_p = np.delete(saraharray, np.where(saraharray[:, 8] != 170), 0)# delete all data not on probe track

            #extract stops
            stopsdata_b = extractstops(dailymouse_b)
            stopsdata_nb = extractstops(dailymouse_nb)
            stopsdata_p = extractstops(dailymouse_p)

            # filter stops (removes stops 0.5 cm after a stop)
            stopsdata_b = filterstops(stopsdata_b)
            stopsdata_nb = filterstops(stopsdata_nb)
            stopsdata_p = filterstops(stopsdata_p)

            try:
                if dayss == 1: # if its a day to analyse, store data
                    if dailymouse_b.size > 0:
                        print("mouse running on ", (rewardzonestart*10)-30, " for ", len(np.unique(dailymouse_b[:,9])), " beaconed trials")

                        df_tmp = pd.DataFrame()
                        average_stops_b = get_average_stops(stopsdata_b)
                        df_tmp['Mouse_id'] = [mouse] *(len(np.unique(stopsdata_b[:,2])))
                        df_tmp['Integration length'] = [(rewardzonestart*10)-30] *(len(np.unique(stopsdata_b[:,2])))
                        df_tmp['Average Stop'] = average_stops_b - 40
                        df_tmp['Trial_type'] = ['beaconed'] *(len(np.unique(stopsdata_b[:,2])))
                        df = pd.concat([df, df_tmp], ignore_index=True)
                    if dailymouse_nb.size > 0:
                        print("mouse running on ", (rewardzonestart*10)-30, " for ", len(np.unique(dailymouse_nb[:,9])), " non beaconed trials")

                        df_tmp = pd.DataFrame()
                        average_stops_nb = get_average_stops(stopsdata_nb)
                        df_tmp['Mouse_id'] = [mouse] *(len(np.unique(stopsdata_nb[:,2])))
                        df_tmp['Integration length'] = [(rewardzonestart*10)-30] *(len(np.unique(stopsdata_nb[:,2])))
                        df_tmp['Average Stop'] = average_stops_nb - 40
                        df_tmp['Trial_type'] = ['non_beaconed'] *(len(np.unique(stopsdata_nb[:,2])))
                        df = pd.concat([df, df_tmp], ignore_index=True)
                    if dailymouse_p.size > 0:
                        print("mouse running on ", (rewardzonestart*10)-30, " for ", len(np.unique(dailymouse_p[:,9])), " probe trials")

                        df_tmp = pd.DataFrame()
                        average_stops_p = get_average_stops(stopsdata_p)
                        df_tmp['Mouse_id'] = [mouse] *(len(np.unique(stopsdata_p[:,2])))
                        df_tmp['Integration length'] = [(rewardzonestart*10)-30] *(len(np.unique(stopsdata_p[:,2])))
                        df_tmp['Average Stop'] = average_stops_p - 40
                        df_tmp['Trial_type'] = ['probe'] *(len(np.unique(stopsdata_p[:,2])))
                        df = pd.concat([df, df_tmp], ignore_index=True)

            except IndexError:
                print("there was an index error ..............")
                continue

    df = df[df["Average Stop"] > 0] # removes negative average stop (these would be before end of first black box)
    df_mean_data = df.groupby(['Trial_type', 'Integration length'])['Average Stop'].mean().reset_index()
    df_mean_mice_data = df.groupby(['Trial_type', 'Integration length', "Mouse_id"])['Average Stop'].mean().reset_index()

    if trial_type == "probe_and_non_beaconed":
        df_mean_data_tmp = df_mean_data[((df_mean_data["Trial_type"] == "probe") | (df_mean_data["Trial_type"] == "non_beaconed"))]
        df_mean_data_tmp = df_mean_data_tmp.groupby(['Integration length']).mean().reset_index()
        df_mean_data_x = np.asarray(df_mean_data_tmp['Integration length'])
        df_mean_data_y = np.asarray(df_mean_data_tmp['Average Stop'])
        df_data_x = np.asarray(df[((df["Trial_type"] == "probe") | (df["Trial_type"] == "non_beaconed"))]['Integration length'])
        df_data_y = np.asarray(df[((df["Trial_type"] == "probe") | (df["Trial_type"] == "non_beaconed"))]['Average Stop'])

        df_mean_mice_data_tmp = df_mean_mice_data[((df_mean_mice_data["Trial_type"] == "probe") | (df_mean_mice_data["Trial_type"] == "non_beaconed"))]
        df_mean_mice_data_x = df_mean_mice_data_tmp["Integration length"]
        df_mean_mice_data_y = df_mean_mice_data_tmp["Average Stop"]

    else:
        df_mean_data_x = np.asarray(df_mean_data[(df_mean_data["Trial_type"] == trial_type)]['Integration length'])
        df_mean_data_y = np.asarray(df_mean_data[(df_mean_data["Trial_type"] == trial_type)]['Average Stop'])
        df_data_x = np.asarray(df[(df["Trial_type"] == trial_type)]['Integration length'])
        df_data_y = np.asarray(df[(df["Trial_type"] == trial_type)]['Average Stop'])

        df_mean_mice_data_tmp = df_mean_mice_data[(df_mean_mice_data["Trial_type"] == trial_type)]
        df_mean_mice_data_x = df_mean_mice_data_tmp["Integration length"]
        df_mean_mice_data_y = df_mean_mice_data_tmp["Average Stop"]

    if normalise:
        df_mean_data_x_beaconed = np.asarray(df_mean_data[(df_mean_data["Trial_type"] == "beaconed")]['Integration length'])
        df_mean_data_y_beaconed = np.asarray(df_mean_data[(df_mean_data["Trial_type"] == "beaconed")]['Average Stop'])

        df_mean_data_diff_beaconed = df_mean_data_x_beaconed - df_mean_data_y_beaconed
        df_mean_data_y = df_mean_data_diff_beaconed + df_mean_data_y
        # calculate some factors based on beacoend

        for i in range(len(df_mean_mice_data_y)):
            df_mean_mice_data_y.iloc[i] = df_mean_data_diff_beaconed[df_mean_data_x_beaconed==df_mean_mice_data_x.iloc[i]] + df_mean_mice_data_y.iloc[i]



    group_model_params = fit_parameters_to_model(df_mean_data_x, df_mean_data_y)
    # plotting model fit
    best_fit_responses = model(test_x, prior_gain=group_model_params[0], lambda_coef=group_model_params[1], k=group_model_params[2])
    # plot optimised response target
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) #stops per trial
    plt.title("All Mice", fontsize="20")
    plt.scatter(df_mean_mice_data_x, df_mean_mice_data_y, color="r", marker="o")
    plt.plot(df_mean_data_x, df_mean_data_y, color="r", marker="v", label="data")
    plt.plot(test_x, best_fit_responses, color="g", label="model")
    #plt.plot(np.arange(0,500), np.arange(0,500), "k--", label="Unity")
    distances = np.arange(0,500)
    reward_zone_size = 30
    ax.fill_between(distances, distances-reward_zone_size/2, distances+reward_zone_size/2, facecolor="k", alpha=0.3)
    plt.xlabel("Target", fontsize=20)
    plt.xlim((0,500))
    plt.ylim((0,500))
    plt.ylabel("Optimal Response", fontsize=20)
    plt.subplots_adjust(left=0.2)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    textstr = '\n'.join((
        r'$\Gamma=%.2f$' % (group_model_params[0], ),
        r'$\lambda=%.2f$' % (group_model_params[1], ),
        r'$\mathrm{k}=%.2f$' % (group_model_params[2], )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
    plt.legend(loc="upper left")
    if normalise:
        plt.savefig(save_path+"/"+trial_type+"_normalised_group_mice_model_fit.png")
    else:
        plt.savefig(save_path+"/"+trial_type+"_group_mice_model_fit.png")
    plt.show()
    plt.close()

    #if normalise:
    #    return

    ppids = np.unique(df_mean_mice_data["Mouse_id"])
    subject_model_params = np.zeros((len(ppids), 3)) # fitting 3 parameters
    for j in range(len(ppids)):
        if trial_type == "probe_and_non_beaconed":
            df_mean_mice_data_tmp = df_mean_mice_data[((df_mean_mice_data["Trial_type"] == "probe") | (df_mean_mice_data["Trial_type"] == "non_beaconed"))]
            df_mean_mice_data_tmp = df_mean_mice_data_tmp.groupby(['Integration length', "Mouse_id"]).mean().reset_index()
            subject_data_mean_x = np.asarray(df_mean_mice_data_tmp[(df_mean_mice_data_tmp["Mouse_id"] == ppids[j])]['Integration length'])
            subject_data_mean_y = np.asarray(df_mean_mice_data_tmp[(df_mean_mice_data_tmp["Mouse_id"] == ppids[j])]['Average Stop'])
            subject_data_x = np.asarray(df[((df["Trial_type"] == "probe") | (df["Trial_type"] == "non_beaconed")) & (df["Mouse_id"] == ppids[j])]['Integration length'])
            subject_data_y = np.asarray(df[((df["Trial_type"] == "probe") | (df["Trial_type"] == "non_beaconed")) & (df["Mouse_id"] == ppids[j])]["Average Stop"])

        else:
            subject_data_mean_x = np.asarray(df_mean_mice_data[(df_mean_mice_data["Trial_type"] == trial_type) & (df_mean_mice_data["Mouse_id"] == ppids[j])]['Integration length'])
            subject_data_mean_y = np.asarray(df_mean_mice_data[(df_mean_mice_data["Trial_type"] == trial_type) & (df_mean_mice_data["Mouse_id"] == ppids[j])]['Average Stop'])
            subject_data_x = np.asarray(df[(df["Trial_type"] == trial_type) & (df["Mouse_id"] == ppids[j])]['Integration length'])
            subject_data_y = np.asarray(df[(df["Trial_type"] == trial_type) & (df["Mouse_id"] == ppids[j])]["Average Stop"])

        if normalise:
            subject_data_mean_x_beaconed = np.asarray(df_mean_mice_data[(df_mean_mice_data["Trial_type"] == "beaconed") & (df_mean_mice_data["Mouse_id"] == ppids[j])]['Integration length'])
            subject_data_mean_y_beaconed = np.asarray(df_mean_mice_data[(df_mean_mice_data["Trial_type"] == "beaconed") & (df_mean_mice_data["Mouse_id"] == ppids[j])]['Average Stop'])

            subject_data_mean_diff_beaconed = subject_data_mean_x_beaconed - subject_data_mean_y_beaconed
            for i in range(len(subject_data_mean_x)):
                subject_data_mean_y[i] = subject_data_mean_diff_beaconed[subject_data_mean_x_beaconed==subject_data_mean_x[i]] + subject_data_mean_y[i]

            for i in range(len(subject_data_x)):
                subject_data_y[i] = subject_data_mean_diff_beaconed[subject_data_mean_x_beaconed==subject_data_x[i]] + subject_data_y[i]


        subject_model_params[j] = fit_parameters_to_model(subject_data_x, subject_data_y)

        # plotting model fit
        best_fit_responses = model(test_x, prior_gain=subject_model_params[j][0], lambda_coef=subject_model_params[j][1], k=subject_model_params[j][2])
        # plot optimised response target
        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(1,1,1) #stops per trial
        plt.title(ppids[j], fontsize="20")
        plt.scatter(subject_data_x, subject_data_y, color="r", marker="o")
        plt.plot(subject_data_mean_x, subject_data_mean_y, "r", label="data")
        plt.plot(test_x, best_fit_responses, "g", label="model")
        distances = np.arange(0,500)
        reward_zone_size = 30
        ax.fill_between(distances, distances-reward_zone_size/2, distances+reward_zone_size/2, facecolor="k", alpha=0.3)
        #plt.plot(np.arange(0,500), np.arange(0,500), "k--", label="Unity")
        plt.xlabel("Target (cm)", fontsize=20)
        plt.xlim((0,500))
        plt.ylim((0,500))
        plt.ylabel("Response (cm)", fontsize=20)
        plt.subplots_adjust(left=0.2)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        textstr = '\n'.join((
            r'$\Gamma=%.2f$' % (subject_model_params[j][0], ),
            r'$\lambda=%.2f$' % (subject_model_params[j][1], ),
            r'$\mathrm{k}=%.2f$' % (subject_model_params[j][2], )))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
        plt.legend(loc="upper left")

        if normalise:
            plt.savefig(save_path+"/"+ppids[j]+"_"+trial_type+"_normalised_group_mice_model_fit.png")
        else:
            plt.savefig(save_path+"/"+ppids[j]+"_"+trial_type+"_model_fit.png")
        plt.show()
        plt.close()

    print("for Gamma, mean = ", np.mean(subject_model_params[:,0]))
    print("for lambda, mean = ", np.mean(subject_model_params[:,1]))
    print("for k, mean = ", np.mean(subject_model_params[:,2]))

    print("for Gamma, std = ", np.std(subject_model_params[:,0]))
    print("for lambda, std = ", np.std(subject_model_params[:,1]))
    print("for k, std = ", np.std(subject_model_params[:,2]))

    print("for Gamma, p=", stats.ttest_1samp(subject_model_params[:,0],1)[1]/2)
    print("for lambda, p=", stats.ttest_1samp(subject_model_params[:,2],1)[1]/2)


def rando_calrissian(save_path):
    test_x = np.arange(0,500, 20)
    df = pd.DataFrame()
    n_trials = 100
    for trial in np.arange(0, n_trials):
        for track_length in [200, 230, 290, 332, 430, 591]:
            # define track length
            if track_length == 200:
                rewardzonestart = 88
            elif track_length == 230:
                rewardzonestart = 118
            if track_length == 290:
                rewardzonestart = 163
            if track_length == 332:
                rewardzonestart = 230.5
            if track_length == 430:
                rewardzonestart = 331
            if track_length == 591:
                rewardzonestart = 481

            df_tmp = pd.DataFrame()
            df_tmp['Integration length'] = [rewardzonestart-30]
            df_tmp['Average Stop'] = np.random.uniform(low=0, high=track_length) -40
            df = pd.concat([df, df_tmp], ignore_index=True)

    df_mean_mice_data_tmp = df.groupby(['Integration length']).mean().reset_index()
    subject_data_mean_x = np.asarray(df_mean_mice_data_tmp['Integration length'])
    subject_data_mean_y = np.asarray(df_mean_mice_data_tmp['Average Stop'])
    subject_data_x = np.asarray(df['Integration length'])
    subject_data_y = np.asarray(df["Average Stop"])
    subject_model_params = fit_parameters_to_model(subject_data_x, subject_data_y)

    # plotting model fit
    best_fit_responses = model(test_x, prior_gain=subject_model_params[0],
                               lambda_coef=subject_model_params[1],
                               k=subject_model_params[2])
    # plot optimised response target
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) #stops per trial
    plt.title("Rando Calrissian", fontsize="20")
    plt.scatter(subject_data_x, subject_data_y, color="r", marker="o")
    plt.plot(subject_data_mean_x, subject_data_mean_y, "r", label="data")
    plt.plot(test_x, best_fit_responses, "g", label="model")
    #plt.plot(np.arange(0,500), np.arange(0,500), "k--", label="Unity")
    distances = np.arange(0,500)
    reward_zone_size = 30
    ax.fill_between(distances, distances-reward_zone_size/2, distances+reward_zone_size/2, facecolor="k", alpha=0.3)
    plt.xlabel("Target (cm)", fontsize=20)
    plt.xlim((0,500))
    plt.ylim((0,500))
    plt.ylabel("Response (cm)", fontsize=20)
    plt.subplots_adjust(left=0.2)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    textstr = '\n'.join((
        r'$\Gamma=%.2f$' % (subject_model_params[0], ),
        r'$\lambda=%.2f$' % (subject_model_params[1], ),
        r'$\mathrm{k}=%.2f$' % (subject_model_params[2], )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.80, 0.05, textstr, transform=ax.transAxes, fontsize=14, bbox=props)
    plt.legend(loc="upper left")
    plt.savefig(save_path+"/rando_model_fit.png")
    plt.show()
    plt.close()

def main():
    print("run here")
    save_path = r"Z:\ActiveProjects\Harry\OculusVR\Figures\angelaki_model"
    #analyse(save_path, trial_type="beaconed", normalise=True)
    #analyse(save_path, trial_type="non_beaconed", normalise=True)
    #analyse(save_path, trial_type="probe", normalise=True)
    #analyse(save_path, trial_type="probe_and_non_beaconed", normalise=True)

    #analyse(save_path, trial_type="beaconed", normalise=False)
    #analyse(save_path, trial_type="non_beaconed", normalise=False)
    #analyse(save_path, trial_type="probe", normalise=False)
    analyse(save_path, trial_type="probe_and_non_beaconed", normalise=False)

    #rando_calrissian(save_path)
if __name__ == '__main__':
    main()