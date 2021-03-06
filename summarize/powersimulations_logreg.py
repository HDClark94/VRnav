import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
import pandas as pd
import time
from scipy.stats import f
np.random.seed(1)

def data():
    # have this make a sigmoid with random number of correct trials each time its called.
    start_time = time.time()

    iterations = 1000
    n_subjects = np.array([2,4,8,10,20,40, 100])
    trials = 60
    track_lengths = np.array([50, 75, 112.5, 168.75, 253.125])
    coef1 = 5
    coef2 = -0.06

    coef3 = 5
    coef4 = -0.062
    n_conditions = 2

    power_condition = []
    power_track_length = []
    power_interaction = []

    # consider condition 1 first
    group1_theo = (np.e**(coef1 + (coef2*track_lengths)))/ \
                  (np.e**(coef1 + (coef2*track_lengths))+1)

    z1 = coef1 + (coef2*track_lengths)
    pr = 1/(1+np.e**(-z1))

    # now consider condition 2
    group2_theo = (np.e**(coef3 + (coef4*track_lengths)))/ \
                  (np.e**(coef3 + (coef4*track_lengths))+1)

    plt.plot(group1_theo, label = "condition1")
    plt.plot(group2_theo, label = "condition2")
    plt.legend()
    plt.show()

    z2 = coef3 + (coef4*track_lengths)
    pr2 = 1/(1+np.e**(-z2))

    for n in n_subjects:
        condition_p = []
        track_length_p = []
        interaction_p = []

        c = np.repeat(np.append(np.ones(len(track_lengths)*n), np.ones(len(track_lengths)*n)*2), trials) # currently hardcoded for only 2 conditions
        #s = np.tile(np.tile(np.transpose(np.tile(np.linspace(1,n,n), (len(track_lengths),1))).flatten(), trials), n_conditions)
        tl = np.tile(track_lengths, n_conditions*n*trials)

        for i in range(iterations):

            y_percentage1 = np.tile(np.random.binomial(1,pr,  (n,len(track_lengths))).flatten(), trials)
            y_percentage2 = np.tile(np.random.binomial(1,pr2, (n,len(track_lengths))).flatten(), trials)
            y = np.append(y_percentage1, y_percentage2) # currently hardcoded for only 2 conditions

            X = np.column_stack((c,tl))
            lom = pg.logistic_regression(X,y)

            condition_p.append(np.nan_to_num(lom[lom.names=="x1"]['pval'].values[0]))
            track_length_p.append(np.nan_to_num(lom[lom.names=="x2"]['pval'].values[0]))
            #interaction_p.append(np.nan_to_num(lom[lom.names==""]['p-unc'].values[0]))

        condition_p = np.array(condition_p)
        track_length_p = np.array(track_length_p)
        #interaction_p = np.array(interaction_p)

        power_condition_n = len(condition_p[condition_p<0.05])/iterations
        power_track_length_n = len(track_length_p[track_length_p<0.05])/iterations
        #power_interaction_n = len(interaction_p[interaction_p<0.05])/iterations

        power_condition.append(power_condition_n)
        power_track_length.append(power_track_length_n)
        #power_interaction.append(power_interaction_n)

        print("it took ", time.time()-start_time, "for 1 simulated loop to run")
        start_time = time.time()

    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1) #stops per trial
    ax.set_title('Power analysis using Multiple Logistic Regression, ', fontsize=20, verticalalignment='bottom', style='italic')  # title

    ax.plot(n_subjects, power_condition, color = 'black', label = 'F = Condition', linewidth = 2)
    ax.plot(n_subjects, power_track_length, color = 'red',   label = 'F = Track Length', linewidth = 2)
    #ax.plot(power_interaction, n_subjects, color = 'blue',  label = 'Interaction', linewidth = 2)

    #ax.set_xlim(0,200)
    #ax.set_ylim(0, nb_y_max+0.01)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.6, left = 0.15, right = 0.82, top = 0.85)
    #fig.text(0.5, 0.04, 'Track Position Relative to Goal (cm)', ha='center', fontsize=16)
    #fig.text(0.05, 0.94, Mouse, ha='center', fontsize=16)
    ax.legend()
    plt.show()
    #fig.savefig(save_path,  dpi=200)
    plt.close()
    #plt.show()


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    data()

if __name__ == '__main__':
    main()
