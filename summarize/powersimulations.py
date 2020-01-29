import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
import pandas as pd
from scipy.stats import f
np.random.seed(1)

def data():
    # have this make a sigmoid with random number of correct trials each time its called.

    df = pd.DataFrame()
    iterations = 10
    n_subjects = 30
    trials = 60
    track_lengths = np.array([50, 75, 112.5, 168.75, 253.125])
    coef1 = 5
    coef2 = -0.03

    coef3 = 4
    coef4 = -0.02

    power_condition = []
    power_track_length = []
    power_interaction = []

    for n in range(0, n_subjects):
        n = n + 1
        condition_p = []
        track_length_p = []
        interaction_p = []

        for i in range(iterations):

            # building sim data
            for j in range(n):
                j=j+1
                # consider condition 1 first
                group1_theo = (np.e**(coef1 + (coef2*track_lengths)))/\
                              (np.e**(coef1 + (coef2*track_lengths))+1)

                z1 = coef1 + (coef2*track_lengths)
                pr = 1/(1+np.e**(-z1))
                y = np.random.binomial(trials,pr, len(track_lengths))
                y_percentage1 = (y/trials)*100

                subject_id_long = np.ones(len(y_percentage1))*j
                condition1_long = np.ones(len(y_percentage1)) # assign condition 1 as condition = 1

                subject_df_condition1 = pd.DataFrame({"subject": subject_id_long, "Condition": condition1_long, 'Track_length': track_lengths,'percentage_corr_trials': y_percentage1})

                # now consider condition 2
                group2_theo = (np.e**(coef3 + (coef4*track_lengths)))/ \
                              (np.e**(coef3 + (coef4*track_lengths))+1)

                z2 = coef3 + (coef4*track_lengths)
                pr2 = 1/(1+np.e**(-z2))
                y2 = np.random.binomial(trials,pr2, len(track_lengths))
                y_percentage2 = (y2/trials)*100

                condition2_long = np.ones(len(y_percentage2))*2 # assign condition 2 as condition = 2
                subject_df_condition2 = pd.DataFrame({"subject": subject_id_long, "Condition": condition2_long, 'Track_length': track_lengths,'percentage_corr_trials': y_percentage2})

                # bring each dataframe from each condition for each subject into main dataframe

                if j==1:
                    df = subject_df_condition1
                    df = df.append(subject_df_condition2)
                else:
                    df = df.append(subject_df_condition1)
                    df = df.append(subject_df_condition2)

            #plt.plot(track_lengths, y_percentage)
            #plt.show()

            aov = pg.rm_anova(dv='percentage_corr_trials',within=['Condition', 'Track_length'],subject='subject', data=df, detailed=True)

            condition_p.append(np.nan_to_num(aov[aov.Source=="Condition"]['p-unc'].values[0]))
            track_length_p.append(np.nan_to_num(aov[aov.Source=="Track_length"]['p-unc'].values[0]))
            interaction_p.append(np.nan_to_num(aov[aov.Source=="Condition * Track_length"]['p-unc'].values[0]))

        condition_p = np.array(condition_p)
        track_length_p = np.array(track_length_p)
        interaction_p = np.array(interaction_p)

        power_condition_n = len(condition_p[condition_p<0.05])/iterations
        power_track_length_n = len(track_length_p[track_length_p<0.05])/iterations
        power_interaction_n = len(interaction_p[interaction_p<0.05])/iterations

        power_condition.append(power_condition_n)
        power_track_length.append(power_track_length_n)
        power_interaction.append(power_interaction_n)

    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1) #stops per trial
    ax.set_title('Power analysis', fontsize=20, verticalalignment='bottom', style='italic')  # title

    ax.plot(power_condition,np.arange(n_subjects),   color = 'black', label = 'F = Condition', linewidth = 2)
    ax.plot(power_track_length,np.arange(n_subjects),color = 'red',   label = 'F = Track Length', linewidth = 2)
    ax.plot(power_interaction,np.arange(n_subjects), color = 'blue',  label = 'Interaction', linewidth = 2)

    ax.set_xlim(0,200)
    #ax.set_ylim(0, nb_y_max+0.01)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.6, left = 0.15, right = 0.82, top = 0.85)
    #fig.text(0.5, 0.04, 'Track Position Relative to Goal (cm)', ha='center', fontsize=16)
    #fig.text(0.05, 0.94, Mouse, ha='center', fontsize=16)
    #ax.legend(loc=(0.99, 0.5))
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
