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

    iterations = 500
    trials = 60
    track_lengths = np.array([50, 75, 112.5, 168.75, 253.125])
    coef1 = 5
    coef2 = -0.05
    coef3 = 4
    coef4 = -0.02
    n_conditions = 2

    n_subjects = np.array([2,4,5,10,20,30,40])
    coef3s =  np.linspace(1, 10, 5)
    coef4s = np.linspace(-0.01, -0.1, 5)

    parameters_powers_conditions = np.zeros((len(n_subjects), len(coef3s), len(coef4s)))
    parameters_powers_track_length = np.zeros((len(n_subjects), len(coef3s), len(coef4s)))
    parameters_powers_interaction = np.zeros((len(n_subjects), len(coef3s), len(coef4s)))

    # consider condition 1 first
    group1_theo = (np.e**(coef1 + (coef2*track_lengths)))/ \
                  (np.e**(coef1 + (coef2*track_lengths))+1)

    z1 = coef1 + (coef2*track_lengths)
    pr = 1/(1+np.e**(-z1))

    # now consider condition 2
    group2_theo = (np.e**(coef3 + (coef4*track_lengths)))/ \
                  (np.e**(coef3 + (coef4*track_lengths))+1)

    z2 = coef3 + (coef4*track_lengths)
    pr2 = 1/(1+np.e**(-z2))


    for counter3, coef3 in enumerate(coef3s):
        for counter4, coef4 in enumerate(coef4s):

            z1 = coef1 + (coef2*track_lengths)
            pr = 1/(1+np.e**(-z1))

            z2 = coef3 + (coef4*track_lengths)
            pr2 = 1/(1+np.e**(-z2))


            for n_counter, n in enumerate(n_subjects):
                condition_p = []
                track_length_p = []
                interaction_p = []

                subject_id_long = np.tile(np.transpose(np.tile(np.linspace(1,n,n), (len(track_lengths),1))).flatten(), n_conditions)
                conditions_long = np.append(np.ones(len(track_lengths)*n), np.ones(len(track_lengths)*n)*2) # currently hardcoded for only 2 conditions
                track_lengths_long = np.tile(track_lengths, n_conditions*n)

                for i in range(iterations):
                    y_percentage1 = ((np.random.binomial(trials,pr,  (n,len(track_lengths)))/trials)*100).flatten()
                    y_percentage2 = ((np.random.binomial(trials,pr2, (n,len(track_lengths)))/trials)*100).flatten()
                    appended_correct = np.append(y_percentage1, y_percentage2) # currently hardcoded for only 2 conditions

                    df = pd.DataFrame({"subject": subject_id_long, "Condition": conditions_long, 'Track_length': track_lengths_long,'percentage_corr_trials': appended_correct})
                    aov = pg.rm_anova(dv='percentage_corr_trials', within=['Condition', 'Track_length'],subject='subject', data=df, detailed=True)

                    condition_p.append(np.nan_to_num(aov[aov.Source=="Condition"]['p-unc'].values[0]))
                    track_length_p.append(np.nan_to_num(aov[aov.Source=="Track_length"]['p-unc'].values[0]))
                    interaction_p.append(np.nan_to_num(aov[aov.Source=="Condition * Track_length"]['p-unc'].values[0]))

                condition_p = np.array(condition_p)
                track_length_p = np.array(track_length_p)
                interaction_p = np.array(interaction_p)

                power_condition = len(condition_p[condition_p<0.05])/iterations
                power_track_length = len(track_length_p[track_length_p<0.05])/iterations
                power_interaction = len(interaction_p[interaction_p<0.05])/iterations

                parameters_powers_conditions[n_counter, counter3, counter4] = power_condition
                parameters_powers_track_length[n_counter, counter3, counter4] = power_track_length
                parameters_powers_interaction[n_counter, counter3, counter4] = power_interaction

                print("it took ", time.time()-start_time, "for 1 simulated loop to run")
                print("currently on ", str(n), "n subjects, ", str(coef3), "coef3 and ", str(coef4), "coef4")
                start_time = time.time()

    #np.save('/mnt/datastore/Harry/OculusVR/Power_analysis/Harry_figs/conditions_assay.npy', parameters_powers_conditions)
    #np.save('/mnt/datastore/Harry/OculusVR/Power_analysis/Harry_figs/track_length.npy', parameters_powers_track_length)
    #np.save('/mnt/datastore/Harry/OculusVR/Power_analysis/Harry_figs/interaction.npy', parameters_powers_interaction)

    np.save(r'Z:\ActiveProjects\Harry\OculusVR\Power_analysis\Harry_figs\conditions_assay.npy', parameters_powers_conditions)
    np.save(r'Z:\ActiveProjects\Harry\OculusVR\Power_analysis\Harry_figs\track_length.npy', parameters_powers_track_length)
    np.save(r'Z:\ActiveProjects\Harry\OculusVR\Power_analysis\Harry_figs\interaction.npy', parameters_powers_interaction)


    '''
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
    '''

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    data()
    #a = np.load('/mnt/datastore/Harry/OculusVR/Power_analysis/Harry_figs/conditions_assay.npy')
    print("hello there")

if __name__ == '__main__':
    main()
