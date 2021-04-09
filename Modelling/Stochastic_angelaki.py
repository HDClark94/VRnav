import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import inv
from filterpy.common import Q_discrete_white_noise
from Modelling.exponential import *
np.random.seed(66)
from summarize.common import *
from Modelling.params import Parameters
params = Parameters(gain=1, dt=1, n_trials=10, N_iterations=500, velocity_max = 2, target_width=13.8,
                    predict_std_R=0.01, measurement_std_Q =0.1, expon_coef=-35, lambda_coef =2, k=0.01)

def run_Stochastic_model(params, save_path):

    gain = params.gain
    dt = params.dt
    n_trials = params.n_trials
    N_iterations = params.N_iterations
    velocity_arange = np.arange(0, params.velocity_max, 0.01) # to sweep over potential velocity range
    target_width = params.target_width
    predict_std_R = params.predict_std_R
    measurement_std_Q = params.measurement_std_Q
    expon_coef= params.expon_coef
    lambda_coef = params.lambda_coef
    k = params.k # proportionality constant

    gt_motor_commands = np.zeros(N_iterations); gt_motor_commands[0] = 1*gain # simple velocity
    gt_velocities = np.cumsum(gt_motor_commands)
    gt_velocities[gt_velocities < 0] = 0
    gt_motor_commands = np.diff(np.append(np.array([0]), gt_velocities))
    gt_velocities = np.cumsum(gt_motor_commands)
    gt_positions = np.append(0, np.cumsum(gt_velocities)[:-1])

    Optimal_response = np.zeros((n_trials, N_iterations))
    response_positions = np.zeros((n_trials, N_iterations))
    Positions = np.zeros((n_trials, N_iterations))
    Velocities = np.zeros((n_trials, N_iterations))
    Position_Errors = np.zeros((n_trials, N_iterations))
    position_uncertainty_error = np.zeros((n_trials, N_iterations))

    time_steps = np.arange(0, len(gt_motor_commands)*dt, dt)
    for n in range(n_trials):
        print("simulating trial, ", n)

        # Initialization of state matrices
        X = np.array([0, 0.0])     # state vector position and velocity

        # Applying the Kalman Filter
        for i in np.arange(0, N_iterations):
            posterior_velocity_estimate = posterior_velocity(gt_velocities[i], expon_coef, measurement_std_Q, velocity_arange)

            Y = posterior_velocity_estimate
            X = np.array([X[0]+Y[0], Y[0]])  # pos vel

            tmp = gt_positions[i]*gt_positions[i]/X # this step extrapolates the response location for the current gt location
            Optimal_response[n, i], error = position_uncertainty(tmp, lambda_coef=lambda_coef, target_width=target_width, k=k)

            Positions[n, i] = X[0]
            Velocities[n, i] = X[1]

    avg_pos = np.mean(Positions, axis=0)
    std_pos = np.std(Positions, axis=0)

    avg_vel = np.mean(Velocities, axis=0)
    std_vel = np.std(Velocities, axis=0)

    avg_optimal_response = np.mean(Optimal_response, axis=0)
    std_optimal_response = np.std(Optimal_response, axis=0)


    '''
    # plot estimated trajectories
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1) #stops per trial
    for i in range(n_trials):
        plt.plot(time_steps, Positions[i, :], "r", alpha=0.1)
    #plt.fill_between(time_steps,avg_pos-std_pos,avg_pos+std_pos, facecolor='red', alpha=0.3)
    plt.plot(time_steps, avg_pos, "g", label="predicted trajectory")
    plt.plot(time_steps, gt_positions, "k--", label="true trajectory")
    plt.xlabel("Time Step", fontsize=20)
    plt.ylabel("X Position", fontsize=20)
    plt.legend()
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()

    fig.savefig(save_path+"\est_trajectory_e_coef"+num2str(expon_coef)
                +"_l_coef"+num2str(lambda_coef)
                +"_R"+num2str(predict_std_R)
                +"_Q"+num2str(measurement_std_Q))

    '''
    # plot belief target
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) #stops per trial
    for i in range(n_trials):
        plt.plot(Positions[i, :], gt_positions, "r", alpha=0.1)

    plt.plot(avg_pos, gt_positions, "g")
    plt.plot(np.arange(0,400), np.arange(0,400), "k--", label="Unity")
    plt.xlabel("Target", fontsize=20)
    plt.ylabel("Response", fontsize=20)
    plt.xlim(0,max(gt_positions))
    plt.ylim(0,max(gt_positions))
    plt.legend()
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()
    fig.savefig(save_path+"\est_belief_e_coef"+num2str(expon_coef)
                +"_l_coef"+num2str(lambda_coef)
                +"_R"+num2str(predict_std_R)
                +"_Q"+num2str(measurement_std_Q))


    # plot true/estimate
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) #stops per trial
    for i in range(n_trials):
        plt.plot(gt_positions, Positions[i, :], "r", alpha=0.1)

    plt.plot(gt_positions, avg_pos, "g")
    plt.plot(np.arange(0,400), np.arange(0,400), "k--", label="Unity")
    plt.xlabel("Gt Location", fontsize=20)
    plt.ylabel("Perceived Location", fontsize=20)
    plt.xlim(0,max(gt_positions))
    plt.ylim(0,max(gt_positions))
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    plt.show()
    fig.savefig(save_path+"\est_response_e_coef"+num2str(expon_coef)
                +"_l_coef"+num2str(lambda_coef)
                +"_R"+num2str(predict_std_R)
                +"_Q"+num2str(measurement_std_Q))


    # plot optimised response target
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) #stops per trial
    for i in range(n_trials):
        plt.plot(gt_positions, Optimal_response[i, :], "r", alpha=0.1)

    plt.plot(gt_positions, avg_optimal_response, "g")
    plt.plot(np.arange(0,400), np.arange(0,400), "k--", label="Unity")
    plt.xlabel("Target", fontsize=20)
    plt.ylabel("Optimised Response", fontsize=20)
    plt.xlim(0,max(gt_positions))
    plt.ylim(0,max(gt_positions))
    #plt.xlim(0,10)
    #plt.ylim(0,10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    plt.show()
    fig.savefig(save_path+"\est_response_e_coef"+num2str(expon_coef)
                +"_l_coef"+num2str(lambda_coef)
                +"_R"+num2str(predict_std_R)
                +"_Q"+num2str(measurement_std_Q))


    # plot estimated velocities
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) #stops per trial
    for i in range(n_trials):
        plt.plot(time_steps, Velocities[i,:], 'b', alpha=0.3)
    plt.fill_between(time_steps,avg_vel-std_vel,avg_vel+std_vel, facecolor='blue', alpha=0.3)
    plt.plot(time_steps, avg_vel, "b",label="predicted trajectory")
    plt.plot(time_steps, gt_velocities, "k--", label="true trajectory")
    plt.xlabel("Time Step", fontsize=20)
    plt.ylabel("X Velocity", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    plt.show()
    plt.savefig("")
    fig.savefig(save_path+"\est_velo_e_coef"+num2str(expon_coef)
                +"_l_coef"+num2str(lambda_coef)
                +"_R"+num2str(predict_std_R)
                +"_Q"+num2str(measurement_std_Q))


def position_uncertainty_recursive(X, lambda_coef=0, target_width=1, k=1):
    # adjust position estimate x by regressing the position uncertainty
    # into the position estimate by the process in uncertainty_exponent()
    step=1
    pos_estimate = X[0].copy()
    faux_target_distances = np.linspace(pos_estimate-(target_width/2), pos_estimate+(target_width/2), 10)
    error_in_uncertainty = 0

    if pos_estimate>step:
        distances = np.arange(step, 1000, step)
        expected_rewards = np.nan_to_num(np.sum(pdf_vectorised(x=faux_target_distances, mu=distances, sigma=k*(distances**lambda_coef)), axis=1))
        distance_at_peak_expected_reward_x = distances[np.argmax(expected_rewards)]

        error_in_uncertainty = pos_estimate-distance_at_peak_expected_reward_x
        pos_estimate = distance_at_peak_expected_reward_x

        return pos_estimate, error_in_uncertainty
    else:
        return pos_estimate, error_in_uncertainty

def model_stochastic(target_distances, likelihood_width=1, expon_coef=1, lambda_coef=1, k=1):

    velocity_arange = np.arange(0, params.velocity_max, 0.01) # to sweep over potential velocity range
    target_width = params.target_width
    gt_velocities = np.ones(params.N_iterations)
    gt_positions = np.append(0, np.cumsum(gt_velocities)[:-1])

    response_distances = []
    for j in range(len(target_distances)):
        not_passed_target = True
        # Initialization of state matrices
        X = np.array([0, 0.0])     # state vector position and velocity
        i=0 # iterator

        posterior_velocity_estimates = posterior_velocity(gt_velocities[i], expon_coef, likelihood_width, velocity_arange)
        posterior_position_estimates = np.cumsum(posterior_velocity_estimates)

        # Applying the Kalman Filter
        while(not_passed_target):
            #TODO consider using recursion to speed this shit up
            X = np.array([X[0]+posterior_velocity_estimates[i], posterior_velocity_estimates[i]])  # pos vel perceived vector

            tmp = gt_positions[i]*gt_positions[i]/X # this step extrapolates the response location for the current gt location
            Optimal_response, _ = position_uncertainty_recursive(tmp, lambda_coef=lambda_coef, target_width=target_width, k=k)
            i+=1

            if gt_positions[i] >= target_distances[j]:
                response_distances.append(Optimal_response)
                not_passed_target = False

    response_distances = np.array(response_distances)
    return response_distances


def main():
    print("run something here")
    params = Parameters(gain=1, dt=1, n_trials=10, N_iterations=500, velocity_max = 2, target_width=0.1,
                        predict_std_R=0.01, measurement_std_Q =0.1, expon_coef=-35, lambda_coef =2, k=0.01)
    save_path = r"C:\Users\44756\PycharmProjects\VRnavh\Modelling\parameter_search"

    run_Stochastic_model(params, save_path)

if __name__ == '__main__':
    main()