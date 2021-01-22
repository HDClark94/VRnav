import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import inv
from filterpy.common import Q_discrete_white_noise
from Modelling.exponential import *
np.random.seed(61)
from summarize.common import *
from Modelling.params import Parameters

def kf_predict(X, P, A, Q, B, U):
    X = dot(A, X) + dot(B, U)
    P = dot(A, dot(P, A.T)) + Q
    return(X, P)

def kf_update(X, P, Y, H, R):
    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    X = X + dot(K, (Y-IM)).T
    P = P - dot(K, dot(IS, K.T))
    return (X,P,K)

def model(params, save_path):

    # in practice the process noise Q and measurement noise R might change with each timestep
    # or it can be assumed they are held constant, (by some constant x multiplied by the identity matrix)

    # these work well
    predict_std_R = 0.1 # 0.1
    measurement_std_Q = 0.1 # 0.1
    expon_coef= -28 #-40 for a nice big slow velocity prior when the stds are both 0.1
    lambda_coef = 5
    k = 1 # proportionality constant

    gain = params.gain
    dt = params.dt                       # gain=1, dt=1, n_t=10, n_i=50, R=0.1,Q=0.1, e_coef=-28, l_coef=5,
    n_trials = params.n_trials
    N_iterations = params.N_iterations
    velocity_arange = np.arange(0, params.velocity_max, 0.01) # to sweep over potential velocity range
    target_width = params.target_width
    predict_std_R = params.predict_std_R
    measurement_std_Q = params.measurement_std_Q
    expon_coef= params.expon_coef
    lambda_coef = params.lambda_coef
    k = params.k # proportionality constant

    gt_motor_commands = np.random.randint(-1,2,N_iterations)*gain
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
        P = np.diag((0.01, 0.01))    # we start with a small co variance matrix as we are quite certain of where we are
        A = np.array([[1, dt], [0, 1]]) # state transition matrix   see here: https://share.cocalc.com/share/7557a5ac1c870f1ec8f01271959b16b49df9d087/Kalman-and-Bayesian-Filters-in-Python/08-Designing-Kalman-Filters.ipynb?viewer=share
        Q = Q_discrete_white_noise(dim=2, dt=dt, var=measurement_std_Q)
        U = np.zeros((X.shape[0],1))
        B = np.array([[0.0], [1/gain]])
        U = np.zeros(1)

        # Measurement matrices
        Y = np.array([[X[0]]])
        H = np.array([[0, 1]])
        R = np.eye(Y.shape[0])*predict_std_R

        # Applying the Kalman Filter
        for i in np.arange(0, N_iterations):

            # assign input and measurement with noise
            U[0] = gt_motor_commands[i]+(predict_std_R*np.random.randn(1))

            posterior_velocity_estimate = posterior_velocity(gt_velocities[i], expon_coef, measurement_std_Q, velocity_arange)

            Y = np.array([[posterior_velocity_estimate]])
            (X, P) = kf_predict(X, P, A, Q, B, U)
            (X, P, K) = kf_update(X, P, Y, H, R)
            X = X.flatten()

            tmp = gt_positions[i]*gt_positions[i]/X # this step extrapolates the response location for the current gt location

            Optimal_response[n, i], error = position_uncertainty(tmp, lambda_coef=lambda_coef, target_width=target_width, k=k)
            #Optimal_response[n, i], error = position_uncertainty(X, lambda_coef=lambda_coef, target_width=target_width, k=k)

            Positions[n, i] = X[0]
            Velocities[n, i] = X[1]
            #response_positions[n, i] = gt_positions[] # location at which one should stop
            Position_Errors[n, i] = abs(gt_positions[i] - X[0])
            position_uncertainty_error[n, i] = error

    avg_pos = np.mean(Positions, axis=0)
    std_pos = np.std(Positions, axis=0)

    avg_vel = np.mean(Velocities, axis=0)
    std_vel = np.std(Velocities, axis=0)

    avg_abs_err = np.mean(Position_Errors, axis=0)
    std_abs_err = np.std(Position_Errors, axis=0)

    avg_optimal_response = np.mean(Optimal_response, axis=0)
    std_optimal_response = np.std(Optimal_response, axis=0)

    avg_response_positions = np.mean(response_positions, axis=0)
    std_response_positions = np.std(response_positions, axis=0)

    avg_position_uncertainty_error = np.mean(position_uncertainty_error, axis=0)
    std_position_uncertainty_error = np.std(position_uncertainty_error, axis=0)

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


    '''
    # plot estimation variance against location
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1) #stops per trial
    plt.plot(gt_positions, std_pos, "r")
    plt.xlabel("Location", fontsize=20)
    plt.ylabel("Std of estimated location", fontsize=20)
    plt.legend()
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()
    
    # and against time
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1) #stops per trial
    plt.plot(time_steps, std_pos, "r")
    plt.xlabel("Time Step", fontsize=20)
    plt.ylabel("Std of estimated location", fontsize=20)
    plt.legend()
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()

    # plotting abolute error with std
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1) #stops per trial
    plt.plot(time_steps, avg_abs_err, "r")
    plt.xlabel("Time Step", fontsize=20)
    plt.fill_between(time_steps,avg_abs_err-std_abs_err,avg_abs_err+std_abs_err, facecolor='red', alpha=0.3)
    plt.ylabel("avg location error", fontsize=20)
    plt.legend()
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()
    
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1) #stops per trial
    plt.plot(time_steps, std_abs_err, "r")
    plt.xlabel("Time Step", fontsize=20)
    plt.ylabel("Std of estimated location error", fontsize=20)
    plt.legend()
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()

    # now plotted against position
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1) #stops per trial
    plt.plot(gt_positions, avg_abs_err, "r")
    plt.xlabel("Location", fontsize=20)
    plt.fill_between(gt_positions,avg_abs_err-std_abs_err,avg_abs_err+std_abs_err, facecolor='red', alpha=0.3)
    plt.ylabel("avg location error", fontsize=20)
    plt.legend()
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()
    
    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1) #stops per trial
    plt.plot(gt_positions, std_abs_err, "r")
    plt.xlabel("Location", fontsize=20)
    plt.ylabel("Std of estimated location error", fontsize=20)
    plt.legend()
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()
    '''


def main():
    print("run something here")
    params = Parameters(gain=1, dt=1, n_trials=10, N_iterations=500, velocity_max = 2, target_width=0.1,
                        predict_std_R=0.01, measurement_std_Q =0.1, expon_coef=-35, lambda_coef =2, k=0.01)
    save_path = r"C:\Users\44756\PycharmProjects\VRnavh\Modelling\parameter_search"

    model(params, save_path)

if __name__ == '__main__':
    main()