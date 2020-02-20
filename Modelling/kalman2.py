import matplotlib.pyplot as plt
from numpy import *
import numpy as np
from numpy.linalg import inv

from numpy import dot, sum, tile, linalg, log, pi, exp
from numpy.linalg import inv, det
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

np.random.seed(64)

# equations taken from https://arxiv.org/ftp/arxiv/papers/1204/1204.0375.pdf

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


def example():
    # in practice the process noise Q and measurement noise R might change with each timestep
    # or it can be assumed they are held constant, (by some constant x multiplied by the identity matrix)
    gain = 1
    dt = 1
    n_trials = 100
    N_time_steps = 200
    n_people = 10
    predict_std_R_mean = 0.1
    measurement_std_Q_mean = 1
    inter_person_std = 0.1

    gt_motor_commands = np.random.randint(-1,2,N_time_steps)
    gt_velocities = np.cumsum(gt_motor_commands)
    gt_velocities[gt_velocities < 0] = 0
    gt_positions = np.cumsum(gt_velocities)

    Positions= np.zeros((n_trials, N_time_steps))
    Velocities= np.zeros((n_trials, N_time_steps))

    Biases = []
    Variances= []

    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1)

    for k in range(n_people):

        # every person has slightly different Q and R parameters
        predict_std_R = np.random.normal(predict_std_R_mean, inter_person_std)
        measurement_std_Q = np.random.normal(measurement_std_Q_mean, inter_person_std)

        for n in range(n_trials):

            # Initialization of state matrices
            X = array([0.0, 0.0])     # state vector position and velocity
            P = diag((0.01, 0.01))    # we start with a small co variance matrix as we are quite certain of where we are
            A = array([[1, dt], [0, 1]])    # state transition matrix
            Q = Q_discrete_white_noise(dim=2, dt=dt, var=measurement_std_Q)
            B = eye(X.shape[0])
            U = zeros((X.shape[0],1))
            B = np.array([[0.0], [1/gain]])
            U = zeros(1)

            # Measurement matrices
            Y = array([[X[0]]])
            H = array([[0, 1]])
            R = eye(Y.shape[0])*predict_std_R

            # Applying the Kalman Filter
            for i in arange(0, N_time_steps):

                # assign input and measurement with noise
                U[0] = gt_motor_commands[i]+(predict_std_R*random.randn(1))
                Y = array([[gt_velocities[i] + (measurement_std_Q*np.random.randn(1)[0])]])

                (X, P) = kf_predict(X, P, A, Q, B, U)
                (X, P, K) = kf_update(X, P, Y, H, R)

                # housekeeping
                X=X.flatten()

                Positions[n, i] = X[0]
                Velocities[n, i] = X[1]

        avg_pos = np.mean(Positions, axis=0)
        var_pos = np.var(Positions, axis=0)

        avg_vel = np.mean(Velocities, axis=0)
        var_vel = np.var(Velocities, axis=0)

        plt.plot(gt_positions, var_pos, "b", alpha=0.7)

        Variances.append(var_pos)
        Biases.append(avg_pos)

    Biases = np.array(Biases)
    Variances = np.array(Variances)

    mean_Bias = np.mean(Biases,axis=0)
    std_Bias = np.std(Biases,axis=0)

    mean_Variances = np.mean(Variances, axis=0)
    std_Variances = np.std(Variances, axis=0)


    plt.plot(gt_positions, mean_Variances, "b")
    plt.fill_between(gt_positions, mean_Variances-std_Variances, mean_Variances+std_Variances, facecolor='blue', alpha=0.3)
    plt.xlabel("Location")
    plt.ylabel("Variance of Estimated Location")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1)
    #plt.plot(gt_positions, mean_Bias, "r")
    for i in range(len(Biases)):
        plt.plot(gt_positions, Biases[i], "r")
    plt.fill_between(gt_positions, mean_Bias-std_Bias, mean_Bias+std_Bias, facecolor='red', alpha=0.3)
    plt.xlabel("Location")
    plt.ylabel("Bias of Estimated Location")
    plt.legend()
    plt.show()

def main():
    print("run something here")
    example()

if __name__ == '__main__':
    main()