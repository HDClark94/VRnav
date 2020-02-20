import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import inv
from filterpy.common import Q_discrete_white_noise
np.random.seed(64)

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
    N_iterations = 50

    predict_std_R = 0.1
    measurement_std_Q = 1

    gt_motor_commands = np.random.randint(-1,2,N_iterations)*gain
    gt_velocities = np.cumsum(gt_motor_commands)
    gt_velocities[gt_velocities < 0] = 0
    gt_positions = np.cumsum(gt_velocities)

    Positions= np.zeros((n_trials, N_iterations))
    Velocities= np.zeros((n_trials, N_iterations))

    time_steps = np.arange(0, len(gt_motor_commands)*dt, dt)
    for n in range(n_trials):

        # Initialization of state matrices
        X = np.array([0.0, 0.0])     # state vector position and velocity
        P = np.diag((0.01, 0.01))    # we start with a small co variance matrix as we are quite certain of where we are
        A = np.array([[1, dt], [0, 1]])    # state transition matrix
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
            Y = np.array([[gt_velocities[i] + (measurement_std_Q*np.random.randn(1)[0])]])

            (X, P) = kf_predict(X, P, A, Q, B, U)
            (X, P, K) = kf_update(X, P, Y, H, R)

            #Y = array([[gt_velocities[i] + (measurement_std_Q*np.random.randn(1)[0])]])

            # housekeeping
            X=X.flatten()

            Positions[n, i] = X[0]
            Velocities[n, i] = X[1]

    avg_pos = np.mean(Positions, axis=0)
    std_pos = np.std(Positions, axis=0)

    avg_vel = np.mean(Velocities, axis=0)
    std_vel = np.std(Velocities, axis=0)

    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1) #stops per trial
    for i in range(n_trials):
        plt.plot(time_steps, Positions[i, :], "r", alpha=0.3)
    plt.fill_between(time_steps,avg_pos-std_pos,avg_pos+std_pos, facecolor='red', alpha=0.3)
    plt.plot(time_steps, avg_pos, "r", label="predicted trajectory")
    plt.plot(time_steps, gt_positions, "k--", label="true trajectory")
    plt.xlabel("Time Step")
    plt.ylabel("X Position")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1) #stops per trial
    for i in range(n_trials):
        plt.plot(time_steps, Velocities[i,:], 'b', alpha=0.3)
    plt.fill_between(time_steps,avg_vel-std_vel,avg_vel+std_vel, facecolor='blue', alpha=0.3)
    plt.plot(time_steps, avg_vel, "b",label="predicted trajectory")
    plt.plot(time_steps, gt_velocities, "k--", label="true trajectory")
    plt.xlabel("Time Step")
    plt.ylabel("X Velocity")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1) #stops per trial
    plt.plot(gt_positions, std_pos, "r")
    plt.xlabel("Location")
    plt.ylabel("Std of estimated location")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1) #stops per trial
    plt.plot(time_steps, std_pos, "r")
    plt.xlabel("Time Step")
    plt.ylabel("Std of estimated location")
    plt.legend()
    plt.show()

def main():
    print("run something here")
    example()

if __name__ == '__main__':
    main()