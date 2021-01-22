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

def kalman_model(target_distances, predict_std_R=0.1, measurement_std_Q=0.1,
          expon_coef=-35, lambda_coef=3, k=1):

    print("testing params ", predict_std_R, " and ", measurement_std_Q, " and ",
          expon_coef, " and ", lambda_coef, " and ", k)

    simple_v = True
    gain = 1
    dt = 1                    # gain=1, dt=1, n_t=10, n_i=50, R=0.1,Q=0.1, e_coef=-28, l_coef=5,
    n_trials = 1
    N_iterations = 10000
    velocity_max = 2 # to sweep over potential velocity range
    target_width = 0.1
    predict_std_R = predict_std_R
    measurement_std_Q = measurement_std_Q
    expon_coef= expon_coef
    lambda_coef = lambda_coef
    k = k # proportionality constant

    if simple_v:
        gt_motor_commands = np.zeros(N_iterations); gt_motor_commands[0] = 1*gain # simple velocity
    else:
        gt_motor_commands = np.random.randint(-1,2,N_iterations)*gain

    velocity_arange = np.arange(-2,velocity_max,0.01)
    gt_velocities = np.cumsum(gt_motor_commands)
    gt_velocities[gt_velocities < 0] = 0
    gt_motor_commands = np.diff(np.append(np.array([0]), gt_velocities))
    gt_velocities = np.cumsum(gt_motor_commands)
    gt_positions = np.append(0, np.cumsum(gt_velocities)[:-1])

    Optimal_response = np.zeros((n_trials, N_iterations))
    Positions = np.zeros((n_trials, N_iterations))
    Velocities = np.zeros((n_trials, N_iterations))
    Position_Errors = np.zeros((n_trials, N_iterations))
    position_uncertainty_error = np.zeros((n_trials, N_iterations))

    response_distances = []

    for n in range(n_trials):
        #print("simulating trial, ", n)

        # Initialization of state matrices
        X = np.array([0, 0.0])     # state vector position and velocity
        P = np.diag((0.01, 0.01))    # we start with a small co variance matrix as we are quite certain of where we are
        A = np.array([[1, dt], [0, 1]]) # state transition matrix   see here: https://share.cocalc.com/share/7557a5ac1c870f1ec8f01271959b16b49df9d087/Kalman-and-Bayesian-Filters-in-Python/08-Designing-Kalman-Filters.ipynb?viewer=share
        Q = Q_discrete_white_noise(dim=2, dt=dt, var=measurement_std_Q)
        U = np.zeros((X.shape[0],1))
        B = np.array([[0.0], [1/gain]])
        U = np.zeros(1)

        # Measurement matrices
        Y = np.array([[X[0] + (measurement_std_Q*np.random.randn(1))]])
        H = np.array([[0, 1]])
        R = np.eye(Y.shape[0])*predict_std_R

        # Applying the Kalman Filter
        target_idx = 0
        i = 0
        while(gt_positions[i] < target_distances[-1]):

            # assign input and measurement with noise
            U[0] = gt_motor_commands[i]+(predict_std_R*np.random.randn(1))
            posterior_velocity_estimate = posterior_velocity(gt_velocities[i], expon_coef, measurement_std_Q, velocity_arange)

            Y = np.array([[posterior_velocity_estimate]])
            (X, P) = kf_predict(X, P, A, Q, B, U)
            (X, P, K) = kf_update(X, P, Y, H, R)
            X = X.flatten()

            Optimal_response[n, i], error = position_uncertainty(X, lambda_coef=lambda_coef, target_width=target_width, k=k)
            Positions[n, i] = X[0]
            Velocities[n, i] = X[1]
            Position_Errors[n, i] = abs(gt_positions[i] - X[0])
            position_uncertainty_error[n, i] = error

            i+=1

            if gt_positions[i] >= target_distances[target_idx]:
                response_distances.append(Optimal_response[n, i-1])
                target_idx+=1

    response_distances = np.array(response_distances)
    print(response_distances)

    print("FINISHED RUN IN KALMAN MODEL, AHAHHHHHHHHHHHHHHHHHHHHHHHHHH")
    return response_distances




def main():
    print("run something here")
    params = Parameters()
    save_path = r"C:\Users\44756\PycharmProjects\VRnavh\Modelling\parameter_search"
    target_distances = np.arange(0,400, 20)

    kalman_model(target_distances=target_distances, predict_std_R=0.1, measurement_std_Q=0.1,
                 expon_coef=-35, lambda_coef=3, k=1)

if __name__ == '__main__':
    main()