import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import inv
from filterpy.common import Q_discrete_white_noise
np.random.seed(65)

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
    base_gain = 1
    dt = 1
    n_trials = 1000
    N_iterations = 200

    bin_size = 2
    bins = np.arange(0, 600, bin_size)
    bin_centres = 0.5*(bins[1:]+bins[:-1])

    fig1 = plt.figure(figsize = (12,4))
    ax1 = fig1.add_subplot(1,1,1) #stops per trial
    fig2 = plt.figure(figsize = (12,4))
    ax2 = fig2.add_subplot(1,1,1) #stops per trial
    fig3 = plt.figure(figsize = (12,4))
    ax3 = fig3.add_subplot(1,1,1) #stops per trial
    fig4 = plt.figure(figsize = (12,4))
    ax4 = fig4.add_subplot(1,1,1) #stops per trial

    Qs = [1, 0.8, 0.6, 0.4, 0.2]
    Rs = [0.5]

    for q in Qs:
        for r in Rs:
            predict_std_R = q
            measurement_std_Q = r
            #gain = np.random.normal(base_gain, 0.5)

            Positions= np.zeros((n_trials, N_iterations))
            Velocities= np.zeros((n_trials, N_iterations))
            Position_Errors = np.zeros((n_trials, N_iterations))
            gt_positions_all = np.zeros((n_trials, N_iterations))

            for n in range(n_trials):
                gt_motor_commands = np.random.randint(-1,2,N_iterations)*gain
                gt_velocities = np.cumsum(gt_motor_commands)
                gt_velocities[gt_velocities < 0] = 0
                gt_motor_commands = np.append(np.array([0]), np.diff(gt_velocities))
                gt_positions = np.cumsum(gt_velocities)
                time_steps = np.arange(0, len(gt_motor_commands)*dt, dt)

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
                    Position_Errors[n, i] = abs(gt_positions[i] - X[0])
                    gt_positions_all[n,i] = gt_positions[i]


            avg_abs_err = np.mean(Position_Errors, axis=0)
            std_abs_err = np.std(Position_Errors, axis=0)

            # plotting absolute error with std
            label1 = "R = " + str(predict_std_R) + ", Q = " + str(measurement_std_Q)
            ax1.plot(time_steps, avg_abs_err, label=label1)
            ax2.plot(time_steps, std_abs_err, label=label1)

            flatten_errors = Position_Errors.flatten()
            flatten_positions = gt_positions_all.flatten()
            #avg_error_by_position = np.histogram(gt_positions_all, bins=bins, weights=Position_Errors)[0] /\
            #                        np.histogram(gt_positions_all, bins=bins, weights=Position_Errors)[0]

            flatten_errors = Position_Errors.flatten()
            flatten_positions = Positions.flatten()

            bin_indices = np.digitize(flatten_positions, bins)

            avg_err_pos = np.zeros(len(bin_centres))*np.nan
            std_err_pos = np.zeros(len(bin_centres))*np.nan

            for j, bin in enumerate(bins):
                bin_errors = flatten_errors[bin_indices==bin]
                if len(bin_errors)>0:
                    avg_err_pos[j] = np.mean(bin_errors)
                    std_err_pos[j] = np.std(bin_errors)

            ax3.plot(bin_centres, avg_err_pos, label=label1)
            ax4.plot(bin_centres, std_err_pos, label=label1)


    ax1.legend()
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("avg location error")
    ax1.set_xlim(left=0)
    ax1.set_ylim(0, 20)
    ax2.legend()
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Std of estimated location error")
    ax2.set_ylim(0,15)
    ax2.set_xlim(left=0)
    ax3.legend()
    ax3.set_xlabel("Location")
    ax3.set_ylabel("Mean absolute Error")
    ax3.set_ylim(0, 20)
    ax3.set_xlim(left=0)
    ax4.legend()
    ax4.set_xlabel("Location")
    ax4.set_ylabel("STD absolute Error")
    ax4.set_ylim(0,15)
    ax4.set_xlim(left=0)
    plt.show()

    # analyse for error and error variance against location


def main():
    print("run something here")
    example()

if __name__ == '__main__':
    main()