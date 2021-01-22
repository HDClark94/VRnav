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

def model(target_distances, prior_gain=1, lambda_coef=1, k=1):
    target_width = 13.8
    #target_width = 0.1

    #print("testing params ", prior_gain, " and ", lambda_coef, " and ", k)
    response_distances = []

    for target_distance in target_distances:
        X = np.array([target_distance, 1])

        Optimal_response, error = position_uncertainty(X, lambda_coef=lambda_coef, target_width=target_width, k=k)
        Optimal_response=prior_gain*Optimal_response
        response_distances.append(Optimal_response)

    response_distances = np.array(response_distances)

    return response_distances



def main():
    print("yep")

    target_distances = [63.2,  94.8, 142.2, 213.3, 320.]
    human_data_means_y = [ 79.30901538, 107.51794553, 147.5121811,  205.56856471, 290.27327138]

    target_distances_model = np.arange(10, 400, 30)
    best_fit_responses = model(target_distances_model, prior_gain=1.7, lambda_coef=1.9, k=0.01)

    # plot optimised response target
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) #stops per trial
    plt.plot(target_distances, human_data_means_y, "r", label="data")
    plt.plot(target_distances_model, best_fit_responses, "g", label="model")
    plt.plot(np.arange(0,400), np.arange(0,400), "k--", label="Unity")
    plt.xlabel("Target", fontsize=20)
    plt.ylabel("Optimal Response", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()