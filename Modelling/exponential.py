# serves to show the range of exponents

import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import inv
import math

np.random.seed(64)
def posterior_velocity(gt_velocity, expon_coef, likelihood_width, velocities):

    likelihood_dist = guassian(velocities,gt_velocity,likelihood_width)/np.sum(guassian(velocities,gt_velocity,likelihood_width))
    prior_dist = ((np.e**(expon_coef*velocities))/np.sum(np.e**(expon_coef*velocities)))
    posterior_dist = (likelihood_dist*prior_dist)/np.sum(likelihood_dist*prior_dist)

    return np.random.choice(velocities, 1, p=posterior_dist)

def exponential_prior(velocity, coef):
    return np.e**(coef*velocity)

def guassian(x, mu, sigma):
    return np.exp(-np.power(x-mu, 2.) / (2 * np.power(sigma,2.)))

def pdf(x, mu, sigma):
    x = (x - mu)/sigma
    return np.exp(-x*x/2.0) / np.sqrt(2.0*np.pi) / sigma

def pdf_vectorised(x, mu, sigma):

    mu_new = np.repeat(mu[:, np.newaxis], len(x), axis=1)
    x_new = (np.repeat(x[:, np.newaxis], len(mu), axis=1).T)
    sigma_new = np.repeat(sigma[:, np.newaxis], len(x), axis=1)

    x_new2 = (x_new - mu_new)/sigma_new
    return np.exp(-x_new2*x_new2/2.0) / np.sqrt(2.0*np.pi) / sigma_new


def test_pdf_vectorised():
    step =0.05
    distances = np.arange(step,10, step)
    target_distances2 = distances
    target_width = 0.02
    coef = 1.5

    optimal_responses = []
    for distance in target_distances2:
        target_distances = distances[np.where(np.logical_and(distances>=distance-target_width,
                                                             distances<=distance+target_width))]
        expected_rewards = []
        for i in range(len(distances)):
            expected_reward = np.sum(pdf(x=target_distances, mu=distances[i], sigma=distances[i]**coef))
            expected_rewards.append(expected_reward)
        expected_rewards = np.array(expected_rewards)
        optimal_responses.append(distances[np.argmax(expected_rewards)])
    optimal_responses=np.array(optimal_responses)

    optimal_responses_vectorised = []
    for distance in target_distances2:
        target_distances = distances[np.where(np.logical_and(distances>=distance-target_width,
                                                             distances<=distance+target_width))]
        expected_rewards = []
        expected_reward = np.sum(pdf_vectorised(x=target_distances, mu=distances, sigma=distances**coef), axis=1)
        expected_rewards.append(expected_reward)
        expected_rewards = np.array(expected_rewards)
        optimal_responses_vectorised.append(distances[np.argmax(expected_rewards)])
    optimal_responses_vectorised=np.array(optimal_responses_vectorised)

    assert np.allclose(optimal_responses, optimal_responses_vectorised, rtol=1e-05, atol=1e-08)
    # it seems to past the test???????????


def exponent_example():
    print("hello there")

    coefs = np.arange(-0.5, 0.05, 0.05)

    fig = plt.figure(figsize = (12,4))
    ax = fig.add_subplot(1,1,1) #stops per trial
    velocities = np.arange(0,100, 0.1)

    for coef in coefs:
        # plot response target
        prior = exponential_prior(velocities, coef)
        normalised_prior = prior/np.sum(prior)
        ax.plot(velocities, normalised_prior, label="coef = "+str(np.round(coef, decimals=2)))

    plt.xlabel("Velocity (VU/s)")
    plt.ylabel("Probability")
    plt.xlim(0,30)

    # add likelihood
    velocity = 7
    Y = velocity +(1*np.random.randn(10000000))
    bin_centres = 0.5*(velocities[1:]+velocities[:-1])
    likelihood = np.histogram(Y, bins=velocities)[0]/np.sum(np.histogram(Y, bins=velocities)[0])
    plt.plot(bin_centres, likelihood)

    posterior = normalised_prior[:-1]*likelihood/np.sum(normalised_prior[:-1]*likelihood)
    plt.plot(bin_centres, posterior, color='yellow')


    #plt.ylim(0,300)
    plt.legend()
    plt.show()

def uncertainty_exponent():
    print("hello there")

    coefs = [0.5, 1, 1.5, 2]
    sample_distances = [1, 2, 4, 8, 16]

    fig, axs = plt.subplots(3)
    step =0.05
    distances = np.arange(step,max(sample_distances)+5, step)

    target_width = 0.5

    for coef in coefs:
        bias = []
        for distance in sample_distances:
            # plot response target
            uncertainty = pdf(x=distances, mu=distance, sigma=distance**coef)
            uncertainty = uncertainty/np.sum(uncertainty)
            axs[0].plot(distances, uncertainty, label="coef = "+str(np.round(coef, decimals=2)))
            axs[0].axvline(distance, 0, 1)

            target_distances = distances[np.where(np.logical_and(distances>=distance-target_width,
                                                                 distances<=distance+target_width))]
            expected_rewards = np.sum(pdf_vectorised(x=target_distances, mu=distances, sigma=distances**coef), axis=1)

            axs[1].plot([distance-target_width,distance+target_width], [0, 0], linewidth=5)
            axs[1].plot(distances, expected_rewards)
            axs[1].axvline(distances[np.argmax(expected_rewards)],0,1)

            print("bias =", str(distances[np.argmax(expected_rewards)]-distance))
            bias.append(distances[np.argmax(expected_rewards)]-distance)

        # for third plot
        optimal_responses = []
        for distance in distances:
            target_distances = distances[np.where(np.logical_and(distances>=distance-target_width,
                                                                 distances<=distance+target_width))]
            expected_rewards = np.sum(pdf_vectorised(x=target_distances, mu=distances, sigma=distances**coef), axis=1)
            optimal_responses.append(distances[np.argmax(expected_rewards)])

        optimal_responses=np.array(optimal_responses)

        axs[2].plot(distances, optimal_responses, label="coef = "+str(coef))

    axs[2].plot(distances, distances, "k--")
    axs[2].set_xlim(0,max(distances))
    axs[2].set_ylim(0,max(distances))
    axs[2].set_xlabel("Distance")
    axs[2].set_ylabel("Optimal Response")

    axs[0].set_xlabel("Distance")
    axs[0].set_ylabel("Probability")
    axs[0].set_xlim(0,max(distances))

    axs[1].set_xlabel("Distance")
    axs[1].set_ylabel("Expected Reward")
    axs[1].set_xlim(0,max(distances))
    #axs[1].set_ylim(0,max(expected_rewards))
    #axs[0].legend()
    plt.show()

    # plot response target
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) #stops per trial
    plt.plot(sample_distances, bias, "g")
    plt.xlabel("Target", fontsize=20)
    plt.ylabel("Error", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    plt.show()

def position_uncertainty(X, lambda_coef=0, target_width=1, k=1):
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


def main():
    print("run something here")
    #exponent_example()

    uncertainty_exponent()

if __name__ == '__main__':
    main()