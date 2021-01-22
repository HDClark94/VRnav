import numpy as np

class Parameters():

    def __init__(self, gain=1, dt=1, n_trials=10, N_iterations=50,
                 velocity_max = 30, target_width=0.1,
                 predict_std_R=0.1, measurement_std_Q = 0.1,expon_coef= -35, lambda_coef = 3, k=1):

        self.gain = gain
        self.dt = dt
        self.n_trials = n_trials
        self.N_iterations = N_iterations
        self.velocity_max = velocity_max
        self.target_width = target_width
        self.predict_std_R = predict_std_R
        self.measurement_std_Q = measurement_std_Q
        self.expon_coef = expon_coef
        self.lambda_coef = lambda_coef
        self.k = k