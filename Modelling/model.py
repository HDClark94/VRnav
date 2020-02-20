import numpy as np
import os




class Env():
    def __init__(self):
        self.min_position = -1.2
        self.max_position = 1.0
        self.max_speed = 0.2
        self.goal_position = 0.5  # to 1 decimal place
        self.goal_width_half = 0.1  # to
        self.start_location = 0
        self.state = 0
        self.rewarded = False

        self.reset()

    def step(self, action):


        self.state += action

        if (self.state[0] > self.goal_position-self.goal_width_half and
                self.state[0] < self.goal_position+self.goal_width_half and action == 0):



    def reset(self):
        self.state = np.array([self.start_location])
        self.rewarded = False





def main():
    print("run something here")

if __name__ == '__main__':
    main()