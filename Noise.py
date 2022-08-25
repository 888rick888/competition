import numpy as np
import time
np.random.seed(1)


EXPLORE = 10000

class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class ActionNoise(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.std_dev = sigma
        self.epsilon = 1
        self.theta = 0.15
        self.dt = 1e-2


    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

    def add_noise(self, action, std):
        noise_np = self.sample(action, std)
        action = action + noise_np
        # action = np.clip(action, -100, 200)
        return action

    def sample(self, action, std):
        action = np.asarray(action)
        self.epsilon -= 1.0 / EXPLORE
        noise = np.zeros(np.shape(action))
        if int(np.shape(action)[1]) == 1:
            for i in range(len(action)):
                noise[i][0] = max(self.epsilon, self.dt) * self.fun(action[i][0], mu=0.0, theta=self.theta, std=std)   #Steering oniy
        if int(np.shape(action)[1]) == 2:
            for i in range(len(action)):
                noise[i][0] = max(self.epsilon, self.dt) * self.fun(action[i][0], mu=0.0, theta=self.theta, std=std)   #Steering
                noise[i][1] = max(self.epsilon, self.dt) * self.fun(action[i][0], mu=0.5, theta=self.theta, std=std)   #acceleration
        return noise
    
    def fun(self, x, mu, theta, std):
        return theta * (mu - x) + std * np.random.randn(1)
