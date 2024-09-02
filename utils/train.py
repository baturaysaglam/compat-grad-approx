import json
import os
import random

import numpy as np
import torch


def identity_func(x):
    return x


def select_action(agent, state, action_space, mu, stochastic=True):
    action_dim, max_action = action_space.shape[0], action_space.high[0]

    if "SAC" in str(agent):
        action, (mean, std) = agent.select_action(state, evaluate=False)
        action = action.clip(-max_action, max_action)
        noise = (action - mean) / std
    else:
        mean = agent.select_action(state)

        noise = np.random.normal(0, max_action * mu, size=action_dim)
        action = (mean + noise).clip(-max_action, max_action)
        noise = (action - mean) / mu
        std = np.ones_like(action) * mu

    if not stochastic:
        action = mean
        noise, std = np.ones_like(action), np.ones_like(action)

    return action, noise, std


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def evaluate_policy(agent, eval_env, seed, eval_episodes=10):
    eval_env.seed(seed + 100)
    eval_env.action_space.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False

        while not done:
            action = agent.select_action(np.array(state))
            action = action[1][0] if isinstance(action, tuple) else action
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    return avg_reward


class DelayedRewardTracker:
    def __init__(self, delay_steps):
        self.delay_steps = delay_steps
        self.rewards = [0] * delay_steps  # Initialize a list to store the last N rewards

    def __call__(self, instant_reward):
        """
        Adds the instant reward to the list and returns the oldest reward (which is the delayed reward).
        """
        delayed_reward = self.rewards.pop(0)  # Get the oldest reward
        self.rewards.append(instant_reward)   # Add the new reward to the end of the list
        return delayed_reward


class NoisyRewardTracker:
    def __init__(self):
        self.min_reward = float('inf')
        self.max_reward = float('-inf')
        self.noise_std_dev = 0.0  # Default value, will be updated based on episode rewards
        self.frac = 0.1

    def update_reward_range(self, reward):
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)

    def calculate_std_dev(self):
        # Calculate the standard deviation as a fraction of the reward range
        # Adjust the fraction as needed
        reward_range = self.max_reward - self.min_reward
        self.noise_std_dev = reward_range * self.frac  # Example: 10% of the range

    def reset(self):
        self.min_reward = float('inf')
        self.max_reward = float('-inf')

    def __call__(self, reward, done):
        """
        Adds the zero-mean Gaussian noise (std. dev. specified by the user) to the reward.
        """
        if done:
            self.calculate_std_dev()
            self.reset()
        else:
            self.update_reward_range(reward)

        # Add Gaussian noise to the reward
        noisy_reward = reward + np.random.normal(0, self.noise_std_dev)
        return noisy_reward


class RandomSparseRewardTracker:
    def __init__(self, reward_probability):
        self.reward_probability = reward_probability

    def __call__(self, instant_reward):
        """
        Randomly decides whether to return the real reward or a zero reward based on the reward probability.
        """
        if random.random() < self.reward_probability:
            return instant_reward  # Give the real reward
        else:
            return 0  # Give a zero reward


class ExperienceReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
        self.device = device

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.mask = np.zeros((max_size, 1))

        self.mu = np.zeros((max_size, action_dim))
        self.exploration_noise = np.zeros((max_size, action_dim))

        self.device = device

    def add(self, state, action, next_state, reward, done, mu, exploration_noise):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.mask[self.ptr] = 1. - done
        self.mu[self.ptr] = mu
        self.exploration_noise[self.ptr] = exploration_noise

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.mask[ind]).to(self.device),
            torch.FloatTensor(self.mu[ind]).to(self.device),
            torch.FloatTensor(self.exploration_noise[ind]).to(self.device)
        )


def get_save_dir(args_dict, results_dir, env_name):
    results_dir = os.path.join(results_dir, args_dict["policy"])

    if not os.path.exists(results_dir + f"/{env_name}"):
        os.makedirs(results_dir + f"/{env_name}")

    with open(os.path.join(results_dir, 'parameters.json'), 'w') as file:
        json.dump(args_dict, file, indent=4)

    results_dir += f"/{env_name}"

    print(f"Saved in {results_dir}\n")

    return results_dir