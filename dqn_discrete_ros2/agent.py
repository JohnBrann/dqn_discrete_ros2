import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from collections import deque
from datetime import datetime, timedelta
import argparse
import itertools
import os

import rclpy
from rclpy.node import Node

from model_msgs.srv import EnvReset
from model_msgs.srv import EnvSetup
from model_msgs.srv import EnvStepCartpole

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.output(x)

class Agent(Node):
    def __init__(self):
        super().__init__('agent_node')

        with open('/home/csrobot/pytorch_ws/src/dqn_discrete_ros2/dqn_discrete_ros2/hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameter_set='cartpole1'
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        self.is_training = hyperparameters['is_training']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # Initialize service clients
        self.env_reset_client = self.create_client(EnvReset, 'env_reset')
        self.env_dim_client = self.create_client(EnvSetup, 'env_setup')
        self.env_step_client = self.create_client(EnvStepCartpole, 'env_step')
        
        while not self.env_reset_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('reset env service not available, waiting again...')
        
        while not self.env_dim_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('env dimension service not available, waiting again...')
        
        while not self.env_step_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('step env service not available, waiting again...')
        
        dim_message = self.send_env_dim_request()

        self.action_space_dim = dim_message.action_dim
        self.state_dim = dim_message.state_dim
        
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def send_env_dim_request(self):
        req = EnvSetup.Request()
        future = self.env_dim_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def send_env_reset_request(self):
        req = EnvReset.Request()
        future = self.env_reset_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def send_env_step_request(self, action):
        req = EnvStepCartpole.Request()
        req.action = action
        future = self.env_step_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        rewards_per_episode = []
        policy_dqn = DQN(self.state_dim, self.action_space_dim, self.fc1_nodes).to(device)

        if is_training:
            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)
            target_dqn = DQN(self.state_dim, self.action_space_dim, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            epsilon_history = []
            step_count = 0
            best_reward = -9999999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():
            self.send_env_reset_request()

            state = self.state_subscriber.step_data()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                if is_training and random.random() < epsilon:
                    action = random.sample(self.action_space_dim)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                        action = action.item()

                state_srv = self.send_env_step_request(action)
                state = np.array([state_srv.cart_pos, state_srv.cart_velocity, state_srv.pole_angle, state_srv.pole_angular_velocity])
                self.step_data = (state, state_srv.reward, state_srv.terminated, state_srv.truncated)
                new_state, reward, terminated, truncated = self.step_data
                episode_reward += reward
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                action = torch.tensor(action, dtype=torch.int64, device=device)
                
                if is_training:
                    memory.append((state, action, new_state, reward, terminated, truncated))
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
        current_q = policy_dqn(states).gather(1, actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main(args=None):
    rclpy.init(args=args)
    agent = Agent()
    agent.run(is_training=True)
    agent.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
