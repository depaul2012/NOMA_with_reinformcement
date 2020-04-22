import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
import time
import random
from pprint import pprint
from itertools import combinations, permutations
from noma_reinforcement import building_network_parameters

from matplotlib import style
style.use("ggplot")

size = 10
episodes = 25000
base_station_reward = 20
base_station_penalty = (50, 100)

epsilon = 0.9
epsilon_decay = 0.9998

show_every = 2500

start_q_table = None
learning_rate = 0.1
discount = 0.95

base_station = (5, 5)
base_station_2 = (-5,-5)
user_locations = [(1, 1), (1, 3), (3, 2), (-4, -1), (-2, -4), (-1, -2)]


class base_station_controller(object):
    def __init__(self):
        self.x, self.y = random.choice(user_locations)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def re_allocation(self):
        agent_move = (self.x, self.y)
        if agent_move not in user_locations:
            agent_move = random.choice(user_locations)
        return agent_move

    def action(self, choice):
        '''
        Gives us 5 total movement options to get to the 5 different locations for. (u1,u2,u3,u4)
        '''
        if choice == 0:  # 0 stands for user u1
            self.move(x=1, y=1)
        elif choice == 1:  # 1 stands for user u2
            self.move(x=1, y=3)
        elif choice == 2:  # 2 stands for user u3
            self.move(x=3, y=2)
        elif choice == 3:  # 3 stands for user u4
            self.move(x=-4, y=-1)
        elif choice == 4:  # 3 stands for user u5
            self.move(x=-2, y=-4)
        elif choice == 5:  # 3 stands for user u4
            self.move(x=-1, y=-2)

    def move(self, x=False, y=False):
        # If no value for x, move randomly
        if not x:
            self.x = random.choice([i[0] for i in user_locations])
        else:
            self.x = x

        # If no value for y, move randomly
        if not y:
            self.y = random.choice([i[1] for i in user_locations])
        else:
            self.y = y

        # If we base stations is trying to serve a user out of bounds
        if self.x < -10:
            self.x = -10
        elif self.x > size + 1:
            self.x = size - 1
        if self.y < -10:
            self.y = -10
        elif self.y > size + 1:
            self.y = size - 1

        agent_move = (self.x, self.y)
        if agent_move not in user_locations:
            agent_move = random.choice(user_locations)
            self.x, self.y = agent_move
        else:
            self.x = self.x
            self.y = self.y


if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for o in permutations(user_locations, 2):  # 5 implying 5 users
        u, v = o
        #print((base_station, u, base_station_2, v))
        q_table[(base_station, u, base_station_2, v)] = [[np.random.uniform(-6, 0) for i in range(6)] for i in range(2)]
    pprint('Initial Q-table keys\n {}'.format(q_table.keys()))
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []
bs1_users = [(x,y) for x,y in user_locations if x>0 and y>0]
bs2_users = [i for i in user_locations if i not in bs1_users]

for episode in range(episodes):
    bs1_player = base_station_controller()
    bs2_player = base_station_controller()
    if episode % show_every == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{show_every} ep mean: {np.mean(episode_rewards[-show_every:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (base_station, bs1_player), (base_station_2, bs2_player)
        print('OBS {}'.format(i), obs[0], obs[1])
        if np.random.random() > epsilon:
            # GET THE ACTION
            #actions = [np.argmax(i) for i in q_table[(obs[0][0], (obs[0][1].x, obs[0][1].y), obs[1][0], (obs[1][1].x, obs[1][1].y))]]
            print(obs[0][0], obs[0][1].x, obs[0][1].y, obs[1][0], obs[1][1].x, obs[1][1].y)
            if (obs[0][1].x + obs[0][1].y) != (obs[1][1].x + obs[1][1].y):
                actions = [np.argmax(i) for i in  q_table[(obs[0][0], (obs[0][1].x, obs[0][1].y), obs[1][0], (obs[1][1].x, obs[1][1].y))]]
        else:
            actions = [np.random.randint(0, 5) for i in range(2)]
        print('action taken', actions)
        # Take the action!
        if actions[0] != actions[1]:
            bs1_player.action(actions[0])
            bs2_player.action(actions[1])
            # the logic
            bs1_associated_user = (bs1_player.x, bs1_player.y)
            bs2_associated_user = (bs2_player.x, bs2_player.y)

            if bs1_associated_user in bs2_users or bs2_associated_user in bs1_users:
                reward = -(base_station_penalty[1])
            elif bs1_associated_user in bs1_users and bs2_associated_user in bs2_users:
                reward = base_station_reward

            ## NOW WE KNOW THE REWARD, LET'S CALC YO
            # first we need to obs immediately after the move.
            new_obs = (base_station, bs1_player), (base_station_2, bs2_player)

            print('New OBS', new_obs[0][0], new_obs[0][1].x, new_obs[0][1].y, new_obs[1][0], new_obs[1][1].x, new_obs[1][1].y)

            q = q_table[(new_obs[0][0], (new_obs[0][1].x, new_obs[0][1].y), new_obs[1][0], (new_obs[1][1].x, new_obs[1][1].y))]
            max_future_q = [np.max(i) for i in q]
            current_q = [q[0][actions[0]],q[1][actions[1]]]

            new_q = []
            if reward == base_station_reward:
                new_q = [base_station_reward/2 for _ in [base_station, base_station_2]]
            else:
                new_q = [(1 - learning_rate) * current_q[i] + learning_rate * (reward + discount * max_future_q[i]) for i in range(2)]

            for i in range(2):
                q_table[((obs[0][0], (obs[0][1].x, obs[0][1].y), obs[1][0], (obs[1][1].x, obs[1][1].y)))][i][actions[i]] = new_q[i]

            episode_reward += reward
            if reward == base_station_reward or reward == -base_station_penalty[1]:
                break

    # print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= epsilon_decay

moving_avg = np.convolve(episode_rewards, np.ones((show_every,)) / show_every, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {show_every}ma")
plt.xlabel("episode #")
plt.show()

with open(f"wireless_communication_qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)