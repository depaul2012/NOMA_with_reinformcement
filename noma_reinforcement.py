import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
import time
import random
from pprint import pprint
import statistics
from itertools import combinations, permutations
import pandas as pd
from matplotlib import style
from tabulate import tabulate
style.use("ggplot")

size = 10
episodes = 2
base_station_reward = 30
base_station_penalty = (50, 120)

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

    def network_params(self):
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


def initialize_q_table():
    if start_q_table is None:
        # initialize the q-table#
        q_table = {}
        for o in permutations(user_locations, 2):  # 5 implying 5 users
            u, v = o
            print((base_station, u, base_station_2, v))
            q_table[(base_station, u, base_station_2, v)] = [[np.random.uniform(-6, 0) for i in range(6)] for i in range(2)]
        #pprint('Initial Q-table keys\n {}'.format(q_table.keys()))
    else:
        with open(start_q_table, "rb") as f:
            q_table = pickle.load(f)
    return q_table

def radius_based_training():
    epsilon = 0.9
    episode_rewards = []
    q_table = initialize_q_table()
    bs1_users = [(x,y) for x,y in user_locations if x>0 and y>0]
    bs2_users = [i for i in user_locations if i not in bs1_users]
    for episode in range(episodes):
        bs1_player = base_station_controller()
        bs2_player = base_station_controller()
        if episode % show_every == 0:
            #print(f"on #{episode}, epsilon is {epsilon}")
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

def building_network_parameters(users, base_station):
    #distance
    d = [euclidean_distance(i, base_station) for i in users]
    #channel_gain
    h = np.random.rand(len(users))
    #noise(AWGN) and variance
    n = np.random.random(len(users))
    v = [statistics.variance(n)] * len(users)

    #an algorithm to allocate power coefficients based on the channel gain
    average_h = np.mean(h.tolist())
    p = []
    for i in h:
        i_p = np.random.randint(10)
        if i < average_h:
            i_p = i * i_p
        p.append(float(i_p))

    dataset = pd.DataFrame()
    columns_of_interest = ['users', 'distrance to base', 'channel', 'power', 'noise', 'variance']
    users_loc = [str(i) for i in users]
    for i,j in enumerate([users_loc, d, h, p, n, v]):
        dataset = pd.concat([dataset, pd.DataFrame(j, columns=[columns_of_interest[i]])], axis=1)

    return dataset

def euclidean_distance(user, base_station):
    x_u, y_u = user
    x_b, y_b = base_station
    return np.sqrt((x_u-x_b)**2 + (y_u-y_b)**2)

def cluster_level_interference(users, bp, current_user):
    I = 0
    for i,u in enumerate(users):
        if u[0] != current_user:
            I += ((u[2]*np.sqrt(u[3]*bp)) + u[-2])
    return I

def basestation_level_interference(clusters, bp, index_cluster):
    I = 0
    for i,u in enumerate(clusters):
        if i != index_cluster:
            I += cluster_level_interference(u, bp, current_user=None)
    return I

def total_transmitted_superposed_signal(users, current_user, sic=False):
    P = 0
    for i,j in enumerate(users):
        if j[0] != current_user:
            P += j[3]
    return P

def initialize_built_network(bs):
    clusters = {}
    for i, j in enumerate(bs):
        cluster = building_network_parameters(user_locations, j)
        clusters[j] = cluster.values
        print('Base station', i + 1, j)
        print(tabulate(cluster, headers='keys', tablefmt='psql'))
    return clusters

def decoding_order(users, i_c, i_b, snr, power):
    D = []
    for i,j in enumerate(users):
        numerator = (snr * (i_c + i_b)) + 1
        denominator = snr * (abs(users[2])**2)
        D.append((users[0], (numerator/denominator)))
    D = sorted(D, key=lambda x:x[1])
    return D

def compute_data_rate(bs, clusters, bps, sic=False):
    data_rates = {}
    clusters = [i[1] for i in clusters]
    for i, curr_cluster in enumerate(clusters):
        for k, user in enumerate(curr_cluster):
            channel_gain = abs(user[2]) ** 2  # numerator
            power_coeffient = user[3]
            distance = user[1]
            # numerator
            numerator = channel_gain * distance * power_coeffient
            '----------------------------------------------------------------------------------------'
            transmitted_superposed_signal = total_transmitted_superposed_signal(users=curr_cluster,
                                                                                current_user=user[0],
                                                                                sic=sic)

            cluster_interference = cluster_level_interference(users=curr_cluster,
                                                              bp=bps[i],
                                                              current_user=user[0])

            baselevel_interference = basestation_level_interference(clusters=clusters,
                                                                    bp=bps[i],
                                                                    index_cluster=i)

            snr = bps[i] / user[-1]
            # denominator
            denominator = (channel_gain * transmitted_superposed_signal) + cluster_interference + baselevel_interference + snr
            '----------------------------------------------------------------------------------------'
            user_r = np.log2(1 + (numerator / denominator))
            data_rates[(bs[i], user[0])] = user_r
           # print('User {} has data rate {}'.format(user[0], user_r))
    return data_rates

#agent takes action of assigning user to base-station
def action_user_base_station_assignment(users, clu):
    us_bs = {}
    for i in range(len(clu)):
        if clu[i][0] not in us_bs:
            us_bs[clu[i][0]] = users[i]
    #example us_bs = {bs_1:1, bs_2:2}
    return us_bs

#agent checks what's the original state of the network
def check_original_user_bs(us_bs, clu):
    us_bs_s = {}
    for u in us_bs:
        for v in clu:
            if us_bs[u] in [i[0] for i in v[1]]:
                us_bs_s[us_bs[u]] = v[0]
                break
    # example us_bs = {1:bs_1, 2:bs_2}
    return us_bs_s

#pass the before(original state) and after state to decide which scenario to implement
def check_scenario(us_bs_s, us_bs):
    us_bs = dict((v,k) for k,v in us_bs.items())
    if us_bs_s == us_bs:
        scenario = 1
    elif len(set(list(us_bs_s.values()))) == 1:
        scenario = 2
    else:
        scenario = 3
    return  scenario


def swap(clu, users, default_settings):
    d, final = [], []
    scenario_1 = False
    scenario_2 = False
    scenario_3 = False
    default_settings = [(k,v) for k,v in default_settings.items()]
    # print('CLU before\n', tabulate(pd.DataFrame(clu), headers='keys', tablefmt='psql'))
    # print(tabulate(pd.DataFrame(default_settings), headers='keys', tablefmt='psql'))

    #print(default_settings)
    step_1 = action_user_base_station_assignment(users=users, clu=clu)
    step_2 = check_original_user_bs(us_bs=step_1, clu=clu)
    scenario = check_scenario(us_bs_s=step_2, us_bs=step_1)

    #scenario 1 (users don't need swapping because they each belong to their respective base stations)
    if scenario == 1:
        scenario_1 = True
        print('SCENARIO 1\n')
        print(step_1)
        print(step_2)
        final = clu

    #scenario 2, all users on one base station
    elif scenario == 2:
        scenario_2 = True
        print('SCENARIO 2\n')
        print(step_1)
        print(step_2)
        for bt in clu:
            bt_users = [i[0] for i in bt[1]]
            if all(i in bt_users for i in list(step_1.values())):
                x = list(step_1.values())
                clu_remainder = remove_user_from_cluster(user=x[-1], c_cluster=bt[1])
                user_removed = x[-1]
                final.append((bt[0], np.array(clu_remainder)))
                break


        for next_bt in clu:
            if next_bt[0] not in [i[0] for i in final]:
                next_user = fecth_user_from_original_network(base=next_bt[0], user=user_removed, network=default_settings)
                cluster_to_append_removed_user = list(next_bt[1])
                cluster_to_append_removed_user.append(next_user)
                final.append((next_bt[0], np.array(cluster_to_append_removed_user)))

    elif scenario == 3:
        print('SCENARIO 3')
        print(step_1)
        print(step_2)
        temp_final = []
        step_2_reversed = dict((v,k) for k,v in step_2.items())
        for o,p in step_2_reversed.items():
            for c in clu:
                if c[0] == o:
                    clu_remainder = remove_user_from_cluster(user=p, c_cluster=c[1])
                    temp_final.append((o, np.array(clu_remainder)))

        for k,v in step_1.items():
            next_user = fecth_user_from_original_network(base=k, user=v, network=default_settings)
            for b in temp_final:
                if k == b[0]:
                    cluster_to_append_swapped_user = list(b[1])
                    cluster_to_append_swapped_user.append(next_user)
                    final.append((k, np.array(cluster_to_append_swapped_user)))

    return final

def fecth_user_from_original_network(base, user, network):
    for i in network:
        if i[0] == base:
            for j in i[1]:
                if j[0] == str(user):
                    user_needed = j
    return user_needed

def remove_user_from_cluster(user, c_cluster):
    retreived_cluster_users = [i for i in c_cluster if i[0] != user]
    return retreived_cluster_users

def reward_function(before_rates, after_rates):
    reward = 0
    for m, n in before_rates.items():
        m_bs, m_user = m
        before_rate = n
        for o, p in after_rates.items():
            o_bs, o_user = o
            after_rate = p
            print('here', o_user, m_user, before_rate, after_rate)
            if o_user == m_user and after_rate > before_rate:
                reward += 5
            elif o_user == m_user and after_rate < before_rate:
                reward -= 20
            else:
                reward += 0
    return  reward


def noma_based_training():
    epsilon = 0.9
    episode_rewards = []
    q_table = initialize_q_table()
    base_stations = [base_station, base_station_2]
    base_stations_powers = [100, 120]
    clusters_in_network = initialize_built_network(bs=base_stations)
    #print(clusters_in_network)

    clu = []
    for i,c in enumerate(clusters_in_network):
        if i == 0:
            c_ = np.array(clusters_in_network[c][:3,:])
        elif i == 1:
            c_ = np.array(clusters_in_network[c][3:,:])
        clu.append((base_stations[i], c_))

    #print(clu.shape)
    data_rates = compute_data_rate(bs=base_stations, clusters=clu, bps=base_stations_powers)
    print('\n')
    pprint(data_rates)
    for episode in range(episodes):
        bs1_player = base_station_controller()
        bs2_player = base_station_controller()
        if episode % show_every == 0:
            #print(f"on #{episode}, epsilon is {epsilon}")
            print(f"{show_every} ep mean: {np.mean(episode_rewards[-show_every:])}")
            show = True
        else:
            show = False
        print('Initial\n', clu)
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
                print(bs1_associated_user, bs2_associated_user)
                swapped_clusters = swap(clu=clu, users=(str(bs1_associated_user), str(bs2_associated_user)), default_settings=clusters_in_network)
                computed_noma_rates = compute_data_rate(bs=base_stations, clusters=swapped_clusters, bps=base_stations_powers)
                print('Swapped\n', swapped_clusters)
                pprint(computed_noma_rates)

                reward = reward_function(before_rates=data_rates, after_rates=computed_noma_rates)
                print('\n\nReward\n\n', reward)
                data_rates = computed_noma_rates
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

if __name__=='__main__':
    #print(initialize_q_table())
    noma_based_training()