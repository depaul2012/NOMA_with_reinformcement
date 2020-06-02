import os
import time
import pandas as pd
import scipy
from scipy import special
import statistics
import random
import numpy as np
import tensorflow as tf
from collections import deque
import argparse
from tabulate import tabulate
from PA_alg import PA_alg
reuse = tf.compat.v1.AUTO_REUSE
dtype = np.float32
flag_fig = True

fd = 10
Ts = 20e-3
Ns = 5e1 #5e1 1e3
n_x = 5
n_y = 5
L = 2
C = 16
meanM = 2   # lamda: average user number in one BS
minM = 4   # maximum user number in one BS
maxM = 4   # maximum user number in one BS
min_dis = 0.01 #km
max_dis = 1. #km
min_p = 5. #dBm
max_p = 38. #dBm
p_n = -114. #dBm

power_num = 10 #action_num
OBSERVE = 100
EPISODE = 10000
TEST_EPISODE = 500
memory_size = 50000
INITIAL_EPSILON = 0.2 
FINAL_EPSILON = 0.0001 
learning_rate = 0.001
train_interval = 10
batch_size = 256
# def Generate_H_set(args):
#     '''
#     Jakes model
#     '''
#     H_set = np.zeros([args.max_users, args.adjacent_users,int(Ns)], dtype=dtype)
#     pho = np.float32(scipy.special.k0(2*np.pi*fd*Ts))
#     H_set[:,:,0] = np.kron(np.sqrt(0.5*(np.random.randn(args.max_users, c)**2+np.random.randn(args.max_users, c)**2)), np.ones((1,maxM), dtype=np.int32))
#     for i in range(1,int(Ns)):
#         H_set[:,:,i] = H_set[:,:,i-1]*pho + np.sqrt((1.-pho**2)*0.5*(np.random.randn(args.max_users, args.adjacent_users)**2+np.random.randn(args.max_users, args.adjacent_users)**2))
#     path_loss = Generate_path_loss()
#     H2_set = np.square(H_set) * np.tile(np.expand_dims(path_loss, axis=2), [1,1,int(Ns)])
#     return H2_set
c = 3*L*(L+1) + 1 # adjascent BS
K = maxM * c # maximum adjascent users, including itself
state_num = 3*C + 2    #  3*K - 1  3*C + 2
N = n_x * n_y # BS number
M = N * maxM # maximum users
W_ = np.ones((M), dtype = dtype)         #[M]
sigma2_ = 1e-3*pow(10., p_n/10.)
maxP = 1e-3*pow(10., max_p/10.)
power_set = np.hstack([np.zeros((1), dtype=dtype), 1e-3*pow(10., np.linspace(min_p, max_p, power_num-1)/10.)])
replay_memory = deque(maxlen = memory_size)
weight_file = os.getcwd() + '\dqn_00.mat'
hist_file = 'C:\Software\workshop\python\Commsys\hist_00'

def euclidean_distance(user, base_station):
    x_u, y_u = user
    x_b, y_b = base_station
    return np.sqrt((x_u-x_b)**2 + (y_u-y_b)**2)

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
        i_p = np.random.randint(1, 10)
        if i < average_h:
            i_p = i + i_p
        p.append(float(i_p))

    dataset = pd.DataFrame()
    columns_of_interest = ['users', 'distrance to base', 'channel', 'power', 'noise', 'variance']
    users_loc = [str(i) for i in users]
    for i,j in enumerate([users_loc, d, h, p, n, v]):
        dataset = pd.concat([dataset, pd.DataFrame(j, columns=[columns_of_interest[i]])], axis=1)

    return dataset

def initialize_built_network(args, bs):
    clusters = {}
    for i, j in enumerate(bs):
        cluster = building_network_parameters(args.user_locations, j)
        clusters[j] = cluster.values
        #print('Base station', i + 1, j)
        #print(tabulate(cluster, headers='keys', tablefmt='psql'))
        #print(np.array(cluster))
    return clusters

def Generate_H_set():
    '''
    Jakes model
    '''
    H_set = np.zeros([M,K,int(Ns)], dtype=dtype)
    pho = np.float32(scipy.special.k0(2*np.pi*fd*Ts))
    H_set[:,:,0] = np.kron(np.sqrt(0.5*(np.random.randn(M, c)**2+np.random.randn(M, c)**2)), np.ones((1,maxM), dtype=np.int32))
    for i in range(1,int(Ns)):
        H_set[:,:,i] = H_set[:,:,i-1]*pho + np.sqrt((1.-pho**2)*0.5*(np.random.randn(M, K)**2+np.random.randn(M, K)**2))
    path_loss = Generate_path_loss()

    H2_set = np.square(H_set) * np.tile(np.expand_dims(path_loss, axis=2), [1,1,int(Ns)])
    print('Path loss --------------------------', H2_set.shape)
    return H2_set
    
def Generate_environment():
    path_matrix = M*np.ones((n_y + 2*L, n_x + 2*L, maxM), dtype = np.int32)
    for i in range(L, n_y+L):
        for j in range(L, n_x+L):
            for l in range(maxM):
                path_matrix[i,j,l] = ((i-L)*n_x + (j-L))*maxM + l
    p_array = np.zeros((M, K), dtype = np.int32)
    for n in range(N):
        i = n//n_x
        j = n%n_x
        Jx = np.zeros((0), dtype = np.int32)
        Jy = np.zeros((0), dtype = np.int32)
        for u in range(i-L, i+L+1):
            v = 2*L+1-np.abs(u-i)
            jx = j - (v-i%2)//2 + np.linspace(0, v-1, num = v, dtype = np.int32) + L
            jy = np.ones((v), dtype = np.int32)*u + L
            Jx = np.hstack((Jx, jx))
            Jy = np.hstack((Jy, jy))
        for l in range(maxM):
            for k in range(c):
                for u in range(maxM):
                    p_array[n*maxM+l,k*maxM+u] = path_matrix[Jy[k],Jx[k],u]
    p_main = p_array[:,(c-1)//2*maxM:(c+1)//2*maxM]
    for n in range(N):
        for l in range(maxM):
            temp = p_main[n*maxM+l,l]
            p_main[n*maxM+l,l] = p_main[n*maxM+l,0]
            p_main[n*maxM+l,0] = temp
    p_inter = np.hstack([p_array[:,:(c-1)//2*maxM], p_array[:,(c+1)//2*maxM:]])
    p_array =  np.hstack([p_main, p_inter])             
     
    user = np.maximum(np.minimum(np.random.poisson(meanM, (N)), maxM), minM)
    user_list = np.zeros((N, maxM), dtype = np.int32)
    for i in range(N):
        user_list[i,:user[i]] = 1
    for k in range(N):
        for i in range(maxM):
            if user_list[k,i] == 0.:
                p_array = np.where(p_array == k*maxM+i, M, p_array)              
    p_list = list()
    for i in range(M):
        p_list_temp = list() 
        for j in range(K):
            p_list_temp.append([p_array[i,j]])
        p_list.append(p_list_temp)               
    return p_array, p_list, user_list

def Generate_path_loss():
    slope = 0.      #0.3
    p_tx = np.zeros((n_y, n_x))
    p_ty = np.zeros((n_y, n_x))
    p_rx = np.zeros((n_y, n_x, maxM))
    p_ry = np.zeros((n_y, n_x, maxM))   
    dis_rx = np.random.uniform(min_dis, max_dis, size = (n_y, n_x, maxM))
    phi_rx = np.random.uniform(-np.pi, np.pi, size = (n_y, n_x, maxM))    
    for i in range(n_y):
        for j in range(n_x):
            p_tx[i,j] = 2*max_dis*j + (i%2)*max_dis
            p_ty[i,j] = np.sqrt(3.)*max_dis*i
            for k in range(maxM):  
                p_rx[i,j,k] = p_tx[i,j] + dis_rx[i,j,k]*np.cos(phi_rx[i,j,k])
                p_ry[i,j,k] = p_ty[i,j] + dis_rx[i,j,k]*np.sin(phi_rx[i,j,k])
    dis = 1e10 * np.ones((M, K), dtype = dtype)
    lognormal = np.zeros((M, K), dtype = dtype)
    for k in range(N):
        for l in range(maxM):
            for i in range(c):
                for j in range(maxM):
                    if p_array[k*maxM+l,i*maxM+j] < M:
                        bs = p_array[k*maxM+l,i*maxM+j]//maxM
                        dx2 = np.square((p_rx[k//n_x][k%n_x][l]-p_tx[bs//n_x][bs%n_x]))
                        dy2 = np.square((p_ry[k//n_x][k%n_x][l]-p_ty[bs//n_x][bs%n_x]))
                        distance = np.sqrt(dx2 + dy2)
                        dis[k*maxM+l,i*maxM+j] = distance
                        std = 8. + slope * (distance - min_dis)
                        lognormal[k*maxM+l,i*maxM+j] = np.random.lognormal(sigma = std)                   
    path_loss = lognormal*pow(10., -(120.9 + 37.6*np.log10(dis))/10.)
    return path_loss

def compute_data_rate(bs, clusters, sic=False):
    data_rates = {}

    for curr_bs,curr_cluster in clusters.items():
        for clu in curr_cluster:
            #print('clu', clu)
            channel_gain, power_coeffient, distance = clu[2], clu[3], clu[1]  # numerator
            # SINR equation numerator
            numerator = 1 * power_coeffient * np.square(np.abs(channel_gain)) #1 signifies the use-bs indicator
            '----------------------------------------------------------------------------------------'
            intra_interference = intra_level_interference(users=curr_cluster, current_user=clu[0])

            inter_interference = 0
            for l in clusters:
                if l != curr_bs:
                    inter_interference += intra_level_interference(users=clusters[l], current_user=clu[0])

            # SINR equation denominator
            denominator = intra_interference + inter_interference + clu[4]**2
            '----------------------------------------------------------------------------------------'
            user_r = np.log2(1 + (numerator / denominator))
            data_rates[(curr_bs, clu[0])] = user_r
           # print('User {} has data rate {}'.format(user[0], user_r))
    return data_rates

def intra_level_interference(users, current_user):
    I = 0
    for i,u in enumerate(users):
        if u[0] != current_user: #signal to user can't be interference
            I += (1 * u[3] * np.square(np.abs(u[2])))
    return I
    
def Calculate_rate():
    maxC = 1000.
    P_extend = tf.concat([P, tf.zeros((1), dtype=dtype)], axis=0)
    P_matrix = tf.gather_nd(P_extend, p_list)
    path_main = tf.multiply(H2[:,0], P_matrix[:,0])
    path_inter = tf.reduce_sum(tf.multiply(H2[:,1:], P_matrix[:,1:]), axis=1)
    sinr = path_main / (path_inter + sigma2)
    sinr = tf.minimum(sinr, maxC)       #capped sinr
    rate = W * tf.log(1. + sinr)/np.log(2)
    rate_extend = tf.concat([rate, tf.zeros((1), dtype=dtype)], axis=0)
    rate_matrix = tf.gather_nd(rate_extend, p_list)
    sinr_norm_inv = H2[:,1:] / tf.tile(H2[:,0:1], [1,K-1])
    sinr_norm_inv = tf.log(1. + sinr_norm_inv)/np.log(2)# log representation
    reward = tf.reduce_sum(rate)
    return rate_matrix, sinr_norm_inv, P_matrix, reward

def Generate_state(rate_last, p_last, sinr_norm_inv):
    '''
    Generate state matrix
    ranking
    state including:
    1.rate[t-1]          [M,K]  rate_last
    2.power[t-1]         [M,K]  p_last
    3.sinr_norm_inv[t]   [M,K-1]  sinr_norm_inv
    '''
#    s_t = np.hstack([rate_last])
#    s_t = np.hstack([rate_last, sinr_norm_inv])

    indices1 = np.tile(np.expand_dims(np.linspace(0, M-1, num=M, dtype=np.int32), axis=1),[1,C])
    indices2 = np.argsort(sinr_norm_inv, axis = 1)[:,-C:]
    rate_last = np.hstack([rate_last[:,0:1], rate_last[indices1, indices2+1]])
    p_last = np.hstack([p_last[:,0:1], p_last[indices1, indices2+1]])
    sinr_norm_inv = sinr_norm_inv[indices1, indices2]
    s_t = np.hstack([rate_last, p_last, sinr_norm_inv])
    return s_t
    
def Variable(shape):  
    w = tf.get_variable('w', shape=shape, initializer = tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable('b', shape=[shape[-1]], initializer = tf.constant_initializer(0.01))    
    return w, b   
    
def Find_params(para_name):
    sets=[]
    for var in tf.trainable_variables():
        if not var.name.find(para_name):
            sets.append(var)
    return sets
    
#def Network(s, a, name):
#    with tf.variable_scope(name + '.0', reuse = reuse):
#        w,b = Variable([state_num, 64])
#        l = tf.nn.relu(tf.matmul(s, w)+b)
#    with tf.variable_scope(name + '.2', reuse = reuse):
#        w,b = Variable([64, power_num])
#        q_hat = tf.matmul(l, w) + b
#    r = tf.reduce_sum(tf.multiply(q_hat, a), reduction_indices = 1)
#    a_hat = tf.argmax(q_hat, 1)
#    list_var = Find_params(name)
#    return q_hat, a_hat, r, list_var
    
def Network(s, a, name):
    with tf.variable_scope(name + '.0', reuse = reuse):
        w,b = Variable([state_num, 128])
        l = tf.nn.relu(tf.matmul(s, w)+b)
    with tf.variable_scope(name + '.1', reuse = reuse):
        w,b = Variable([128, 64])
        l = tf.nn.relu(tf.matmul(l, w)+b)
    with tf.variable_scope(name + '.2', reuse = reuse):
        w,b = Variable([64, power_num])
        q_hat = tf.matmul(l, w) + b
    r = tf.reduce_sum(tf.multiply(q_hat, a), reduction_indices = 1)
    a_hat = tf.argmax(q_hat, 1)
    list_var = Find_params(name)
    return q_hat, a_hat, r, list_var
    
def Loss(y, r):
    cost = tf.reduce_mean(tf.square((y - r)))
    return cost
    
def Optimizer(cost, var_list):
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(learning_rate, global_step = global_step,
                                    decay_steps = EPISODE, decay_rate = 0.1)
    add_global = global_step.assign_add(1)
    with tf.variable_scope('opt', reuse = reuse):
        train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost, var_list = var_list)
    return train_op, add_global
        
def Save_store(s_t, a_t, r_t, s_next):
    r_t = np.tile(r_t, (M))
    p_t = np.zeros((M, power_num), dtype = dtype)
    p_t[range(M), a_t] = 1.
    for i in range(M):    
        replay_memory.append((s_t[i], p_t[i], r_t[i], s_next[i]))
    
def Sample():
    minibatch = random.sample(replay_memory, batch_size)
    batch_s = [d[0] for d in minibatch]
    batch_a = [d[1] for d in minibatch]
    batch_r = [d[2] for d in minibatch]
    return batch_s, batch_a, batch_r

def Select_action(sess, s_t, episode):
    if episode > OBSERVE:
        epsilon = INITIAL_EPSILON - (episode - OBSERVE) * (INITIAL_EPSILON - FINAL_EPSILON) / (EPISODE - OBSERVE) 
    elif episode <= OBSERVE:
        epsilon = INITIAL_EPSILON
    else:
        epsilon = 0.
    q_hat_ = sess.run(q_main, feed_dict={s: s_t}) #[M, power_num]
    # print('q_hat', q_hat_)
    best_action = np.argmax(q_hat_, axis = 1)
    random_index = np.array(np.random.uniform(size = (M)) < epsilon, dtype = np.int32)
    random_action = np.random.randint(0, high = power_num, size = (M))
    action_set = np.vstack([best_action, random_action])
    power_index = action_set[random_index, range(M)] #[M]
    power = power_set[power_index] # W
    return power, power_index  

def Step(p_t, H2_t):
    rate_last, sinr_norm_, p_last, reward_ = sess.run([rate_matrix, sinr_norm_inv, P_matrix, reward], 
        feed_dict={P: p_t, H2: H2_t, W: W_, sigma2: sigma2_})
    s_next = Generate_state(rate_last, p_last, sinr_norm_)
    return s_next, reward_  
    
def Experience_replay(sess):
    batch_s, batch_a, batch_r = Sample()
    sess.run(train_main, feed_dict = {s : batch_s, a : batch_a, y : batch_r})
    
def Save(weight_file):
    dict_name={}
    for var in list_main: 
        dict_name[var.name]=var.eval()
    scipy.io.savemat(weight_file, dict_name)
    
def Network_ini(theta):
    update=[]
    for var in list_main:
        update.append(tf.assign(tf.get_default_graph().get_tensor_by_name(var.name),tf.constant(np.reshape(theta[var.name],var.shape))))
    return update
    
def Initial_para():
    H2_set = Generate_H_set()
    s_next, _ = Step(np.zeros([M], dtype = dtype), H2_set[:,:,0])
    return H2_set, s_next  

def Plot_environment():
    if flag_fig:
        import matplotlib.pyplot as plt
        plt.close('all')
        p_tx = np.zeros((n_y, n_x))
        p_ty = np.zeros((n_y, n_x))
        p_rx = np.zeros((n_y, n_x, maxM))
        p_ry = np.zeros((n_y, n_x, maxM))   
        dis_r = np.random.uniform(min_dis, max_dis, size = (n_y, n_x, maxM))
        phi_r = np.random.uniform(-np.pi, np.pi, size = (n_y, n_x, maxM))    
        for i in range(n_y):
            for j in range(n_x):
                p_tx[i,j] = 2*max_dis*j + (i%2)*max_dis
                p_ty[i,j] = np.sqrt(3.)*max_dis*i
                for k in range(maxM):  
                    p_rx[i,j,k] = p_tx[i,j] + dis_r[i,j,k]*np.cos(phi_r[i,j,k])
                    p_ry[i,j,k] = p_ty[i,j] + dis_r[i,j,k]*np.sin(phi_r[i,j,k])
                    
        plt.close('all')
        plt.figure(1)

        for j in range(n_x):
            for i in range(n_y):
                for k in range(6):
                    x_t = [p_tx[i,j]+2/np.sqrt(3.)*max_dis*np.sin(np.pi/3*k), p_tx[i,j]+2/np.sqrt(3.)*max_dis*np.sin(np.pi/3*(k+1))]
                    y_t = [p_ty[i,j]+2/np.sqrt(3.)*max_dis*np.cos(np.pi/3*k), p_ty[i,j]+2/np.sqrt(3.)*max_dis*np.cos(np.pi/3*(k+1))]
                    plt.plot(x_t, y_t, color="black")
        for i in range(n_y):
            for j in range(n_x):
                for l in range(maxM):
                    if user_list[i*n_x+j,l]:      
                        rx = plt.scatter(p_rx[i,j,l], p_ry[i,j,l], marker='o', label='2', s=25, color='orange')
                        plt.text(p_rx[i,j,l]+0.03, p_ry[i,j,l]+0.03, '%d' %((i*n_x+j)*maxM+l), ha ='center', va = 'bottom', fontsize=12)
        tx = plt.scatter(p_tx, p_ty, marker='x', label='1', s=45)
        for i in range(n_y):
            for j in range(n_x):
                plt.text(p_tx[i,j]+0.1, p_ty[i,j]+0.1, '%d' %(i*n_x+j), ha ='center', va = 'bottom', fontsize=15, color = 'r')
        plt.legend([tx, rx], ["BS", "User"])
        plt.xlabel('X axis (km)')
        plt.ylabel('Y axis (km)')
        plt.axis('equal')
        plt.show()
#        
#        if train_hist: 
#            window = 101
#            train_hist = Smooth(np.array(train_hist), window)
#            plt.figure(2)
#            plt.plot(range(EPISODE), train_hist)
#            plt.show()
    
def Train_episode(sess, episode): 
    reward_dqn_list = list()
    H2_set, s_t = Initial_para()
    for step_index in range(int(Ns)):
        p_t, a_t = Select_action(sess, s_t, episode)
        s_next, r_ = Step(p_t, H2_set[:,:,step_index])
        Save_store(s_t, a_t, r_, s_next)
        if episode > OBSERVE:
            if step_index % train_interval == 0:
                Experience_replay(sess)
        s_t = s_next
        reward_dqn_list.append(r_)
    if episode > OBSERVE:
        sess.run(add_global)
    dqn_mean = sum(reward_dqn_list)/(Ns*M) # bps/Hz per link
    return dqn_mean
    
def Test_episode(sess, episode):
    '''
    1.DQN
    2.FP
    3.WMMSE
    4.Maximum Power
    5.Random Power
    '''
    reward_dqn_list = list()
    reward_fp_list = list()
    reward_wmmse_list = list()
    reward_mp_list = list()
    reward_rp_list = list()
    H2_set, s_t = Initial_para()
    for step_index in range(int(Ns)):
        q_hat_ = sess.run(q_main, feed_dict={s: s_t}) #[N, power_num]
        p_t = power_set[np.argmax(q_hat_, axis = 1)] # W
        s_next, r_ = Step(p_t, H2_set[:,:,step_index])
        s_t = s_next
        reward_dqn_list.append(r_)
        pa_alg_set.Load_data(H2_set[:,:,step_index], p_array)
        fp_alg, wmmse_alg, mp_alg, rp_alg = pa_alg_set.Calculate()
        r_fp = sess.run(reward, feed_dict={P: fp_alg, H2: H2_set[:,:,step_index], W: W_, sigma2: sigma2_})
        r_wmmse = sess.run(reward, feed_dict={P: wmmse_alg, H2: H2_set[:,:,step_index], W: W_, sigma2: sigma2_})
        r_mp = sess.run(reward, feed_dict={P: mp_alg, H2: H2_set[:,:,step_index], W: W_, sigma2: sigma2_})
        r_rp = sess.run(reward, feed_dict={P: rp_alg, H2: H2_set[:,:,step_index], W: W_, sigma2: sigma2_})
        reward_fp_list.append(r_fp)
        reward_wmmse_list.append(r_wmmse)
        reward_mp_list.append(r_mp)
        reward_rp_list.append(r_rp)      
    dqn_mean = sum(reward_dqn_list)/(Ns*M)
    fp_mean = sum(reward_fp_list)/(Ns*M)
    wmmse_mean = sum(reward_wmmse_list)/(Ns*M)
    mp_mean = sum(reward_mp_list)/(Ns*M)
    rp_mean = sum(reward_rp_list)/(Ns*M)
    return dqn_mean, fp_mean, wmmse_mean, mp_mean, rp_mean
    
def Train(sess):
    st = time.time()
    dqn_hist = list() 
    for k in range(1, EPISODE+1):
        dqn_hist.append(Train_episode(sess, k))
        if k % 100 == 0: 
            print("Episode(train):%d   DQN: %.3f   Time cost: %.2fs" 
                  %(k, np.mean(dqn_hist[-100:]), time.time()-st))
            st = time.time()
    Save(weight_file) 
    return dqn_hist
        
def Test(sess):
    sess.run(load)
    dqn_hist = list()
    fp_hist = list()
    wmmse_hist = list()
    mp_hist = list()
    rp_hist = list()
    for k in range(1, TEST_EPISODE+1):
        dqn_mean, fp_mean, wmmse_mean, mp_mean, rp_mean = Test_episode(sess, k)
        dqn_hist.append(dqn_mean)
        fp_hist.append(fp_mean)
        wmmse_hist.append(wmmse_mean)
        mp_hist.append(mp_mean)
        rp_hist.append(rp_mean)
    print("Test: DQN: %.3f  FP: %.3f  WMMSE: %.3f  MP: %.3f  RP: %.3f" 
          %(np.mean(dqn_hist), np.mean(fp_hist), np.mean(wmmse_hist), np.mean(mp_hist), np.mean(rp_hist)))
    print("%.2f, %.2f, %.2f, %.2f, %.2f" 
          %(np.mean(dqn_hist), np.mean(fp_hist), np.mean(wmmse_hist), np.mean(mp_hist), np.mean(rp_hist)))
    return dqn_hist, fp_hist, wmmse_hist, mp_hist, rp_hist
    
def Test_one(sess):
    sess.run(load)
    reward_dqn_list = list()
    reward_fp_list = list()
    reward_wmmse_list = list()
    reward_mp_list = list()
    reward_rp_list = list()
    H2_set, s_t = Initial_para()
    for step_index in range(int(Ns)):
        q_hat_ = sess.run(q_main, feed_dict={s: s_t}) #[N, power_num]
        p_t = power_set[np.argmax(q_hat_, axis = 1)] # W
        s_next, r_ = Step(p_t, H2_set[:,:,step_index])
        s_t = s_next
        reward_dqn_list.append(r_/M)
        pa_alg_set.Load_data(H2_set[:,:,step_index], p_array)
        fp_alg, wmmse_alg, mp_alg, rp_alg = pa_alg_set.Calculate()
        r_fp = sess.run(reward, feed_dict={P: fp_alg, H2: H2_set[:,:,step_index], W: W_, sigma2: sigma2_})
        r_wmmse = sess.run(reward, feed_dict={P: wmmse_alg, H2: H2_set[:,:,step_index], W: W_, sigma2: sigma2_})
        r_mp = sess.run(reward, feed_dict={P: mp_alg, H2: H2_set[:,:,step_index], W: W_, sigma2: sigma2_})
        r_rp = sess.run(reward, feed_dict={P: rp_alg, H2: H2_set[:,:,step_index], W: W_, sigma2: sigma2_})
        reward_fp_list.append(r_fp/M)
        reward_wmmse_list.append(r_wmmse/M)
        reward_mp_list.append(r_mp/M)
        reward_rp_list.append(r_rp/M)      
    if flag_fig:
        import matplotlib.pyplot as plt
        plt.figure(3)
        window = 11
        y = list()
        y.append(Smooth(np.array(reward_dqn_list), window))
        y.append(Smooth(np.array(reward_fp_list), window))
        y.append(Smooth(np.array(reward_wmmse_list), window))
        y.append(Smooth(np.array(reward_mp_list), window))
        y.append(Smooth(np.array(reward_rp_list), window))
        label=['DQN','FP','WMMSE','Random power','Maximal power']
        color = ['royalblue', 'orangered', 'lawngreen', 'gold', 'olive']
        linestyle = ['-', '--', '-.', ':', '--']
        p = list()
        for k in range(5):
            p_temp, = plt.plot(range(int(Ns)), y[k], color = color[k], linestyle = linestyle[k], label = label[k])
            p.append(p_temp)
        plt.legend(loc = 7)
        plt.xlabel('Time slot')
        plt.ylabel('Average rate (bps)')
        plt.grid()
        plt.show()
        print("Test: DQN: %.2f  FP: %.2f  WMMSE: %.2f  MP: %.2f  RP: %.2f" 
          %(np.mean(reward_dqn_list), np.mean(reward_fp_list), np.mean(reward_wmmse_list), np.mean(reward_mp_list), np.mean(reward_rp_list)))

def Bar_plot():
    import matplotlib.pyplot as plt
    plt.figure(4)
    data =  [[3.19, 2.97, 2.91, 2.00, 1.99],
             [2.22, 1.97, 2.01, 1.02, 1.02],
             [1.53, 1.33, 1.38, 0.51, 0.51],
             [1.22, 1.05, 1.10, 0.35, 0.35]]       
    data = np.array(data).T
    name_list = ['K=1','K=2','K=4','K=6']
    num = len(data[0])
    set_size = len(data)
    total_width = 3
    width = total_width / num
    label=['DQN','FP','WMMSE','Random power','Maximal power']
    color = ['royalblue', 'orangered', 'lawngreen', 'gold', 'olive']
    x = np.linspace(0, 16, num = num, dtype = dtype)
    plt.xticks(x+total_width/2, name_list)
    bar = list()
    for k in range(set_size):
        bar.append(plt.bar(x, data[k], width = width, label = label[k],fc = color[k]))
        x = x + width
    for k in range(set_size): 
        for n in range(num):    
            height = bar[k][n].get_height()
            if k == 4:
                plt.text(bar[k][n].get_x() + bar[k][n].get_width()/2, height+0.1, data[k][n], ha='center', va='bottom')
            else:
                plt.text(bar[k][n].get_x() + bar[k][n].get_width()/2, height, data[k][n], ha='center', va='bottom')
    plt.legend(loc = 1)
    plt.xlabel('User number per cell')
    plt.ylabel('Average rate (bps)')
    plt.show()

def Smooth(a, window):
    out0 = np.convolve(a, np.ones(window, dtype=np.int32),'valid')/window
    r = np.arange(1, window-1, 2)
    start = np.cumsum(a[:window-1])[::2]/r
    stop = (np.cumsum(a[:-window:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))

if __name__ == "__main__":
    par = argparse.ArgumentParser()
    par.add_argument("--base_station", default=(5, 5), type=int, help="base station location in the environment")
    par.add_argument("--base_station_2", default=(-5, -5), type=int, help="base station location in the environment")
    par.add_argument("--initial_k", default=3, type=bool, help="Initial max number of users per base station")
    par.add_argument("--user_locations", default=[(1, 1), (1, 3), (3, 2), (-4, -1), (-2, -4), (-1, -2)])
    args = par.parse_args()
    base_stations = [args.base_station, args.base_station_2]
    clusters_in_network = initialize_built_network(args, bs=base_stations)

    j,clu = 0,{}
    k_init = args.initial_k
    for bs,us in clusters_in_network.items():
        if j == 0:
            c_ = us[:k_init, :]
        else:
            c_ = us[k_init:, :]
            k_init += args.initial_k
        j += 1
        clu[bs] = c_

    # s = tf.compat.v1.placeholder(shape=[None, state_num], dtype=dtype)
    # a = tf.compat.v1.placeholder(shape=[None, power_num], dtype=dtype)
    # y = tf.compat.v1.placeholder(shape=[None], dtype=dtype)
    # q_main, a_main, r, list_main = Network(s, a, 'main')
    # cost = Loss(y, r)
    # train_main, add_global = Optimizer(cost, list_main)
    # print('q_main\n', cost)
    # print(tabulate(clusters_in_network, headers='keys', tablefmt='psql'))
    with tf.Graph().as_default():
        H2 = tf.compat.v1.placeholder(shape = [None, K], dtype = dtype)
        P = tf.compat.v1.placeholder(shape = [None], dtype = dtype)
        W = tf.compat.v1.placeholder(shape = [None], dtype = dtype)
        sigma2 = tf.compat.v1.placeholder(dtype = dtype)
        p_array, p_list, user_list = Generate_environment()
        rate_matrix, sinr_norm_inv, P_matrix, reward = Calculate_rate()
        data_rates = compute_data_rate(bs=base_stations, clusters=clu)
        # print(p_array,'\n\n', p_list, len(p_list), '\n\n', user_list, '\n\n', rate_matrix, '\n')
        # print(reward)
        s = tf.compat.v1.placeholder(shape = [None, state_num], dtype = dtype)
        a = tf.compat.v1.placeholder(shape = [None, power_num], dtype = dtype)
        y = tf.compat.v1.placeholder(shape = [None], dtype = dtype)
        q_main, a_main, r, list_main = Network(s, a, 'main')
        for i in list_main:
            print(i)
        cost = Loss(y, r)

        train_main, add_global = Optimizer(cost, list_main)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            train_hist = Train(sess)
            #scipy.io.savemat(hist_file, {'train_hist': train_hist})

#        Plot_environment()
#        Bar_plot()
#         pa_alg_set = PA_alg(M, K, maxP)
#         load = Network_ini(scipy.io.loadmat(weight_file))
#         with tf.Session() as sess:
#             tf.global_variables_initializer().run()
#             Test_one(sess)
# #            dqn_hist, fp_hist, wmmse_hist, mp_hist, rp_hist = Test(sess)
# #
#
#
            