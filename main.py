import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
import pdb
from scipy import sparse
import pickle

from systems import *
from networks import *
from reinforce_policy import ReinforcePolicy
from reinforce_policy import *
import graphtools as gt


#######################
## UTILITY FUNCTIONS ##
#######################
def moving_average(data, window=10):
    cumsum = np.cumsum(data)
    moving_sum = cumsum[window:] - cumsum[:-window]
    moving_avg = moving_sum / window


    moving_avg = np.concatenate((np.zeros(window), moving_avg))
    moving_avg[:window] = cumsum[:window] / window
    return moving_avg

def sample_graph(self,batch_size,mu,A,state_dim):
    samples = np.random.exponential(mu, size=(batch_size, state_dim, state_dim)) 
    samples = (samples + np.transpose(samples,(0,2,1)))/2
    PP = samples[None,:,:] * A
    return PP[0]

def sample_one_graph(self,mu,A,state_dim):
    samples = np.random.exponential(mu, size=(state_dim, state_dim)) 
    samples = (samples + np.transpose(samples))/2
    return samples * A

def run_sys(sys, policy, num_iter, batch_size=64,save_file="test8.mat"):

    history_dict = {'lambd': [],
                    's': [],
                       'f0': [],
                       'f1': [],
                       'A': [],
                       'p': [],
                       'S': [],
                       'x': []}
    history_dict['A'].append(sys.A)

    for k in range(num_iter):
        

        if k%1000 == 0:
            print("Iteration " + str(k))

        # sample state and actions
        states = sys.sample(batch_size)
        graph_state = sys.sample_graph(batch_size)
        actions = policy.get_action(states, graph_state)

        # compute reward and constraint violation
        capacity = sys.compute_capacity(states,actions,graph_state)
        f0 = sys.reward(states,actions,graph_state, capacity=capacity)
        f1 = sys.constraint(states,actions,graph_state, capacity=capacity)

        # save training data from iteration k
        if k%1000 == 0:
            history_dict['S'].append(graph_state)
            history_dict['p'].append(actions)
            history_dict['x'].append(states)
        if k%100 == 0:
            history_dict['lambd'].append(policy.lambd)
            history_dict['s'].append(policy.slack)
            history_dict['f0'].append(np.mean(f0))
            history_dict['f1'].append(np.mean(f1,axis=0))
  
        # performing learning step
        policy.learn(states, actions, f0, f1, graph_state)

    #save_policy(policy,save_file)

    return history_dict

#############################################
############ Save NN architecture ###############
##############################################
def save_policy(policy, filename):
    data_save = {}
    index = 0
        
    tvars = tf.trainable_variables()
    num_layers = int(len(tvars)/2)
    variable_names = []
    for i in np.arange(num_layers):
        variable_names.append("weight"+str(i))
        variable_names.append("bias"+str(i))
    variable_names.append("weight"+str(num_layers))

    tvars_vals = policy.sess.run(tvars)
    for var, val in zip(tvars, tvars_vals):
        data_save[variable_names[index]] = val
        index +=1

    scipy.io.savemat(filename,data_save)

#############################################
############ Save simulation data ###############
##############################################
def save_data(data, filename):
    scipy.io.savemat(filename, data)



####################
## TEST FUNCTIONS ##
####################
def interference_test():
    mu = 2 # parameter for exponential distribution of wireless channel distribution
    num_channels = 10 # number of wireless channels (action_dim and state_dim)
    pmax = num_channels # maximum power allocation

    batch_size = 100

    mu2 = 4

    pl = 2.2
    A = build_adhoc_network(num_channels,pl)

    sys = SumCapacity_Interference(num_channels, A, pmax=pmax,mu=mu, sigma=1)

    distribution = BernoulliDistribution(sys.action_dim, upper_bound=4.0)
    lambda_lr = .01
    theta_lr = 3e-3
    reinforce_policy = ReinforcePolicy(sys,
        model_builder=regnn_model,
        distribution=distribution, lambda_lr = lambda_lr, theta_lr = theta_lr, batch_size = batch_size)

    history_dict = run_sys(sys, reinforce_policy, 40000, batch_size=batch_size, save_file = "gnn5_" + str(num_channels) + ".mat")

    save_data(history_dict, "test3.mat")


def cellular_test():
    mu = 1 # parameter for exponential distribution of wireless channel distribution

    n = 5
    k = 10
    num_channels = int(n*k)

    pmax = num_channels # maximum power allocation

    batch_size = 64

    pl = 2.2
    A, assign = build_cellular_network(n,k,pl)


    sys = SumCapacity_Interference(num_channels, A, pmax=pmax,mu=mu, sigma=1, cell = True, assign = assign)
    distribution = BernoulliDistribution(sys.action_dim, upper_bound=4.0)
    lambda_lr = 0.001
    theta_lr = 5e-3
    reinforce_policy = ReinforcePolicy(sys, model_builder=regnn_model, distribution=distribution, lambda_lr = lambda_lr, theta_lr = theta_lr, batch_size = batch_size)

    history_dict = run_sys(sys, reinforce_policy, 40000, batch_size=batch_size, save_file = "cell_network.mat")

    save_data(history_dict, "wireless_net_interference_data_gnn_50_34.mat")


def aggregation_test():
    mu = 2 # parameter for exponential distribution of wireless channel distribution
    num_channels = 10 # number of wireless channels (action_dim and state_dim)
    pmax = num_channels # maximum power allocation

    batch_size = 100

    mu2 = 4


    GSO = 'Adjacency'
    pl = 2.2
    A = build_adhoc_network(num_channels,pl)

    sys = SumCapacity_Interference(num_channels, A, pmax=pmax,mu=mu, sigma=1)

  #  distribution = TruncatedGaussianDistribution(sys.action_dim, 
  #      lower_bound=np.ones(sys.action_dim)*0.0, 
  #      upper_bound=np.ones(sys.action_dim)*8.0)

    distribution = BernoulliDistribution(sys.action_dim, upper_bound=4.0)
    lambda_lr = .001
    theta_lr = 5e-4
    reinforce_policy = ReinforcePolicy(sys,
        model_builder=gnn_model,
        distribution=distribution, lambda_lr = lambda_lr, theta_lr = theta_lr, batch_size=batch_size, archit = "aggregation")


    history_dict = run_sys(sys, reinforce_policy, 20000, batch_size=batch_size, save_file = "agnn_" + str(num_channels) + ".mat",  subsample = subsample)

    save_data(history_dict, "wireless_net_distributed_data_gnn_" + str(num_channels) +".mat")

    


if __name__ == '__main__':
    import argparse
    import sys

    rn = np.random.randint(2**20)
    rn1 = np.random.randint(2**20)
    tf.set_random_seed(rn)
    np.random.seed(rn1)


    interference_test()
   # fairness_test()
    #cellular_test()
#



