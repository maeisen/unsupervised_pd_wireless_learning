import numpy as np
import tensorflow as tf
import math
import scipy
import scipy.stats
import graphtools as gt
#import datatools as dt
import pdb
from architecture import cnngs
from policy_distributions import *
from scipy import sparse


#######################################################
### Fully connected NN (multi-layer perceptron) model##
#######################################################
def mlp_model(state_dim, action_dim, batch_size=64, num_param=1, layers=[64, 32], archit='none'):
    with tf.variable_scope('policy'):
        state_input = tf.placeholder(tf.float32, [batch_size, state_dim], name='state_input')
        graph_input = tf.placeholder(tf.float32, [batch_size, state_dim, state_dim], name='graph_input')
        is_train = tf.placeholder(tf.bool, name="is_train")

        net = tf.reshape(graph_input,[-1, state_dim*state_dim])
        for idx, layer in enumerate(layers):
            net = tf.contrib.layers.fully_connected(net,
                layer,
                activation_fn=tf.nn.relu,
                scope='layer'+str(idx))

        output = tf.contrib.layers.fully_connected(net,
            action_dim*num_param,
            activation_fn=None,
            scope='output')
        output = tf.layers.batch_normalization(output, training=is_train)

    return state_input, graph_input, is_train, output


#######################################################
### Random edge graph neural network (GNN) model##
#######################################################
def regnn_model(state_dim, action_dim, batch_size = 64, num_param=1, layers=[4]*10, archit = 'no_pooling'):

    L = len(layers)

    A = np.eye(state_dim)
    GSO = 'Adjacency'

    if archit == "aggregation":
        A = [A, [4]]

    pool = 'nopool'
    gnn = cnngs(GSO, A,  # Graph parameters
            layers, [1]*(L-1) + [int(num_param)], [1]*L, [int(action_dim*num_param)], # Architecture
            'temp', './', archit = archit,decay_steps=1, pool=pool)

    
    graph_input = tf.placeholder(tf.float32, [batch_size, state_dim,state_dim], name='graph_input')
    S = graph_input

    state_input = tf.placeholder(tf.float32, [batch_size, state_dim], name='state_input')
    is_train = tf.placeholder(tf.bool, name="is_train")


    x = tf.expand_dims(state_input, 2)  # T x N x F=1 or N x F=1

    dropout = 1
   # T, N, F = x.get_shape()
    if archit == 'aggregation':
        maxP = min(S.shape[1],20)
        x = gt.collect_at_node3(x,S,[gnn.R],maxP)
    with tf.variable_scope('policy'):
        for l in range(L):
            with tf.variable_scope('gsconv{}'.format(l+1)):
                if gnn.archit == 'hybrid':
                    # Padding:
                    Tx, Nx, Fx = x.get_shape()
                    Tx, Nx, Fx = int(Tx), int(Nx), int(Fx)
                    if Nx < N:
                        x = tf.pad(x, [[0,0],[0,int(N-Nx)],[0,0]])
                    # Diffusion:
                    RR = [int(x) for x in range(gnn.R[l])]
                    x = gt.collect_at_node(x,S,RR,gnn.P[l])

                with tf.name_scope('filter'):
                    Tx, Nx, Fx = x.get_shape()
                    x = gnn.filter(x, l, S)
                  #  x = gnn.batch_norm(x)
                with tf.name_scope('pooling'):
                    x = gnn.pool(x, l)
                with tf.name_scope('nonlin'):
                    if l<L-1:
                        x = gnn.nonlin(x)
        T, N, F = x.get_shape()
       # x = tf.reshape(x, [int(T), int(N*F)])  # T x M (Recall M = N*F)
        x = tf.transpose(x, perm=[0,2,1])
        x = tf.reshape(x, [int(T), int(N*F)])  # T x M (Recall M = N*F)
        for l in range(len(gnn.M)-1):
            with tf.variable_scope('fc{}'.format(l+1)):
                x = gnn.fc(x, l)
                x = tf.nn.dropout(x, dropout)
        output = gnn.batch_norm(x)
        output = x

    return state_input, graph_input, is_train, output


##############################################
##############################################
##### Initialize learning policy ############
##############################################
### inputs: ##################################
## sys - system class (defined in systems.py) 
## model_builder - NN model (defined in reinforce_policy.py)
## distribution - policy distribution (defined in policy_distributions.py)
## sess - TF session
## lambda_lr - learning rate for dual parameter
## slack_lr - learning rate for counterfactual slack (if cf = True)
## theta_lr - learning rate for policy parameter
## batch_size - learning batch size
## cf - True if including counterfactual slack update
class ReinforcePolicy(object):
    def __init__(self,
        sys,
        model_builder=regnn_model,
        distribution=None,
        sess=None,
        lambda_lr = 0.005,
        slack_lr = 0.005,
        theta_lr = 5e-4,
        batch_size = 64,
        archit='no_pooling',
        cf = False):

        self.state_dim = sys.state_dim
        self.action_dim = sys.action_dim
        self.constraint_dim = sys.constraint_dim
        self.lambd = 1*np.ones((sys.constraint_dim, 1))
        self.slack = np.zeros((sys.constraint_dim,1))

        self.is_train = True

        self.archit = archit

        self.model_builder = model_builder
        self.dist = distribution
        self.batch_size = batch_size

        self.cf = cf

        self.lambd_lr = lambda_lr
        self.theta_lr = theta_lr
        self.slack_lr = slack_lr

        self.stats = RunningStats(64*100)
        self._build_model(self.state_dim, self.action_dim, model_builder, distribution)

        if sess == None:
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.InteractiveSession(config=config)
            tf.global_variables_initializer().run()
        else:
            self.sess = sess

    def _build_model(self, state_dim, action_dim, model_builder, distribution):

        self.state_input, self.graph_input, self.is_train, self.output = model_builder(state_dim, action_dim, num_param = self.dist.num_param, batch_size = self.batch_size, archit = self.archit)

        tvars = tf.trainable_variables()
        self.g_vars = [var for var in tvars if 'policy/' in var.name]

        self.selected_action = tf.placeholder(tf.float32, [None, action_dim], name='selected_action')

        self.log_probs, self.params = self.dist.log_prob(self.output, self.selected_action)

        self.cost = tf.placeholder(tf.float32, [None], name='cost')

        self.loss = self.log_probs * self.cost
        self.loss = tf.reduce_mean(self.loss)

        lr = self.theta_lr
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(lr)
            self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.g_vars)
        #    self.c_gs = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in self.gradients]
            self.optimize = self.optimizer.apply_gradients(self.gradients)


    def normalize_gso(self,S):
        norms = np.linalg.norm(S,ord=2,axis=(1,2))
        Snorm = S/norms[:,None,None]
        return Snorm

    def normalize_inputs(self,inputs):
        input2 = inputs - inputs.mean(axis=1).reshape(-1, 1)
        return input2

    def get_action(self, inputs, S, training=True):
        Sn = self.normalize_gso(S)
        c_inputs = self.normalize_inputs(inputs)

        fd = {self.state_input: c_inputs, self.graph_input: Sn, self.is_train: training}
        params = self.sess.run(self.params, feed_dict=fd)

        action = self.dist.get_action(params)


        return action

    def get_mean_action(self, inputs, S, training=False):
        Sn = self.normalize_gso(S)
        c_inputs = self.normalize_inputs(inputs)
        fd = {self.state_input: c_inputs, self.graph_input: Sn, self.is_train: training}

        params = self.sess.run(self.params, feed_dict=fd)
        action = self.dist.get_mean_action(params)

        return action

    def random_action(self, inputs, S):
        action = self.dist.random_action(self.batch_size)

        return action

    def learn(self, inputs, actions, reward, constraint, S):
        """
        Args:
            inputs (TYPE): N by m
            actions (TYPE): N by m
            reward (TYPE): N by 1
            constraint (TYPE): N by p

        Returns:
            TYPE: Description
        """
        cost = reward + np.dot(constraint, self.lambd)/self.constraint_dim
        cost = np.reshape(cost, (-1))

        self.stats.push_list(cost)
        cost_minus_baseline = cost - self.stats.get_mean()

        # improve policy weights
        # policy gradient step
        Sn = self.normalize_gso(S)
        c_inputs = self.normalize_inputs(inputs)
        fd = {self.state_input: c_inputs,
              self.graph_input: Sn,
              self.selected_action: actions,
              self.is_train: True,
              self.cost: cost_minus_baseline}
        loss, _ = self.sess.run([self.loss, self.optimize], feed_dict=fd)

        if np.any(np.isnan(loss)):
            pdb.set_trace()

        # gradient ascent step on lambda
        delta_lambd = np.mean(constraint, axis=0) - self.slack[:,0]
        delta_lambd = np.reshape(delta_lambd, (-1, 1))
        old_lambd = np.copy(self.lambd)
        self.lambd += delta_lambd * self.lambd_lr
        self.lambd = np.maximum(self.lambd, 0)

        # perform counterfactual slack update
        if self.cf:
            delta_slack = old_lambd - self.slack
            self.slack += delta_slack * self.slack_lr
            self.slack = np.maximum(self.slack,0)
        
      # decrease dual learning rate
      #  self.lambd_lr *= 0.9998

        return loss

