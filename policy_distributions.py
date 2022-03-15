import numpy as np
import tensorflow as tf
import math
import scipy
import scipy.stats
import graphtools as gt
#import datatools as dt
import pdb
from scipy import sparse


# slow implmenetation of running average
class RunningStats(object):
    def __init__(self, N):
        self.N = N
        self.vals = []
        self.num_filled = 0

    def push(self, val):
        if self.num_filled == self.N:
            self.vals.pop(0)
            self.vals.append(val)
        else:
            self.vals.append(val)
            self.num_filled += 1

    def push_list(self, vals):
        num_vals = len(vals)

        self.vals.extend(vals)
        self.num_filled += num_vals
        if self.num_filled >= self.N:
            diff = self.num_filled - self.N
            self.num_filled = self.N
            self.vals = self.vals[diff:]

    def get_mean(self):
        return np.mean(self.vals[:self.num_filled])

    def get_std(self):
        return np.std(self.vals[:self.num_filled])

    def get_mean_n(self, n):
        start = max(0, self.num_filled-n)
        return np.mean(self.vals[start:self.num_filled])



class ProbabilityAction(object):
    def __init__(self, num_param, action_dim):
        """
        Class that implements various probabilistic actions

        Args:
            num_param (TYPE): number of parameters for a single distribution (P)
            example: single variable gaussian will have two parameters: mean and variance
        """
        self.num_param = num_param
        self.action_dim = action_dim

    def log_prob(self, params, selected_action):
        """
        Given a batch of distribution parameters
        and selected actions.
        Compute log probability of those actions

        Args:
            params (TYPE): N by (A*P) Tensor
            selected_action (TYPE): N by A Tensor

        Returns:
            Length N Tensor of log probabilities, N by (A*P) Tensor of corrected parameters
        """
        raise Exception("Not Implemented")

    def get_action(self, params):
        """

        Args:
            params (TYPE): N by (A*P) Numpy array

        Returns: N by A Numpy array of sampled actions
        """
        raise Exception("Not Implemented")

    def get_mean_action(self, params):
        """

        Args:
            params (TYPE): N by (A*P) Numpy array

        Returns: N by A Numpy array of sampled actions
        """
        raise Exception("Not Implemented")

    def random_action(self, N):
        """

        Args:
            params (TYPE): N by (A*P) Numpy array

        Returns: N by A Numpy array of sampled actions
        """
        raise Exception("Not Implemented")

#####################################################
########## Bernoulli  ###############################
######################################################
class BernoulliDistribution(ProbabilityAction):
    def __init__(self, action_dim, upper_bound):
        super(BernoulliDistribution,self).__init__(1, action_dim)
        self.upper_bound = upper_bound

    def log_prob(self, params, selected_action):

        p = tf.nn.sigmoid(params)*(.99-.01) + .01

        transmit_action = selected_action/self.upper_bound

        output = p

        log_probs2 = (transmit_action) * tf.log(p) + (1.0 - transmit_action)*tf.log(1-p)
        log_probs = tf.reduce_sum(log_probs2, axis=1)

        return log_probs, output

    def get_action(self, params):
        action = np.random.binomial(1,params)*self.upper_bound
        return action

    def random_action(self,N):
        action = np.random.binomial(1,0.3,size=(N,self.action_dim))*self.upper_bound
        return action

    def get_mean_action(self, params):
        action = (params>= 0.5).astype(float) * self.upper_bound
        return action

#####################################################
########## Gaussian  ###############################
######################################################
class GaussianDistribution(ProbabilityAction):
    def __init__(self, action_dim):
        super(GaussianDistribution,self).__init__(2, action_dim)

    def log_prob(self, params, selected_action):
        mean = tf.gather(params, np.array(range(self.action_dim)), axis=1, name='mean')
        std = tf.gather(params, np.array(range(self.action_dim, 2*self.action_dim)), axis=1, name='std')

        std = tf.nn.sigmoid(std) * .1 + .0001

        output = tf.concat([mean, std], axis=1)

        dist = tf.distributions.Normal(mean, std)
        log_probs = dist.log_prob(selected_action)

        log_probs = tf.reduce_sum(log_probs, axis=1)
        return log_probs, output

    def get_action(self, params):
        mean = params[:, :self.action_dim]
        std = params[:, self.action_dim:]

        action = scipy.stats.norm.rvs(loc=mean, scale=std)
        return action

    def get_mean_action(self, params):
        mean = params[:, :self.action_dim]
        std = params[:, self.action_dim:]

        action = mean
        return action

    def random_action(self, N):
        action = scipy.stats.norm.rvs(loc=0, scale=5, size=(N, self.action_dim) )
        return action

#####################################################
########## Beta  ###############################
######################################################
class BetaDistribution(ProbabilityAction):
    def __init__(self, action_dim, lower_bound, upper_bound):
        super(BetaDistribution,self).__init__(2, action_dim)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)

    def log_prob(self, params, selected_action):
        alpha = tf.gather(params, np.array(range(self.action_dim)), axis=1, name='alpha')
        beta = tf.gather(params, np.array(range(self.action_dim, 2*self.action_dim)), axis=1, name='beta')

        alpha = tf.nn.softplus(alpha) + 1
        beta = tf.nn.softplus(beta) + 1 # TODO: add a little epsilon?

        output = tf.concat([alpha, beta], axis=1)
        dist = tf.distributions.Beta(alpha, beta)

        log_probs = dist.log_prob(selected_action/self.upper_bound)

        log_probs = tf.reduce_sum(log_probs, axis=1)
        return log_probs, output

    def get_action(self, params):
        alpha = params[:, :self.action_dim]
        beta = params[:, self.action_dim:]

        N = params.shape[0]

        lower_bound = np.vstack([self.lower_bound for _ in range(N)]) + 1e-6
        upper_bound = np.vstack([self.upper_bound for _ in range(N)]) - 1e-6

        action = np.random.beta(alpha,beta)*(upper_bound - lower_bound) + lower_bound

        return action

    def get_mean_action(self, params):
        alpha = params[:, :self.action_dim]
        beta = params[:, self.action_dim:]

        N = params.shape[0]

        lower_bound = np.vstack([self.lower_bound for _ in range(N)]) + 1e-6
        upper_bound = np.vstack([self.upper_bound for _ in range(N)]) - 1e-6

        action = alpha / (alpha+ beta) *(upper_bound - lower_bound) + lower_bound

    def random_action(self, N):
        action = np.random.uniform(low=self.lower_bound, high=self.upper_bound,size=(N,self.action_dim))

        return action

#####################################################
########## Truncated Gaussian Distribution  ###############################
######################################################
class TruncatedGaussianDistribution(ProbabilityAction):
    def __init__(self, action_dim, lower_bound, upper_bound):
        super(TruncatedGaussianDistribution,self).__init__(2, action_dim)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)


    def log_prob(self, params, selected_action):
        mean = tf.gather(params, np.array(range(self.action_dim)), axis=1, name='mean')
        std = tf.gather(params, np.array(range(self.action_dim, 2*self.action_dim)), axis=1, name='std')

        mean = tf.nn.sigmoid(mean) * (self.upper_bound - self.lower_bound) + self.lower_bound
        std = tf.nn.sigmoid(std) * 0.5 + .01 # TODO: add a little epsilon?
       # std = 0.3

        output = tf.concat([mean, std], axis=1)


        dist = tf.distributions.Normal(mean, std)

        log_probs = dist.log_prob(selected_action) - tf.log(dist.cdf(self.upper_bound) - dist.cdf(self.lower_bound)) - tf.log(std)
        log_probs = tf.reduce_sum(log_probs, axis=1)
        return log_probs, output

    def get_action(self, params):
        mean = params[:, :self.action_dim]
        std = params[:, self.action_dim:]

        N = params.shape[0]

        lower_bound = (np.vstack([self.lower_bound for _ in range(N)]) - mean) / std
        upper_bound = (np.vstack([self.upper_bound for _ in range(N)]) - mean) / std

        action = scipy.stats.truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std)

        return action

    def get_mean_action(self, params):
        mean = params[:, :self.action_dim]
        std = params[:, self.action_dim:]

        action = mean

        return action

    def random_action(self, N):
        action = np.random.uniform(low=self.lower_bound, high=self.upper_bound,size=(N,self.action_dim))

        return action


#####################################################
########## Beta + Bernoulli  ###############################
######################################################
class BetaBernoulliDistribution(ProbabilityAction):
    def __init__(self, action_dim, lower_bound, upper_bound):
        super(BetaBernoulliDistribution,self).__init__(3/2, action_dim)
        self.beta = BetaDistribution(int(action_dim/2), lower_bound,upper_bound)
        self.bernoulli = BernoulliDistribution(int(action_dim/2), 1.0)

    def log_prob(self, params, selected_action):

        pt1 = int(self.action_dim)
        pt2 = int(3/2*self.action_dim)

        beta_p = tf.gather(params, np.array(range(pt1)), axis=1, name='gp')
        bernoulli_p = tf.gather(params, np.array(range(pt1, pt2)), axis=1, name='bp')

        pt1 = int(self.action_dim/2)
        pt2 = int(self.action_dim)


        rate_action = tf.gather(selected_action, np.array(range(pt1)), axis=1, name='rate')
        transmit_action = tf.gather(selected_action, np.array(range(pt1,pt2)), axis=1, name='transmit')

        log_probs1, output1 = self.beta.log_prob(beta_p,rate_action)
        log_probs2, output2 = self.bernoulli.log_prob(bernoulli_p,transmit_action)

        output = tf.concat([output1, output2], axis=1)

        log_probs = log_probs1 + log_probs2

        return log_probs, output

    def get_action(self, params):
        pt1 = int(self.action_dim)
        pt2 = int(3/2*self.action_dim)

        beta_p = params[:,:pt1]
        bernoulli_p = params[:,pt1:pt2]

        action1 = self.beta.get_action(beta_p)
        action2 = self.bernoulli.get_action(bernoulli_p)

        action = np.concatenate([action1, action2], axis=1)

        return action

    def get_mean_action(self, params):
        pt1 = int(self.action_dim)
        pt2 = int(3/2*self.action_dim)

        beta_p = params[:,:pt1]
        bernoulli_p = params[:,pt1:pt2]

        action1 = self.beta.get_mean_action(beta_p)
        action2 = self.bernoulli.get_mean_action(bernoulli_p)

        action = np.concatenate([action1, action2], axis=1)

        return action

    def random_action(self, N):

        action1 = self.beta.random_action(N)
        action2 = self.bernoulli.random_action(N)

        action = np.concatenate([action1, action2], axis=1)

        return action

#####################################################
########## Truncated Gaussian + Bernoulli  ###############################
######################################################
class TGaussianBernoulliDistribution(ProbabilityAction):
    def __init__(self, action_dim, lower_bound, upper_bound):
        super(TGaussianBernoulliDistribution,self).__init__(3/2, action_dim)
        self.gaussian = TruncatedGaussianDistribution(int(action_dim/2), lower_bound,upper_bound)
        self.bernoulli = BernoulliDistribution(int(action_dim/2), 1.0)

    def log_prob(self, params, selected_action):

        pt1 = int(self.action_dim)
        pt2 = int(3/2*self.action_dim)

        gaussian_p = tf.gather(params, np.array(range(pt1)), axis=1, name='gp')
        bernoulli_p = tf.gather(params, np.array(range(pt1, pt2)), axis=1, name='bp')

        pt1 = int(self.action_dim/2)
        pt2 = int(self.action_dim)


        rate_action = tf.gather(selected_action, np.array(range(pt1)), axis=1, name='rate')
        transmit_action = tf.gather(selected_action, np.array(range(pt1,pt2)), axis=1, name='transmit')

        log_probs1, output1 = self.gaussian.log_prob(gaussian_p,rate_action)
        log_probs2, output2 = self.bernoulli.log_prob(bernoulli_p,transmit_action)

        output = tf.concat([output1, output2], axis=1)

        log_probs = log_probs1 + log_probs2

        return log_probs, output

    def get_action(self, params):
        pt1 = int(self.action_dim)
        pt2 = int(3/2*self.action_dim)

        gaussian_p = params[:,:pt1]
        bernoulli_p = params[:,pt1:pt2]

        action1 = self.gaussian.get_action(gaussian_p)
        action2 = self.bernoulli.get_action(bernoulli_p)

        action = np.concatenate([action1, action2], axis=1)

        return action

    def get_mean_action(self, params):
        pt1 = int(self.action_dim)
        pt2 = int(3/2*self.action_dim)

        gaussian_p = params[:,:pt1]
        bernoulli_p = params[:,pt1:pt2]

        action1 = self.gaussian.get_mean_action(gaussian_p)
        action2 = self.bernoulli.get_mean_action(bernoulli_p)

        action = np.concatenate([action1, action2], axis=1)

        return action

    def random_action(self, N):

        action1 = self.beta.random_action(N)
        action2 = self.bernoulli.random_action(N)

        action = np.concatenate([action1, action2], axis=1)

        return action