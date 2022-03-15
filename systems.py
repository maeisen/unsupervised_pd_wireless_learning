import numpy as np
import scipy.io
import pdb
from scipy import sparse
from functools import partial

class ProbabilisticSystem(object):
    def __init__(self, state_dim, action_dim, constraint_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraint_dim = constraint_dim


    def sample(self, batch_size):
        raise Exception("Not yet implemented")

    def reward(self, state, action):
        """
        Args:
            state (TYPE):  state, N by state_dim matrix
            action (TYPE): resource allocation, N by action_dim matrix
        Returns:
            TYPE: N by 1 matrix of constraint violations
        """
        raise Exception("Not yet implemented")

    def constraint(self, state, action):
        """
        Args:
            state (TYPE):  state, N by state_dim matrix
            action (TYPE): resource allocation, N by action_dim matrix
        Returns:
            TYPE: N by constraint_dim matrix of constraint violations
        """
        raise Exception("Not yet implemented")

    def _exponential_sample(self, batch_size, mu = None):
        if mu == None:
            mu = self.mu
        samples = np.random.exponential(mu, size=(batch_size, self.state_dim))
        return samples
    def _uniform_sample(self,batch_size):
        samples = np.ones((batch_size, self.state_dim))
        return samples


#################################################################################
# Maximize sum-rate over interference network with avergae max power constraint #
#################################################################################
class SumCapacity_Interference(ProbabilisticSystem):
    def __init__(self, num_channels, A, pmax, mu=2, sigma=1, cell=False, assign=0):
        super(SumCapacity_Interference,self).__init__(num_channels, num_channels,1)

        self.pmax = pmax          # upper bound for average sum power
        self.mu = mu              # parameter of fast fading distribution
        self.sigma = sigma        # AWGN noise
        self.A = A                # path loss fading matrix

        self.cell = cell          # True if using multi-cell network
        self.assign = assign      # base station assignmnets if using multi-cell network


    def sample(self, batch_size):
        # No node state, sample uniform vector
        return self._uniform_sample(batch_size)

    def sample_graph(self,batch_size):
        # sample fast fading state
        samples = np.random.exponential(self.mu, size=(batch_size, self.A.shape[0], self.A.shape[1])) 
        PP = samples[None,:,:] * self.A
        A = PP[0]
        A[A < 0.001] = 0.0
        
        if self.cell:
            A2 = self.build_cell_graph(A,self.assign)
            return A2
        else:
            return A

    def build_cell_graph(self,A,assign):
        A2 = np.zeros((A.shape[0],A.shape[1],A.shape[1]))
        for i in np.arange(A.shape[1]):
            A2[:,i,:] = A[:,:,assign[i]]
        return A2

    def compute_capacity(self,state,action,S):
        state_m = S
        capacity_num = np.expand_dims(np.diagonal(state_m,axis1=1,axis2=2) * action, 2)
        capacity_den = np.matmul(state_m,np.expand_dims(action,2)) - capacity_num + self.sigma
        capacity_temp = np.log(capacity_num/capacity_den + 1)
        return capacity_temp

    # objective function - sum rate
    def reward(self, state, action, S, capacity=None):
        capacity2 = -np.sum(capacity,axis=1)
        return capacity2

    # constraint function - average sum power < pmax
    def constraint(self, state, action,S, capacity=None):
        lhs = np.array([np.sum(action, axis=1) - self.pmax])
        return lhs.T


#################################################################################
# Maximize sum-rate over interference network with varying rate constraint #
#################################################################################
class SumCapacity_Data(ProbabilisticSystem):
    def __init__(self, num_channels, A, pmax, mu=2, mu2 = 1, sigma=1, cell=False, assign=0):
        super(SumCapacity_Data,self).__init__(num_channels, num_channels,num_channels)

        self.pmax = pmax          # upper bound for average sum power
        self.mu = mu              # parameter of fast fading distribution
        self.mu2 = mu2            # parameter of node state distribution
        self.sigma = sigma        # AWGN noise
        self.A = A                # path loss fading matrix

        self.cell = cell          # True if using multi-cell network
        self.assign = assign      # base station assignmnets if using multi-cell network


    def sample(self, batch_size):
        # sample node state
        return self._exponential_sample(batch_size, mu=self.mu2)

    def sample_graph(self,batch_size):
        # sample fast fading state
        samples = np.random.exponential(self.mu, size=(batch_size, self.A.shape[0], self.A.shape[1])) 
        PP = samples[None,:,:] * self.A
        A = PP[0]
        A[A < 0.001] = 0.0
        
        if self.cell:
            A2 = self.build_cell_graph(A,self.assign)
            return A2
        else:
            return A

    def build_cell_graph(self,A,assign):
        A2 = np.zeros((A.shape[0],A.shape[1],A.shape[1]))
        for i in np.arange(A.shape[1]):
            A2[:,i,:] = A[:,:,assign[i]]
        return A2

    def compute_capacity(self,state,action,S):
        state_m = S
        capacity_num = np.expand_dims(np.diagonal(state_m,axis1=1,axis2=2) * action, 2)
        capacity_den = np.matmul(state_m,np.expand_dims(action,2)) - capacity_num + self.sigma
        capacity_temp = np.log(capacity_num/capacity_den + 1)
        return capacity_temp

    # objective function - sum log rate
    def reward(self, state, action, S, capacity=None):
        capacity2 = -np.sum(capacity,axis=1)
        return capacity2

    # constraint - minimum capacity    
    def constraint(self, state, action, S, capacity=None):
        lhs2 = state - capacity
        return np.reshape(lhs2,(-1,self.constraint_dim))


