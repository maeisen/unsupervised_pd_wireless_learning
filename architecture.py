# Stop patching Deffe's code. Create a new one based on his.
# Version 01. 03/19/2018.

# Check which ones of these I really need.
import tensorflow as tf
import sklearn
import scipy.sparse
import numpy as np
import os, time, collections, shutil
import graphtools as gt
import pdb
import datatools as dt

# Common methods for all models

class base_model(object):

	def __init__(self):
		self.regularizers = []

	# High-level interface which runs the constructed computational graph.

	def predict(self, data, labels=None, sess=None):
		loss = 0
		size = data.shape[0]
		predictions = np.empty(size)
		sess = self._get_session(sess)
		for begin in range(0, size, self.batch_size):
			end = begin + self.batch_size
			end = min([end, size])

			batch_data = np.zeros((self.batch_size, data.shape[1]))
			tmp_data = data[begin:end,:]
			if type(tmp_data) is not np.ndarray:
				tmp_data = tmp_data.toarray()  # convert sparse matrices
			batch_data[:end-begin] = tmp_data
			feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}

			# Compute loss if labels are given.
			if labels is not None:
				batch_labels = np.zeros(self.batch_size)
				batch_labels[:end-begin] = labels[begin:end]
				feed_dict[self.ph_labels] = batch_labels
				batch_pred, batch_loss = sess.run(
						[self.op_prediction, self.op_loss], feed_dict)
				loss += batch_loss
			else:
				batch_pred = sess.run(self.op_prediction, feed_dict)

			predictions[begin:end] = batch_pred[:end-begin]

		if labels is not None:
			return predictions, loss * self.batch_size / size
		else:
			return predictions

	def evaluate(self, data, labels, sess=None):
		"""
		Runs one evaluation against the full epoch of data.
		Return the precision and the number of correct predictions.
		Batch evaluation saves memory and enables this to run on smaller
		GPUs.
		sess: the session in which the model has been trained.
		op: the Tensor that returns the number of correct predictions.
		data: size T x N
			T: number of samples
			N: number of nodes
		labels: size T
			T: number of samples
		"""
		t_process, t_wall = time.process_time(), time.time()
		predictions, loss = self.predict(data, labels, sess)
		#print(predictions)
		ncorrects = sum(predictions == labels)
		accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
		f1 = 100 * sklearn.metrics.f1_score(
				labels, predictions, average='weighted')
		string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
				accuracy, ncorrects, len(labels), f1, loss)
		if sess is None:
			string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(
					time.process_time()-t_process, time.time()-t_wall)
		return string, accuracy, f1, loss

	def fit(self, train_data, train_labels, val_data, val_labels):
		t_process, t_wall = time.process_time(), time.time()
		sess = tf.Session(graph=self.graph)
		shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
		writer = tf.summary.FileWriter(
				self._get_path('summaries'), self.graph)
		shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
		os.makedirs(self._get_path('checkpoints'), exist_ok = True)
		path = os.path.join(self._get_path('checkpoints'), 'model')
		sess.run(self.op_init)

		# Training.
		accuracies = []
		losses = []
		indices = collections.deque()
		num_steps = int(
				self.num_epochs * train_data.shape[0] / self.batch_size)

		for step in range(1, num_steps+1):
			# Be sure to have used all the samples before using one a
			# second time.
			if len(indices) < self.batch_size:
				indices.extend(np.random.permutation(train_data.shape[0]))
			idx = [indices.popleft() for i in range(self.batch_size)]

			batch_data, batch_labels = train_data[idx,:], train_labels[idx]

			if type(batch_data) is not np.ndarray:
				batch_data = batch_data.toarray()  # convert sparse matrices
			feed_dict = {self.ph_data: batch_data, 
					self.ph_labels: batch_labels, 
					self.ph_dropout: self.dropout}
			learning_rate, loss_average = sess.run(
					[self.op_train, self.op_loss_average], feed_dict)

			# Periodical evaluation of the model.
			if step % self.eval_frequency == 0 or step == num_steps:
				epoch = step * self.batch_size / train_data.shape[0]
				print('[{}] step {} / {} (epoch {:.2f} / {}):'.format(
						self.name, step, num_steps, epoch, self.num_epochs))
				print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(
						learning_rate, loss_average))
				string, accuracy, f1, loss = self.evaluate(
						val_data, val_labels, sess)
				accuracies.append(accuracy)
				losses.append(loss)
				print('  validation {}'.format(string))
				print('  time: {:.0f}s (wall {:.0f}s)'.format(
						time.process_time()-t_process, time.time()-t_wall))

				# Summaries for TensorBoard.
				summary = tf.Summary()
				summary.ParseFromString(sess.run(self.op_summary, feed_dict))
				summary.value.add(
						tag='validation/accuracy', simple_value=accuracy)
				summary.value.add(tag='validation/f1', simple_value=f1)
				summary.value.add(tag='validation/loss', simple_value=loss)
				writer.add_summary(summary, step)

				# Save model parameters (for evaluation).
				self.op_saver.save(sess, path, global_step=step)

		print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(
				max(accuracies), np.mean(accuracies[-10:])))
		writer.close()
		sess.close()

		t_step = (time.time() - t_wall) / num_steps
		
		return accuracies, losses, t_step

	def get_var(self, name):
		sess = self._get_session()
		var = self.graph.get_tensor_by_name(name + ':0')
		val = sess.run(var)
		sess.close()
		return val

	# Methods to construct the computational graph.

	def build_graph(self, N):
		"""Build the computational graph of the model."""
		self.graph = tf.Graph()
		with self.graph.as_default():

			# Inputs.
			with tf.name_scope('inputs'):
				self.ph_data = tf.placeholder(
						tf.float32, (self.batch_size, N), 'data')
				self.ph_labels = tf.placeholder(
						tf.int32, (self.batch_size), 'labels')
				self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

			# Model.
			op_logits = self.inference(self.ph_data, self.ph_dropout)
			self.op_logits = op_logits
			self.op_loss, self.op_loss_average = self.loss(
					op_logits, self.ph_labels, self.regularization)
			self.op_train = self.training(self.op_loss, self.learning_rate,
					self.decay_steps, self.decay_rate, self.momentum)
			self.op_prediction = self.prediction(op_logits)

			# Initialize variables, i.e. weights and biases.
			self.op_init = tf.global_variables_initializer()

			# Summaries for TensorBoard and Save for model parameters.
			self.op_summary = tf.summary.merge_all()
			self.op_saver = tf.train.Saver(max_to_keep=5)

		self.graph.finalize()

	def inference(self, data, dropout):
		"""
		It builds the model, i.e. the computational graph, as far as
		is required for running the network forward to make predictions,
		i.e. return logits given raw data.
		data: size T x N
			T: number of samples
			N: number of nodes/features
		training: we may want to discriminate the two, e.g. for dropout.
			True: the model is built for training.
			False: the model is built for evaluation.
		"""
		# TODO: optimizations for sparse data
		logits = self._inference(data, dropout)
		return logits

	def probabilities(self, logits):
		"""Return the probability of a sample to belong to each class."""
		with tf.name_scope('probabilities'):
			probabilities = tf.nn.softmax(logits)
			return probabilities

	def prediction(self, logits):
		"""Return the predicted classes."""
		with tf.name_scope('prediction'):
			prediction = tf.argmax(logits, axis=1)
			return prediction

	def loss(self, logits, labels, regularization):
		"""
		Adds to the inference model the layers required to generate loss.
		"""
		with tf.name_scope('loss'):
			with tf.name_scope('cross_entropy'):
				labels = tf.to_int64(labels)
				cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
				cross_entropy = tf.reduce_mean(cross_entropy)
			with tf.name_scope('regularization'):
				regularization *= tf.add_n(self.regularizers)
			loss = cross_entropy + regularization

			# Summaries for TensorBoard.
			tf.summary.scalar('loss/cross_entropy', cross_entropy)
			tf.summary.scalar('loss/regularization', regularization)
			tf.summary.scalar('loss/total', loss)
			with tf.name_scope('averages'):
				averages = tf.train.ExponentialMovingAverage(0.9)
				op_averages = averages.apply(
						[cross_entropy, regularization, loss])
				tf.summary.scalar('loss/avg/cross_entropy', 
						averages.average(cross_entropy))
				tf.summary.scalar('loss/avg/regularization', 
						averages.average(regularization))
				tf.summary.scalar('loss/avg/total', averages.average(loss))
				with tf.control_dependencies([op_averages]):
					loss_average = tf.identity(
							averages.average(loss), name='control')
			return loss, loss_average

	def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
		"""
		Adds to the loss model the Ops required to generate and apply
		gradients.
		"""
		
		with tf.name_scope('training'):
			# Learning rate.
			global_step = tf.Variable(0, name='global_step', trainable=False)
			if decay_rate != 1:
				learning_rate = tf.train.exponential_decay(
						learning_rate, global_step, decay_steps, 
						decay_rate, staircase=True)
			tf.summary.scalar('learning_rate', learning_rate)
			# Optimizer.
			if momentum == 0:
				if self.train_method == "SGD":
					optimizer = tf.train.GradientDescentOptimizer(learning_rate)
				elif self.train_method == "ADAM":
					optimizer = tf.train.AdamOptimizer(learning_rate)
			else:
				optimizer = tf.train.MomentumOptimizer(
						learning_rate, momentum)
			grads = optimizer.compute_gradients(loss)
			op_gradients = optimizer.apply_gradients(
					grads, global_step=global_step)
			# Histograms.
			for grad, var in grads:
				if grad is None:
					print('warning: {} has no gradient'.format(var.op.name))
				else:
					tf.summary.histogram(var.op.name + '/gradients', grad)
			# The op return the learning rate.
			with tf.control_dependencies([op_gradients]):
				op_train = tf.identity(learning_rate, name='control')
			return op_train

	# Helper methods.

	def _get_path(self, folder):
		path = os.path.dirname(os.path.realpath(__file__))
		return os.path.join(path, '..', folder, self.dir_name)

	def _get_session(self, sess=None):
		"""Restore parameters if no session given."""
		if sess is None:
			sess = tf.Session(graph=self.graph)
			filename = tf.train.latest_checkpoint(
					self._get_path('checkpoints'))
			self.op_saver.restore(sess, filename)
		return sess

	def _weight_variable(self, shape, regularization=True):
		initial = tf.truncated_normal_initializer(0, 0.1)
		var = tf.get_variable(
				'weights', shape, tf.float32, initializer=initial)
		if regularization:
			self.regularizers.append(tf.nn.l2_loss(var))
		tf.summary.histogram(var.op.name, var)
		return var

	def _bias_variable(self, shape, regularization=True):
		initial = tf.constant_initializer(0.1)
		var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
		if regularization:
			self.regularizers.append(tf.nn.l2_loss(var))
		tf.summary.histogram(var.op.name, var)
		return var

	def _batch_norm_variable(self):
		initial_gamma = tf.constant_initializer(1)
		initial_beta = tf.constant_initializer(0.0)

		gamma = tf.get_variable('gamma', [1], tf.float32, initializer=initial_gamma)
		beta = tf.get_variable('beta', [1], tf.float32, initializer=initial_beta)

		return gamma, beta

	def _conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

class cnngs(base_model):
	"""
	<DESCRIPTION>
	Parameters I'm including:
	
	Graph parameters:
	GSO: 'Laplacian', 'Adjacency'
	S: graph shift operator and selected nodes or list of GSOs
	
	Architecture:
	archit: 'clustering'
			'selection'
			'aggregation'
			'hybrid'
			'no-pooling'

	filter: 'chebyshev' (and all the others)
			'lsigf'
			'nvgf'
			'pmimo'
			'qmimo'
			'pqmimo'
	
	pool:	'max'
			'average'
			'median'
	
	nonlin: 'relu'
			'abs'
	
	K: Number of filter taps on each layer (list)
	F: Number of output features on each layer (list)
	a: Aggregating neighborhood (list)
	M: Number of hidden neurons on FC layer.
	
	Training:
	train_method: 'ADAM'
				  'SGD'
	num_epochs
	learning_rate
	decay_rate
	decay_steps
	momentum
	
	Regularization:
	regularization
	dropout
	batch_size
	eval_frequency
	
	File handling:
	name
	dir_name
	"""
	
	def __init__(self,
			# Non-default arguments:
			GSO, S, # Graph parameters
			K, F, a, M, # Architecture
			name, dir_name, # Organization
			# Default arguments:
			archit = 'selection', filter = 'lsigf', pool = 'maxpool', 
			nonlin = 'b1relu', # Architecture
			train_method = 'ADAM', num_epochs = 20, learning_rate = 0.1,
			decay_rate = 0.95, decay_steps = None, momentum =0.9, # Training
			regularization = 0, dropout = 1, batch_size = 100,
			eval_frequency = 200, # Regularization
			):
		
		super(cnngs,self).__init__() # Inheritance
		
		if isinstance(S,list):
			N1, N2 = S[0].shape
			assert N1 == N2
			N = [N1]
			del N1, N2
			if isinstance(S[1],list): # 'selection', 'aggregation' 
										# or 'hybrid'
				N = N + S[1]
				if len(S) == 3:
					if isinstance(S[2],list): # 'hybrid'
						P = S[2]
			else:
				for i in range(1,len(S)): # 'clustering'
					N1, N2 = S[i].shape
					assert N1 == N2
					N = N + [N1]
					del N1, N2
		else: # 'no-pooling'
			N1, N2 = S.shape
			assert N1 == N2
			N = N1
			del N1, N2

		L = len(K) # Number of (convolutional) layers
		
		# Check the parameters into the function are correct:
		assert len(F) == len(a) == L
		
		if not isinstance(S,list): # 'no-pooling'
			S = [S] * L
			N = [N] * (L+1)
		# Since S was not a list, this means that the same S is used 
		# throughout all the layers, so make it a list and copy it in each 
		# list element
		R = []
		P = []
		# Striping of graph parameters:
		if archit == 'clustering':
			SN = N
			origS = S
			N = [N[0]]
			S = [S[0]]
			for l in range(L):
				nextN = int(N[l]/a[l])
				nextIdx = int(np.nonzero(np.array(SN) == nextN)[0])
				N += [nextN]
				S += [origS[nextIdx]]
			assert len(S) >= L
			if pool == 'maxpool':
				pool = 'mpool1'
			R = []
			P = []
		elif archit == 'selection':
			assert len(N) >= L
			assert len(N) <= L+1
			if len(N) == L:
				print('    Obs.: Last convolutional layer will not use ' + \
						'downsampling; last value of a ignored.')
				N = N + [N[-1]]
			R = S[1] # Get all the lists of selected nodes
			assert np.all(R) <= N[0]
			P = []
		elif archit == 'hybrid':
			R = S[1]
			P = S[2]
			assert len(N) >= L
			assert len(N) <= L+1
			S = S[0]
			if pool == 'maxpool' and nonlin == 'b1relu':
				filter = 'regcnn_max_relu'
			pool = 'nopool'
			nonlin = 'identity'
		elif archit == 'aggregation':
			R = S[1] # Get the only selected node
			assert len(R) == 1
			R = R[0]
			assert R <= N[0]
			filter = 'regconv'
			if pool == 'maxpool':
				pool = 'mregpool'
			assert N[1] == S[1][0] == R
			S = S[0]
			N = [N[0]]
			for l in range(L):
				N = N + [int(N[l]/a[l])]
			P = []
		elif archit == 'no-pooling':
			pool = 'nopool'
			R = []
			P = []
		# If it is 'no-pooling' then S just has one element, which is the
		# matrix, thus it need not be asserted as in 'clustering'
		
#		print('  architecture/S = {}'.format(S))
		print('  architecture/L = {}'.format(L))
		print('  architecture/N = {}'.format(N))
		


		###########################################################
		###########################################################
		###########################################################
		### Print count of parameters
		n_param = 0
		print('CNNGS Architecture: {} ({})'.format(name, archit))
		print('  input: M_0 = N = {}'.format(N[0]))
		if archit == 'hybrid':
			for i in range(L):
				print('  l_{0}: gsconv_{0}'.format(i+1))
				lastF = F[i-1][-1] if i>0 else 1
				print('    input dimension : M_{0} = F_{0} N_{0}' \
						' = {1:2d} * {2:2d} = {3}'.
						format(i, lastF, N[i], lastF*N[i]))
				print('    output dimension: M_{0} = F_{0} N_{0}' \
						' = {1:2d} * {2:2d} = {3}'.
						format(i+1, F[i][-1], N[i+1], F[i][-1]*N[i+1]))
				l_param = 0
				print('    parameters_{} detail:'.format(i+1))
				for ll in range(len(K[i])):
					if i == 0 and ll == 0:
						llastF = 1
					elif ll == 0:
						llastF = F[i-1][-1]
					else:
						llastF = F[i][ll-1]
					print('      parameters_({0},{1}): K_({0},{1}) '\
						'F_({0},{1}) F_({0},{2})' \
						' = {3} * {4} * {5} = {6}'.
						format(i+1,ll+1, ll, K[i][ll], F[i][ll], llastF, 
							K[i][ll]*F[i][ll]*llastF))
					l_param += K[i][ll]*F[i][ll]*llastF
				print('    parameters = parameters_{0} N_{0} = '\
						'{1} * {2} = {3}'
						.format(i+1,l_param, N[i+1], l_param * N[i+1]))
				n_param += l_param * N[i+1]
			lastM = F[i][-1]*N[i+1]
		else:
			for i in range(L):
				print('  l_{0}: gsconv_{0}'.format(i+1))
				lastF = F[i-1] if i>0 else 1
				print('    input dimension : M_{0} = F_{0} N_{0}' \
						' = {1:2d} * {2:2d} = {3}'.
						format(i, lastF, N[i], lastF*N[i]))
				print('    output dimension: M_{0} = F_{0} N_{0}' \
						' = {1:2d} * {2:2d} = {3}'.
						format(i+1, F[i], N[i+1], F[i]*N[i+1]))
				print('    parameters: K_{0} F_{0} F_{1}' \
						' = {2} * {3} * {4} = {5}'.
						format(i+1, i, K[i], F[i], lastF, K[i]*F[i]*lastF))
				n_param += K[i]*F[i]*lastF
			#lastM = F[i]*N[i+1]
			lastM = 0
		for i in range(len(M)):
			fc_name = 'softmax' if i == len(M)-1 else 'fc_{}'.\
					format(i+1)
			print('  l_{}: {}'.format(L+i+1, fc_name))
			print('    input dimension : M_{0} = {1}'.
					format(L+i, lastM))
			print('    output dimension: M_{0} = {1}'.
					format(L+i+1, M[i]))
			print('    parameters: M_{0} M_{1} = {2} * {3} = {4}'.
					format(L+i+1, L+i, M[i], lastM, M[i]*lastM))
			n_param += M[i]*lastM
			lastM = M[i]
		print('  Total parameters = {}'.format(n_param))
		
		print(' ')
		###########################################################
		###########################################################
		###########################################################

		### Create list of S for lsigfs of selection.
		if archit == 'selection':
			S0 = S[0]
			S = []
			for i in range(L):
				thisS = gt.GSO_powers_selected(S0,N[i],K[i])
				S = S + [thisS]
		
		### Store attributes and bind operations.
		# Graph parameters
		self.GSO, self.S, self.N, self.R, self.P = GSO, S, N, R, P
		# Architecture
		self.archit = archit
		self.filter = getattr(self,filter)
		self.pool = getattr(self,pool)
		self.nonlin = getattr(self,nonlin)

		self.K, self.F, self.a, self.M = K, F, a, M
		# Training
		self.train_method, self.num_epochs, self.learning_rate = \
				train_method, num_epochs, learning_rate
		self.decay_rate, self. decay_steps, self.momentum = \
				decay_rate, decay_steps, momentum
		# Regularization
		self.regularization, self.dropout, self.batch_size =  \
				regularization, dropout, batch_size
		self.eval_frequency = eval_frequency
		# Organization
		self.name = name
		self.dir_name = dir_name

		# count number of parameters
		self.n_param = n_param
		
		# Build the computational graph.
		#self.build_graph(N[0])

	def chebyshev5(self, x, l):
		# x: data sample
		# l: convolutional layer {l = 0,...,L-1}
		# Copy values to use in this function
		S = self.S[l]
		K = self.K[l]
		Fout = self.F[l]
		T, N, Fin = x.get_shape()
		T, N, Fin = int(T), int(N), int(Fin)
		# T: number of training samples
		# N: number of nodes (intrinsic dimension)
		# Fin: number of input features
		# TODO: Work with S a list as well
		S = scipy.sparse.csr_matrix(S)
#		S = gt.rescale_L(S, lmax=2)
		S = S.tocoo()
		indices = np.column_stack((S.row, S.col))
		S = tf.SparseTensor(indices, S.data, S.shape)
		S = tf.sparse_reorder(S)
		# Transform to Chebyshev basis
		x0 = tf.transpose(x, perm=[1, 2, 0])  # N x Fin x T
		x0 = tf.reshape(x0, [N, Fin*T])  # N x Fin*T
		x = tf.expand_dims(x0, 0)  # 1 x N x Fin*T
		def concat(x, x_):
			x_ = tf.expand_dims(x_, 0)  # 1 x N x Fin*T
			return tf.concat([x, x_], axis=0)  # K x N x Fin*T
		if K > 1:
			x1 = tf.sparse_tensor_dense_matmul(S, x0)
			x = concat(x, x1)
		for k in range(2, K):
			x2 = 2 * tf.sparse_tensor_dense_matmul(S, x1) - x0  # N x Fin*T
			x = concat(x, x2)
			x0, x1 = x1, x2
		x = tf.reshape(x, [K, N, Fin, T])  # K x N x Fin x T
		x = tf.transpose(x, perm=[3,1,2,0])  # T x N x Fin x K
		x = tf.reshape(x, [T*N, Fin*K])  # T*N x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank 
        	# per feature pair.
		H = self._weight_variable([Fin*K, Fout], regularization=False)
		x = tf.matmul(x, H)  # T*N x Fout
		return tf.reshape(x, [T, N, Fout])  # T x N x Fout

	def lsigf(self, x, l, S):
		# x: data sample
		# l: convolutional layer {l = 0,...,L-1}
		# Copy values to use in this function
		K = self.K[l]
		Fout = self.F[l]
		T2, N, Fin = x.get_shape()
		T2 = tf.shape(x)[0]
		T = tf.shape(S)[0]

		if not isinstance(S,list):
			S = gt.GSO_powers_selected3(S,N,K)
		assert len(S) == K

		#S0 = S[0] # Needed to check whether it is the identity or not.


		# tensorS = []
		# # Create sparse tensor matrix for each power of the GSO
		# for it in range(len(S)):
		# 	thisS = S[it]
		# 	zero = tf.constant(0, dtype=tf.float32)
		# 	where = tf.not_equal(thisS, zero)
		# 	indices = tf.where(where)
		# 	values = tf.gather_nd(thisS, indices)
		# 	thisS = tf.SparseTensor(indices, values, thisS.shape)

		# 	thisS = tf.cast(tf.sparse_reorder(thisS),tf.float32)
		# 	tensorS = tensorS + [thisS]
		# S = tensorS
		# del thisS, tensorS
		

		# Filter
	
		#x0 = tf.transpose(x, perm=[1, 2, 0])  # N x Fin x T
		# N matrices of Fin x T: each matrix contains Fin features at node
		# N for every training sample T
		#x0 = tf.reshape(x0, [N, Fin*T2])  # N x Fin*T

		x0 = tf.identity(x) # T x N X Fin
		x = tf.expand_dims(x, 0)  # 1 x N x Fin*T
		# 1 matrix of size N x Fin*T
		def concat(x, x_):
			# Takes matrix x_ of size N x Fin*T and expands one dimension
			x_ = tf.expand_dims(x_, 0)  # 1 x N x Fin*T
			# Adds it to the other K-1 matrices to get
			return tf.concat([x, x_], axis=0)  # K x N x Fin*T
#		if K > 1:
		for k in range(1,K):

			# So, we start with x0 as in N x Fin*T matrix.
			#x1 = tf.sparse_tensor_dense_matmul(S[k], x0) # T x N x Fin'
			x1 = tf.matmul(S[k], x0)
			x = concat(x, x1)

		x = tf.transpose(x, perm=[3,1,2,0])  # T x N x Fin x K
		x = tf.reshape(x, [T*N, Fin*K])  # T*N x Fin*K
		H = self._weight_variable([Fin*K, Fout], regularization=True)
		x = tf.matmul(x, H)  # T*N x Fout
		return tf.reshape(x, [T, N, Fout])  # T x N x Fout

	def regconv(self,x,l,S):
		K = self.K[l]
		Fout = self.F[l]
		T, N, Fin = x.get_shape()
		T, N, Fin = int(T), int(N), int(Fin)
		H = self._weight_variable([K, Fin, Fout], regularization=False)
		x = tf.nn.conv1d(x, H, stride = 1, padding = 'SAME')
		return x
	
	def regcnn_max_relu(self,x,l):
		K = self.K[l]
		F = self.F[l]
		a = self.a[l]
		R = self.R[l]
		P = self.P[l]
		L = len(K)
		T, RP, Fin = x.get_shape()
		T, RP, Fin = int(T), int(RP), int(Fin)
		assert RP == R*P
		del RP
		z = x
		for r in range(R):
			xr = tf.slice(z, [0,P*r,0], [T,P,Fin]) # T x P x Fin
			Fin_r = Fin
			for ll in range(L):
				with tf.variable_scope('node{}regcnn{}'.format(r,ll+1)):
					with tf.name_scope('conv'):
						H = self._weight_variable([K[ll], Fin_r, F[ll]],
								regularization = False)
						xr = tf.nn.conv1d(xr, H, 
								stride = 1, padding = 'SAME') # T x P x Fout
					with tf.name_scope('pooling'):
						if ll == L-1:
							# Last layer _has_ to pool to one node (and 
							#	F features)
							_, Px, _ = xr.get_shape()
							Px = int(Px)
							a[ll] = Px
						if a[ll] > 1:
							xr = tf.expand_dims(xr, axis = 2) # TxPx1xF
							xr = tf.nn.max_pool(xr, ksize=[1,a[ll],1,1],
									strides=[1,a[ll],1,1], padding='SAME')
							xr = tf.squeeze(xr, [2])
					with tf.name_scope('nonlin'):
						b = self._bias_variable([1, 1, int(F[ll])],
								regularization=False)
						xr = tf.nn.relu(xr + b)
				Fin_r = F[ll]
			if r == 0:
				x = xr
			else:
				x = tf.concat([x,xr], axis = 1)
		return x	
	
	# Defferrard max pooling:
	# TODO: Can I combine both max poolings in one?
	def mpool1(self, x, l):
		p = int(self.N[l]/self.N[l+1])
		"""Max pooling of size p. Should be a power of 2."""
		if p > 1:
			x = tf.expand_dims(x, 3)  # T x N x F x 1
			x = tf.nn.max_pool(
					x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
			x = tf.squeeze(x, [3])  # T x N/p x F
			return x
		else:
			return x
	
	def maxpool(self, x, l):
		# x: T x N x F matrix for sample
		# l: convolutional layer {l = 0,...,L-1}
		# Copy useful values
		if self.N[l] == self.N[l+1]:
			return x
		S = self.S[0][1] # Original GSO
		N  = S.shape[0] # Number of nodes in original GSO
		a = self.a[l] # Neighbors to consider
		Nout = self.N[l+1] # Number of nodes to keep (consecutive nodes,
		# because GSO was correctly ordered)
		T, Nin, F = x.get_shape()
		T = tf.shape(x)[0]
		#T, Nin, F = int(T), int(Nin), int(F)
		# zero-padding:
		if Nin < N:
			Nzeros = N-Nin
			x = tf.pad(x, [[0,0],[0,Nzeros],[0,0]])
		# Apply pooling only in the nodes that will be kept (first Nout)
		x0 = tf.slice(x, [0,0,0], [T,Nout,F])
		# Create space to put the value of the neighbors
		x0 = tf.expand_dims(x0, 2) # T x N x 1 x F
		# Look for neighborhood:
		R = gt.neighborhood_reach(S,Nout,a) # Indices for the alpha-hop 
		# Get rid of diagonal elements (its own value, we already have that)
		for r in range(Nout):
			R[r] = [R[r][rr] for rr in range(len(R[r])) if R[r][rr] != r]
		# neighborhood of each of the Nout nodes
		Rmax = max([len(R[i]) for i in range(Nout)]) # max. length of neighb.
		# Complete with zeros (same node, so as not to alter the max) if 
		# neighbors are "missing" (if it has less neighbors)
		for r in range(len(R)):
			if len(R[r]) < Rmax:
				R[r] = R[r] + [r] * (Rmax-len(R[r]))
		xg = tf.gather(x, R, axis = 1) # get neighboring nodes, for all 
		# features and all training samples T
		# xg has dimension T x N x Rmax x F
		x0 = tf.concat([x0,xg], axis=2) # T x N x (Rmax+1) x F
		x = tf.nn.max_pool(x0, 
				ksize = [1,1,Rmax+1,1], 
				strides = [1,1,Rmax+1,1], 
				padding = 'SAME') # T x N x 1 x F
		return tf.squeeze(x, [2]) # T x N x F
	
	def mregpool(self,x,l):
		a = self.a[l] # Neighbors to consider
		Nout = self.N[l+1] # Number of nodes to keep (consecutive nodes,
		# because GSO was correctly ordered)
		T, Nin, F = x.get_shape()
		T, Nin, F = int(T), int(Nin), int(F)
		if Nin/Nout < 1:
			a = 1
		elif a != int(Nin/Nout):
			a = int(Nin/Nout)
		if a == 1:
			return x
		x = tf.expand_dims(x, axis = 2) # T x N x 1 x F
		x = tf.nn.max_pool(
				x, ksize=[1,a,1,1], strides=[1,a,1,1], padding='SAME')
		x = tf.squeeze(x, axis=[2])
		return x
		
	def nopool(self, x, l):
		return x
	
	def b1relu(self, x):
		"""Bias and ReLU. One bias per filter."""
		"""One bias per feature, same for all nodes."""
		T, N, F = x.get_shape()
		b = self._bias_variable([1, 1, int(F)], regularization=False)
		return tf.nn.relu(x + b)

	def b2relu(self, x):
		"""Bias and ReLU. One bias per vertex per filter."""
		T, N, F = x.get_shape()
		b = self._bias_variable([1, int(N), int(F)], regularization=False)
		return tf.nn.relu(x + b)
	
	def b1abs(self, x):
		"""Bias and Abs value. One bias per filter."""
		T, N, F = x.get_shape()
		b = self._bias_variable([1, 1, int(F)], regularization=False)
		return tf.abs(x + b)

	def b2abs(self, x):
		"""Bias and Abs value. One bias per vertex per filter."""
		T, N, F = x.get_shape()
		b = self._bias_variable([1, int(N), int(F)], regularization=False)
		return tf.abs(x + b)
	
	def identity(self,x):
		return x
	
	def fc(self, x, l, relu=True):
		# x: data sample
		# l: convolutional layer {l = 0,...,L-1}
		# Copy values to use in this function
		Mout = self.M[l]
		"""Fully connected layer with Mout features."""
		T, Min = x.get_shape()
		A = self._weight_variable([int(Min), Mout], regularization=True)
		b = self._bias_variable([Mout], regularization=True)
		x = tf.matmul(x, A) + b
		return tf.nn.relu(x) if relu else x

	def batch_norm(self, x):
		# x: data sample
		# l: convolutional layer {l = 0,...,L-1}
		# Copy values to use in this function
		mu, var = tf.nn.moments(x, axes=[0])
		X_norm = (x - mu) / tf.sqrt(var + 1e-8)

		gamma, beta = self._batch_norm_variable()

		out = gamma * X_norm + beta
		return out
	
	def _inference(self, x, dropout):
		# Graph convolutional layers.
		# x is of size T x N:
		#	T: number of training samples
		#	N: number of nodes
		x = tf.expand_dims(x, 2)  # T x N x F=1
		T, N, F = x.get_shape()
		if self.archit == 'aggregation':
			maxP = min((self.S).shape[0],20)
			x = gt.collect_at_node(x,self.S,[self.R],maxP)
		L = len(self.K) # Number of (convolutional) layers
		for l in range(L):
			with tf.variable_scope('gsconv{}'.format(l+1)):
				if self.archit == 'hybrid':
					# Padding:
					Tx, Nx, Fx = x.get_shape()
					Tx, Nx, Fx = int(Tx), int(Nx), int(Fx)
					if Nx < N:
					 	x = tf.pad(x, [[0,0],[0,int(N-Nx)],[0,0]])
					# Diffusion:
					RR = [int(x) for x in range(self.R[l])]
					x = gt.collect_at_node(x,self.S,RR,self.P[l])
				with tf.name_scope('filter'):
					x = self.filter(x, l)
				with tf.name_scope('pooling'):
					x = self.pool(x, l)
				with tf.name_scope('nonlin'):
					x = self.nonlin(x)
		T, N, F = x.get_shape()
		x = tf.reshape(x, [int(T), int(N*F)])  # T x M (Recall M = N*F)
		for l in range(len(self.M)-1):
			with tf.variable_scope('fc{}'.format(l+1)):
				x = self.fc(x, l)
				x = tf.nn.dropout(x, dropout)
		# Logits linear layer, i.e. softmax without normalization.
		with tf.variable_scope('logits'):
			x = self.fc(x, len(self.M)-1, relu=False)
		return x
