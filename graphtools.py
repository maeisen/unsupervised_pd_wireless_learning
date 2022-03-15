import numpy as np
import scipy.sparse
import sklearn.metrics
import sklearn.neighbors
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import scipy.spatial.distance
import sys
import datatools
import tensorflow as tf

def create_ER(n,p):
	# Function provided in MATLAB(R) by Dr. S. Segarra.

	FLAG_CONNECTED = 0
	while FLAG_CONNECTED == 0:
		G = np.random.rand(n,n)
		G = G < p
		G = G.astype(float)
		G = np.triu(G,1)
		G = G + G.conj().T
		L = np.diag(np.sum(G,axis=0))-G
		D,_ = np.linalg.eig(L)
		n_connected_components = (np.absolute(D)<=1e-6)
		if np.sum(n_connected_components.astype(float)) == 1.:
			FLAG_CONNECTED = 1
	return G

def create_SBM(n,p):
	n_c = n.shape[0] # Number of communities
	FLAG_CONNECTED = 0
	P = np.zeros([np.sum(n),np.sum(n)])
	for it1 in range(n_c):
		P_row1 = np.sum(n[0:it1])
		P_row2 = np.sum(n[0:(it1+1)])
		P[P_row1:P_row2,P_row1:P_row2] = np.tile(
				p[it1,it1],(n[it1,0],n[it1,0]))
		for it2 in range(it1+1,n_c):
			P_col1 = np.sum(n[0:it2])
			P_col2 = np.sum(n[0:(it2+1)])
			P[P_row1:P_row2,P_col1:P_col2] = np.tile(
					p[it1,it2], (n[it1,0],n[it2,0]))
			P[P_col1:P_col2,P_row1:P_row2] = np.tile(
					p[it1,it2], (n[it2,0],n[it1,0]))
	while FLAG_CONNECTED == 0:
		G = np.random.rand(np.sum(n),np.sum(n)) < P
		G = G.astype(np.float32)
		G = np.triu(G,1);
		G = G + G.T
		L = np.diag(np.sum(G,axis=0))-G
		D,_ = np.linalg.eig(L)
		n_connected_components = (np.absolute(D)<=1e-6)
		if np.sum(n_connected_components.astype(float)) == 1.:
			FLAG_CONNECTED = 1
	return G
	
def shuffle_degree_SBM(A,N):
	print('  shuffle_degree_SBM/A = ')
	print(A)
	print('  shuffle_degree_SBM/N = {}'.format(N))
	N_c = N.shape[0] # Number of communities
	print('  shuffle_degree_SBM/N_c = {}'.format(N_c))
	deg = A.dot(np.ones([int(sum(N)),1])) # Degree of each node
	print('  shuffle_degree_SBM/deg = {}'.format(deg))
	perm = []
	for it in range(N_c):
		firstN = int(sum(N[0:it])) if it > 0 else 0
		lastN = int(sum(N[0:it+1])) if it > 0 else int(N[it])
		print(firstN)
		print(lastN)
		deg_c = deg[firstN:lastN]
		print('  shuffle_degree_SBM/deg_c{} ='.format(it))
		print(deg_c)
		nodes_c = np.squeeze(np.argsort(-deg_c, axis=0).T).tolist()
		print('  shuffle_degree_SBM/nodes_c{} = {}'.format(it,nodes_c))
		node_c = nodes_c[0]+firstN
		print('  shuffle_degree_SBM/nodes_c{} = {}'.format(it,node_c))
		perm = perm + [node_c]
	print('  shuffle_degree_SBM/perm = {}'.format(perm))
	remaining = [x for x in range(int(sum(N))) if x not in perm]
	perm = perm + remaining
	print('  shuffle_degree_SBM/perm = {}'.format(perm))
	A = A[perm][:,perm]
	print('  shuffle_degree_SBM/A = ')
	print(A)
	deg = A.dot(np.ones([int(sum(N)),1])) # Degree of each node
	print('  shuffle_degree_SBM/deg = ')
	print(deg)
	return A

def create_Adc(n):
	Adc = np.zeros((n,n))
	aux = np.eye(n)
	Adc[0:n-1,:] = aux[1:n,:]
	Adc[n-1,:] = aux[0,:]
	return Adc

def create_image_grid(n,m):
	A1 = create_Adc(n)
	A1[n-1,0] = 0.
	A1 = A1.conj().T + A1
	A2 = create_Adc(m)
	A2 [m-1,0] = 0.
	A2 = A2.conj().T + A2
	A = np.kron(A2,np.eye(n)) + np.kron(np.eye(m),A1)
	return A
	
def degree_order(A,K):
	"""
	Compute the permutation ordering based on degree
	A: Adjacency matrix
	K: How many neighbors to discard when selecting next node (for K=0 is 
	simple degree-based ordering)
	"""
	# Check input
	assert K >= 0
	N = A.shape[0]
	M = A.shape[1]
	assert N == M
	del M
	deg = A.dot(np.ones([N,1])) # Degree of each node
	allnodes = np.squeeze(np.argsort(-deg, axis=0).T).tolist()
	if K == 0:
		nodelist = allnodes
		return nodelist
	else:
		nodelist = [allnodes[0]]
		dmax = max(np.linalg.eigvalsh(A.todense()))
		Ak = np.linalg.matrix_power((A/dmax).todense(),K) + np.eye(N)
		for i in range(N-1):
			reach = np.nonzero(Ak[nodelist,:])[1].tolist()
			remaining = [x for x in allnodes if x not in reach]
			if remaining:
				nodelist = nodelist + [remaining[0]]
			else:
				remaining = [x for x in allnodes if x not in nodelist]
				if isinstance(remaining,list):
					nodelist = nodelist + remaining
				else:
					nodelist = nodelist + [remaining]
				break
	assert len(nodelist) == N
	return nodelist
				

def GSO_powers_selected(S,R,K):
	"""
	Create a list of matrices S_l^k = D_l*S^k*D_l'
	S: Full GSO representing entire graph
	R: number of selected nodes (the first ones)
	K: power
	"""
	# Thoroughly checked on the 8x8 ER case.
	N = S.shape[0] # Number of nodes
#	Sin = scipy.sparse.csr_matrix(scipy.sparse.eye(N)) # First element
	Sin = np.eye(N)
	Sin = Sin.astype(float)
	S = S.todense()
	Sl = [Sin[0:R][:,0:R]] # Save first element
	for it in range(K-1):
		Sin = Sin.dot(S)
		Sl = Sl + [Sin[0:R][:,0:R]]
	return Sl

def GSO_powers_selected2(S,R,K):
	"""
	Create a list of matrices S_l^k = D_l*S^k*D_l'
	S: Full GSO representing entire graph
	R: number of selected nodes (the first ones)
	K: power
	"""
	# Thoroughly checked on the 8x8 ER case.
	N = S.shape[1] # Number of nodes
	N = N.value
	Sin = tf.eye(N,num_columns=N)
	Sl = [Sin[0:R][:,0:R]] # Save first element
	for it in range(K-1):
		Sin = tf.matmul(S,Sin)
		Sl = Sl + [Sin[0:R][:,0:R]]
	return Sl

def GSO_powers_selected3(S,R,K):
	"""
	Create a list of matrices S_l^k = D_l*S^k*D_l'
	S: Full GSO representing entire graph
	R: number of selected nodes (the first ones)
	K: power
	"""
	# Thoroughly checked on the 8x8 ER case.
	N = S.shape[1] # Number of nodes
	T = S.shape[0]
	N = N.value
	T = T.value
	Sin = tf.eye(N,batch_shape=[T])
	Sl = [Sin[:,0:R][:,:,0:R]] # Save first element
	for it in range(K-1):
		Sin = tf.matmul(S,Sin)
		Sl = Sl + [Sin[:,0:R][:,:,0:R]]
	return Sl

def collect_at_node3(x,S,R,P):
	"""
	Collect Diffusion Information at Node
		x: Input (T x N x F; T matrices of size N x F, where N is number of nodes and F is number of features)
		S: Original GSO with which to perform diffusion, of size T x N x N.
		R: List of selected nodes at which to collect the diffusion
		P: Number of diffusions to collect (S^{P-1}; P = 1 returns the value at the node)
	"""
	N = S.shape[1]
	T, Nx, F = x.get_shape()
	T, Nx, F = int(T), int(Nx), int(F)
	assert Nx == N
	del Nx
#	x = tf.transpose(x, perm=[1,2,0]) # N x F x T
#	x = tf.reshape(x, [N, F*T]) # N x F*T
	# Convert S in a SparseTensor:
#	S = scipy.sparse.csr_matrix(S)
#	S = S.tocoo()
#	indices = np.column_stack((S.row, S.col))
#	S = tf.SparseTensor(indices, S.data, S.shape)
#	S = tf.sparse_reorder(S)
	x0 = x # First one, N x F*T
	x = tf.transpose(x, perm=[1,2,0]) # N x F x T
	x = tf.reshape(x, [N, F*T]) # N x F*T
	x = tf.expand_dims(x, axis=0) # 1 x N x F*T
	for k in range(1,P):
		x0 = tf.matmul(S, x0) # T x N x F
		x02 = tf.transpose(x0, perm=[1,2,0]) # N x F x T
		x02 = tf.reshape(x02, [N, F*T]) # N x F*T
		x = tf.concat([x,tf.expand_dims(x02,0)], axis = 0)
	# P (shifts) x N (nodes) x F*T
	x = tf.gather(x, R, axis = 1) # P x R x F*T
	x = tf.reshape(x, [P,len(R),F,T]) # P x R x F x T
	x = tf.transpose(x, perm = [3,2,1,0]) # T x F x R x P
	x = tf.reshape(x, [T,F,len(R)*P]) # T x F x R*P
	x = tf.transpose(x, perm=[0,2,1]) # T x R*P x F
	return x
####

def neighborhood_reach(S,R,a):
	"""
	S: Full GSO
	R: Consecutive nodes to keep
	a: neighborhood to look for
	return: list of sets of neighboring nodes of each selected node
	"""
	N = S.shape[0]
	#Spow = scipy.sparse.csr_matrix(scipy.sparse.eye(N))
	Spow = tf.eye(N,num_columns=N, dtype=tf.float32)
	reach = compute_reach(Spow,R)

	S = S.astype(np.float32)
	for aa in range(a):
		#Spow = Spow.dot(S)
		Spow = tf.matmul(S,Spow)
		thisreach = compute_reach(Spow,R)
		for rr in range(R):
			# Add elements:
			reach[rr] = reach[rr]+thisreach[rr]
			# Avoid repeated elements:
			reach[rr] = list(set(reach[rr]))
	return reach

def compute_reach(S,R):
	"""
	S: Matrix to compute reach to
	R: which elements of the matrix
	"""
	reach = [None] * R
	# Position of nonzero elements in S
	for i in range(R):
		#nze =  S[i,:].nonzero() # Nonzero elements
		#reach[i] = nze[1].tolist()

		zero = tf.constant(0, dtype=tf.float32)
		nze = tf.not_equal(S[i,:], zero)
		nze = tf.to_float(nze)
		reach[i] = nze[1]
	return reach
	
def determine_selection_parameters(S,R):
	"""
	Determine parmeters K and a to cover all the data
	S: GSO
	R: number of nodes selected on each layer
	"""
	L = len(R) # Number of layers
	N = S.shape[0] # Number of nodes
	K = []
	a = []
	allnodes = [x for x in range(N)]
	for r in R:
		nodes_r = [x for x in range(r)]
		Sc = scipy.sparse.csr_matrix(scipy.sparse.eye(N))
		for k in range(N-1):
			Sc = Sc.dot(S)
			reach = np.nonzero(Sc[0:r,:])[1].tolist()
			remaining = [x for x in allnodes if x not in reach]
			if not remaining:
				break
		K = K + [k+2]
		
		a_r = []
		for rr in nodes_r:
			Sc = scipy.sparse.csr_matrix(scipy.sparse.eye(N))
			for k in range(N-1):
				Sc = Sc.dot(S)
				reach_r = np.nonzero(Sc[rr,:])[1].tolist()
				remaining_r = [x for x in nodes_r if x not in reach]
				if len(remaining_r) < len(nodes_r):
					a_r = a_r + [k+1]
					break
		a = a + [max(a_r)]
	return K,a

def collect_at_node(x,S,R,P):
	"""
	Collect Diffusion Information at Node
		x: Input (T x N x F; T matrices of size N x F, where N is number of nodes and F is number of features)
		S: Original GSO with which to perform diffusion, of size N x N.
		R: List of selected nodes at which to collect the diffusion
		P: Number of diffusions to collect (S^{P-1}; P = 1 returns the value at the node)
	"""
	N = S.shape[0]
	T, Nx, F = x.get_shape()
	T, Nx, F = int(T), int(Nx), int(F)
	assert Nx == N
	del Nx
	x = tf.transpose(x, perm=[1,2,0]) # N x F x T
	x = tf.reshape(x, [N, F*T]) # N x F*T
	# Convert S in a SparseTensor:
	S = scipy.sparse.csr_matrix(S)
	S = S.tocoo()
	indices = np.column_stack((S.row, S.col))
	S = tf.SparseTensor(indices, S.data, S.shape)
	S = tf.sparse_reorder(S)
	x0 = x # First one, N x F*T
	x = tf.expand_dims(x, axis=0) # 1 x N x F*T
	for k in range(1,P):
		x0 = tf.sparse_tensor_dense_matmul(S, x0) # 1 x N x F*T
		x = tf.concat([x,tf.expand_dims(x0,0)], axis = 0)
	# P (shifts) x N (nodes) x F*T
	x = tf.gather(x, R, axis = 1) # P x R x F*T
	x = tf.reshape(x, [P,len(R),F,T]) # P x R x F x T
	x = tf.transpose(x, perm = [3,2,1,0]) # T x F x R x P
	x = tf.reshape(x, [T,F,len(R)*P]) # T x F x R*P
	x = tf.transpose(x, perm=[0,2,1]) # T x R*P x F
	return x
####



def coarsen(A, levels, self_connections=False):
	"""
	Coarsen a graph, represented by its adjacency matrix A, at multiple
	levels.
	"""
	graphs, parents = metis(A, levels)
	perms = compute_perm(parents)

	for i, A in enumerate(graphs):
		M, M = A.shape

		if not self_connections:
			A = A.tocoo()
			A.setdiag(0)

		if i < levels:
			A = perm_adjacency(A, perms[i])

		A = A.tocsr()
		A.eliminate_zeros()
		graphs[i] = A

		Mnew, Mnew = A.shape
#		print('Layer {0}: M_{0} = |V| = {1} nodes ({2} added),'
#				'|E| = {3} edges'.format(i, Mnew, Mnew-M, A.nnz//2))

	return graphs, perms[0] if levels > 0 else None

def metis(W, levels, rid=None):
	"""
	Coarsen a graph multiple times using the METIS algorithm.
	INPUT
	W: symmetric sparse weight (adjacency) matrix
	levels: the number of coarsened graphs
	OUTPUT
	graph[0]: original graph of size N_1
	graph[2]: coarser graph of size N_2 < N_1
	graph[levels]: coarsest graph of Size N_levels < ... < N_2 < N_1
	parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
		which indicate the parents in the coarser graph[i+1]
	nd_sz{i} is a vector of size N_i that contains the size of the supernode
		in the graph{i}
	NOTE
		if "graph" is a list of length k, then "parents" will be a list of
		length k-1
    """

	N, N = W.shape
	if rid is None:
		rid = np.random.permutation(range(N))
	parents = []
	degree = W.sum(axis=0) - W.diagonal()
	graphs = []
	graphs.append(W)
	#supernode_size = np.ones(N)
	#nd_sz = [supernode_size]
	#count = 0

	#while N > maxsize:
	for _ in range(levels):

		#count += 1

		# CHOOSE THE WEIGHTS FOR THE PAIRING
		# weights = ones(N,1)       # metis weights
		weights = degree            # graclus weights
		# weights = supernode_size  # other possibility
		weights = np.array(weights).squeeze()

		# PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
		idx_row, idx_col, val = scipy.sparse.find(W)
		perm = np.argsort(idx_row)
		rr = idx_row[perm]
		cc = idx_col[perm]
		vv = val[perm]
		cluster_id = metis_one_level(rr,cc,vv,rid,weights)  # rr is ordered
		parents.append(cluster_id)

		# TO DO
		# COMPUTE THE SIZE OF THE SUPERNODES AND THEIR DEGREE 
		#supernode_size = full(   sparse(cluster_id,  ones(N,1) ,
		#	supernode_size )     )
		#print(cluster_id)
		#print(supernode_size)
		#nd_sz{count+1}=supernode_size;

		# COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
		nrr = cluster_id[rr]
		ncc = cluster_id[cc]
		nvv = vv
		Nnew = cluster_id.max() + 1
		# CSR is more appropriate: row,val pairs appear multiple times
		W = scipy.sparse.csr_matrix((nvv,(nrr,ncc)), shape=(Nnew,Nnew))
		W.eliminate_zeros()
		# Add new graph to the list of all coarsened graphs
		graphs.append(W)
		N, N = W.shape

		# COMPUTE THE DEGREE (OMIT OR NOT SELF LOOPS)
		degree = W.sum(axis=0)
		#degree = W.sum(axis=0) - W.diagonal()

		# CHOOSE THE ORDER IN WHICH VERTICES WILL BE VISTED AT THE NEXT PASS
		#[~, rid]=sort(ss);     # arthur strategy
		#[~, rid]=sort(supernode_size);    #  thomas strategy
		#rid=randperm(N);                  #  metis/graclus strategy
		ss = np.array(W.sum(axis=0)).squeeze()
		rid = np.argsort(ss)

	return graphs, parents
	
# Coarsen a graph given by rr,cc,vv.  rr is assumed to be ordered
def metis_one_level(rr,cc,vv,rid,weights):

	nnz = rr.shape[0]
	N = rr[nnz-1] + 1

	marked = np.zeros(N, np.bool)
	rowstart = np.zeros(N, np.int32)
	rowlength = np.zeros(N, np.int32)
	cluster_id = np.zeros(N, np.int32)

	oldval = rr[0]
	count = 0
	clustercount = 0

	for ii in range(nnz):
		rowlength[count] = rowlength[count] + 1
		if rr[ii] > oldval:
			oldval = rr[ii]
			rowstart[count+1] = ii
			count = count + 1

	for ii in range(N):
		tid = rid[ii]
		if not marked[tid]:
			wmax = 0.0
			rs = rowstart[tid]
			marked[tid] = True
			bestneighbor = -1
			for jj in range(rowlength[tid]):
				nid = cc[rs+jj]
				if marked[nid]:
					tval = 0.0
				else:
					tval = vv[rs+jj] * (1.0/weights[tid] + 1.0/weights[nid])
				if tval > wmax:
					wmax = tval
					bestneighbor = nid

			cluster_id[tid] = clustercount

			if bestneighbor > -1:
				cluster_id[bestneighbor] = clustercount
				marked[bestneighbor] = True

			clustercount += 1

	return cluster_id

def compute_perm(parents):
	"""
	Return a list of indices to reorder the adjacency and data matrices so
	that the union of two neighbors from layer to layer forms a binary tree.
	"""

	# Order of last layer is random (chosen by the clustering algorithm).
	indices = []
	if len(parents) > 0:
		M_last = max(parents[-1]) + 1
		indices.append(list(range(M_last)))

	for parent in parents[::-1]:
		#print('parent: {}'.format(parent))

		# Fake nodes go after real ones.
		pool_singeltons = len(parent)

		indices_layer = []
		for i in indices[-1]:
			indices_node = list(np.where(parent == i)[0])
			assert 0 <= len(indices_node) <= 2
			#print('indices_node: {}'.format(indices_node))

			# Add a node to go with a singelton.
			if len(indices_node) is 1:
				indices_node.append(pool_singeltons)
				pool_singeltons += 1
				#print('new singelton: {}'.format(indices_node))
			# Add two nodes as children of a singelton in the parent.
			elif len(indices_node) is 0:
				indices_node.append(pool_singeltons+0)
				indices_node.append(pool_singeltons+1)
				pool_singeltons += 2
				#print('singelton childrens: {}'.format(indices_node))

			indices_layer.extend(indices_node)
		indices.append(indices_layer)

	# Sanity checks.
	for i,indices_layer in enumerate(indices):
		M = M_last*2**i
		# Reduction by 2 at each layer (binary tree).
		assert len(indices[0] == M)
		# The new ordering does not omit an indice.
		assert sorted(indices_layer) == list(range(M))

	return indices[::-1]

assert (compute_perm([np.array([4,1,1,2,2,3,0,0,3]),np.array([2,1,0,1,0])])
		== [[3,4,0,9,1,2,5,8,6,7,10,11],[2,4,1,3,0,5],[0,1,2]])

def perm_data(x, indices):
	"""
	Permute data matrix, i.e. exchange node ids,
	so that binary unions form the clustering tree.
	"""
	if indices is None:
		return x

	N, M = x.shape
	Mnew = len(indices)
	assert Mnew >= M
	xnew = np.empty((N, Mnew))
	for i,j in enumerate(indices):
		# Existing vertex, i.e. real data.
		if j < M:
			xnew[:,i] = x[:,j]
		# Fake vertex because of singeltons.
		# They will stay 0 so that max pooling chooses the singelton.
		# Or -infty ?
		else:
			xnew[:,i] = np.zeros(N)
	return xnew

def perm_adjacency(A, indices):
	"""
	Permute adjacency matrix, i.e. exchange node ids,
	so that binary unions form the clustering tree.
	"""
	if indices is None:
		return A

	M, M = A.shape
	Mnew = len(indices)
	assert Mnew >= M
	A = A.tocoo()

	# Add Mnew - M isolated vertices.
	if Mnew > M:
		rows = scipy.sparse.coo_matrix((Mnew-M, M), dtype=np.float32)
		cols = scipy.sparse.coo_matrix((Mnew, Mnew-M), dtype=np.float32)
		A = scipy.sparse.vstack([A, rows])
		A = scipy.sparse.hstack([A, cols])

    # Permute the rows and the columns.
	perm = np.argsort(indices)
	A.row = np.array(perm)[A.row]
	A.col = np.array(perm)[A.col]

	# assert np.abs(A - A.T).mean() < 1e-9
	assert type(A) is scipy.sparse.coo.coo_matrix
	return A

####

def grid(m, dtype=np.float32):
	"""Return the embedding of a grid graph."""
	M = m**2
	x = np.linspace(0, 1, m, dtype=dtype)
	y = np.linspace(0, 1, m, dtype=dtype)
	xx, yy = np.meshgrid(x, y)
	z = np.empty((M, 2), dtype)
	z[:, 0] = xx.reshape(M)
	z[:, 1] = yy.reshape(M)
	return z

def distance_scipy_spatial(z, k=4, metric='euclidean'):
	"""Compute exact pairwise distances."""
	d = scipy.spatial.distance.pdist(z, metric)
	d = scipy.spatial.distance.squareform(d)
	# k-NN graph.
	idx = np.argsort(d)[:, 1:k+1]
	#!!! fgama 2017/18/10 Comment:
	# np.argsort(d) sorts the rows of the matrix d in increasing 
	# distance. By selecting from 1 to k+1 we are leaving out the 0 
	# distance (the diagonal) and going up to the first k elements.
	# This helps in building the k-NN since selects, for each node, the
	# k-nn following the distance
	#!!!
	d.sort()
	d = d[:, 1:k+1]
	return d, idx

def distance_sklearn_metrics(z, k=4, metric='euclidean'):
	"""Compute exact pairwise distances."""
	### FGAMA, 2017/08/22: Parallelization is not adequately
	# supported by Mac. One solution is to use a forkserver, but I'm not
	# sure how it's done. In any case, just as a patch here, I will 
	# avoid parallelization altogether when working on a Mac.
	if sys.platform == 'darwin':
		n_jobs = 1
	else:
		n_jobs = -2
	
	d = sklearn.metrics.pairwise.pairwise_distances(
			z, metric=metric, n_jobs=n_jobs)
	# k-NN graph.
	idx = np.argsort(d)[:, 1:k+1]
	d.sort()
	d = d[:, 1:k+1]
	return d, idx

def distance_lshforest(z, k=4, metric='cosine'):
	"""Return an approximation of the k-nearest cosine distances."""
	assert metric is 'cosine'
	lshf = sklearn.neighbors.LSHForest()
	lshf.fit(z)
	dist, idx = lshf.kneighbors(z, n_neighbors=k+1)
	assert dist.min() < 1e-10
	dist[dist < 0] = 0
	return dist, idx

# TODO: other ANNs s.a. NMSLIB, EFANNA, FLANN, Annoy, sklearn neighbors, PANN

def adjacency(dist, idx):
	"""Return the adjacency matrix of a kNN graph."""
	M, k = dist.shape
	assert M, k == idx.shape
	assert dist.min() >= 0

	# Weights.
	sigma2 = np.mean(dist[:, -1])**2
	dist = np.exp(- dist**2 / sigma2)

	# Weight matrix.
	I = np.arange(0, M).repeat(k)
	J = idx.reshape(M*k)
	V = dist.reshape(M*k)
	W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

	# No self-connections.
	W.setdiag(0)

	# Non-directed graph.
	bigger = W.T > W
	W = W - W.multiply(bigger) + W.T.multiply(bigger)

	assert W.nnz % 2 == 0
	assert np.abs(W - W.T).mean() < 1e-10
	assert type(W) is scipy.sparse.csr.csr_matrix
	return W

def replace_random_edges(A, noise_level):
	"""Replace randomly chosen edges by random edges."""
	M, M = A.shape
	n = int(noise_level * A.nnz // 2)

	indices = np.random.permutation(A.nnz//2)[:n]
	rows = np.random.randint(0, M, n)
	cols = np.random.randint(0, M, n)
	vals = np.random.uniform(0, 1, n)
	assert len(indices) == len(rows) == len(cols) == len(vals)

	A_coo = scipy.sparse.triu(A, format='coo')
	assert A_coo.nnz == A.nnz // 2
	assert A_coo.nnz >= n
	A = A.tolil()

	for idx, row, col, val in zip(indices, rows, cols, vals):
		old_row = A_coo.row[idx]
		old_col = A_coo.col[idx]

		A[old_row, old_col] = 0
		A[old_col, old_row] = 0
		A[row, col] = 1
		A[col, row] = 1

	A.setdiag(0)
	A = A.tocsr()
	A.eliminate_zeros()
	return A

def laplacian(W, normalized=True):
	"""Return the Laplacian of the weigth matrix."""

	# Degree matrix.
	d = W.sum(axis=0)

	# Laplacian matrix.
	if not normalized:
		D = scipy.sparse.diags(d.A.squeeze(), 0)
		L = D - W
	else:
		d += np.spacing(np.array(0, W.dtype))
		d = 1 / np.sqrt(d)
		D = scipy.sparse.diags(d.A.squeeze(), 0)
		I = scipy.sparse.identity(d.size, dtype=W.dtype)
		L = I - D * W * D

	# assert np.abs(L - L.T).mean() < 1e-9
	assert type(L) is scipy.sparse.csr.csr_matrix
	return L

def lmax(L, normalized=True):
	"""Upper-bound on the spectrum."""
	if normalized:
		return 2
	else:
		return scipy.sparse.linalg.eigsh(
				L, k=1, which='LM', return_eigenvectors=False)[0]

def fourier(L, algo='eigh', k=1):
	"""Return the Fourier basis, i.e. the EVD of the Laplacian."""

	def sort(lamb, U):
		idx = lamb.argsort()
		return lamb[idx], U[:, idx]

	if algo is 'eig':
		lamb, U = np.linalg.eig(L.toarray())
		lamb, U = sort(lamb, U)
	elif algo is 'eigh':
		lamb, U = np.linalg.eigh(L.toarray())
	elif algo is 'eigs':
		lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM')
		lamb, U = sort(lamb, U)
	elif algo is 'eigsh':
		lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

	return lamb, U

def plot_spectrum(L, algo='eig'):
	"""Plot the spectrum of a list of multi-scale Laplacians L."""
	# Algo is eig to be sure to get all eigenvalues.
	plt.figure(figsize=(17, 5))
	for i, lap in enumerate(L):
		lamb, U = fourier(lap, algo)
		step = 2**i
		x = range(step//2, L[0].shape[0], step)
		lb = 'L_{} spectrum in [{:1.2e}, {:1.2e}]'.format(i, lamb[0], lamb[-1])
		plt.plot(x, lamb, '.', label=lb)
	plt.legend(loc='best')
	plt.xlim(0, L[0].shape[0])
	plt.ylim(ymin=0)

def lanczos(L, X, K):
	"""
	Given the graph Laplacian and a data matrix, return a data matrix which can
	be multiplied by the filter coefficients to filter X using the Lanczos
	polynomial approximation.
	"""
	M, N = X.shape
	assert L.dtype == X.dtype

	def basis(L, X, K):
		"""
		Lanczos algorithm which computes the orthogonal matrix V and the
		tri-diagonal matrix H.
		"""
		a = np.empty((K, N), L.dtype)
		b = np.zeros((K, N), L.dtype)
		V = np.empty((K, M, N), L.dtype)
		V[0, ...] = X / np.linalg.norm(X, axis=0)
		for k in range(K-1):
			W = L.dot(V[k, ...])
			a[k, :] = np.sum(W * V[k, ...], axis=0)
			W = W - a[k, :] * V[k, ...] - (
					b[k, :] * V[k-1, ...] if k > 0 else 0)
			b[k+1, :] = np.linalg.norm(W, axis=0)
			V[k+1, ...] = W / b[k+1, :]
		a[K-1, :] = np.sum(L.dot(V[K-1, ...]) * V[K-1, ...], axis=0)
		return V, a, b

	def diag_H(a, b, K):
		"""Diagonalize the tri-diagonal H matrix."""
		H = np.zeros((K*K, N), a.dtype)
		H[:K**2:K+1, :] = a
		H[1:(K-1)*K:K+1, :] = b[1:, :]
		H.shape = (K, K, N)
		Q = np.linalg.eigh(H.T, UPLO='L')[1]
		Q = np.swapaxes(Q, 1, 2).T
		return Q

	V, a, b = basis(L, X, K)
	Q = diag_H(a, b, K)
	Xt = np.empty((K, M, N), L.dtype)
	for n in range(N):
		Xt[..., n] = Q[..., n].T.dot(V[..., n])
	Xt *= Q[0, :, np.newaxis, :]
	Xt *= np.linalg.norm(X, axis=0)
	return Xt  # Q[0, ...]

def rescale_L(L, lmax=2):
	"""Rescale the Laplacian eigenvalues in [-1,1]."""
	M, M = L.shape
	I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
	L /= lmax / 2
	L -= I
	return L

def chebyshev(L, X, K):
	"""Return T_k X where T_k are the Chebyshev polynomials of order up to K.
	Complexity is O(KMN)."""
	M, N = X.shape
	assert L.dtype == X.dtype

	# L = rescale_L(L, lmax)
	# Xt = T @ X: MxM @ MxN.
	Xt = np.empty((K, M, N), L.dtype)
	# Xt_0 = T_0 X = I X = X.
	Xt[0, ...] = X
	# Xt_1 = T_1 X = L X.
	if K > 1:
		Xt[1, ...] = L.dot(X)
	# Xt_k = 2 L Xt_k-1 - Xt_k-2.
	for k in range(2, K):
		Xt[k, ...] = 2 * L.dot(Xt[k-1, ...]) - Xt[k-2, ...]
	return Xt
