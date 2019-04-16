from dgl.data.sbm import sbm
import numpy as np
import scipy as sp
from scipy.cluster.vq import kmeans2
from scipy.io import loadmat
import scipy.sparse.linalg as linalg
import scipy.sparse as sps
from sklearn.metrics import f1_score

n_blocks = 4
block_size = 5
n = n_blocks * block_size
p = 7
q = 1
A = sbm(n_blocks, block_size, p, q)
y = np.zeros([n, n_blocks])
for i in range(n_blocks):
    y[i * block_size : (i + 1) * block_size, i] = 1

'''
f = open('karate.adjlist')
adjlist = [list(map(int, line.split(' '))) for line in f.readlines()]
indptr = np.cumsum([0] + list(map(len, adjlist)))
indices = sum(adjlist, [])
data = np.ones(len(indices))
n = len(adjlist)
A = sps.csr_matrix((data, indices, indptr), shape=[n, n])
'''

'''
mat = loadmat('blogcatalog.mat')
# mat = loadmat('youtube.mat')
A = mat['network']
n = A.shape[0]
y = mat['group'].todense()
'''

D = sps.diags(A.tocsr().dot(np.ones(n)))
r = (sp.sum(D) / n) ** 0.5
Hp = (r ** 2 - 1) * sps.eye(n) - r * A + D
Hm = (r ** 2 - 1) * sps.eye(n) + r * A + D
wp, vp = linalg.eigsh(Hp, k=n - 1, which='SA')
print(sorted(wp))
wm, vm = linalg.eigsh(Hm, k=n - 1, which='SA')
print(sorted(wm))
_, y_bar = kmeans2(np.hstack([vp, vm]), y.shape[1])
print(f1_score(np.argmax(y, axis=1), y_bar, average='micro'))
print(f1_score(np.argmax(y, axis=1), y_bar, average='macro'))
