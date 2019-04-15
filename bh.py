from dgl.data.sbm import sbm
import numpy as np
import scipy as sp
from scipy.cluster.vq import kmeans2
from scipy.io import loadmat
import scipy.sparse.linalg as linalg
import scipy.sparse as sps
from sklearn.metrics import f1_score

'''
mat = loadmat('blogcatalog.mat')
# mat = loadmat('youtube.mat')
A = mat['network']
n = A.shape[0]
y = mat['group'].todense()
'''

n_blocks = 4
block_size = 100
n = n_blocks * block_size
p = 7 / n
q = 1 / n
A = sbm(n_blocks, block_size, p, q)
y = np.zeros([n, n_blocks])
for i in range(n_blocks):
    y[i * block_size : (i + 1) * block_size, i] = 1

D = sps.diags(A.tocsr().dot(np.ones(n)))
r2 = sp.sum(D) / n
Hp = (r2 - 1) * sps.eye(n) - r2 ** 0.5 * A + D
Hm = (r2 - 1) * sps.eye(n) + r2 ** 0.5 * A + D
wp, vp = linalg.eigsh(Hp, k=2, which='SA')
print(sorted(wp))
wm, vm = linalg.eigsh(Hm, k=2, which='SA')
print(sorted(wm))
_, y_bar = kmeans2(np.hstack([vp, vm]), y.shape[1])
print(f1_score(np.argmax(y, axis=1), y_bar, average='micro'))
print(f1_score(np.argmax(y, axis=1), y_bar, average='macro'))
