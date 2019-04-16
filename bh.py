import numpy as np
import scipy as sp
from scipy.cluster.vq import kmeans2
from scipy.io import loadmat
import scipy.sparse.linalg as linalg
import scipy.sparse as sps
from sklearn.metrics import f1_score

# TODO format
mat = loadmat('blogcatalog.mat')
# mat = loadmat('youtube.mat')
A = mat['network']
n = A.shape[0]
D = sps.diags(A.tocsr().dot(np.ones(n)))
y = mat['group'].todense()
r2 = sp.sum(D) / n
Hp = (r2 - 1) * sps.eye(n) - r2 ** 0.5 * A + D
Hm = (r2 - 1) * sps.eye(n) + r2 ** 0.5 * A + D
wp, vp = linalg.eigsh(Hp, k=2, which='SA')
print(sorted(wp))
wm, vm = linalg.eigsh(Hm, k=2, which='SA')
print(sorted(wm))
_, y_bar = kmeans2(np.hstack([vp, vm]), y.shape[1])
print(y.shape, y_bar.shape)
print(f1_score(np.argmax(y, axis=1), y_bar, average='micro'))
print(f1_score(np.argmax(y, axis=1), y_bar, average='macro'))
