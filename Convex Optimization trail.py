import scipy
import scipy.ndimage as spimg
import numpy as np
import PIL.Image as Img
import scipy.fftpack as spfft
import cvxpy as cvx
import cv2
import math


def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


ori = Img.open('1.jpg')
ori = np.array(ori)
r_num, c_num, d_num = ori.shape
X_orig = np.zeros((r_num, c_num))
for i in range(0, r_num):
    for j in range(0, c_num):
        X_orig[i, j] = ori[i, j, 0] + 256 * ori[i, j, 1] + 256*256 * ori[i, j, 2]


X = spimg.zoom(X_orig, 0.5)

ny, nx = X.shape

k = round(nx * ny * 0.5)
ri = np.random.choice(nx * ny, k, replace=False)
b = X.T.flat[ri]

A = np.kron(
    spfft.idct(np.identity(nx), norm='ortho', axis=0),
    spfft.idct(np.identity(ny), norm='ortho', axis=0)
)
A = A[ri, :]  # same as phi times kron

vx = cvx.Variable(nx * ny)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [A@vx == b]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)
print(vx.shape)
Xat2 = np.array(vx.value).squeeze()
print(Xat2)
Xat = Xat2.reshape(nx, ny).T  # stack columns
Xa = idct2(Xat)
print(Xa.shape)

img_sampled = np.zeros((ny, nx, d_num))
for i in range(0, ny):
    for j in range(0, nx):
        img_sampled[i, j, 2] = math.floor(Xa[i, j]/256/256)
        img_sampled[i, j, 1] = math.floor((Xa[i, j]-img_sampled[i, j, 2]*256*256)/256)
        img_sampled[i, j, 0] = Xa[i, j]-img_sampled[i, j, 2]*256*256-img_sampled[i, j, 1]*256


for i in range(0, ny):
    for j in range(0, nx):
        for k in range(0, d_num):
            if img_sampled[i, j, k] == 0:
                img_sampled[i, j, k] = 255

Result = Img.fromarray(img_sampled, mode='RGB')
Result.show()
# confirm solution
if not np.allclose(X.T.flat[ri], Xa.T.flat[ri]):
    print('Warning: values at sample indices don\'t match original.')

# create images of mask (for visualization)
mask = np.zeros(X.shape)
print(mask.shape)
mask.T.flat[ri] = 255
Xm = 255 * np.ones(X.shape)
Xm.T.flat[ri] = X.T.flat[ri]

Xm = Img.fromarray(Xm, mode='L')
Xm.show()