# 1D Gaussian
from math import sqrt, exp, pi
def gaussianPDF(mu, sigma, x):
    return 1.0/(sqrt(2.0*pi)*sigma)*exp((mu-x)*(x-mu)/(2*sigma*sigma))

from matplotlib.pylab import *
from numpy import arange
samples = arange(-5., 5., .05)
figure(1)
plot(samples, [gaussianPDF(0.0, 1.0, x) for x in samples], lw=2.)
grid()
draw()
show()

# 2D Gaussian
from numpy import exp, dot, linalg, matrix, array
def gaussianPDF_2D(mu, Sigma, x):
    d = 1.0/sqrt(linalg.det(2.0*pi*Sigma))*exp(-0.5*dot(dot(x-mu,linalg.inv(Sigma)),x-mu))
    return float(d)

Sigma = matrix('1., 0.; 0., 1.')
mu = array([0., 0.])

from numpy import meshgrid, zeros
x = arange(-5., 5., .05)
y = arange(-5., 5., .05)
X, Y = meshgrid(x, y)
Z = zeros(X.shape)
nx, ny = X.shape
for i in xrange(nx):
    for j in xrange(ny):
        Z[i,j] = gaussianPDF_2D(mu, Sigma,
                array([X[i,j], Y[i,j]]))

figure(2, figsize=(9,9))
contour(X, Y, Z)
draw()
show()

from mpl_toolkits.mplot3d import Axes3D
fig = figure(3)
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z)
draw()
show()

from pylab import imread, imshow
img = imread('gradient.png')
figure()

from matplotlib.pyplot import subplot
from scipy import ndimage
figure()
subplot(2,2,1)
imshow(img)
subplot(2,2,2)
imshow(ndimage.rotate(img, 90))
subplot(2,2,3)
nored = img.copy()
nored[:,:,0] = zeros(nored[:,:,0].shape)
imshow(nored)
subplot(2,2,4)
onlyred = img.copy()
onlyred[:,:,1:] = zeros(onlyred[:,:,1:].shape)
imshow(onlyred)
show()
