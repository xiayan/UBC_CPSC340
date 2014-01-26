from scipy import *
from pylab import *

img = imread("lebron.png")[:,:,0]
gray()
figure(1)
imshow(img)

m,n = img.shape
U,s,Vt = svd(img)
# make S a diagonal matrix
S = resize(s, [m,1]) * eye(m,n)

k = 0
esum = sum(s)

retention = 0.99

for i in xrange(1, m+1):
    recover = sum(s[0:i])/esum
    if recover >= retention:
        k = i
        break

print "To retain %f variance, %d out of %d dimensions kept" % (retention, k, m)

figure(2)
imshow(dot(U[:,1:k], dot(S[1:k,1:k], Vt[1:k,:])))
show()
