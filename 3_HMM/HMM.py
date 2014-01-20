import numpy as np

def prediction(theta, px):
    return np.dot(px, theta)

def updating(y, px, phi):
    x_y = phi[:, y]
    px = px * x_y
    return px / sum(px)

def HMM(px0, theta, phi, y, T):

    px = px0.copy()

    for t in xrange(T):
        if t < len(y):
            # update and prediction
            px = prediction(theta, px)
            px = updating(y[t], px, phi)
        else:
            # prediction only
            px = prediction(theta, px)
        print "After step: %d" % (t+1)
        print px
        print '\n'
    return px

def main():
    # distribution at t0
    px0 = np.array([0.4, 0.6])

    # transition matrix of hidden states
    theta = np.array([[0.8, 0.2],
                      [0.1, 0.9]])

    # inference matrix
    phi = np.array([[0.1, 0.2, 0.4, 0.0, 0.3],
                    [0.3, 0.0, 0.3, 0.3, 0.1]])

    # Observations
    y = np.array([3, 3, 0, 4, 2])
    T = 8

    # distribution after Y observations
    px = HMM(px0, theta, phi, y, T)
    print "Final:"
    print(px)

main()
