import numpy as np
## Support Vector Machine
def fit(X,Y):
    X = np.c_[np.ones((X.shape[0], 1)), X]
    #print(X.shape)
    w = np.random.uniform(0, 1.0, size=(1,3))
    wasl =w
    w=w.T
    epochs = 1
    alpha = 0.001
    while (epochs < 10000):
        lambda_ = 1 / epochs

        for index,ix in enumerate(X):
            ix=ix.reshape(1,3)
            flag = Y[index] * (np.dot(ix, w)) >= 1
            if flag:
                w -= alpha * (2 * lambda_ * w)
            else:
                wasl += alpha * (Y[index]*ix) - (2 * lambda_ * wasl)
        epochs+=1


    y_pred = np.sign(np.dot(X,w))
    return y_pred,w