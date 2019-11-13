import numpy as np

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        N, T, D = x.shape
        self.x = x
        out = np.dot(x.reshape(N * T, -1), W) + b
        return out.reshape(N ,T, -1)

    def backward(self, dout):
        W, b = self.params
        N, T, D = self.x.shape
        dout = dout.reshape(N * T, -1)
        self.grads[0][...] = np.dot(self.x.reshape(N * T, -1).T, dout)
        self.grads[1][...] = np.sum(dout, axis=0)
        dx = np.dot(dout, W.T)
        return dx.reshape(N, T, -1)
