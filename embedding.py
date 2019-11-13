import numpy as np

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        return W[idx]

    def backward(self, dout):
        dw, = self.grads
        dw[...] = 0
        np.add.at(dw, self.idx, dout)

class TEmbedding:
    def __init__(self, W):
        self.W = W
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape
        out = np.empty((N, T, D), dtype='f')
        self.layers = []
        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)
        return out

    def backward(self, dout):
        N, T, D = dout.shape
        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]
        self.grads[0][...] = grad
