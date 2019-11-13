import numpy as np

class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        h_next = np.tanh(np.dot(h_prev, Wh) + np.dot(x, Wx) + b)
        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache
        dt = dh_next * (1 - h_next ** 2)
        self.grads[0][...] = np.dot(x.T, dt)
        self.grads[1][...] = np.dot(h_prev.T, dt)
        self.grads[2][...] = np.sum(dt, axis=0)
        return np.dot(dt, Wx.T), np.dot(dt, Wh.T)

class TRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers, self.h, self.dh = None, None, None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        _, H = Wx.shape
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')
        if self.stateful == False or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        for t in range(T):
            layer = RNN(Wx, Wh, b)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape
        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx
            for i, g in enumerate(layer.grads):
                grads[i] += g
        for i, g in enumerate(grads):
            self.grads[i][...] = g
        self.dh = dh
        return dxs

    def set_state(self, h=None):
        self.H = h
