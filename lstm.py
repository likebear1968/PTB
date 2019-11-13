import numpy as np

class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        v = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        f = self.sigmoid(v[:, : H])
        g = np.tanh(v[:, H: 2 * H])
        i = self.sigmoid(v[:, 2 * H: 3 * H])
        o = self.sigmoid(v[:, 3 * H:])

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tc_next = np.tanh(c_next)
        ds = dc_next + (dh_next * o) * (1 - tc_next ** 2)

        do = dh_next * tc_next * (o * (1 - o))
        di = ds * g * (i * (1 - i))
        dg = ds * i * (1 - g ** 2)
        df = ds * c_prev * (f * (1 - f))

        dv = np.hstack((df, dg, di, do))

        self.grads[0][...] = np.dot(x.T, dv)
        self.grads[1][...] = np.dot(h_prev.T, dv)
        self.grads[2][...] = dv.sum(axis=0)

        return np.dot(dv, Wx.T), np.dot(dv, Wh.T), ds * f

class TLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers, self.h, self.c, self.dh = None, None, None, None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H, _ = Wh.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if self.stateful == False or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if self.stateful == False or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(Wx, Wh, b)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, _ = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0
        grads = [0, 0, 0]

        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, g in enumerate(layer.grads):
                grads[i] += g
        for i, g in enumerate(grads):
            self.grads[i][...] = g
        self.dh = dh
        return dxs

    def set_state(self, h=None, c=None):
        self.h, self.c = h, c
