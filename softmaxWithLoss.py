import numpy as np

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def softmax(self, x):
        if x.ndim == 2:
            x -= x.max(axis=1, keepdims=True)
            x = np.exp(x)
            x /= x.sum(axis=1, keepdims=True)
        elif x.ndim == 1:
            x -= np.max(x)
            x = np.exp(x) / np.sum(np.exp(x))
        return x

    def forward(self, xs, ts):
        N, T, V = xs.shape
        if ts.ndim == 3:
            # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2)
        # バッチ分と時系列分をまとめる
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = (ts != self.ignore_label)
        ys = self.softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask
        loss = -np.sum(ls)
        loss /= mask.sum()
        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache
        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]
        return dx.reshape(N, T, V)
