import numpy as np

class Trainer:
    def __init__(self, model, optimizer):
        self.model, self.optimizer = model, optimizer
        self.time_idx = 0

    def get_batch(self, xs, ts, batch_size, time_size):
        bx = np.empty((batch_size, time_size), dtype='i')
        bt = np.empty((batch_size, time_size), dtype='i')
        data_size = len(xs)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]
        for t in range(time_size):
            for i, o in enumerate(offsets):
                bx[i, t] = xs[(o + self.time_idx) % data_size]
                bt[i, t] = ts[(o + self.time_idx) % data_size]
            self.time_idx += 1
        return bx, bt

    def remove_duplicate(self, params, grads):
        params, grads = params[:], grads[:]
        while True:
            flg = False
            for i in range(0, len(params) - 1):
                for j in range(i + 1, len(params)):
                    if params[i] is params[j]:
                        # 重みを共有する場合
                        grads[i] += grads[j]
                        params.pop(j)
                        grads.pop(j)
                        flg = True
                    elif params[i].ndim == 2 and params[j].ndim == 2 and params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                        # 転置行列として重みを共有する場合（weight tying）
                        grads[i] += grads[j].T
                        params.pop(j)
                        grads.pop(j)
                        flg = True
                    if flg: break
                if flg: break
            if not flg: break
        return params, grads

    def fit(self, xs, ts, epoch_size, batch_size, time_size):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        for ep in range(epoch_size):
            total_loss = 0
            loss_count = 0
            for itr in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)
                y = self.model.predict(batch_x, True)
                loss = self.model.loss.forward(y, batch_t)
                self.model.backward()
                params, grads = self.remove_duplicate(self.model.params, self.model.grads)
                self.optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1
            if (ep+1) % 10 == 0:
                print('loss %.2f | perplexity %.4f' %(total_loss / loss_count, np.exp(total_loss / loss_count)))
