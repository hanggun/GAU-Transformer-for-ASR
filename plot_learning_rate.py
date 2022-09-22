import numpy as np
import matplotlib.pyplot as plt
from bert4keras.backend import K
import tensorflow as tf
from tqdm import tqdm


class CustomSchedule:
    def __init__(self, d_model, warmup_steps=4000, max_lr=None, start_step=0):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.start_step = start_step

    def __call__(self, step):
        step = step + self.start_step
        arg1 = 1/np.sqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = 1/np.sqrt(self.d_model) * np.min([arg1, arg2])
        if self.max_lr is not None:
            return np.min([self.max_lr, lr])
        return lr


def piecewise_linear(t, schedule, from_zero=True):
    """分段线性函数
    其中schedule是形如{1000: 1, 2000: 0.1}的字典，
    表示 t ∈ [0, 1000]时，输出从0均匀增加至1，而
    t ∈ [1000, 2000]时，输出从1均匀降低到0.1，最后
    t > 2000时，保持0.1不变。
    """
    schedule = sorted(schedule.items())
    if from_zero and schedule[0][0] != 0:
        schedule = [(0, 0.0)] + schedule

    t = K.cast(t, K.floatx())
    x = (t * 0 + 1) * schedule[0][1]
    for i in range(len(schedule)):
        t_begin = schedule[i][0]
        x_begin = x
        if i != len(schedule) - 1:
            dx = schedule[i + 1][1] - schedule[i][1]
            dt = schedule[i + 1][0] - schedule[i][0]
            slope = 1.0 * dx / dt
            x = schedule[i][1] + slope * (t - t_begin)
        else:
            x = (t * 0 + 1) * schedule[i][1]
        x = K.switch(t >= t_begin, x, x_begin)

    return x


def transformer_schedule(t, start_step, warmup_steps, d_model):
    t = K.cast(t, K.floatx())
    d_model = K.cast(d_model, K.floatx())
    step = t + start_step
    arg1 = 1 / K.sqrt(step)
    arg2 = step * (warmup_steps ** -1.5)
    lr = 1 / K.sqrt(d_model) * K.minimum(arg1, arg2)
    return lr

schedule = CustomSchedule(512, warmup_steps=4000)
lrs = [schedule.__call__(x) for x in range(100000)]
print(lrs[40000])
plt.plot(lrs)
plt.show()
# sess = tf.Session()
# lrs = []
# for t in tqdm(range(100)):
#     lrs.append(sess.run(transformer_schedule(t, 0., 10., 512.)))
# print(lrs)
# plt.plot(lrs)
# plt.show()