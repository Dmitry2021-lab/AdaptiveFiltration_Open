#lms - Least mean squares adaptive filter
#nlms - Normalized least mean squares adaptive filter
#nlmf - normalized least mean fourth adaptive filter
# Here is a classical models

import numpy as np
import random
import Wave
import datetime
import os

class TimeDomainAdaptiveFilters():
    def __init__(self):
        self.order_lms = 5
        self.mu_lms = 0.01

        self.order_nlms = 5
        self.mu_nlms = 0.01
        self.eps_nlms = 1.

        self.order_nlmf = 5
        self.mu_nlmf = 0.01
        self.eps_nlmf = 1.

    def lms_init(self, order = 100, mu = 0.01, type = 'zero'):
        self.order_lms = order
        self.mu_lms = mu
        error_lms = 1

        if type == 'zero':
            wight = np.zeros(order)
        elif type == 'random':
            wight = np.random.sample(order)

        return wight, error_lms

    def lms_adapt( self, x, wight, e ):
        wight += self.mu_lms * e * x
        return wight

    def lms_run( self, d, x, w ):
        y = np.dot(w, x)
        e = d - y
        return y, e

    def nlms_init(self, order = 100, mu = 0.01, eps = 1.,type = 'zero'):
        self.order_nlms = order
        self.mu_nlms = mu
        self.eps_nlms = eps
        error = 1

        if type == 'zero':
            wight = np.zeros(order)
        elif type == 'random':
            wight = np.random.sample(order)

        return wight, error

    def nlms_adapt( self, x, wight, e ):
        mu = self.mu_nlms / (self.eps_nlms + np.dot(x, x))
        wight += mu * e * x
        return wight

    def nlms_run( self, d, x, w ):
        y = np.dot(w, x)
        e = d - y
        return y, e

    def nlmf_init(self, order = 100, mu = 1e-8, eps = 1.,type = 'zero'):
        self.order_nlmf = order
        self.mu_nlmf = mu
        self.eps_nlmf = eps
        error = 1

        if type == 'zero':
            wight = np.zeros(order)
        elif type == 'random':
            wight = np.random.sample(order)

        return wight, error

    def nlmf_adapt(self, x, wight, e):
        mu = self.mu_nlmf / (self.eps_nlmf + np.dot(x, x))
        wight += mu * x * e ** 3
        return wight

    def nlmf_run( self, d, x, w ):
        y = np.dot(w, x)
        e = d - y
        return y, e

ORDER = 512
if __name__ == '__main__':
    wv = Wave.PyWav()
    dir = os.getcwd()

    f = TimeDomainAdaptiveFilters()
    w,e = f.nlms_init(order = ORDER)
    NoiseFile = dir +'\\SeparateOutput\\'+'micSplin7.wav'
    UsefullFile = dir +'\\SeparateOutput\\'+'micSplin1.wav'
    noise = np.asarray(wv.ReadWav(NoiseFile), dtype=np.float32)
    signal = np.asarray(wv.ReadWav(UsefullFile), dtype=np.float32)
    now = datetime.datetime.now()
    Buff = []
    for i in range(0, len(signal) - ORDER, 1):
        x = noise[i:i + ORDER]
        d = signal[i + ORDER]
        w = f.nlms_adapt(x, w, e)
        y,e = f.nlms_run(d,x,w)

        Buff.append(e)
    print('spend time = ', datetime.datetime.now() - now)
    wv.WriteWav(dir + '/ResultWaves/' + 'Out_LMS_' + str(f.order_nlms) + '.wav', Buff)



