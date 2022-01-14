#FIR - finite impulse response
'''
J. Shynk Frequency-domain and multirate adaptive filtering
// 1992 Mathematics, Computer Science IEEE Signal Processing Magazine
Overlap-Add algorithm (Fig 6)
'''

import numpy as np
import random
import Wave
import datetime
import os


def Correlation( ns_data, sgn_data, start, finish, acceptable_shift_left, acceptable_shift_right ):
	max = -1.e10
	index = -1
	N = len(ns_data)
	a = np.asarray(ns_data[start:finish])
	if start - acceptable_shift_left < 0:
		return 0
	delt = finish - start
	for i in range(start - acceptable_shift_left, start + acceptable_shift_right):
		b = np.asarray(sgn_data[i:delt + i])
		cor = np.sum(a * b)
		if cor > max:
			max = cor
			index = i - start
	return index


class OverAddFDAF():
	def __init__( self, dft, mu, eps, alf ):
		self.dir = os.getcwd()
		self.N = dft
		self.N2 = 2 * dft #size of Fourier Transform window
		self.alf = alf
		self.lmb = 1 - self.alf
		self.mu = mu
		self.eps = eps
		self.zero = np.asarray([0 for i in range(self.N)])
		self.Old_SgnFft = np.asarray([0 for i in range(self.N2)])
		self.mtrJ = np.asarray([(-1)**i for i in range(self.N2)])

		self.power = np.asarray([random.random() for i in range(self.N2)])

	def initWight( self, type='random' ):
		error = np.asarray([1 for i in range(self.N)], dtype=np.float32)
		wight = np.asarray([])
		if type == 'zero':
			wight = np.asarray([0 for i in range(self.N2)], dtype=np.float32)
		elif type == 'random':
			wight = np.asarray([random.random() for i in range(self.N2)])
		return wight, error

	def OverAddFft( self, x ):
		x2 = np.concatenate((x, self.zero))
		_SgnFft = np.fft.fft(x2)
		SgnFft = _SgnFft + self.mtrJ*self.Old_SgnFft
		self.Old_SgnFft = _SgnFft
		return SgnFft

	def BinPower( self, SgnFft ):
		pwr = SgnFft * SgnFft.conjugate()
		self.power = self.alf * self.power + self.lmb * pwr
		mu = self.mu / (self.power + self.eps)
		return mu

	def adapt( self, e, sfft, W ):
		mu = self.BinPower(sfft)
		tfft = sfft.conjugate()
		full_e = np.concatenate((e, self.zero))
		efft = np.fft.fft(full_e)
		dW = efft * tfft * mu
		W = W + dW
		return W

	# d array has size N, x array has size N, W array has size 2N
	def Run( self, x, W, d, sfft ):
		Y = sfft * W
		z = np.fft.ifft(Y)
		y = np.real(z[:self.N] / self.N2)
		e = d - y
		return y, e

MU = 79
EPS = 0
DFT = 512
ALF = 0.71
RATE = 16000

if __name__ == '__main__':
	Buff = []
	wv = Wave.PyWav()
	f = OverAddFDAF(DFT, MU, EPS, ALF)

	NoiseFile = f.dir + '\\Waves\\' + 'micSplin7.wav'
	UsefullFile = f.dir + '\\Waves\\' + 'micSplin1.wav'

	signal = np.asarray(wv.ReadWav(UsefullFile), dtype=np.float32)
	noise = np.asarray(wv.ReadWav(NoiseFile), dtype=np.float32)
	Shift = Correlation(noise, signal, RATE, 3 * RATE, 0, 300)
	W, e = f.initWight(type='zero')

	now = datetime.datetime.now()
	for i in range(0, len(signal) - f.N - Shift, f.N):
		x = noise[i:i + f.N]
		d = signal[i + Shift:i + f.N + Shift]
		sfft = f.OverAddFft(x)
		y, e = f.Run(x, W, d, sfft)
		W = f.adapt(e, sfft, W)
		Buff = [*Buff, *e]
	print('spend time = ', datetime.datetime.now() - now)
	wv.WriteWav(f.dir + '/ResultWaves/' + 'Out_DF_' + str(f.N) + '.wav', Buff)
