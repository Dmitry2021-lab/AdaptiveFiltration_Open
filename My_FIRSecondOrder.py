#JIA-SIEN SO0 AND KHEE K. PANG Multidelay Block Frequency Domain Adaptive Filter
#I added a second order

import numpy as np
import random
import Wave
import datetime
import os

Params = {
	'mu' : 79, #это лучший параметр
	'oldmu' : 0.1, #это лучший параметр
	'alf': 0.71, #это лучший параметр
	'eps': 0.01,
	'dft'  : 512,
	'rate' : 16000
}

class SecondOrderFilter():
	def __init__( self):
		self.iterr = 0
		self.N2 = 2 * Params['dft']
		self.dir = os.getcwd()
		self.old_x = np.asarray([0 for i in range(Params['dft'])])
		self.zero = np.asarray([0 for i in range(Params['dft'])])
		self.power = np.asarray([0 for i in range(self.N2)])
		self.OldSgnFft1 = np.asarray([0 for i in range(self.N2)])
		self.oldMU1 = np.asarray([0 for i in range(self.N2)])

	def initWight( self, type='random' ):
		error = np.asarray([1 for i in range(Params['dft'])], dtype=np.float32)
		wight1 = np.asarray([])
		wight2 = np.asarray([])
		if type == 'zero':
			wight1 = np.asarray([0 for i in range(self.N2)], dtype=np.float32)
			wight2 = np.asarray([0 for i in range(self.N2)], dtype=np.float32)
		elif type == 'random':
			wight1 = np.asarray([random.random() for i in range(self.N2)])
			wight2 = np.asarray([random.random() for i in range(self.N2)])
		else:
			print('Заполнить массив весов!!!!')
		return wight1, wight2, error

	def OverFft( self, x ):
		x2 = np.concatenate((self.old_x, x))
		SgnFft = np.fft.fft(x2)

		self.OldSgnFft1 = SgnFft

		self.old_x = x
		return SgnFft

	def Run( self, W1, W2, d, sfft  ):
		Y = sfft * W1 + self.OldSgnFft1*W2
		z = np.fft.ifft(Y)
		y = np.real(z[Params['dft']:] / self.N2)
		e = d - y
		return y, e

	def adapt( self, W1, W2, e, x_fft ):
		full_e = np.concatenate((self.zero, e))
		e_fft = np.fft.fft(full_e)
		cx_fft = x_fft.conjugate()

		mu = self.BinPower(cx_fft)
		dW1 = mu * e_fft * cx_fft
		dW2 = self.oldMU1 * e_fft * self.OldSgnFft1.conjugate()
		W1 = W1 + dW1
		W2 = W2 + dW2
		self.oldMU1 = Params['oldmu']*mu
		return W1, W2

	def BinPower( self, SgnFft ):
		pwr = SgnFft * SgnFft.conjugate()
		self.power = Params['alf'] * self.power + (1-Params['alf']) * pwr
		mu = Params['mu'] / (self.power + Params['eps'])
		return mu

if __name__ == '__main__':
	Buff = []
	wv = Wave.PyWav()
	f = SecondOrderFilter()

	NoiseFile = f.dir + '\\SeparateOutput\\' + 'micSplin7.wav'
	UsefullFile = f.dir + '\\SeparateOutput\\' + 'micSplin1.wav'

	signal = np.asarray(wv.ReadWav(UsefullFile), dtype=np.float32)
	noise = np.asarray(wv.ReadWav(NoiseFile), dtype=np.float32)
	W1, W2, e = f.initWight(type='zero')
	now = datetime.datetime.now()
	for i in range(0, len(signal) - Params['dft'], Params['dft']):
		x = noise[i:i + Params['dft']]
		d = signal[i:i + Params['dft']]
		xfft = f.OverFft(x)
		y, e = f.Run(W1, W2, d, xfft)
		W1, W2 = f.adapt(W1, W2, e, xfft)
		Buff = [*Buff, *e]
	print('spend time = ', datetime.datetime.now() - now)
	wv.WriteWav(f.dir + '/ResultWaves/' + 'Out_DF_' + str(Params['dft']) + '.wav', Buff)