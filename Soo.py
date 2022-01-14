#JIA-SIEN SO0 AND KHEE K. PANG Multidelay Block Frequency Domain Adaptive Filter


import numpy as np
import random
import Wave
import datetime
import os

Params = {
	'mu' : 79, #best parametr
	'alf': 0.71, #best parametr
	'eps': 0.01,
	'dft'  : 512,
	'rate' : 16000
}

class Soo():
	def __init__( self):
		self.iterr = 0
		self.N2 = 2 * Params['dft'] #size of Fourier Transform window
		self.dir = os.getcwd()
		self.old_x = np.asarray([0 for i in range(Params['dft'])])
		self.zero = np.asarray([0 for i in range(Params['dft'])])
		self.power = np.asarray([0 for i in range(self.N2)])
		self.lmb = 1 - Params['alf']


	def initWight( self, type='random' ):
		error = np.asarray([1 for i in range(Params['dft'])], dtype=np.float32)
		wight = np.asarray([])
		if type == 'zero':
			wight = np.asarray([0 for i in range(self.N2)], dtype=np.float32)
		elif type == 'random':
			wight = np.asarray([random.random() for i in range(self.N2)])
		else:
			print('Create array!!!!')
		return wight, error

	def OverFft( self, x ):
		x2 = np.concatenate((self.old_x, x))
		SgnFft = np.fft.fft(x2)
		self.old_x = x
		return SgnFft

	def Run( self, x, W, d, sfft ):
		Y = sfft * W
		z = np.fft.ifft(Y)
		y = np.real(z[Params['dft']:] / self.N2)
		e = d - y
		return y, e

	def adapt( self, e, x_fft, W, y ):
		full_e = np.concatenate((self.zero, e))
		e_fft = np.fft.fft(full_e)
		cx_fft = x_fft.conjugate()
		mu = self.BinPower(cx_fft)
		dW = mu * e_fft * cx_fft
		W = W + dW
		return W

	def BinPower( self, SgnFft ):
		pwr = SgnFft * SgnFft.conjugate()
		self.power = Params['alf'] * self.power + self.lmb * pwr
		mu = Params['mu'] / (self.power + Params['eps'])
		return mu

if __name__ == '__main__':
	Buff = []
	wv = Wave.PyWav()
	f = Soo()

	NoiseFile = f.dir + '\\SeparateOutput\\' + 'micSplin7.wav'
	UsefullFile = f.dir + '\\SeparateOutput\\' + 'micSplin1.wav'

	signal = np.asarray(wv.ReadWav(UsefullFile), dtype=np.float32)
	noise = np.asarray(wv.ReadWav(NoiseFile), dtype=np.float32)
	W, e = f.initWight(type='zero')
	now = datetime.datetime.now()
	for i in range(0, len(signal) - Params['dft'], Params['dft']):
		x = noise[i:i + Params['dft']]
		d = signal[i:i + Params['dft']]
		xfft = f.OverFft(x)
		y, e = f.Run(x, W, d, xfft)
		W = f.adapt(e, xfft, W, y)
		Buff = [*Buff, *e]
	print('spend time = ', datetime.datetime.now() - now)
	wv.WriteWav(f.dir + '/ResultWaves/' + 'Out_DF_' + str(Params['dft']) + '.wav', Buff)