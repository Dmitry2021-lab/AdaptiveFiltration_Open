#IIR - infinite impulse response
'''
J. Shynk Frequency-domain and multirate adaptive filtering
// 1992 Mathematics, Computer Science IEEE Signal Processing Magazine
OverlapSave algorithm (Fig 12)
'''

import numpy as np
import random
import Wave
import datetime
import os

Params = {
	'mu_x' : 79,
	'mu_d' : 79,
	'alf_x': 0.71,
	'alf_d': 0.71,
	'eps_x': 0.5,
	'eps_d': 0.5,
	'dft'  : 512,
	'rate' : 16000
}

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

class IIR_OverSave():
	def __init__( self):
		self.N2 = 2 * Params['dft']
		self.lmb_d = 1 - Params['alf_d']
		self.lmb_x = 1 - Params['alf_x']
		self.old_x = np.zeros(Params['dft'])
		self.old_d = np.zeros(Params['dft'])

		self.power_x = np.random.sample(self.N2)
		self.power_d = np.random.sample(self.N2)
		self.zero = np.zeros(Params['dft'])
		self.oldWX = np.random.sample(self.N2)
		self.oldWD = np.random.sample(self.N2)

	def initWight( self, type='random' ):
		error = np.random.sample(Params['dft'])
		if type == 'zero':
			wight_x =np.zeros(self.N2)
			wight_d = np.zeros(self.N2)
		elif type == 'random':
			wight_x = np.random.sample(self.N2)
			wight_d = np.random.sample(self.N2)
		return wight_x, wight_d, error

	def x_OverFft( self, x ):
		x2 = np.concatenate((self.old_x, x))
		XFft = np.fft.fft(x2)
		self.old_x = x
		return XFft

	def d_OverFft( self, d ):
		d2 = np.concatenate((self.old_d, d))
		DFft = np.fft.fft(d2)
		self.old_d = d
		return DFft

	def xBinPower( self, XFft ):
		pwr = XFft * XFft.conjugate()
		self.power_x = Params['alf_x'] * self.power_x + self.lmb_x * pwr
		mu_x = Params['mu_x'] / (self.power_x + Params['eps_x'])
		return mu_x

	def dBinPower( self, DFft ):
		pwr = DFft * DFft.conjugate()
		self.power_d = Params['alf_d'] * self.power_d + self.lmb_d * pwr
		mu_d = Params['mu_d'] / (self.power_d + Params['eps_d'])
		return mu_d

	def adapt_d( self, d_fft, DW, efft ):
		mu = self.dBinPower(d_fft)
		tfft = d_fft.conjugate()
		dW = efft * tfft * mu
		DW = DW + dW
		return DW

	def adapt_x( self, x_fft, XW, efft ):
		mu = self.xBinPower(x_fft)
		tfft = x_fft.conjugate()
		dW = efft * tfft * mu
		XW = XW + dW
		return XW

	# d массив размером N, x массив размером N, W массив размером 2N
	def Run( self, d, XW, xfft, DW, dfft ):
		Y = xfft * XW + dfft * DW
		z = np.fft.ifft(Y)
		y = np.real(z[Params['dft']:] / self.N2)
		e = d - y
		full_e = np.concatenate((self.zero, e))
		efft = np.fft.fft(full_e)
		return y, e, efft


if __name__ == '__main__':
	Buff = []
	Buff1 = []
	wv = Wave.PyWav()
	f = IIR_OverSave()
	dir = os.getcwd()

	NoiseFile = dir + '\\SeparateOutput\\' + 'micSplin7.wav'
	UsefullFile = dir + '\\SeparateOutput\\' + 'micSplin1.wav'

	signal = np.asarray(wv.ReadWav(UsefullFile), dtype=np.float32)
	noise = np.asarray(wv.ReadWav(NoiseFile), dtype=np.float32)
	Shift = Correlation(noise, signal, Params['rate'], 3 * Params['rate'], 0, 300)
	XW, DW, e = f.initWight(type='zero')
	now = datetime.datetime.now()
	for i in range(0, len(signal) - Params['dft'] - Shift, Params['dft']):
		x = noise[i:i + Params['dft']]
		d = signal[i + Shift:i + Params['dft'] + Shift]
		xfft = f.x_OverFft(x)
		dfft = f.d_OverFft(d)
		y, e, efft = f.Run(d, XW, xfft, DW, dfft)
		XW = f.adapt_x(xfft, XW, efft)
		DW = f.adapt_d(dfft, DW, efft)
		enh = 5*e
		Buff = [*Buff, *enh]
	print('spend time = ', datetime.datetime.now() - now)
	wv.WriteWav(dir + '/ResultWaves/' + 'Out_DF_' + str(Params['dft']) + '.wav', Buff)
