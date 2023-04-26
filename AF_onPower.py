import numpy as np
import random

import self as self

import Wave
import datetime
import os

ORDER = 5

Params = {
	'mu' : 79, #это лучший параметр
	'oldmu' : 0.1, #это лучший параметр
	'alf': 0.71, #это лучший параметр
	'eps': 0.01,
	'dft'  : 512,
	'rate' : 16000
}

class AFP_Filter():
	def __init__(self):
		self.iterr = 0
		self.INTERVAL = 2 * Params['dft']
		self.DFT = Params['dft']
		self.power = [np.zeros(self.DFT) for i in range(ORDER)]

	def initWight( self, type='random' ):
		error = np.random.sample(self.DFT)
		if type == 'zero':
			wight = [np.zeros((self.DFT,self.DFT)) for i in range(ORDER)]
		elif type == 'random':
			wight = [np.random.sample((self.DFT,self.DFT)) for i in range(ORDER)]
		return wight, error

	def GetPower(self, x, d):
		for i in range(ORDER-1):
			self.power[i] = self.power[i+1]
		self.power[ORDER-1] = np.abs(np.fft.rfft(x))
		D = np.abs(np.fft.rfft(d))
		return D

	def Run( self):
		pass
		#return y, e

	def adapt( self):
		pass
		#return W1, W2

if __name__ == '__main__':
	Buff = []
	wv = Wave.PyWav()
	f = AFP_Filter()
	dir = os.getcwd()

	NoiseFile = dir + '\\SeparateOutput\\' + 'micSplin7.wav'
	UsefullFile = dir + '\\SeparateOutput\\' + 'micSplin1.wav'

	signal = np.asarray(wv.ReadWav(UsefullFile), dtype=np.float32)
	noise = np.asarray(wv.ReadWav(NoiseFile), dtype=np.float32)
	W, e = f.initWight(type='zero')
	now = datetime.datetime.now()
	for i in range(0, len(signal) - self.INTERVAL, self.DFT):
		x = noise[i:i + self.INTERVAL]
		d = signal[i:i + self.INTERVAL]
		#Buff = [*Buff, *e]
	print('spend time = ', datetime.datetime.now() - now)
	wv.WriteWav(dir + '/ResultWaves/' + 'Out_DF_' + str(Params['dft']) + '.wav', Buff)