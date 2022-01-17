import wave
import struct
import numpy as np

class PyWav:
    def __init__(self):
        self.samplerate = 16000
        self.length = 0
        self.bytes = 2
        #self.Buff = []
        self.bytes

    def ReadWav(self, nameWav):
        waveFile = wave.open(nameWav, 'r')
        self.samplerate = waveFile.getframerate()
        self.length = waveFile.getnframes()
        self.bytes = waveFile.getsampwidth()
        Buff = []
        for i in range(0, self.length):
            waveData = waveFile.readframes(1)
            data = struct.unpack("<h", waveData)
            Buff.append(data[0])

        waveFile.close()
        return Buff

    def WriteWav(self, nameWav, Buff):
        wf = wave.open(nameWav, 'w')
        nchannels = 1
        nframes = len(Buff)
        comptype = "NONE"
        compname = "not compressed"

        wf.setparams((nchannels, self.bytes, self.samplerate, nframes, comptype, compname))

        for el in Buff:
            # write the audio frames to file
            data = struct.pack('<h', np.int16(el))
            wf.writeframes(data)
        wf.close()