import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import sys
import wave
import pylab
#from spectrogram2 import plotstft


# Generate and plot a constant-Q power spectrum

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

class SpectrogramBuilder():
    def __init__(self, WavPath, DestinationPath):
        self.__wavPath = WavPath
        self.__wav_files = self.get_wavs()
        self.__destinationPath = DestinationPath

    def get_wavs(self):
        ret = []
        for wav_file in os.listdir(self.__wavPath):
            if wav_file.endswith(".wav"):
                ret.append(os.path.join(self.__wavPath, wav_file))

        return ret

    def build_spectrograms(self):
        for wavfile in self.__wav_files:
            self.graph_spectrogram(wavfile)

    def __build_spectrogram(self, filePath):
        sample_rate, samples = wavfile.read(filePath)
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

        #print(spectrogram)
        color_tuple = spectrogram.transpose((1,0,2)).reshape((spectrogram.shape[0]*spectrogram.shape[1],spectrogram.shape[2]))/255.0

        plt.pcolormesh(times, frequencies, color_tuple)
        plt.imshow(spectrogram)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    def graph_spectrogram(self, wav_file):
        # Q Transform
        y, sr = librosa.load(wav_file)
        C = np.abs(librosa.cqt(y, sr=sr))
        librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                                sr=sr)#, x_axis='time', y_axis='cqt_hz')
        #plt.colorbar(format='%+2.0f dB')
        #plt.title('spectrogram of %r' % wav_file)
        plt.tight_layout()
        fileName = 'spectrogram_{0}.png'.format(wav_file.split("\\")[-1])
        plt.savefig(os.path.join(self.__destinationPath, fileName), bbox_inches="tight")
        plt.close('all')

    def get_wav_info(self, wav_file):
        wav = wave.open(wav_file, 'r')
        frames = wav.readframes(-1)
        sound_info = pylab.fromstring(frames, 'int16')
        frame_rate = wav.getframerate()
        wav.close()
        return sound_info, frame_rate
