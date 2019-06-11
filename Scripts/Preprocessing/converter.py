from subprocess import call
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
'''
fluidsynth.exe -F test.raw sound.sf2 test.midi
sox -b 16 -c 2 -r 44000 -t raw -e signed test.raw test.wav
'''
def cqt(inputFile, outputFile):
  plt.figure(figsize=(7.5, 3.75))
  y, sr = librosa.load(inputFile)
  C = librosa.cqt(y, sr=sr)
  librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                            sr=sr)
  plt.axis('off')
  plt.savefig(outputFile, bbox_inches="tight")
  plt.close('all')

class Converter:
    def __init__(self, sf, fs, sox):
        self.__sf = sf
        self.__fluidSynth = fs
        self.__sox = sox

    def MidiToWav(self, midiFile, outputFile):
        rawTemp = outputFile[:-4] + ".raw"
        call([self.__fluidSynth, '-F', rawTemp, self.__sf, midiFile])
        call([self.__sox, '-b', '16', '-c', '2', '-r', '44000', '-t', 'raw', '-e', 'signed', rawTemp, outputFile])
        os.remove(rawTemp)

    def WavToSpec(self, wavFile, outputFile):
        cqt(wavFile, outputFile)
