from __future__ import division

from collections import defaultdict
import sys, os
import argparse
import numpy as np
#import pandas as pd
import pretty_midi
import librosa
import librosa.display
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import madmom
from PIL import Image
from Utils.cfg_reader import CfgReader
import re
from Utils.logger import LoggerFactory

class Preprocessor:
    def __init__(self, cfg):
        self.__cfg = cfg
        self.__logger = LoggerFactory.getLogger(__file__)

        self.data_dir = cfg.getValue("DataDir")

        self.window_size = cfg.getValue("WindowSize")
        self.sr = cfg.getValue("SamplingRate")
        self.min_midi = cfg.getValue("MinMidiValue")
        self.max_midi = cfg.getValue("MaxMidiValue")
        self.hop_length = cfg.getValue("HopLength")
        self.bin_multiple = cfg.getValue("BinMultiple")
        self.trainPercentage = 0.7
        self.valPercentage = 0.1
        self.testPercentage = 0.2
        self.midiUtils = None
        self.converter = None

    def setMidiUtil(self, newMidi):
        self.midiUtils = newMidi

    def setConverter(self, newConverter):
        self.converter = newConverter

    def setCfg(self, newCfg):
        self.__cfg = newCfg

    @staticmethod
    def joinAndCreate(basePath,new):
        newPath = os.path.join(basePath,new)
        if not os.path.exists(newPath):
            os.makedirs(newPath, exist_ok=True)
        return newPath

    def Wav2NpInputOld(self, fileName, bin_multiple, spec_type='cqt'):
        self.__logger.logInfo("Converting wav file {0} into np array".format(fileName))

        bins_per_octave = 12 * bin_multiple #should be a multiple of 12
        n_bins = (self.max_midi - self.min_midi + 1) * bin_multiple

        y, _ = librosa.load(fileName,self.sr)
        S = librosa.cqt(y,fmin=librosa.midi_to_hz(self.min_midi), sr=self.sr, hop_length=self.hop_length,
                          bins_per_octave=bins_per_octave, n_bins=n_bins)

        S = S.T
        S = np.abs(S)

        minDB = np.min(S)
        S = np.pad(S, ((self.window_size//2,self.window_size//2),(0,0)), 'constant', constant_values=minDB)

        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                    sr=self.sr)
        plt.savefig(fileName + ".png", bbox_inches="tight")
        plt.close('all')

        windows = []
        # IMPORTANT NOTE:
        # Since we pad the the spectrogram frame,
        # the onset frames are actually `offset` frames.
        # To obtain a window of the center frame at each true index, we take a slice from i to i+window_size
        # starting at frame 0 of the padded spectrogram
        #print(S.shape)
        for i in range(S.shape[0]-self.window_size+1):
            w = S[i:i+self.window_size,:]
            #print(len(w[0]))
            #librosa.display.specshow(librosa.amplitude_to_db(w, ref=np.max),
                                        #sr=self.sr)
            #plt.savefig(fileName + "{0}.png".format(i), bbox_inches="tight")
            #plt.close('all')
            #input()
            windows.append(w)

        x = np.array(windows)
        return x

    def Midi2NpOutputOld(self, pm_mid,times):
        piano_roll = pm_mid.get_piano_roll(fs=self.sr,times=times)[self.min_midi:self.max_midi+1].T
        piano_roll[piano_roll > 0] = 1
        print(len(piano_roll))
        return piano_roll

    def preprocessOneOld(self, path):
        self.__logger.logInfo("Begin preprocessing")
        pretty_midi.pretty_midi.MAX_TICK = 1e10

        return self.Wav2NpInput(path,bin_multiple=self.bin_multiple)


    def preprocessOld(self):
        framecnt, addCount, errCount = 0, 0, 0
        inputs, outputs = [], []
        self.__logger.logInfo("Begin preprocessing")

        # There's an incompatibility between pretty-midi and MAPS dataset so this is needed
        # https://github.com/craffel/pretty-midi/issues/112
        pretty_midi.pretty_midi.MAX_TICK = 1e10

        for s in os.listdir(self.data_dir):
            subdir = os.path.join(self.data_dir, s)
            if not os.path.isdir(subdir):
                continue

            for file in os.listdir(subdir):
                if file.endswith(".wav"):
                    prefix = os.path.join(subdir, file.split(".wav")[0])
                    audioFileName = prefix + ".wav"
                    midiFileName = prefix + ".mid"
                    txtFileName = prefix + ".txt"

                    inputnp = self.Wav2NpInput(audioFileName,bin_multiple=self.bin_multiple)

                    prettyMidi = pretty_midi.PrettyMIDI(midiFileName)
                    times = librosa.frames_to_time(np.arange(inputnp.shape[0]),sr=self.sr,hop_length=self.hop_length)
                    outputnp = self.Midi2NpOutput(prettyMidi,times)
                    print(inputnp.shape)
                    print(outputnp.shape)

                    if inputnp.shape[0] == outputnp.shape[0]:
                        self.__logger.logInfo("adding to dataset fprefix {}".format(prefix))

                        addCount += 1
                        framecnt += inputnp.shape[0]
                        self.__logger.logInfo("framecnt is {}".format(framecnt))

                        inputs.append(inputnp)
                        outputs.append(outputnp)
                    else:
                        self.__logger.logError("error for fprefix {}".format(prefix))
                        self.__logger.logError(str(inputnp.shape))
                        self.__logger.logError(str(outputnp.shape))

                        errCount += 1

            if addCount:
                path = os.path.join(self.__cfg.getValue("OutputDir"), self.__cfg.getValue("ModelName"))
                data_path = os.path.join(path,'data')
                train_path = self.joinAndCreate(data_path,'train')
                test_path = self.joinAndCreate(data_path,'test')
                val_path = self.joinAndCreate(data_path,'val')
                train_path = self.joinAndCreate(train_path, s)
                test_path = self.joinAndCreate(test_path, s)
                val_path = self.joinAndCreate(val_path, s)

                train_inputs = np.concatenate(inputs[:int(len(inputs)*self.trainPercentage)])
                train_outputs = np.concatenate(outputs[:int(len(outputs)*self.trainPercentage)])

                mmi = np.memmap(filename=os.path.join(train_path,'input.dat'), mode='w+',shape=train_inputs.shape)
                mmi[:] = train_inputs[:]
                mmo = np.memmap(filename=os.path.join(train_path,'output.dat'), mode='w+',shape=train_outputs.shape)
                mmo[:] = train_outputs[:]
                del mmi
                del mmo

                val_inputs = np.concatenate(inputs[int(len(inputs)*self.trainPercentage):int(len(inputs)*(self.valPercentage + self.trainPercentage))])
                val_outputs = np.concatenate(outputs[int(len(inputs)*self.trainPercentage):int(len(outputs)*(self.valPercentage + self.trainPercentage))])

                mmi = np.memmap(filename=os.path.join(val_path,'input.dat'), mode='w+',shape=val_inputs.shape)
                mmi[:] = val_inputs[:]
                mmo = np.memmap(filename=os.path.join(val_path,'output.dat'), mode='w+',shape=val_outputs.shape)
                mmo[:] = val_outputs[:]
                del mmi
                del mmo

                test_inputs = np.concatenate(inputs[int(len(inputs)*(1-self.testPercentage)):])
                test_outputs = np.concatenate(outputs[int(len(outputs)*(1-self.testPercentage)):])

                mmi = np.memmap(filename=os.path.join(test_path,'input.dat'), mode='w+',shape=test_inputs.shape)
                mmi[:] = test_inputs[:]
                mmo = np.memmap(filename=os.path.join(test_path,'output.dat'), mode='w+',shape=test_outputs.shape)
                mmo[:] = test_outputs[:]
                del mmi
                del mmo

    def preprocessOne(self, inPath, outPath):
        self.midiUtils.clearChunks()
        tempo = self.midiUtils.split_midi(inPath, outPath)
        chunks = self.midiUtils.getChunks()
        self.converter.MidiToWav("test.mid", "test.wav")

        for source in chunks.keys():
            for midi in chunks[source]:
                print(midi)
                self.converter.MidiToWav(midi, midi[:-4] + ".wav")
                self.converter.WavToSpec(midi[:-4] + ".wav", midi[:-4] + ".jpg")

        x = []
        filenums = []

        for filename in os.listdir(outPath):
            if not filename.endswith(".jpg"):
                continue

            im = Image.open(os.path.join(outPath, filename))
            im = im.crop((14, 13, 594, 301))
            resize = im.resize((49, 145), Image.NEAREST)
            resize.load()
            arr = np.asarray(resize, dtype="float32")

            x.append(arr)

            filenums.append(int(filename.split("_")[-1].split(".")[0]))

        x = np.array(x)
        x /= 255.0
        return x, filenums, tempo

    def preprocessOneWav(self, inPath, outPath):
        chunks = self.midiUtils.split_wav(inPath, outPath)
        print(len(chunks))
        for source in chunks:
            self.converter.WavToSpec(source, source[:-4] + ".jpg")

        x = []
        filenums = []

        for filename in os.listdir(outPath):
            if not filename.endswith(".jpg"):
                continue

            im = Image.open(os.path.join(outPath, filename))
            im = im.crop((14, 13, 594, 301))
            resize = im.resize((49, 145), Image.NEAREST)
            resize.load()
            arr = np.asarray(resize, dtype="float32")

            x.append(arr)

            filenums.append(int(filename.split("_")[-1].split(".")[0]))

        x = np.array(x)
        x /= 255.0
        return x, filenums


    def preprocess(self):
        self.__logger.logInfo("Begin preprocessing")
        self.midiUtils.split_all()

        for midi in os.listdir(self.__cfg.getValue("OutputDir")):
            print(midi)
            midi = os.path.join(self.__cfg.getValue("OutputDir"), midi)
            self.converter.MidiToWav(midi, midi[:-4] + ".wav")
            self.converter.WavToSpec(midi[:-4] + ".wav", midi[:-4] + ".jpg")

        return
        chunks = self.midiUtils.getChunks()
        for source in chunks.keys():
            for midi in chunks[source]:
                print(midi)
                self.converter.MidiToWav(midi, midi[:-4] + ".wav")
                self.converter.WavToSpec(midi[:-4] + ".wav", midi[:-4] + ".jpg")
