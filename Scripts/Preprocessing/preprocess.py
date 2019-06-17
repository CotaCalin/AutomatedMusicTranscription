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
import mido
from Utils.logger import LoggerFactory
import Utils.utils as utils

class Preprocessor:
    def __init__(self, cfg):
        self.__cfg = cfg
        self.__logger = LoggerFactory.getLogger(__file__)

        self.data_dir = cfg.getValue("DataDir")

        self.midiUtils = None
        self.converter = None
        self.prediction_note = "pred_note.txt"

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

    def preprocessOne(self, inPath, outPath):
        tempo = 44000
        tpb = 3

        if os.path.exists(os.path.join(outPath, self.prediction_note)):
            with open(os.path.join(outPath, self.prediction_note)) as f:
                lines = f.read().splitlines()
                tempo = int(lines[0])
                tpb = int(lines[1])
        else:
            self.midiUtils.clearChunks()
            tempo, tpb = self.midiUtils.split_midi(inPath, outPath)
            chunks = self.midiUtils.getChunks()
            self.converter.MidiToWav(inPath, os.path.join(outPath, "original.wav"))
            with open(os.path.join(outPath, self.prediction_note), 'w') as f:
                f.write(str(tempo) + "\n" + str(tpb))


            for source in chunks.keys():
                for midi in chunks[source]:
                    self.__logger.logInfo("Converting " + midi)
                    self.converter.MidiToWav(midi, midi[:-4] + ".wav")
                    self.converter.WavToSpec(midi[:-4] + ".wav", midi[:-4] + ".jpg")

        x = []
        filenums = []
        jpg = [x for x in os.listdir(outPath) if x.endswith(".jpg")]

        for filename in jpg:
            self.__logger.logInfo("Loading " + filename)

            im = Image.open(os.path.join(outPath, filename))
            im = im.crop((14, 13, 594, 301))
            resize = im.resize((49, 145), Image.NEAREST)
            resize.load()
            arr = np.asarray(resize, dtype="float32")

            x.append(arr)

            filenums.append(int(filename.split("_")[-1].split(".")[0]))

        x = np.array(x)
        x /= 255.0
        return x, filenums, tempo, tpb

    def preprocessOneWav(self, inPath, outPath):
        chunks, tempo = self.midiUtils.split_wav(inPath, outPath)
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
        return x, filenums, mido.bpm2tempo(tempo)


    def preprocess(self):
        self.__logger.logInfo("Begin preprocessing")
        self.midiUtils.split_all(True)

        for midi in os.listdir(self.__cfg.getValue("OutputDir")):
            if not midi.endswith(".mid"):
                continue

            print(midi)
            midi = os.path.join(self.__cfg.getValue("OutputDir"), midi)
            if os.path.isfile(midi[:-4] + ".jpg"):
                continue
            self.converter.MidiToWav(midi, midi[:-4] + ".wav")
            self.converter.WavToSpec(midi[:-4] + ".wav", midi[:-4] + ".jpg")

        return
        chunks = self.midiUtils.getChunks()
        for source in chunks.keys():
            for midi in chunks[source]:
                print(midi)
                self.converter.MidiToWav(midi, midi[:-4] + ".wav")
                self.converter.WavToSpec(midi[:-4] + ".wav", midi[:-4] + ".jpg")
