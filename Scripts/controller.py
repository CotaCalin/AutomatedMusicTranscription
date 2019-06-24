from Utils.logger import LoggerFactory
import traceback
import numpy as np
from Preprocessing.split_midi import MidiUtils
from Preprocessing.converter import Converter
import Utils.utils as utils
import pretty_midi
import os

class Controller:
    def __init__(self, cfg, preprocessor, trainer):
        self.__logger = LoggerFactory.getLogger(__file__)
        self.__cfg = cfg
        self.__trainer = trainer
        self.__trainer.setCfg(cfg)
        self.__preprocessor = preprocessor
        self.__lastPred = ""
        self.__lastPredOut = ""

    def setCfg(self, newCfg):
        self.__cfg.updateCfg(newCfg)
        self.__trainer.setCfg(newCfg)
        self.__preprocessor.setCfg(newCfg)

    def getLastPredOut(self):
        return self.__lastPredOut

    def loadModel(self):
        if not self.__trainer.isLoaded():
            self.__trainer.setup()
            self.__trainer.loadModel()

    def predict(self, path, progressBar=None):
        try:
            pDir = self.__cfg.getValue("PredictionDir")
            songName = path.split("/")[-1][:-4]
            predictionOutput = os.path.join(pDir, songName)
            if not os.path.isdir(predictionOutput):
                os.makedirs(predictionOutput)

            util = MidiUtils(self.__cfg.getValue("DataDir"), predictionOutput)
            self.__preprocessor.setMidiUtil(util)
            c = Converter(sf=self.__cfg.getValue("Sf"),
                        fs=self.__cfg.getValue("Fs"),
                        sox=self.__cfg.getValue("sox"))
            self.__preprocessor.setConverter(c)
            x_pred, filenums, tempo = None, None, None
            y_pred = []

            if path.endswith("wav"):
                x_pred, filenums, tempo = self.__preprocessor.preprocessOneWav(path, predictionOutput)
            else:
                x_pred, filenums ,tempo, tpb= self.__preprocessor.preprocessOne(path, predictionOutput)

            if not self.__trainer.isLoaded():
                self.__trainer.setup()
            result = self.__trainer.predict(x_pred)

            self.__lastPred = path

            self.handle_prediction(result, filenums, tempo, tpb, predictionOutput)
        except Exception as e:
            self.__logger.logError(e)
            traceback.print_exc()

    def handle_prediction(self, result, filenums, tempo, tpb, predictionOutput):
        notes_unsorted = [result[n] for n in range(len(result))]

        notes = [x for _,x in sorted(zip(filenums, notes_unsorted))]
        for n in range(len(notes)):

            actives = []
            print(sorted(notes[n], reverse=True)[:5])
            for i in range(len(notes[n])):
                if notes[n][i] > 0.25:
                    actives.append(i)
            notes[n] = actives

        i=0
        one_hots=[]
        for actives in notes:
            one_hot = np.zeros((128, 1))
            for note in actives:
                one_hot[note, :] = 1

            one_hots.append(one_hot)

        prOut = os.path.join(predictionOutput, "Prediction")

        if not os.path.isdir(prOut):
            os.makedirs(prOut)

        mid = utils.one_hots_to_pretty_midi(one_hots, tempo, tpb)
        mid.save(os.path.join(prOut, 'predict.mid'))

        c = Converter(sf=self.__cfg.getValue("Sf"),
                    fs=self.__cfg.getValue("Fs"),
                    sox=self.__cfg.getValue("sox"))
        c.MidiToWav(os.path.join(prOut, 'predict.mid'), os.path.join(prOut, 'predict.wav'))
        c.MidiToSheet(os.path.join(prOut, 'predict.mid'), os.path.join(prOut, 'predict_sheet'))
        c.MidiToWav(self.__lastPred, os.path.join(prOut, 'original.wav'))
        c.MidiToSheet(self.__lastPred, os.path.join(prOut, 'original_sheet'))
        self.__lastPredOut = prOut

    def train(self):
        try:
            utils = MidiUtils(self.__cfg.getValue("DataDir"), self.__cfg.getValue("OutputDir"))
            self.__preprocessor.setMidiUtil(utils)
            c = Converter(sf=self.__cfg.getValue("Sf"),
                        fs=self.__cfg.getValue("Fs"),
                        sox=self.__cfg.getValue("sox"))
            self.__preprocessor.setConverter(c)
            self.__preprocessor.preprocess()
            self.__trainer.setup()
            self.__trainer.train()

        except Exception as e:
            self.__logger.logError(e)
            traceback.print_exc()
