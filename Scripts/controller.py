from Utils.logger import LoggerFactory
import traceback
import numpy as np
from Preprocessing.split_midi import MidiUtils
from Preprocessing.converter import Converter
import Utils.utils as utils

class Controller:
    def __init__(self, cfg, preprocessor, trainer):
        self.__logger = LoggerFactory.getLogger(__file__)
        self.__cfg = cfg
        self.__trainer = trainer
        self.__trainer.setCfg(cfg)
        self.__preprocessor = preprocessor

    def setCfg(self, newCfg):
        self.__cfg.updateCfg(newCfg)
        self.__trainer.setCfg(newCfg)
        self.__preprocessor.setCfg(newCfg)

    def predict(self, path):
        try:
            utils = MidiUtils(self.__cfg.getValue("DataDir"), self.__cfg.getValue("OutputDir"))
            self.__preprocessor.setMidiUtil(utils)
            c = Converter(sf=self.__cfg.getValue("Sf"),
                        fs=self.__cfg.getValue("Fs"),
                        sox=self.__cfg.getValue("sox"))
            self.__preprocessor.setConverter(c)
            x_pred, filenums ,tempo= self.__preprocessor.preprocessOne(path, "temp")
            #x_pred, filenums = self.__preprocessor.preprocessOneWav(path, "temp")

            print(len(x_pred))
            self.__trainer.setup()
            result = self.__trainer.predict(x_pred)

            self.handle_prediction(result, filenums, tempo)
        except Exception as e:
            self.__logger.logError(e)
            traceback.print_exc()

    def handle_prediction(self, result, filenums, tempo):
        print(result)
        print(len(result))
        resultCpy = result


        #notes_unsorted = [np.argmax(result[n]) for n in range(len(result))]
        notes_unsorted = [result[n] for n in range(len(result))]

        print(notes_unsorted)
        #print(filenums)
        notes = [x for _,x in sorted(zip(filenums, notes_unsorted))]
        for n in range(len(notes)):
            print(np.argmax(notes[n]))
            print(sorted(notes[n], reverse=True)[:10])
            #input()
            actives = []
            print(notes[n])
            print(type(notes[n]))
            #l = notes[n].tolist()
            for i in range(len(notes[n])):
                if notes[n][i] > 0.5:
                    actives.append(i)

            notes[n] = actives

        print(notes)
        input()
        i=0
        one_hots=[]
        for actives in notes:
            one_hot = np.zeros((128, 1))
            for note in actives:
                one_hot[note, :] = 1

            one_hots.append(one_hot)

        mid = utils.one_hots_to_pretty_midi(one_hots, tempo)
        mid.write('predict.mid')

        #u = MidiUtils(self.__cfg.getValue("DataDir"), self.__cfg.getValue("OutputDir"))
        #u.merge_midi("d:\\git\\licenta\\AutomatedMusicTranscription\\Scripts\\sample", "predict.mid", tempo)

        c = Converter(sf=self.__cfg.getValue("Sf"),
                    fs=self.__cfg.getValue("Fs"),
                    sox=self.__cfg.getValue("sox"))
        c.MidiToWav("predict.mid", "predict.wav")

    def train(self):
        try:
            utils = MidiUtils(self.__cfg.getValue("DataDir"), self.__cfg.getValue("OutputDir"))
            self.__preprocessor.setMidiUtil(utils)
            c = Converter(sf=self.__cfg.getValue("Sf"),
                        fs=self.__cfg.getValue("Fs"),
                        sox=self.__cfg.getValue("sox"))
            self.__preprocessor.setConverter(c)
            #self.__preprocessor.preprocess()
            self.__trainer.setup()
            self.__trainer.train()

        except Exception as e:
            self.__logger.logError(e)
            traceback.print_exc()
