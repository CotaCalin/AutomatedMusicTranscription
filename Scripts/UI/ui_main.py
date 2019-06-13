
import os
from Utils.logger import LoggerFactory
from Validators.path_validator import PathValidator

class UIMain:
    def __init__(self, controller):
        self.__logger = LoggerFactory.getLogger(__file__)
        self.__controller = controller
        self.__handlers = {
             "0" : self.__exit,
             "1" : self.__train,
             "2" : self.__predict
        }

        self.__trainHandler = {
            "0" : controller.train,
            "1" : self.__trainNewCfg
        }
        self.run()

    @staticmethod
    def __printMenu():
        s =  "Automatic Music Transcription - Thesis Menu:\n"
        s += "\t 0 Exit\n"
        s += "\t 1 Train\n"
        s += "\t 2 Predict\n"
        print(s)

    @staticmethod
    def __printTrainMenu():
        s =  "Let's train baby:\n"
        s += "\t 0 Default Cfg\n"
        s += "\t 1 New Cfg Path\n"
        print(s)

    def __handleInput(self, handler):
        i = input("Choose: ")

        if i not in handler.keys():
            print("Unknown command. Try again")
            return

        try:
            handler[i]()
        except Exception as e:
            self.__logger.logError(e)

    def __exit(self):
        print("Exitting. Bye bye")
        exit(0)

    def __predict(self):
        i = input("Song Path(wav or midi format):")
        try:
            PathValidator(i)
            self.__logger.logInfo("Begin prediction for {0}".format(i))
            self.__controller.predict(i)
        except Exception as e:
            self.__logger.logError(e)

    def __train(self):
        self.__printTrainMenu()
        self.__handleInput(self.__trainHandler)

    def __trainNewCfg(self):
        i = input("New Cfg Path:")
        try:
            PathValidator(i)
            self.__logger.logInfo("Begin training with new cfg {0}".format(i))
            self.__controller.setCfg(i)
            self.__controller.train()
        except Exception as e:
            self.__logger.logError(e)

    def run(self):
        while True:
            self.__printMenu()
            self.__handleInput(self.__handlers)
