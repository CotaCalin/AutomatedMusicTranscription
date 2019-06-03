from Utils.logger import LoggerFactory

class Controller:
    def __init__(self, cfg, trainer):
        self.__logger = LoggerFactory.getLogger(__file__)
        self.__cfg = cfg
        self.__trainer = trainer
        self.__trainer.setCfg(cfg)

    def setCfg(self, newCfg):
        self.__cfg.updateCfg(newCfg)
        self.__trainer.setCfg(newCfg)

    def train(self):
        try:
            self.__trainer.setup()
            self.__trainer.train()

        except Exception as e:
            self.__logger.logError(e)
