
from UI.ui_main import UIMain
from controller import Controller
from Utils.cfg_reader import CfgReader
from Model.model_factory import ModelFactory
from trainer import Trainer

DEFAULT_CFG = "../licenta.cfg"

if "__main__" == __name__:
    factory = ModelFactory()
    cfg = CfgReader(DEFAULT_CFG)
    trainer = Trainer(factory)
    cont = Controller(cfg, trainer)
    ui = UIMain(cont)
