import os
import sys
import json

DEFAULT_CFG = "../../licenta.cfg"

class CfgReader():
    def __init__(self, cfg_path=DEFAULT_CFG):
        self._path = cfg_path
        self._js = ""
        with open(cfg_path, "r") as f:
            self._js = json.load(f)

    def getValue(self, key):
        return self._js[key]

if "__main__" == __name__:
    cfg = CfgReader()
    print(cfg.getValue("OutputDir"))
    print(cfg.getValue("Root"))