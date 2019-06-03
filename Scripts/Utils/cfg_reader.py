import os
import sys
import json


class CfgReader():
    def __init__(self, cfg_path):
        self._path = cfg_path
        self._js = ""
        with open(cfg_path, "r") as f:
            self._js = json.load(f)

    def getValue(self, key):
        return self._js[key]

    def updateCfg(self, newCfg):
        with open(newCfg, "r") as f:
            self._js = json.load(f)

        self._path = newCfg

if "__main__" == __name__:
    cfg = CfgReader()
    print(cfg.getValue("OutputDir"))
    print(cfg.getValue("Root"))
