from Validators.path_validator import PathValidator
from Preprocessing.preprocess import Preprocessor
from Utils.cfg_reader import CfgReader
from Preprocessing.split_midi import MidiUtils
from Preprocessing.converter import Converter
import os

def testValidatorExist():
    try:
        PathValidator("d:\\datasets\\licenta_demo\\BACH No-03.mid")
        print("Testcase testValidatorExist passed")
    except Exception as e:
        print("Testcase testValidatorExist failed with error: \n" + str(e))


def testValidatorException():
    try:
        PathValidator("d:\\datasets\\licenta_demo\\BACH No-03.mid", isDir=True)
        print("Testcase testValidatorException failed")
    except Exception as e:
        print("Testcase testValidatorException passed")

def testValidatorException2():
    try:
        PathValidator("d:\\datasets\\licenta_demo\\BACH No-03.mid2")
        print("Testcase testValidatorException2 failed")
    except Exception as e:
        print("Testcase testValidatorException2 passed")

def testPreprocessor():
    cfg = CfgReader("d:\\git\\licenta\\AutomatedMusicTranscription\\Scripts\\testing\\testing.cfg")
    p = Preprocessor(cfg)

    util = MidiUtils(cfg.getValue("DataDir"), "D:\\git\\licenta\\AutomatedMusicTranscription\\Scripts\\testing")
    p.setMidiUtil(util)
    c = Converter(sf=cfg.getValue("Sf"),
                fs=cfg.getValue("Fs"),
                sox=cfg.getValue("sox"))
    p.setConverter(c)
    p.preprocessOne("d:\\datasets\\licenta_demo\\trol.mid", "D:\\git\\licenta\\AutomatedMusicTranscription\\Scripts\\testing")

    jpgs = [x for x in os.listdir("D:\\git\\licenta\\AutomatedMusicTranscription\\Scripts\\testing") if x.endswith(".jpg")]

    expected_len = 127

    if len(jpgs) == expected_len:
        print("Testcase testPreprocessor passed")
    else:
        print("Testcase testPreprocessor failed\n" + "{0} != {1}".format(expected_len, len(jpgs)))


testcases = [
    testValidatorExist,
    testValidatorException,
    testValidatorException2,
    testPreprocessor
]

def testAll():
    for tc in testcases:
        tc()

if __name__ == "__main__":
    testAll()
