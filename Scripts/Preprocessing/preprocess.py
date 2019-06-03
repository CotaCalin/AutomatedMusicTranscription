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

from Utils.cfg_reader import CfgReader
from Utils.logger import LoggerFactory

cfg = CfgReader()
logger = LoggerFactory.getLogger(__file__)
data_dir = '../input_dir/'

window_size = cfg.getValue("WindowSize")
sr = cfg.getValue("SamplingRate")
min_midi = cfg.getValue("MinMidiValue")
max_midi = cfg.getValue("MaxMidiValue")
hop_length = cfg.getValue("HopLength")

def joinAndCreate(basePath,new):
    newPath = os.path.join(basePath,new)
    if not os.path.exists(newPath):
        os.mkdir(newPath)
    return newPath

def Wav2NpInput(fileName, bin_multiple, spec_type='cqt'):
    logger.logInfo("Converting wav file {0} into np array".format(fileName))

    bins_per_octave = 12 * bin_multiple #should be a multiple of 12
    n_bins = (max_midi - min_midi + 1) * bin_multiple

    y, _ = librosa.load(fileName,sr)
    S = librosa.cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                      bins_per_octave=bins_per_octave, n_bins=n_bins)

    #y = madmom.audio.signal.Signal(fileName, sample_rate=sr, num_channels=1)
    #S = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(y,fmin=librosa.midi_to_hz(min_midi),
    #                            hop_size=hop_length, num_bands=bins_per_octave, fft_size=4096)

    S = S.T

    S = np.abs(S)
    minDB = np.min(S)
    S = np.pad(S, ((window_size//2,window_size//2),(0,0)), 'constant', constant_values=minDB)

    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                sr=sr)
    plt.savefig(fileName + ".png", bbox_inches="tight")
    plt.close('all')

    windows = []
    # IMPORTANT NOTE:
    # Since we pad the the spectrogram frame,
    # the onset frames are actually `offset` frames.
    # To obtain a window of the center frame at each true index, we take a slice from i to i+window_size
    # starting at frame 0 of the padded spectrogram
    for i in range(S.shape[0]-window_size+1):
        w = S[i:i+window_size,:]
        windows.append(w)

    x = np.array(windows)
    return x

def Midi2NpOutput(pm_mid,times):
    piano_roll = pm_mid.get_piano_roll(fs=sr,times=times)[min_midi:max_midi+1].T
    piano_roll[piano_roll > 0] = 1
    return piano_roll

def organize(args):
    valCnt = 1
    testPrefix = 'ENS'

    path = os.path.join('models',args['model_name'])
    dpath = os.path.join(path,'data')

    train_path = joinAndCreate(dpath,'train')
    test_path = joinAndCreate(dpath,'test')
    val_path = joinAndCreate(dpath,'val')

    for ddir in os.listdir(dpath):
        if os.path.isdir(os.path.join(dpath,ddir)) and not isSplitFolder(ddir):
            #print h5file
            if ddir.startswith(testPrefix):
                os.rename(os.path.join(dpath,ddir), os.path.join(test_path,ddir))
            elif valCnt > 0:
                os.rename(os.path.join(dpath,ddir), os.path.join(val_path,ddir))
                valCnt -= 1
            else:
                os.rename(os.path.join(dpath,ddir), os.path.join(train_path,ddir))

def preprocess(args):
    bin_multiple = cfg.getValue("BinMultiple")

    framecnt, addCount, errCount = 0, 0, 0
    inputs, outputs = [], []

    # There's an incompatibility between pretty-midi and MAPS dataset so this is needed
    # https://github.com/craffel/pretty-midi/issues/112
    pretty_midi.pretty_midi.MAX_TICK = 1e10

    data_dir = args["data_dir"]

    for file in os.listdir(data_dir):
        if file.endswith(".wav"):
            prefix = os.path.join(data_dir, file.split(".wav")[0])
            audioFileName = prefix + ".wav"
            midiFileName = prefix + ".mid"
            txtFileName = prefix + ".txt"

            inputnp = Wav2NpInput(audioFileName,bin_multiple=bin_multiple)

            prettyMidi = pretty_midi.PrettyMIDI(midiFileName)
            times = librosa.frames_to_time(np.arange(inputnp.shape[0]),sr=sr,hop_length=hop_length)
            outputnp = Midi2NpOutput(prettyMidi,times)

            if inputnp.shape[0] == outputnp.shape[0]:
                logger.logInfo("adding to dataset fprefix {}".format(prefix))

                addCount += 1
                framecnt += inputnp.shape[0]
                logger.logInfo("framecnt is {}".format(framecnt))

                inputs.append(inputnp)
                outputs.append(outputnp)
            else:
                logger.logError("error for fprefix {}".format(prefix))
                logger.logError(str(inputnp.shape))
                logger.logError(str(outputnp.shape))

                errCount += 1

    subdir = data_dir
    if addCount:
        inputs = np.concatenate(inputs)
        outputs = np.concatenate(outputs)

        fn = subdir.split('/')[-1]
        if not fn:
            fn = subdir.split('/')[-2]
        # save inputs,outputs to hdf5 file
        #print(fn)
        fnpath = joinAndCreate('d:\\git\\licenta\\AutomatedMusicTranscription\\Scripts\\Model\\models\\new\\data\\train','')

        mmi = np.memmap(filename=os.path.join(fnpath,'input.dat'), mode='w+',shape=inputs.shape)
        mmi[:] = inputs[:]
        mmo = np.memmap(filename=os.path.join(fnpath,'output.dat'), mode='w+',shape=outputs.shape)
        mmo[:] = outputs[:]
        del mmi
        del mmo



if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Preprocess MIDI/Audio file pairs into ingestible data')

    parser.add_argument('data_dir',
                        help='Path to data dir, searched recursively, used for naming HDF5 file')

    args = vars(parser.parse_args())
    print(args)

    preprocess(args)
