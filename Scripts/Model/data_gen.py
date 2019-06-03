from collections import defaultdict
import sys, os
import argparse

import madmom
import numpy as np
import pandas as pd
import pretty_midi
import librosa
import h5py
import math

import numpy as np

def readmm(d,args):
    ipath = os.path.join(d,'input.dat')
    note_range = 88
    n_bins = 3 * note_range
    window_size = 7
    mmi = np.memmap(ipath, mode='r')
    i = np.reshape(mmi,(-1,window_size,n_bins))
    opath = os.path.join(d,'output.dat')
    mmo = np.memmap(opath, mode='r')
    o = np.reshape(mmo,(-1,note_range))
    return i,o

class DataGen:
    def __init__(self, dirpath, batch_size,num_files=1, args=None):
        print('initializing gen for '+dirpath)

        self.mmdirs =  os.listdir(dirpath)
        self.spe = 0 #steps per epoch
        self.dir = dirpath

        for mmdir in self.mmdirs:
            _,outputs = readmm(os.path.join(self.dir,mmdir),args)
            self.spe += len(outputs) // batch_size
        self.num_files = num_files

        self.batch_size = batch_size
        self.current_file_idx = 0
        print('starting with ', self.mmdirs[self.current_file_idx:self.current_file_idx+self.num_files])
        for j in range(self.num_files):
            mmdir = os.path.join(self.dir,self.mmdirs[self.current_file_idx+j])
            i,o = readmm(mmdir,args)
            if j == 0:
                self.inputs,self.outputs = i,o
                print('set inputs,outputs')
            else:
                self.inputs = np.concatenate((self.inputs,i))
                self.outputs = np.concatenate((self.outputs,o))
                print('concatenated')
            self.current_file_idx = (self.current_file_idx + 1) % len(self.mmdirs)
        self.i = 0

    def steps(self):
        return self.spe

    def next(self):
        while True:
            if (self.i+1)*self.batch_size > self.inputs.shape[0]:
                #return rest and then switch files
                x,y = self.inputs[self.i*self.batch_size:],self.outputs[self.i*self.batch_size:]
                self.i = 0
                if len(self.mmdirs) > 1: # no need to open any new files if we only deal with one, like for validation
                    print('switching to ', self.mmdirs[self.current_file_idx:self.current_file_idx+self.num_files])
                    for j in range(self.num_files):
                        mmdir = os.path.join(self.dir,self.mmdirs[self.current_file_idx+j])
                        i,o = readmm(mmdir,args)
                        if j == 0:
                            self.inputs,self.output = i,o
                        else:
                            self.inputs = np.concatenate((self.inputs,i))
                            self.outputs = np.concatenate((self.outputs,o))
                        self.current_file_idx = (self.current_file_idx + 1) % len(self.mmdirs)

            else:
                x,y = self.inputs[self.i*self.batch_size:(self.i+1)*self.batch_size],self.outputs[self.i*self.batch_size:(self.i+1)*self.batch_size]
                self.i += 1
            yield x,y
