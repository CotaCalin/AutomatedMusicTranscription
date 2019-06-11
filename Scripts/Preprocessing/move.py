import os
import sys
import wave
import json

from pydub import AudioSegment
from pydub.utils import make_chunks
from spectrogram import SpectrogramBuilder

CHUNKS_OUTPUT_DIR = ""
SPECTRO_OUTPUT_DIR = ""
OUTPUT_DIR_LEN = 4

chunk_length_ms = 1000 / 16 # pydub calculates in millisec
chunk_length_s = chunk_length_ms / 1000

def BuildGlobals(rootDir):
    global CHUNKS_OUTPUT_DIR, SPECTRO_OUTPUT_DIR

    CHUNKS_OUTPUT_DIR = os.path.join(rootDir, "ChunksOutput")
    SPECTRO_OUTPUT_DIR = os.path.join(rootDir, "Spectrograms")
    current_iteration = 0
    if os.path.isdir(SPECTRO_OUTPUT_DIR):
        current_iteration = max([int(x) for x in os.listdir(SPECTRO_OUTPUT_DIR)])

    current_iteration = max(current_iteration, 0) + 1

    CHUNKS_OUTPUT_DIR = os.path.join(CHUNKS_OUTPUT_DIR, str(current_iteration))
    SPECTRO_OUTPUT_DIR = os.path.join(SPECTRO_OUTPUT_DIR, str(current_iteration))
    print("Current iteration is: %d" % (current_iteration))

def CreateOutputDirs():
    if not os.path.isdir(CHUNKS_OUTPUT_DIR):
        os.makedirs(CHUNKS_OUTPUT_DIR)
    if not os.path.isdir(SPECTRO_OUTPUT_DIR):
        os.makedirs(SPECTRO_OUTPUT_DIR)

def GetNotesForChunk(Index, MidiDict):
    chunkTimeStart = Index * chunk_length_s
    chunkTimeEnd = chunkTimeStart + chunk_length_s
    starts = []
    ends = []
    notes = []

    for i in range(len(MidiDict["OnsetTime"])):
        if chunkTimeStart >= MidiDict["OnsetTime"][i] or (i >= 1 and chunkTimeStart >= MidiDict["OnsetTime"][i - 1]):
            if chunkTimeStart >= MidiDict["OffsetTime"][i]:
                continue
            starts.append(MidiDict["OnsetTime"][i])
            if chunkTimeEnd <= MidiDict["OffsetTime"][i] or (i < len(MidiDict["OffsetTime"])-1 and chunkTimeEnd <= MidiDict["OffsetTime"][i+1]):
                ends.append(MidiDict["OffsetTime"][i])
                notes.append(MidiDict["MidiPitch"][i])

    return starts,ends,notes

def splitInput(inputPath):
    content = os.listdir(inputPath)
    for file in content:
        #print(file)
        fileName = file.split(".")[0]
        #note = fileName.split("-")[-1]
        #print(note[:-1])
        if not file.endswith(".wav"):
            continue

        filePath = os.path.join(cwd, file)
        print(filePath)
        txtFilePath = filePath.split(".wav")[0] + ".txt"
        midiDict = {
                "OnsetTime" : [],
                "OffsetTime" : [],
                "MidiPitch" : []
                }

        #with open(txtFilePath) as f:
            #print(f.read())
            #for line in f.read().splitlines():
            #    if "OnsetTime" in line:
            #        continue
#
             #   line = line.replace("\t", " ")
            #    line = line.split(" ")
             #   midiDict["OnsetTime"].append(float(line[0]))
              ##  midiDict["OffsetTime"].append(float(line[1]))
              #  midiDict["MidiPitch"].append(int(line[2]))

        myaudio = AudioSegment.from_file(filePath , "wav")

        chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
        #Export all of the individual chunks as wav files

        for i, chunk in enumerate(chunks):
            start, end, notes = GetNotesForChunk(i, midiDict)
            dumpDict = {
                    "OnsetTime" : start,
                    "OffsetTime" : end,
                    "MidiPitch" : notes
                    }

            chunk_name = "{0}_{1}.wav".format(fileName, i)
            print("exporting", chunk_name)
            destination = os.path.join(CHUNKS_OUTPUT_DIR, chunk_name)
            chunk.export(destination, format="wav")

            #with open(destination+".txt", 'w') as f:
            #    json.dump(dumpDict, f, sort_keys=True,
            #            indent=4, separators=(',', ': '))


        #input()

if "__main__" == __name__:
    print("Usage: " + str(sys.argv))

    cwd = os.getcwd()

    BuildGlobals(cwd)
    CreateOutputDirs()

    if len(sys.argv) < 2:
        exit()
    cwd = sys.argv[1]

    splitInput(cwd)

    #builder = SpectrogramBuilder("d:\\git\\licenta\\AutomatedMusicTranscription\\Scripts\\Preprocessing", SPECTRO_OUTPUT_DIR)
    #builder.build_spectrograms()
