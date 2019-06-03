from mido import MidiFile
from logger import LoggerFactory
import os
import sys
import midi2audio
from midi2audio import FluidSynth

logger = None

def playSongs(inputPath):
    songs = os.listdir(inputPath)

    for song in songs:
        print(song)
        songPath = os.path.join(inputPath, song)
        mid = MidiFile(songPath)
        synth = FluidSynth()

        print(songPath)
        synth.play_midi(songPath)

        print(mid.length)
        #for i, track in enumerate(mid.tracks):
        #    print('Track {}: {}'.format(i, track.name))
        #    for msg in track:
        #        print(msg)

if "__main__" == __name__:
    logger = LoggerFactory.getLogger(__file__)
    logger.logInfo("Usage:" + str(sys.argv))

    if len(sys.argv) != 2:
        logger.logError("Wrong input")

    inputPath = sys.argv[1]
    if not os.path.isdir(inputPath):
        logger.logError("Input dir is not a directory")
        exit()

    playSongs(inputPath)
