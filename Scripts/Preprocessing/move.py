import os
import sys
import wave
from pydub import AudioSegment
from pydub.utils import make_chunks
from spectrogram import SpectrogramBuilder

if "__main__" == __name__:
    print("Usage: " + str(sys.argv))


    cwd = os.getcwd()
    CHUNKS_OUTPUT_DIR = os.path.join(cwd, "Output")
    SPECTRO_OUTPUT_DIR = os.path.join(cwd, "Spectrograms")
    if not os.path.isdir(CHUNKS_OUTPUT_DIR):
        os.mkdir(CHUNKS_OUTPUT_DIR)
    if not os.path.isdir(SPECTRO_OUTPUT_DIR):
        os.mkdir(SPECTRO_OUTPUT_DIR)
    if len(sys.argv) < 2:
        exit()
    cwd = sys.argv[1]

    content = os.listdir(cwd)
    for file in content:
        #print(file)
        fileName = file.split(".")[0]
        #note = fileName.split("-")[-1]
        #print(note[:-1])
        if not file.endswith(".wav"):
            continue

        filePath = os.path.join(cwd, file)
        print(filePath)

        myaudio = AudioSegment.from_file(filePath , "wav")
        chunk_length_ms = 1000 / 16 # pydub calculates in millisec
        chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

        #Export all of the individual chunks as wav files

        for i, chunk in enumerate(chunks):
            chunk_name = "{0}_{1}.wav".format(fileName, i)
            print("exporting", chunk_name)
            destination = os.path.join(CHUNKS_OUTPUT_DIR, chunk_name)
            chunk.export(destination, format="wav")

        #input()
    builder = SpectrogramBuilder(CHUNKS_OUTPUT_DIR, SPECTRO_OUTPUT_DIR)
    builder.build_spectrograms()
