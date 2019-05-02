import os
import sys
import wave
from pydub import AudioSegment
from pydub.utils import make_chunks
from spectrogram import SpectrogramBuilder

if "__main__" == __name__:
    print("Usage: " + str(sys.argv))
    
    
    cwd = os.getcwd()    
    OUTPUT_DIR = os.path.join(cwd, "Output")
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if len(sys.argv) > 1:
        cwd = sys.argv[1]
        
    content = os.listdir(cwd)
    for file in content:
        print(file)
        fileName = file.split(".")[0]
        note = fileName.split("-")[-1]
        print(note[:-1])
        
        filePath = os.path.join(cwd, file)
        
        # with wave.open(filePath, 'rb') as f:
        #     print(f.getnchannels())
        #     print(f.getsampwidth())
        #     print(f.getframerate())
        #     print(f.getnframes())
        #     print(f.getcomptype())
        #     print(f.getparams())
        #     print(f.readframes(100))
            
        #     frameRate = f.getframerate()
        #     numFrames = f.getnframes()
        #     duration = numFrames/frameRate
        #     print(duration)

        myaudio = AudioSegment.from_file(filePath , "wav") 
        chunk_length_ms = 1000 # pydub calculates in millisec
        chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

        #Export all of the individual chunks as wav files

        # for i, chunk in enumerate(chunks):
        #     chunk_name = "{0}_{1}.wav".format(fileName, i)
        #     print("exporting", chunk_name)
        #     destination = os.path.join(OUTPUT_DIR, chunk_name)
        #     chunk.export(destination, format="wav")
            
        #input()
    builder = SpectrogramBuilder(cwd)
    builder.build_spectrograms()