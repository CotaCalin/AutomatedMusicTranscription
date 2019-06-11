import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from os import listdir
from os.path import isfile, split, join, isdir
from pydub import AudioSegment
from pydub.utils import make_chunks
import os

class MidiUtils:
    def __init__(self, sourceDir, destinationDir):
        self.__destinationDir = destinationDir
        self.__sourceDir = sourceDir
        self.__midis = []
        self.__default_tempo = 500000
        self.__target_segment_len = 1 / 16
        self.__chunks = {}
        self.getMidis()

    def getChunks(self):
        return self.__chunks

    def getMidiFiles(self):
        return self.__midis

    def getMidis(self):
        if not isdir(self.__sourceDir):
            raise Exception("{0} is not a valid Directory".format(self.__sourceDir))

        #print(listdir(self.__sourceDir))
        self.__midis = [join(self.__sourceDir, x) for x in listdir(self.__sourceDir) if x.endswith('.mid')]
        #print(self.__midis)

    def split_all(self):
        print(self.__midis)
        for mid in self.__midis:
            self.split_midi(mid)

    def clearChunks(self):
        self.chunks = {}

    def split_midi(self, mid_file, destinationDir=""):
        '''Split midi file into many chunks'''
        if destinationDir is "":
            destinationDir = self.__destinationDir

        song_name = split(mid_file)[-1][:-4]
        print(song_name)
        mid = MidiFile(mid_file)
        self.__chunks[mid_file] = []

        # identify the meta messages
        metas = []
        tempo = self.__default_tempo
        for msg in mid:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            if msg.is_meta:
                print(msg)
                metas.append(msg)

        target = MidiFile()
        track = MidiTrack()
        for i in range(len(metas)):
            metas[i].time = int(mido.second2tick(metas[i].time, 960, tempo))
        track.extend(metas)
        target.tracks.append(track)
        for msg in mid:
        # Skip non-note related messages
            if msg.is_meta:
                continue
            msg.time = int(mido.second2tick(msg.time, 960, tempo))

            track.append(msg)

        target.save("test.mid")
        input()

        mid = MidiFile("test.mid")
        self.__chunks["test.mid"] = []

        # identify the meta messages
        metas = []
        tempo = self.__default_tempo
        for msg in mid:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            if msg.is_meta:
                print(msg)
                metas.append(msg)

        for i in range(len(metas)):
            metas[i].time = int(mido.second2tick(metas[i].time, mid.ticks_per_beat, tempo))

        target = MidiFile()
        track = MidiTrack()
        track.extend(metas)
        target.tracks.append(track)

        prefix = 0
        time_elapsed = 0
        for msg in mid:
        # Skip non-note related messages
            if msg.is_meta:
                continue

            time_elapsed += msg.time

            if msg.type is not 'end_of_track':
                print(msg.time)
                msg.time = int(mido.second2tick(msg.time, mid.ticks_per_beat, tempo))
                print(mido.tick2second(msg.time, mid.ticks_per_beat, tempo))
                track.append(msg)
                for msg in track:
                    print(msg)
            if msg.type is 'end_of_track' or time_elapsed >= self.__target_segment_len:
                track.append(MetaMessage('end_of_track'))
                dest = join(destinationDir, song_name + '_{}.mid'.format(prefix))
                target.ticks_per_beat = mid.ticks_per_beat
                target.save(dest)
                self.__chunks[mid_file].append(dest)
                target = MidiFile()
                track = MidiTrack()
                track.extend(metas)
                target.tracks.append(track)
                time_elapsed = 0
                prefix += 1

        return tempo


    def merge_midi(self, input_dir, output, tempo_override=None):
        '''Merge midi files into one'''
        midis = []
        for midi in listdir(input_dir):
            midis.append(join(input_dir, midi))

        pairs = [(int(x[:-4].split('_')[-1]), x) for x in midis]
        pairs = sorted(pairs, key=lambda x: x[0])
        midis = [join(input_dir, x[1]) for x in pairs]

        mid = MidiFile(midis[0])
        # identify the meta messages
        metas = []
        # tempo = default_tempo
        tempo = self.__default_tempo
        if tempo_override:
            tempo = tempo_override
        for msg in mid:
            if msg.type is 'set_tempo':
                tempo = msg.tempo
                print(tempo)
                input()
            if msg.is_meta:
                metas.append(msg)
        for meta in metas:
            meta.time = int(mido.second2tick(meta.time, mid.ticks_per_beat, tempo))

        target = MidiFile()
        track = MidiTrack()
        track.extend(metas)
        target.tracks.append(track)
        for midi in midis:
            mid = MidiFile(midi)
            for msg in mid:
                if msg.is_meta:
                    continue
                if msg.type is not 'end_of_track':
                    msg.time = int(mido.second2tick(msg.time, mid.ticks_per_beat, tempo))
                    track.append(msg)

        track.append(MetaMessage('end_of_track'))
        target.ticks_per_beat = mid.ticks_per_beat
        target.save(output)
        for msg in target:
            print(msg)

    def split_wav(self, inputPath ,destinationDir=""):
        if destinationDir is "":
            destinationDir = self.__destinationDir
        chunk_length_ms = 1000 / 4 # pydub calculates in millisec

        myaudio = AudioSegment.from_file(inputPath , "wav")
        chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
        #Export all of the individual chunks as wav files

        exported = []
        for i, chunk in enumerate(chunks):
            chunk_name = "{0}_{1}.wav".format(inputPath.split("\\")[-1], i)
            destination = os.path.join(destinationDir, chunk_name)
            print(destination)
            chunk.export(destination, format="wav")
            exported.append(destination)

        return exported



'''
def main():
    a = MidiUtils("d:\\git\\licenta\\AutomatedMusicTranscription\\Scripts\\Preprocessing\\test", "d:\\git\\licenta\\AutomatedMusicTranscription\\Scripts\\Preprocessing\\output_test")
    a.merge_midi("d:\\git\\licenta\\AutomatedMusicTranscription\\Scripts\\test_predict", "test_predict.mid")
    #return
    #a.split_all()
    c = Converter(sf="d:\\datasets\\test\\KeppysSteinwayPianoLite.sf2",
                fs="d:\\tania\\vcpkg-master\\packages\\fluidsynth_x86-windows\\tools\\fluidsynth\\fluidsynth.exe",
                sox="c:\\Program Files (x86)\\sox-14-4-2\\sox.exe")
    c.MidiToWav("d:\\git\\licenta\\AutomatedMusicTranscription\\Scripts\\Preprocessing\\test_predict.mid", "d:\\git\\licenta\\AutomatedMusicTranscription\\Scripts\\Preprocessing\\test_predict.wav")
    return
    chunks = a.getChunks()
    for source in chunks.keys():
        for midi in chunks[source]:
            print(midi)
            c.MidiToWav(midi, midi[:-4] + ".wav")
            c.WavToSpec(midi[:-4] + ".wav", midi[:-4] + ".jpg")

if __name__ == '__main__':
    main()
'''
