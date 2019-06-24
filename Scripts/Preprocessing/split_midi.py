import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from os import listdir
from os.path import isfile, split, join, isdir
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.silence import detect_nonsilent
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

        self.__midis = [join(self.__sourceDir, x) for x in listdir(self.__sourceDir) if x.endswith('.mid')]

    def split_all(self, train=False):
        print(self.__midis)
        for mid in self.__midis:
            if train:
                self.split_midi_train(mid)
            else:
                self.split_midi(mid)

    def clearChunks(self):
        self.chunks = {}

    def split_midi(self, mid_file, destinationDir=""):
        '''Split midi file into many chunks'''
        if destinationDir is "":
            destinationDir = self.__destinationDir

        print(destinationDir)
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

        mid = MidiFile("test.mid")
        self.__chunks["test.mid"] = []

        # identify the meta messages
        metas = []
        tempo = self.__default_tempo
        tpb = mid.ticks_per_beat
        for msg in mid:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            if msg.is_meta:
                metas.append(msg)

        for i in range(len(metas)):
            metas[i].time = 0

        target = MidiFile()
        track = MidiTrack()
        track.extend(metas)
        target.tracks.append(track)

        prefix = 0
        time_elapsed = 0
        ct = 0
        for msg in mid:
        # Skip non-note related messages
            if msg.is_meta:
                continue
            time_elapsed += msg.time

            t = msg.time
            if msg.type is not 'end_of_track':
                msg.time = int(mido.second2tick(msg.time, mid.ticks_per_beat, tempo))
                track.append(msg)
                if msg.type in ['note_on', 'note_off']:
                    ct+=1

            if msg.type is 'end_of_track' or time_elapsed >= self.__target_segment_len:
                track.append(MetaMessage('end_of_track'))
                target.ticks_per_beat = mid.ticks_per_beat

                if ct > 0:
                    for i in range(int(t/self.__target_segment_len)):
                        dest = join(destinationDir, song_name + '_{}.mid'.format(prefix))
                        target.save(dest)
                        prefix += 1
                        self.__chunks[mid_file].append(dest)

                target = MidiFile()
                track = MidiTrack()
                track.extend(metas)
                target.tracks.append(track)
                time_elapsed = 0
                ct = 0

        return tempo, tpb

    def split_midi_train(self, mid_file, destinationDir=""):
        if destinationDir is "":
            destinationDir = self.__destinationDir

        song_name = split(mid_file)[-1][:-4]
        print(song_name)
        mid = MidiFile(mid_file)
        self.__chunks[mid_file] = []

        metas = []
        tempo = self.__default_tempo
        for msg in mid:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            if msg.is_meta:
                metas.append(msg)

        target = MidiFile()
        track = MidiTrack()
        for i in range(len(metas)):
            metas[i].time = int(mido.second2tick(metas[i].time, 960, tempo))
        track.extend(metas)
        target.tracks.append(track)
        for msg in mid:
            if msg.is_meta:
                continue
            msg.time = int(mido.second2tick(msg.time, 960, tempo))

            track.append(msg)

        target.save("original.mid")

        mid = MidiFile("original.mid")
        self.__chunks["original.mid"] = []

        metas = []
        tempo = self.__default_tempo
        for msg in mid:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            if msg.is_meta:
                metas.append(msg)

        for i in range(len(metas)):
            metas[i].time = 0

        target = MidiFile()
        track = MidiTrack()
        track.extend(metas)
        target.tracks.append(track)

        prefix = 0
        time_elapsed = 0
        ct = 0
        for msg in mid:
        # Skip non-note related messages
            if msg.is_meta:
                continue

            time_elapsed += msg.time

            if msg.type is not 'end_of_track':
                msg.time = int(mido.second2tick(msg.time, mid.ticks_per_beat, tempo))
                track.append(msg)
                if msg.type in ['note_on', 'note_off']:
                    ct+=1

            if msg.type is 'end_of_track' or time_elapsed >= self.__target_segment_len:
                track.append(MetaMessage('end_of_track'))
                target.ticks_per_beat = mid.ticks_per_beat

                if ct > 0:
                    dest = join(destinationDir, song_name + '_{}.mid'.format(prefix))
                    target.save(dest)
                    prefix += 1
                    self.__chunks[mid_file].append(dest)

                target = MidiFile()
                track = MidiTrack()
                track.extend(metas)
                target.tracks.append(track)
                time_elapsed = 0
                ct = 0

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
        metas = []
        tempo = self.__default_tempo
        if tempo_override:
            tempo = tempo_override
        for msg in mid:
            if msg.type is 'set_tempo':
                tempo = msg.tempo
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
        chunk_length_ms = 1000 / 16 # pydub calculates in millisec

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

        # reduce loudness of sounds over 120Hz (focus on bass drum, etc)
        myaudio = myaudio.low_pass_filter(120.0)

        # we'll call a beat: anything above average loudness
        beat_loudness = myaudio.dBFS

        # the fastest tempo we'll allow is 240 bpm (60000ms / 240beats)
        minimum_silence = int(60000 / 240.0)

        nonsilent_times = detect_nonsilent(myaudio, minimum_silence, beat_loudness)

        spaces_between_beats = []
        last_t = nonsilent_times[0][0]

        for peak_start, _ in nonsilent_times[1:]:
            spaces_between_beats.append(peak_start - last_t)
            last_t = peak_start

        # We'll base our guess on the median space between beats
        spaces_between_beats = sorted(spaces_between_beats)
        space = spaces_between_beats[int(len(spaces_between_beats) / 2)]

        bpm = 60000 / space
        print(bpm)

        return exported, bpm
