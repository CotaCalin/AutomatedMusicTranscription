from __future__ import division
"""
Simple function for converting Pretty MIDI object into one-hot encoding
/ piano-roll-like to be used for machine learning.
"""
import pretty_midi
import numpy as np
import sys
import mido
import argparse

def pretty_midi_to_one_hot(pm, fs=18):
    """Compute a one hot matrix of a pretty midi object
    Parameters
    ----------
    pm : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    Returns
    -------
    one_hot : np.ndarray, shape=(128,times.shape[0])
        Piano roll of this instrument. 1 represents Note Ons,
        -1 represents Note offs, 0 represents constant/do-nothing
    """

    # Allocate a matrix of zeros - we will add in as we go
    one_hots = []

    if len(pm.instruments) < 1:
        return 0

    for instrument in pm.instruments:
        one_hot = np.zeros((128, int(fs*instrument.get_end_time())+1))
        for note in instrument.notes:
            one_hot[note.pitch, int(note.start*fs)] = 1
            one_hot[note.pitch, int(note.end*fs)] = 0
        one_hots.append(one_hot)

    one_hot = np.zeros((128, np.max([o.shape[1] for o in one_hots])))
    for o in one_hots:
        one_hot[:, :o.shape[1]] += o

    one_hot = np.clip(one_hot,-1,1)
    return one_hot

def one_hots_to_pretty_midi(one_hots, tempo, tpb, fs=18, program=1,bpm=120):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,time)
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    bpm : int
        Beats per minute, used to decide when to re-emphasize notes left on.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    print(tempo)
    print(mido.tempo2bpm(tempo))
    bpm = mido.tempo2bpm(tempo)
    #tempo = int(tempo*0.8)
    #pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    #instrument = pretty_midi.Instrument(program=program)
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    one_hot = np.zeros((128, len(one_hots)))
    for i in range(len(one_hots)):
        for j in range(len(one_hots[i])):
            if one_hots[i][j] == 1:
                one_hot[j][i] = one_hots[i][j]

    notes, frames = one_hot.shape
    #print(notes)
    #print(frames)
    #mid.ticks_per_beat = tpb * 2


    frame_t = one_hot.T
    last_msg = 0
    toGenerate = []
    for frame in range(frames):
        current_notes = []
        current_notes_begin = {}
        current_notes_end = {}

        for j in range(len(frame_t[frame])):
            if frame_t[frame][j] == 1:
                current_notes.append(j)
                current_notes_begin[j] = frame
                current_notes_end[j] = frame + 1
                for k in range(frame+1, frames):
                    if frame_t[k][j] == 1:
                        current_notes_end[j] += 1
                        frame_t[k][j] = 0
                    else:
                        break
        m = 0
        for note in current_notes:
            #print(current_notes_begin[note]*1/16)
            #print(current_notes_end[note]*1/16)
            #print()
            #pm_note = pretty_midi.Note(
            #        velocity=100,
            #        pitch=note,
            #        start=current_notes_begin[note]*1/16,
            #        end=current_notes_end[note]*1/16)
            #instrument.notes.append(pm_note)
            #track.append(mido.Message('note_on', note=note,
            #time=int(mido.second2tick(current_notes_begin[note]*1/16 - last_msg,
            #mid.ticks_per_beat, tempo))))
            #track.append(mido.Message('note_off', note=note,
            #time=int(mido.second2tick((current_notes_end[note] - current_notes_begin[note])*1/16,
            #mid.ticks_per_beat, tempo))))
            #print(current_notes_begin[note])
            #print(1/16)
            #input()
            begin = (note, current_notes_begin[note]*1/16, 'begin')
            end = (note, current_notes_end[note]*1/16, 'end')
            toGenerate.append(begin)
            toGenerate.append(end)
            #print(mido.second2tick(current_notes_begin[note]*1/16 - last_msg,
            #mid.ticks_per_beat, tempo))
            #print(current_notes_begin[note]*1/16)
            #print(last_msg)
            #print(current_notes_begin[note]*1/16 - last_msg)
            #input()

            #m = current_notes_begin[note]*1/16 - last_msg

            #last_msg += m

    last_msg = 0

    sortedNotes = sorted(toGenerate, key=lambda x: x[1])
    for note in sortedNotes:
        print(note)
        if note[2] == 'end':
            track.append(mido.Message('note_off', note=note[0],
            time=int(mido.second2tick(note[1] - last_msg, mid.ticks_per_beat, int(tempo*1.5)))))
        else:
            track.append(mido.Message('note_on', note=note[0],
            time=int(mido.second2tick(note[1] - last_msg, mid.ticks_per_beat, int(tempo*1.5)))))
        last_msg = note[1]

    track.append(mido.MetaMessage('end_of_track'))
    for msg in track:
        print(msg)
        
    return mid

def slice_to_categories(piano_roll):
    notes_list = np.zeros(128)
    notes = np.nonzero(piano_roll)[0]
    notes = np.unique(notes)

    for note in notes:
        notes_list[note] = 1

    return notes_list
