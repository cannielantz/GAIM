#Converter code taken from https://github.com/burliEnterprises/tensorflow-music-generator/blob/master/midi_manipulation.py

import midi
import numpy as np

min = 24
max = 102
span = max-min

#Convert midi file to note state matrix
def midi2matrix(midifile, squash=True, span=span):
    pattern = midi.read_midifile(midifile)

    timeleft = [track[0].tick for track in pattern] #Tick down
    posns = [0 for track in pattern] #Positions

    stateMatrix = []
    time = 0 #Start time

    state = [[0,0] for x in range(span)]
    stateMatrix.append(state)
    condition = True

    while condition:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(span)]
            stateMatrix.append(state)

        for i in range(len(timeleft)):
            if not condition:
                break

            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]

                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < min) or (evt.pitch >= max):
                        pass #Note is out of bounds
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-min] = [0, 0]
                        else:
                            state[evt.pitch-min] = [1, 1]

                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        out = stateMatrix
                        condition = False
                        break
                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    S = np.array(stateMatrix)
    stateMatrix = np.hstack((S[:, :, 0], S[:, :, 1]))
    stateMatrix = np.asarray(stateMatrix).tolist()
    return stateMatrix

def matrix2midi(statematrix, name='game_song', span=span):
    statematrix = np.array(statematrix)

    if not len(statematrix.shape) == 3:
        statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:]))
    pattern = midi.pattern()
    track = midi.Track()
    pattern.append(track)

    span = max - min
    tickscale = 55

    lastcmdtime = 0
    prevstate = [[0, 0] for x in range(span)]

    for time, state in enumerate(statematrix + [prevstate[:]]):
        offNotes = []
        onNotes = []

        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick = (time - lastcmdtime)*tickscale, pitch=note+min))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick = (time - lastcmdtime)*tickscale, velocity = 40, pitch = note + min))
            lastcmdtime = time

        prevstate = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.midi".format(name), pattern)



