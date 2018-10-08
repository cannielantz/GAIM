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


