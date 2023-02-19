from data_preprocess.data_prep import Midi_RNN, load_midi_sequentially, create_midi_from_sequential_notes

midi_rnn = Midi_RNN(seq_length=16)

midi_notes = load_midi_sequentially("data_preprocess/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_05_Track05_wav.midi")
print('loaded')

create_midi_from_sequential_notes(midi_notes, 'result.midi')
print('saved')
