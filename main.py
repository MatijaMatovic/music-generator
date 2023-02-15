from data_preprocess.data_prep import Midi_RNN

midi_rnn = Midi_RNN(seq_length=16)

midi_rnn.parser("C:\\FAKS\MASTER\\Neuronske mreze\\music-generator\\data_preprocess\\maestro-v3.0.0\\2004\\MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_05_Track05_wav.midi")

print(midi_rnn.file_notes)

midi_rnn.create_midi(midi_rnn.file_notes[0], 'midi_result')

