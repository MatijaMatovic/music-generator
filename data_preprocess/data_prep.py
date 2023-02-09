import pypianoroll as pianoroll
import glob
import torch
from torch.utils.data import DataLoader,Dataset 
import numpy as np
PATH = '..\\data_preprocess\\maestro-v3.0.0\\'
FILES = '*\\*.midi'

def load_midi_files(path, compress = False):
    """
    reads midi files from path
    returns an array of n_songs length, where
    each elem is (n_notes, 128) tensor
    """
    files = glob.glob(path + FILES)
    combined_pianorolls = []
    pianorolls_lenghts = []
    for i, file in enumerate(files):
        multitrack = pianoroll.read(file)
        piano_read = multitrack.tracks[0].pianoroll #only one track per song 
        piano_read = torch.tensor(piano_read, dtype=torch.float32)
        pianorolls_lenghts.append(piano_read.shape[0])
        piano_read /= 127. #normalize the values
        combined_pianorolls.append(piano_read) #add the song to the list
        print(f'Read {i} / {len(files)}')
    if compress:
        combined_pianorolls[combined_pianorolls > 0.2] = 1.0
    return combined_pianorolls, torch.tensor(pianorolls_lenghts)

# _, lengths = load_midi_files(PATH)
# print(lengths)


class MidiDataset(Dataset):
    def __init__(self, data, len_array, seq_length = 16):
        """
        midis is an array of n_songs length
        where each elem is a tensor of (n_notes, 128)

        len_array is an array of the lenghts in notes of each song i.e
        number of notes per song

        seq_length - the number of notes we fetch from each song for training
        """
        self.midis = data
        self.len_array = len_array
        self.seq_length = seq_length
        self.cumsum = torch.cumsum(len_array, dim=0)
    def __len__(self):
        """
        this is the number of sequences of seq_length that fit in the
        array 
        """
        return int(torch.sum(self.len_array, dim=0) / self.seq_length)
    def __getitem__(self, index):
        """
        index is the index of the sequence we want to get
        we are looking for the song it belongs to

        sumsum is the cumulative sum of the lengths of songs
        we find the index of the first element in the cumsum that is greater than
        index * seq_Length because that is the index of the song our sequence belongs to

        start of seq in song is

        (index - cumsum_prev % 16)

        and we take (index - cumsum_prev % 16) * 16

        """
        song_index = (self.cumsum > index*self.seq_length).nonzero()[0,0].item()
        note_index = index * self.seq_length
        coef = 0 if song_index == 0 else 1
        seq_start = (note_index - coef * self.cumsum[song_index-1]).item()  # videti je li song_idx == 0
        seq_end = seq_start + self.seq_length  # videti je li preskace kraj pesme
        if seq_end + coef * self.cumsum[song_index - 1] > self.cumsum[song_index]:
            return self.midis[song_index][-self.seq_length:, :]
        
        return self.midis[song_index][seq_start : seq_end, :]




def get_loader():
    dataset, len_array = load_midi_files(PATH, compress= False)
    dataset = MidiDataset(dataset, len_array, seq_length = 16)
    loader = DataLoader(dataset, batch_size = 64, drop_last=True)
    return loader