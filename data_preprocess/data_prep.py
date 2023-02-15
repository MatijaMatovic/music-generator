import pypianoroll as pianoroll
import glob
import torch
from torch.utils.data import DataLoader,Dataset
from Levenshtein import distance
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from music21 import *
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from tqdm import tqdm
import random

import numpy as np
PATH = '..\\data_preprocess\\maestro-v3.0.0\\'
FILES = '2004\\*.midi'
USEABLE_KEYS = [i+":" for i in "BCDFGHIKLMmNOPQRrSsTUVWwXZ"]

ABC_INPUT = '.\\trainset\\abc\\'
ABC_OUTPUT_CLEAN = 'clean_abc'
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



class Midi_RNN():
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.file_notes = []
        self.trainseq = []
        self.transfer_dict = dict()
        self.dict_n = 0
    def parser(self, folder_name):
        """
        get the notes and chord from Midi files
        """
        for file in glob.glob(folder_name):
            midi = converter.parse(file)

            print(f'Parsing {file}.....')

            notes = []
            for element in midi.flat.elements:
                if isinstance(element, note.Rest) and element.offset != 0:
                    notes.append(f'R|{float(element.duration.quarterLength)}')
                if isinstance(element, note.Note):
                    notes.append(f'{str(element.pitch)}|{float(element.duration.quarterLength)}')
                if isinstance(element, chord.Chord):
                    temp = (','.join(str(p) + '|' + str(float(element.duration.quarterLength)) for p, n in zip(element.pitches, element.notes)))
                    notes.append(temp)
            self.file_notes.append(notes)
        note_set = sorted(set(n for notes in self.file_notes for n in notes))
        self.dict_n = len(note_set)
        self.transfer_dict = dict((n, number) for number, n in enumerate(note_set))

    def prepare_sequence(self):
        """
        Preps the sequence for the NN
        """
        for notes in self.file_notes:
            for i in range(len(notes) - self.seq_length):
                self.trainseq.append([self.transfer_dict[n] for n in notes[i:i + self.seq_length]])
        self.trainseq = np.array(self.trainseq)
        self.trainseq = (self.trainseq - float(self.dict_n) / 2) / (float(self.dict_n) / 2) # normalizing features to 0 - 1 range

        return self.trainseq

    def create_midi(self, predicition_output, filename):
        """
        create a Midi file from the prediction of the NN
        """
        offset = 0
        midi_stream = stream.Stream()

        for pattern in predicition_output:
            if 'R' in pattern:
                _, quarter_len = pattern.split('|')
                n = note.Rest()
                n.quarterLength = float(quarter_len)
                midi_stream.append(n)
            elif ',' in pattern:
                val_quarter_array = pattern.split(',')
                notes = []
                for elem in val_quarter_array:
                    val, quarter_len = elem.split('|')
                    n = note.Note(val)
                    n.quarterLength = float(quarter_len)
                    n.storedInstrument = instrument.Piano()
                    notes.append(n)
                c = chord.Chord(notes)
                #c.offset = offset
                midi_stream.append(c)
            else:
                val, quarter_len = pattern.split('|')
                n = note.Note(val)
                n.quarterLength = float(quarter_len)
                n.storedInstrument = instrument.Piano()
                midi_stream.append(n)

            # offset update


        midi_stream.write('midi', fp=f'{filename}.mid')


class ABCDataset(Dataset):
    def __init__(self, data, ctx_bars_num = 8, tgt_bars_num = 8, bos_id = 2, eos_id = 3, is_test = False ):
        self.notes = []
        self.keys = []

        for(keys, notes) in data:
            if notes is None:
                continue

            self.keys.append(keys)
            self.notes.append(notes)

        self.ctx_bars_num = ctx_bars_num
        self.tgt_bars_num = tgt_bars_num
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.is_test = is_test

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        notes = self.notes[idx]
        keys = self.keys[idx]

        if not self.is_test:
            split_idx = 8
            ctx_notes = notes[split_idx - self.ctx_bars_num : split_idx]
            tgt_notes = notes[split_idx : split_idx + self.tgt_bars_num]
        else:
            ctx_notes = notes
            tgt_notes = []

        ctx_tokens = [self.bos_id] + keys
        tgt_tokens = [self.bos_id]

        for bar in ctx_notes:
            ctx_tokens += bar
        for bar in tgt_notes:
            tgt_tokens += bar

        ctx_tokens += [self.eos_id]
        tgt_tokens += [self.eos_id]

        ctx_tokens = torch.tensor(ctx_tokens, dtype = torch.long)
        tgt_tokens = torch.tensor(tgt_tokens, dtype = torch.long)

        return {"features" : ctx_tokens, "target" : tgt_tokens}

def read_abc(path):
    keys = []
    notes = []
    with open(path) as rf:
        for line in rf:
            line = line.strip()
            if line.startswith("%"):
                continue
            if any([line.startswith(key) for key in USEABLE_KEYS]):
                keys.append(line)
            else:
                notes.append(line)

    keys = " ".join(keys)
    notes = "".join(notes).strip()
    notes = notes.replace(" ", "")

    if notes.endswith("|"):
        notes = notes[:-1]

    notes = notes.replace("[", " [")
    notes = notes.replace("]", "] ")
    notes = notes.replace("(", " (")
    notes = notes.replace(")", ") ")
    notes = notes.replace("|", " | ")
    notes = notes.strip()
    notes = " ".join(notes.split(" "))

    if not keys and not notes:
        return None, None

    return keys, notes

def collate_function(batch):
    features = [i["features"] for i in batch]
    target = [i["target"] for i in batch]

    feature_lens = [len(i) for i in features]
    target_lens = [len(i) for i in target]

    max_feature_len = max(feature_lens)
    max_target_len = max(target_lens)

    feature_mask = torch.tensor([[1] * l + [0] * (max_feature_len - 1) for l in feature_lens], dtype=torch.bool)
    target_mask = torch.tensor([[1] * l + [0]*(max_target_len - 1) for l in target_lens], dtype=torch.bool)

    features_padded = pad_sequence(features, batch_first=True)
    target_padded = pad_sequence(target, batch_first=True)

    return {"input_ids": features_padded,
            "decoder_input_ids": target_padded,
            "labels": target_padded,
            "attention_mask": feature_mask,
            "decoder_attention_mask": target_mask}

def bars_similarity(bar1, bar2):
    distances = []
    for n1 in bar1:
        distances.append(min([distance(n1, n2) / (len(n1) + len(n2)) for n2 in bar2]))
    return sum(distances) / len(distances)

def get_num_repeats(bars):
    num_repeats = 0
    for i, b1 in enumerate(bars):
        for j, b2 in enumerate(bars):
            if i != j:
                num_repeats += int(b1 == b2)
    return num_repeats

def clean_abc_data():
    input_dir = Path(ABC_INPUT)
    output_dir = Path(ABC_OUTPUT_CLEAN)
    output_dir.mkdir(exist_ok=True)
    file_index = 0

    for i in tqdm(list(input_dir.glob("*.abc"))):
        keys, abc = read_abc(i)
        if abc is None:
            continue

        abc = abc.replace(" ", "").split("|")
        num_bars = len(abc) // 16

        if num_bars == 0:
            continue

        for j in range(num_bars):
            bar1 = abc[j*8 : (j+1)*8]
            bar2 = abc[(j+1)*8 : (j+2)*8]

            if len(bar1) + len(bar2) != 16:
                continue
            if get_num_repeats(bar2) > 4:
                continue
            if "x8" in "|\n".join(bar1 + bar2):
                continue

            sim = bars_similarity(bar1, bar2)
            if sim < 0.45:
                continue

            with open(output_dir.joinpath(f"{file_index}.abc"), "w") as f:
                new_abc = keys.replace(" ", "\n") + "\n" + "|\n".join(bar1 + bar2)
                f.write(new_abc)
            file_index += 1

def main():
    clean_abc_data()

if __name__ == "__main__":
    main()