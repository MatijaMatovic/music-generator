from data_preprocess.data_prep import read_abc
from tqdm import tqdm
import youtokentome as yttm
from pathlib import Path

DATA_PATH = '..\\data_preprocess\\trainset\\abc\\'
CORPUS_PATH = 'train_corpus'
VOCAB_SIZE = 3000
MODEL_PATH = 'abc.yttm'
def train_tokenizer():
    train_files = list(Path(DATA_PATH).glob("*.abc"))
    with open(CORPUS_PATH, "w") as f:
        for file in tqdm(train_files):
            (keys, notes) = read_abc(file)
            f.write(f"{keys}\n{notes}\n")

    yttm.BPE.train(data = CORPUS_PATH, vocab_size= VOCAB_SIZE, model = MODEL_PATH )

    return MODEL_PATH

def main():
    train_tokenizer()

if __name__ == '__main__':
    main()