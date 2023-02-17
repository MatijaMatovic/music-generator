import torch
from tqdm import tqdm
from models.Transformer import get_model
from data_preprocess.data_prep import read_abc, collate_function
from data_preprocess.data_prep import ABCDataset
import youtokentome as yttm
from transformers import Trainer, TrainingArguments
from pathlib import Path


TRAIN_DIR = '..\\data_preprocess\\clean_abc\\'
TOKENIZER = '.\\abc.yttm'
EPOCH = 1000
BATCH_SIZE = 5
SAVE_STEPS = 100
GRAD_ACC_STEPS = 16
N_WORKERS = 0
MIN_SEQ_LEN = 16
MAX_SEQ_LEN = 512
CHECKPOINT = None
OUTPUT_DIR = 'ABCModel'
check = 'store_true'

def get_train_files(dir):

    direc = Path(dir)

    return list(direc.glob("*.abc"))

def train_trans():
    training_args = TrainingArguments(output_dir=OUTPUT_DIR, overwrite_output_dir=True, num_train_epochs=EPOCH,
                                      per_device_train_batch_size=BATCH_SIZE, save_steps=SAVE_STEPS, save_total_limit=10, gradient_accumulation_steps=GRAD_ACC_STEPS,
                                      dataloader_num_workers=N_WORKERS)
    print("Loading tokenizer...")
    tokenizer = yttm.BPE(TOKENIZER)
    print('Loading model...')
    model = get_model(vocab_size=tokenizer.vocab_size())
    print('List training files...')
    train_paths = get_train_files(TRAIN_DIR)
    train_paths = train_paths[:10000]
    print('Loading train text...')
    train_data = []
    for p in tqdm(train_paths):
        (keys, notes) = read_abc(p)
        if keys is None:
            continue

        keys_tokens = tokenizer.encode(keys)
        bars = notes.split(' | ')
        notes_tokens = [tokenizer.encode(i + " | ") for i in bars]

        sequence_len = sum(len(i) for i in notes_tokens)
        if not(MIN_SEQ_LEN < sequence_len < MAX_SEQ_LEN):
            continue

        train_data.append((keys_tokens, notes_tokens))

    print('Making dataset...')
    train_dataset = ABCDataset(train_data)

    if CHECKPOINT:
        state_dict = torch.load(CHECKPOINT)
        model.load_state_dict(state_dict)

    trainer = Trainer(model=model, args=training_args, data_collator=collate_function, train_dataset=train_dataset)
    print('Start training...')
    trainer.train()


def main():
    train_trans()


if __name__ == '__main__':
    main()
