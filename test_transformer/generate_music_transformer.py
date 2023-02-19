from data_preprocess.data_prep import read_abc
from models.Transformer import get_model
from training.train_transformer import get_train_files
from tqdm import tqdm
from pathlib import Path
import torch
import youtokentome as yttm

DATA_PATH = '..\\data_preprocess\\testset\\abc\\'
TOKENIZER = 'abc.yttm'
CHECKPOINT = '..\\ABCModel/checkpoint-3/pytorch_model.bin'
OUTPUT_DIR = 'predict_abc_songs'
def predict_next_notes(model, tokenizer, keys, notes):
    keys_tokens = tokenizer.encode(keys)
    notes_tokens = tokenizer.encode(notes)

    if len(keys_tokens) + len(notes_tokens) > 510:
        notes_tokens = notes_tokens[len(notes_tokens) - len(keys_tokens) - 510:]

    context_tokens = [2] + keys_tokens + notes_tokens + [3]
    context_tokens = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0)

    if torch.cuda.is_available():
        context_tokens = context_tokens.cuda()

    bad_words_ids = []
    bad_words = ["x8 | "]
    for w in bad_words:
        bad_words_ids.append(tokenizer.encode(bad_words)[0])

    gen_tokens = model.generate(input_ids=context_tokens, max_length=320, min_length=32, early_stopping = False, num_beams=20,
                                bos_token_id=2, eos_token_id=3, no_repeat_ngram_size=15, pad_token_id=0, bad_words_ids = bad_words_ids)

    gen_tokens = gen_tokens[0].tolist()

    notes = tokenizer.decode(gen_tokens, ignore_ids = [0,1,2,3])[0]
    notes = notes.replace(" ","").replace("|", "|\n")

    return notes

def predict(model, tokenizer, text_path, output_dir):
    keys, notes = read_abc(text_path)
    new_path = output_dir.joinpath(text_path.name)

    predicted_tokens = predict_next_notes(model, tokenizer, keys, notes)

    with open(text_path) as f:
        abc_text = f.read()

    with open(new_path, "w") as f:
        f.write(abc_text + predicted_tokens)

    return new_path

def generate():
    test_paths = get_train_files(DATA_PATH)
    test_paths = sorted(test_paths)

    tokenizer = yttm.BPE(TOKENIZER)
    model = get_model(tokenizer.vocab_size())
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()

    print('Started generating...')

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    for p in tqdm(test_paths):
        abc_path = predict(model, tokenizer, p, output_dir)
        print(abc_path)
