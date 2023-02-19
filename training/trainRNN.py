import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch import optim
from data_preprocess.data_prep import get_sequential_loader

from models.C_RNN_GAN import GenCRnn, DiscCRnn

DATA_DIR = 'data_preprocess/maestro-v3.0.0/2004'
CHKPT_DIR = 'saved/models'
GENERATED_DIR = 'saved/songs'

G_FILENAME = 'c_rnn_gan_gen.pth'
D_FILENAME = 'c_rnn_gan_disc.pth'

G_LRN_RATE = 0.001
D_LRN_RATE = 0.001
MAX_GRAD_NORM = 5.

NOTE_FEATURES = 4
MAX_SEQ_LEN = 32
BATCH_SIZE = 32

EPSILON = 1e-40  # to approximate zero, to prevent undefined results


class GenLoss(nn.Module):
    def __init__(self):
        super(GenLoss, self).__init__()

    def forward(self, logist_gen):
        logits_gen = torch.clamp(logist_gen, EPSILON, 1.0)
        batch_loss = -torch.log(logits_gen)

        return torch.mean(batch_loss)


class DiscLoss(nn.Module):
    def __init__(self, label_smoothing=False):
        super(DiscLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits_real, logits_gen):
        """
            Discriminator loss

            logits_real: logits from discriminator, when input is real
            logits_gen: logits from discriminator, when input is from generator

            loss = -(y*log(p) + (1-y)*log(1-p))
        """

        logits_real = torch.clamp(logits_real, EPSILON, 1.)
        d_loss_real = -torch.log(logits_real)

        if self.label_smoothing:
            p_fake = torch.clamp((1-logits_real), EPSILON, 1.)
            d_loss_fake = -torch.log(p_fake)
            d_loss_real = 0.9*d_loss_real + 0.1*d_loss_fake

        logits_gen = torch.clamp((1-logits_gen), EPSILON, 1.)
        d_loss_gen = -torch.log(logits_gen)

        batch_loss = d_loss_real + d_loss_gen
        return torch.mean(batch_loss)


def run_training(gen, disc, optimizer, criterion, data_loader, num_epochs=100,
                 g_pretraining_epochs=0, d_pretraining_epochs=0, conditional_freezing=False):
    gen = gen.train()
    disc = disc.train()

    for epoch in range(num_epochs):

        g_pretraining = g_pretraining_epochs > 0
        d_pretraining = d_pretraining_epochs > 0

        loss = {}
        g_total_loss = 0.
        d_total_loss = 0.
        num_corrects = 0
        num_samples = 0

        g_avg_loss, d_avg_loss = 0., 0.
        d_accuracy = 0.

        freeze_g = g_pretraining
        freeze_d = d_pretraining or (conditional_freezing and d_accuracy >= 95.)

        if g_pretraining: g_pretraining_epochs -= 1
        if d_pretraining: d_pretraining_epochs -= 1

        for i, sequence_batch in enumerate(data_loader):

            # Each bach is independent (not a continuation of previous batch)
            # so we reset states for each batch
            # TODO: zameniti BATCH_SIZE sa sequence.shape[?]
            g_states = gen.init_hidden(BATCH_SIZE)
            d_states = disc.init_hidden(BATCH_SIZE)

            if not freeze_g:
                optimizer['g'].zero_grad()

            z = torch.rand((BATCH_SIZE, MAX_SEQ_LEN, NOTE_FEATURES))  # random inputs for generator

            g_features, _ = gen(z, g_states)

            if isinstance(criterion['g'], GenLoss):
                d_logits_gen, _, _ = disc(g_features, d_states)
                loss['g'] = criterion['g'](d_logits_gen)
            else:  # feature matching
                # feed real and generated input to discriminator
                _, d_features_real, _ = disc(sequence_batch, d_states)
                _, d_features_gen, _ = disc(g_features, d_states)
                loss['g'] = criterion['g'](d_features_real, d_features_gen)

            if not freeze_g:
                loss['g'].backward()
                nn.utils.clip_grad_norm(gen.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer['g'].setp()

            if not freeze_d:
                optimizer['d'].zero_grad()

            d_logits_real, _, _ = disc(sequence_batch, d_states)
            d_logits_gen, _, _ = disc(g_features.detach(), d_states)
            loss['d'] = criterion['g'](d_logits_real, d_logits_gen)
            if not freeze_d:
                loss['d'].backward()
                nn.utils.clip_grad_norm(disc.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer['d'].step()

            g_total_loss += loss['g'].item()
            d_total_loss += loss['d'].item()
            num_corrects += (d_logits_real > 0.5).sum().item() + (d_logits_gen < 0.5).sum()
            num_samples += BATCH_SIZE  # TODO: zameniti BATCH_SIZE sa sequence.shape[?]

        if num_samples > 0:
            g_avg_loss = g_total_loss / num_samples
            d_avg_loss = d_total_loss / num_samples
            d_accuracy = 100 * num_corrects / (2 * num_samples)  # 2 because (real + generated)

        if g_pretraining or d_pretraining:
            print(
                "Pretraining Ep. %d/%d " % (epoch+1, num_epochs), "[Freeze G: ", freeze_g, ", Freeze D: ", freeze_d, "]"
            )
        else:
            print("Epoch %d/%d " % (epoch+1, num_epochs), "[Freeze G: ", freeze_g, ", Freeze D: ", freeze_d, "]")

        print(f"\t[Training] GEN loss: {g_avg_loss}, DISC loss: {d_avg_loss}, DISC acc: {d_accuracy}")


def main(args):
    data_loader = get_sequential_loader(batch_size=BATCH_SIZE, seq_length=MAX_SEQ_LEN)

    gen = GenCRnn(note_features=NOTE_FEATURES, seq_len=MAX_SEQ_LEN)
    disc = DiscCRnn(note_features=NOTE_FEATURES, seq_len=MAX_SEQ_LEN)

    if args.use_sgd:
        optimizer = {
            'g': optim.SGD(gen.parameters(), lr=args.g_lr, momentum=0.9),
            'd': optim.SGD(disc.parameters(), lr=args.d_lr, momentum=0.9)
        }
    else:
        optimizer = {
            'g': optim.Adam(gen.parameters(), lr=args.g_lr),
            'd': optim.Adam(disc.parameters(), lr=args.d_lr)
        }

    criterion = {
        'g': nn.MSELoss(reduction='sum') if args.feature_matching else GenLoss(),
        'd': DiscLoss(args.label_smoothing)
    }

    if args.load_gen:
        chkpt = torch.load(os.path.join(CHKPT_DIR, G_FILENAME))
        gen.load_state_dict(chkpt)
        print('Continuing training of generator...')

    if args.load_disc:
        chkpt = torch.load(os.path.join(CHKPT_DIR, D_FILENAME))
        disc.load_state_dict(chkpt)
        print('Continuing training of discriminator...')

    g_pretraining_epochs = 0 if args.no_pretraining else args.g_pretraining_epochs
    d_pretraining_epochs = 0 if args.no_pretraining else args.d_pretraining_epochs

    run_training(
        gen=gen,
        disc=disc,
        optimizer=optimizer,
        criterion=criterion,
        data_loader=data_loader,
        num_epochs=args.num_epochs,
        g_pretraining_epochs=g_pretraining_epochs,
        d_pretraining_epochs=d_pretraining_epochs,
        conditional_freezing=args.conditional_freezing
    )

    if not args.no_save_gen:
        torch.save(gen.state_dict(), os.path.join(CHKPT_DIR, G_FILENAME))
        print('Saved generator!')

    if not args.no_save_disc:
        torch.save(disc.state_dict(), os.path.join(CHKPT_DIR, D_FILENAME))


if __name__ == '__main__':
    ARG_PARSER = ArgumentParser()
    ARG_PARSER.add_argument('--load_gen', action='store_false')
    ARG_PARSER.add_argument('--load_disc', action='store_false')
    ARG_PARSER.add_argument('--no_save_gen', action='store_false')
    ARG_PARSER.add_argument('--no_save_disc', action='store_false')

    ARG_PARSER.add_argument('--num_epochs', default=500, type=int)
    ARG_PARSER.add_argument('--seq_len', default=32, type=int)
    ARG_PARSER.add_argument('--batch_size', default=32, type=int)
    ARG_PARSER.add_argument('--use_sgd', action='store_true')
    ARG_PARSER.add_argument('--g_lr', default=0.001, type=float)
    ARG_PARSER.add_argument('--d_lr', default=0.001, type=float)

    ARG_PARSER.add_argument('--no_pretraining', action='store_false')
    ARG_PARSER.add_argument('--g_pretraining_epochs', default=10, type=int)
    ARG_PARSER.add_argument('--d_pretraining_epochs', default=10, type=int)
    ARG_PARSER.add_argument('--conditional_freezing', action='store_true')
    ARG_PARSER.add_argument('--label_smoothing', action='store_true')
    ARG_PARSER.add_argument('--feature_matching', action='store_true')

    ARGS = ARG_PARSER.parse_args()
    MAX_SEQ_LEN = ARGS.seq_len
    BATCH_SIZE = ARGS.batch_size

    main(ARGS)

