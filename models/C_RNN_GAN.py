import torch
import torch.nn as nn

SEQUENCE_LENGTH = 5
NOTE_FEATURES = 4

HIDDEN_SIZE = 6


class GenCRnn(nn.Module):
    def __init__(self, note_features, seq_len, hidden_size=340):
        super(GenCRnn, self).__init__()
        self.hidden_size = hidden_size
        self.note_features = note_features
        self.seq_len = seq_len

        # First layer according to paper
        self.fully_connected1 = nn.Linear(in_features=(2*note_features), out_features=hidden_size)
        self.relu = nn.ReLU()
        self.lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=0.2)

        # Second layer according to paper
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.fully_connected2 = nn.Linear(in_features=hidden_size, out_features=note_features)

    def forward(self, x, hidden):
        batch_size, seq_len, note_features = x.shape

        # Split to seq_len * (batch_size, num_feats)
        # so each feature is split into batches by element
        x = torch.split(x, 1, dim=1)
        x = [x_step.squeeze(dim=1) for x_step in x]

        # Creates a dummy previous output for first timestep
        # according to paper, previous output is concatenated with current
        # in each time step
        prev_gen = torch.rand((batch_size, note_features))

        hidden1, hidden2 = hidden  # (h1, c1), (h2, c2)
        generated_features = []
        for x_step in x:
            # concatenates current input features and previous timestep
            # as according to paper
            concat_in = torch.cat((x_step, prev_gen), dim=-1)
            out = self.relu(self.fully_connected1(concat_in))
            h1, c1 = self.lstm1(out, hidden1)
            h1 = self.dropout(h1)

            h2, c2 = self.lstm_cell2(h1, hidden2)
            prev_gen = self.fully_connected2(h2)
            generated_features.append(prev_gen)

            hidden1 = (h1, c1)
            hidden2 = (h2, c2)

        # seq_len * (batch_size * num_feats) -> (batch_size * seq_len * num_feats)
        generated_features = torch.stack(generated_features, dim=1)

        hidden = (hidden1, hidden2)
        return generated_features, hidden

    def init_hidden(self, batch_size):
        """
        Initializes initial hidden states.
        Because the model starts with a ReLU, this is not automatically done by the LSTM
        (I think)
        """
        weight = next(self.parameters()).data

        initial_hidden = ((torch.zeros(batch_size, self.hidden_size, dtype=weight.dtype),
                           torch.zeros(batch_size, self.hidden_size, dtype=weight.dtype)),
                          (torch.zeros(batch_size, self.hidden_size, dtype=weight.dtype),
                           torch.zeros(batch_size, self.hidden_size, dtype=weight.dtype)))

        return initial_hidden


class DiscCRnn(nn.Module):
    def __init__(self, note_features, seq_len, hidden_size=340, bidirectional=True):
        super(DiscCRnn, self).__init__()
        self.note_features = note_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_size=note_features, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, dropout=0.2, bidirectional=bidirectional)

        self.fully_connected = nn.Linear(in_features=2*hidden_size, out_features=1)

    def forward(self, x, hidden):
        drop_in = self.dropout(x)

        lstm_out, hidden = self.lstm(drop_in, hidden)

        out = self.fully_connected(lstm_out)
        out = torch.sigmoid(out)  # sigmoid is used according to paper

        num_dims = len(out.shape)
        reduction_dims = tuple(range(1, num_dims))
        out = torch.mean(out, dim=reduction_dims)

        return out, lstm_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        initial_hidden = (torch.zeros((2*2, batch_size, self.hidden_size), dtype=weight.dtype),
                  torch.zeros((2*2, batch_size, self.hidden_size), dtype=weight.dtype))

        return initial_hidden

