import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ContextualEncoder(nn.Module):
    def __init__(self, u_dim, g_dim, args):
        super(ContextualEncoder, self).__init__()
        self.input_size = u_dim
        self.hidden_dim = g_dim
        self.device = args.device
        self.dropout = nn.Dropout(args.drop_rate)
        self.args = args

        self.nhead = 1
        for h in range(7, 15):
            if self.input_size % h == 0:
                self.nhead = h
                break

        self.encoding_layer = nn.Embedding(110, self.input_size)
        self.LayerNorm = nn.LayerNorm(self.input_size)

        self.use_transformer = False
        if args.rnn == "lstm":
            self.rnn = nn.LSTM(
                self.input_size,
                self.hidden_dim // 2,
                dropout=args.drop_rate,
                bidirectional=True,
                num_layers=2,
                batch_first=True,
            )
        elif args.rnn == "gru":
            self.rnn = nn.GRU(
                self.input_size,
                self.hidden_dim // 2,
                dropout=args.drop_rate,
                bidirectional=True,
                num_layers=2,
                batch_first=True,
            )
        elif args.rnn == "transformer":
            print("SeqContext-> USING Transformer")
            self.use_transformer = True
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=self.input_size,
                nhead=self.nhead,
                dropout=args.drop_rate,
                batch_first=True,
            )
            self.transformer_encoder = torch.nn.TransformerEncoder(
                encoder_layer, num_layers=2
            )
            self.transformer_out = torch.nn.Linear(
                self.input_size, self.hidden_dim, bias=True
            )
            print("args.drop_rate:", args.drop_rate)

    def forward(self, text_len_tensor, text_tensor):
        if self.use_transformer:
            rnn_out = self.transformer_encoder(text_tensor)
            rnn_out = self.transformer_out(rnn_out)
        else:
            packed = pack_padded_sequence(
                text_tensor, text_len_tensor, batch_first=True, enforce_sorted=False
            )
            rnn_out, (_, _) = self.rnn(packed, None)
            rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        return rnn_out

    def swish(self, x):
        """https://arxiv.org/abs/1710.05941"""
        return x * torch.sigmoid(x)
