import argparse
import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoderLayer, TransformerEncoder

restypes = [
    'A', 'R', 'N', 'D', 'C',
    'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
unsure_restype = 'X'
unknown_restype = 'U'

def make_dataset(data_config, train_rate=0.7, valid_rate=0.2):
    data_path = data_config.data_path
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    total_number = len(data)
    train_sep = int(total_number * train_rate)
    valid_sep = int(total_number * (train_rate + valid_rate))

    train_data_dicts = data[:train_sep]
    valid_data_dicts = data[train_sep:valid_sep]
    test_data_dicts = data[valid_sep:]

    train_dataset = DisProtDataset(train_data_dicts)
    valid_dataset = DisProtDataset(valid_data_dicts)
    test_dataset = DisProtDataset(test_data_dicts)

    return train_dataset, valid_dataset, test_dataset


class DisProtDataset(Dataset):
    def __init__(self, dict_data):
        sequences = [d['sequence'] for d in dict_data]
        labels = [d['label'] for d in dict_data]
        assert len(sequences) == len(labels)

        self.sequences = sequences
        self.labels = labels
        self.residue_mapping = {'X':20}
        self.residue_mapping.update(dict(zip(restypes, range(len(restypes)))))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.zeros(len(self.sequences[idx]), len(self.residue_mapping))
        for i, c in enumerate(self.sequences[idx]):
            if c not in restypes:
                c = 'X'
            sequence[i][self.residue_mapping[c]] = 1

        label = torch.tensor([int(c) for c in self.labels[idx]], dtype=torch.long)
        return sequence, label


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=40):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x + self.pe[:, : x.size(1)]
        elif len(x.shape) == 4:
            x = x + self.pe[:, :x.size(1), None, :]
        return self.dropout(x)


class DisProtModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.d_model = model_config.d_model
        self.n_head = model_config.n_head
        self.n_layer = model_config.n_layer

        self.input_layer = nn.Linear(model_config.i_dim, self.d_model)
        self.position_embed = PositionalEncoding(self.d_model, max_len=20000)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.dropout_in = nn.Dropout(p=0.1)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            activation='gelu',
            batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=self.n_layer)
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.d_model, model_config.o_dim)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x  = self.position_embed(x)
        x = self.input_norm(x)
        x = self.dropout_in(x)
        x = self.transformer(x)
        x = self.output_layer(x)
        return x


def metric_fn(pred, gt):
    pred = pred.detach().cpu()
    gt = gt.detach().cpu()
    pred_labels = torch.argmax(pred, dim=-1).view(-1)
    gt_labels = gt.view(-1)
    score = f1_score(y_true=gt_labels, y_pred=pred_labels, average='micro')
    return score


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser('IDRs prediction')
    parser.add_argument('--config_path', default='./config.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)

    train_dataset, valid_dataset, test_dataset = make_dataset(config.data)
    train_dataloader = DataLoader(dataset=train_dataset, **config.train.dataloader)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)

    model = DisProtModel(config.model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=config.train.optimizer.lr,
                                weight_decay=config.train.optimizer.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    model.eval()
    metric = 0.
    with torch.no_grad():
        for sequence, label in valid_dataloader:
            sequence = sequence.to(device)
            label = label.to(device)
            pred = model(sequence)
            metric += metric_fn(pred, label)
    print("init f1_score:", metric / len(valid_dataloader))

    for epoch in range(config.train.epochs):
        # train loop
        progress_bar = tqdm(
            train_dataloader,
            initial=0,
            desc=f"epoch:{epoch:03d}",
        )
        model.train()
        total_loss = 0.
        for sequence, label in progress_bar:
            sequence = sequence.to(device)
            label = label.to(device)

            pred = model(sequence)
            loss = loss_fn(pred.permute(0, 2, 1), label)
            progress_bar.set_postfix(loss=loss.item())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(train_dataloader)

        # valid loop
        model.eval()
        metric = 0.
        with torch.no_grad():
            for sequence, label in valid_dataloader:
                sequence = sequence.to(device)
                label = label.to(device)
                pred = model(sequence)
                metric += metric_fn(pred, label)
        print(f"avg_training_loss: {avg_loss}, f1_score: {metric / len(valid_dataloader)}")