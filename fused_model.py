import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel


# HELPERS

def upsample(x, out_size):
    return F.interpolate(x, size=out_size, mode='linear', align_corners=False)


def bn_act_conv(in_, out_, kernel_size=3, stride=1, bias=False, act=nn.GELU):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride,
                  padding=padding, bias=bias),
        nn.BatchNorm1d(out_),
        act()
    )


def conv_only(in_, out_, kernel_size=3, stride=1, bias=True):
    padding = kernel_size // 2
    return nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias)


# CONTEXT GATED BLOCK

class ContextGatedBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.V           = nn.Linear(input_dim, hidden_dim)
        self.U           = nn.Linear(input_dim, hidden_dim)
        self.global_proj = nn.Linear(input_dim, hidden_dim)
        self.w           = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: [B, L, input_dim]
        gate        = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        global_feat = self.global_proj(torch.mean(x, dim=1, keepdim=True))
        combined    = gate + global_feat.expand(-1, x.size(1), -1)
        return self.w(combined) + x


# FCN

class FCNsignalMod(nn.Module):

    def __init__(self, motiflen=10, n_features=0, dropout=0.2):
        super().__init__()
        in_ch          = 4
        self.has_extra = n_features > 0

        if self.has_extra:
            self.feature_proj = nn.Conv1d(n_features, 32, kernel_size=1)
            in_ch += 32

        # Encoder
        self.conv1 = bn_act_conv(in_ch, 32, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = bn_act_conv(32, 32, kernel_size=7)
        self.pool2 = nn.MaxPool1d(2)

        # GRU + ContextGatedBlock
        self.gru_hidden        = 48
        self.gru               = nn.GRU(32, self.gru_hidden, batch_first=True,
                                        bidirectional=True)
        self.ContextGatedBlock = ContextGatedBlock(self.gru_hidden * 2,
                                                   hidden_dim=32)
        # Decoder
        self.blend2  = bn_act_conv(self.gru_hidden * 2 + 32, 32)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.GRU):
                for name, param in layer.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        if self.has_extra:
            extra = self.feature_proj(x[:, 4:, :])
            x     = torch.cat([x[:, :4, :], extra], dim=1)

        skip1 = x                               # [B, in_ch, L]

        out = self.conv1(x)
        out = self.pool1(out)
        out = self.dropout(out)

        skip2 = out                             # [B, 32, L/2]
        out   = self.conv2(out)
        if out.size(1) == skip2.size(1):
            out = out + skip2
        out = self.pool2(out)
        out = self.dropout(out)                 # [B, 32, L/4]

        # GRU + ContextGatedBlock
        out = out.permute(0, 2, 1)             # [B, L/4, 32]
        out, _ = self.gru(out)                 # [B, L/4, 96]
        out = self.dropout(out)
        out = self.ContextGatedBlock(out)      # [B, L/4, 96]
        out = out.permute(0, 2, 1)             # [B, 96, L/4]

        # Decoder — upsample back to full resolution
        out = upsample(out, skip2.size(-1))    # [B, 96, L/2]
        out = torch.cat([out, skip2], dim=1)   # [B, 128, L/2]
        out = self.blend2(out)                 # [B, 32, L/2]
        out = upsample(out, skip1.size(-1))    # [B, 32, L]

        return out                             


# FUSED MODEL

class FusedModel(nn.Module):

    BERT_PATH = "/home/ubuntu/data/dnabert2"

    def __init__(self, motiflen=10, n_features=0, dropout=0.2): #, freeze_bert=False
        super().__init__()

        # sub-models
        self.fcn  = FCNsignalMod(motiflen=motiflen,
                                 n_features=n_features,
                                 dropout=dropout)
        self.bert = AutoModel.from_pretrained(self.BERT_PATH,
                                              trust_remote_code=True)
        # if freeze_bert:
        #     for p in self.bert.parameters():
        #         p.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # project BERT 768-d → 128-d → 32-d to match FCN feature width
        self.bert_proj = nn.Sequential(
            nn.Linear(768, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
        )

        # fused token-level classifier  ([L, 64] → [L, 1])
        self.token_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # sequence-level classifier  (CLS [768] → [1])
        self.seq_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 1),
        )

        self._init_fusion_weights()

    def _init_fusion_weights(self):
        for layer in [self.bert_proj[0],        # 768→128
                    self.bert_proj[3],        # 128→32
                    self.token_classifier[0],
                    self.token_classifier[3],
                    self.seq_classifier[1]]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    @staticmethod
    def _bert_nuc_tensors(hidden_states, offset_mapping):
        all_nuc = []
        for b in range(hidden_states.shape[0]):
            offsets    = offset_mapping[b]       # [T, 2]
            embeddings = hidden_states[b]        # [T, 768]

            nucleotide_embeddings = []
            for i in range(1, len(offsets)):
                start = offsets[i][0].item()
                end   = offsets[i][1].item()
                if start == 0 and end == 0:
                    break                        # padding sentinel
                token_emb = embeddings[i]
                for _ in range(end - start):     # one entry per nucleotide
                    nucleotide_embeddings.append(token_emb)

            if nucleotide_embeddings:
                nuc_tensor = torch.stack(nucleotide_embeddings)  # [L_i, 768]
            else:
                nuc_tensor = embeddings.new_zeros((0, 768))

            all_nuc.append(nuc_tensor)

        return all_nuc   # list of [L_i, 768]

    def forward(self, fcn_input, input_ids, attention_mask, offset_mapping):

        # FCN stream — independent
        fcn_feat     = self.fcn(fcn_input)                   # [B, 32, L]
        fcn_nuc_list = [fcn_feat[b].permute(1, 0)            # [L, 32]
                        for b in range(fcn_feat.size(0))]

        # BERT stream — independent
        bert_out      = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask)
        hidden        = bert_out[0]                          # [B, T, 768]
        bert_nuc_list = self._bert_nuc_tensors(hidden, offset_mapping)
        # list of [L_i, 768]

        # Sequence-level head (BERT CLS token)
        cls             = self.dropout(hidden[:, 0, :])      # [B, 768]
        logits_sequence = self.seq_classifier(cls)           # [B, 1]

        # Fusion — per-sample
        fused_logits = []
        for fcn_nuc, bert_nuc in zip(fcn_nuc_list, bert_nuc_list):
            # fcn_nuc  : [L,    32]   FCN features
            # bert_nuc : [L_i, 768]   BERT features

            bert_nuc = self.dropout(self.bert_proj(bert_nuc))  # [L_i, 32]
            L        = min(fcn_nuc.size(0), bert_nuc.size(0))
            fcn_nuc  = fcn_nuc[:L]                             # [L, 32]
            bert_nuc = bert_nuc[:L]                            # [L, 32]

            fused  = torch.cat([fcn_nuc, bert_nuc], dim=-1)    # [L, 64]
            logits = self.token_classifier(fused)               # [L, 1]
            fused_logits.append(logits)

        return logits_sequence, fused_logits


# DATASET

class FusedDNADataset(Dataset):
    def __init__(self, bertsnr_txt, fcn_npz, tokenizer, max_len=512):

        self.sequences    = []
        self.bert_labels  = []
        self.seq_labels   = []

        with open(bertsnr_txt) as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                self.sequences.append(parts[0])
                self.bert_labels.append(
                    np.array(list(parts[1]), dtype=np.float32))
                self.seq_labels.append(float(parts[2]))

        fcn = np.load(fcn_npz)
        self.one_hots   = fcn['data']                      # [N, 4, L]
        self.fcn_signal = fcn['signal'].squeeze(1)         # [N, L]

        # load extra BigWig features if present 
        if 'extra' in fcn:
            self.extra_feats = fcn['extra']                # [N, F, L]
            self.n_features  = self.extra_feats.shape[1]
        else:
            self.extra_feats = None
            self.n_features  = 0
        

        assert len(self.sequences) == len(self.one_hots), \
            "bertsnr2 and fcn files have different number of samples"

        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq       = self.sequences[idx]
        seq_label = self.seq_labels[idx]

        #  concatenate extra features onto one-hot 
        one_hot = torch.tensor(self.one_hots[idx], dtype=torch.float32)  # [4, L]
        if self.extra_feats is not None:
            extra   = torch.tensor(self.extra_feats[idx], dtype=torch.float32)  # [F, L]
            one_hot = torch.cat([one_hot, extra], dim=0)                         # [4+F, L]
        

        fcn_signal = torch.tensor(self.fcn_signal[idx], dtype=torch.float32)  # [L]

        encoding = self.tokenizer(
            seq,
            max_length             = self.max_len,
            padding                = 'max_length',
            truncation             = True,
            return_tensors         = 'pt',
            return_offsets_mapping = True,
        )

        bert_nuc = self.bert_labels[idx]
        bert_lbl = np.zeros(self.max_len, dtype=np.float32)
        bert_lbl[:len(bert_nuc)] = bert_nuc[:self.max_len]

        return {
            'one_hot'        : one_hot,          # [4+F, L] or [4, L] if no extras
            'fcn_signal'     : fcn_signal,
            'input_ids'      : encoding['input_ids'].squeeze(0),
            'attention_mask' : encoding['attention_mask'].squeeze(0),
            'offset_mapping' : encoding['offset_mapping'].squeeze(0),
            'bert_labels'    : torch.tensor(bert_lbl, dtype=torch.float32),
            'seq_label'      : torch.tensor(seq_label, dtype=torch.float32),
        }