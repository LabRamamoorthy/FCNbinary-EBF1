import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper functions
def upsample(x, out_size):
    return F.interpolate(x, size=out_size, mode='linear', align_corners=False)


def bn_act_conv(in_, out_, kernel_size=3, stride=1, bias=False, act=nn.GELU):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm1d(out_),
        act()
    )


def conv_only(in_, out_, kernel_size=3, stride=1, bias=True):
    padding = kernel_size // 2
    return nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

# Attention module
class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, input_dim)   

    def forward(self, x):
        gate = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        out = self.w(gate)          
        return out, gate



class FCNsignalMod(nn.Module):
    def __init__(self, motiflen=12, n_features=0, dropout=0.2):
        super().__init__()

        in_ch = 4  # DNA one-hot
        self.has_extra = n_features > 0

        # Project extra features to 32 channels if present
        if self.has_extra:
            self.feature_proj = nn.Conv1d(n_features, 32, kernel_size=1)
            in_ch += 32

        # Encoder
        self.conv1 = bn_act_conv(in_ch, 32, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = bn_act_conv(32, 32, kernel_size=7)
        self.pool2 = nn.MaxPool1d(2)

        # GRU + Attention
        self.gru_hidden = 48
        self.gru = nn.GRU(32, self.gru_hidden, batch_first=True, bidirectional=True)
        self.attention = Attention(input_dim=self.gru_hidden * 2, hidden_dim=16)
        self.res_proj = nn.Linear(32, 96)

        # Decoder / blending
        self.blend2 = bn_act_conv(self.gru_hidden * 2, 32)
        self.blend1 = conv_only(32, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
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

    def forward(self, x):
        # x: [B, C, L] (C = 4 or 4 + n_features)
        if self.has_extra:
            extra = self.feature_proj(x[:, 4:, :])  # project extra features
            x = torch.cat([x[:, :4, :], extra], dim=1)

        skip1 = x
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.dropout(out)

        skip2 = out
        out = self.conv2(out)
        if out.size(1) == skip2.size(1):  
            out = out + skip2
        out = self.pool2(out)
        out = self.dropout(out)

        # GRU + Attention
        out = out.permute(0, 2, 1)      # [B, L, 32]
        residual = out                  # [B, L, 32]
        out, _ = self.gru(out)          # [B, L, 96]
        out_att, _ = self.attention(out)  # [B, L, 96]
        residual = self.res_proj(residual)  # [B, L, 96]
        out = out_att + residual        
        out = out.permute(0, 2, 1)     

        # Decoder
        out = upsample(out, skip2.size(-1))
        out = self.blend2(out)
        out = upsample(out, skip1.size(-1))
        out = self.blend1(out)

        return out.squeeze(1)  # [B, L]



# class Attention(nn.Module):
#     def __init__(self, input_dim, hidden_dim=16):
#         super().__init__()
#         self.attn = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, x):
#         # x: [B, L, C]
#         # Compute raw attention scores
#         scores = self.attn(x)  # shape [B, L, 1]
        
#         # Softmax across sequence length
#         weights = torch.softmax(scores, dim=1)  # sum over L = 1
        
#         # Multiply input by attention weights
#         out = x * weights
#         return out, weights
