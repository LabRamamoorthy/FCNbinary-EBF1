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

# ContextGatedBlock module
class ContextGatedBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        # Local gating components
        self.V = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        
        # Global context
        self.global_proj = nn.Linear(input_dim, hidden_dim)
        
        # Output projection
        self.w = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: [B, L, input_dim]
        
        # Local gated features
        gate = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        
        # Global context (average pooling over sequence)
        global_feat = torch.mean(x, dim=1, keepdim=True)  # [B, 1, input_dim]
        global_feat = self.global_proj(global_feat)  # [B, 1, hidden_dim]
        global_feat = global_feat.expand(-1, x.size(1), -1)  # Broadcast to [B, L, hidden_dim]
        
        # Combine local and global
        combined = gate + global_feat
        out = self.w(combined) + x  # Residual connection
        
        return out



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

        # GRU + ContextGatedBlock
        self.gru_hidden = 48
        self.gru = nn.GRU(32, self.gru_hidden, batch_first=True, bidirectional=True)
        self.ContextGatedBlock = ContextGatedBlock(input_dim=self.gru_hidden * 2, hidden_dim=32)

        # Decoder / blending
        self.blend2 = bn_act_conv(self.gru_hidden * 2 + 32, 32)
        self.blend1 = conv_only(32 + in_ch, 1)

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
            elif isinstance(layer, nn.GRU):
                for name, param in layer.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

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

        # GRU + ContextGatedBlock
        out = out.permute(0, 2, 1)      # [B, L, 32]
        out, _ = self.gru(out)          # [B, L, 96]
        out = self.dropout(out) 
        out = self.ContextGatedBlock(out)       # [B, L, 96] 
        out = out.permute(0, 2, 1)      # [B, 96, L]

        # Decoder - upsample back to original resolution
        out = upsample(out, skip2.size(-1))
        out = torch.cat([out, skip2], dim=1)  # Concatenate skip connection
        out = self.blend2(out)

        out = upsample(out, skip1.size(-1))
        out = torch.cat([out, skip1], dim=1)  # Concatenate skip connection
        out = self.blend1(out)

        return out.squeeze(1)  # [B, L]