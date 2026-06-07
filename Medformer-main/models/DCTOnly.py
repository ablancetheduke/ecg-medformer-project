"""
DCTOnly — 纯DCT变换频域 + MLP (消融: 无Medformer时域分支)
对比 FrequencyOnly(FFT) 和 WaveletOnly, 验证纯DCT单支的分类能力。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dct
import numpy as np


class DCTOnlyEncoder(nn.Module):
    def __init__(self, seq_len, enc_in, d_model, dropout):
        super().__init__()
        self.flatten_dim = seq_len * enc_in
        self.encoder = nn.Sequential(
            nn.Linear(self.flatten_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, 5)

    def forward(self, x_enc):
        B, T, C = x_enc.shape
        device = x_enc.device
        x_np = x_enc.cpu().numpy()
        dct_coeffs = dct(x_np, type=2, axis=1, norm='ortho')
        dct_flat = dct_coeffs.reshape(B, -1)
        freq_feature = torch.from_numpy(dct_flat.astype(np.float32)).to(device)
        freq_feature = self.encoder(freq_feature)
        freq_feature = self.dropout(freq_feature)
        return self.projection(freq_feature)


class Model(DCTOnlyEncoder):
    def __init__(self, configs):
        super().__init__(
            configs.seq_len, configs.enc_in, configs.d_model,
            configs.dropout
        )
        self.task_name = configs.task_name

    def classification(self, x_enc, x_mark_enc):
        return self.forward(x_enc)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return super().forward(x_enc)
