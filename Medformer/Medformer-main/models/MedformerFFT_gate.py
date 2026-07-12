"""
MedformerFFT with Gated Fusion.
Gate = sigmoid(W * [time_feat, freq_feat])
Fused = gate * time_feat + (1-gate) * freq_feat
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import ListPatchEmbedding
from layers.Medformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import MedformerLayer


class FrequencyEncoder(nn.Module):
    def __init__(self, seq_len, enc_in, d_model, dropout):
        super().__init__()
        freq_bins = seq_len // 2 + 1
        self.flatten_dim = freq_bins * enc_in
        self.encoder = nn.Sequential(
            nn.Linear(self.flatten_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )

    def forward(self, x_enc):
        freq_mag = torch.abs(torch.fft.rfft(x_enc, dim=1))
        freq_feature = freq_mag.reshape(freq_mag.shape[0], -1)
        return self.encoder(freq_feature)


class TimeFrequencyModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.single_channel = configs.single_channel

        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        augmentations = configs.augmentations.split(",")

        self.enc_embedding = ListPatchEmbedding(
            configs.enc_in, configs.d_model, configs.seq_len,
            patch_len_list, stride_list, configs.dropout,
            augmentations, configs.single_channel,
        )
        self.encoder = Encoder(
            [EncoderLayer(
                MedformerLayer(len(patch_len_list), configs.d_model,
                    configs.n_heads, configs.dropout,
                    configs.output_attention, configs.no_inter_attn),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation)
             for _ in range(configs.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.time_feature_dim = (
            configs.d_model * len(patch_num_list)
            * (1 if not self.single_channel else configs.enc_in)
        )
        self.frequency_encoder = FrequencyEncoder(
            configs.seq_len, configs.enc_in, configs.d_model, configs.dropout
        )

        # Gate fusion: project freq to same dim as time, then learnable gate
        self.freq_proj = nn.Linear(configs.d_model, self.time_feature_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.time_feature_dim * 2, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, 1),
            nn.Sigmoid(),
        )
        self.projection = nn.Linear(self.time_feature_dim, configs.num_class)

    def extract_time_feature(self, x_enc):
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))
        output = self.act(enc_out)
        output = self.dropout(output)
        return output.reshape(output.shape[0], -1)

    def classification(self, x_enc, x_mark_enc):
        time_feature = self.extract_time_feature(x_enc)
        freq_feature = self.frequency_encoder(x_enc)
        freq_feature = self.freq_proj(freq_feature)  # project to same dim

        # Gated fusion
        combined = torch.cat([time_feature, freq_feature], dim=-1)
        gate_weight = self.gate(combined)  # [B, 1]
        fused = gate_weight * time_feature + (1 - gate_weight) * freq_feature

        return self.projection(fused)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        raise NotImplementedError("MedformerFFT_gate only supports classification.")


class Model(TimeFrequencyModel):
    pass
