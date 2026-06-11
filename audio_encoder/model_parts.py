"""
Model architecture and loss components extracted from train_split_2ch_pre.py
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainClusteringLoss(nn.Module):
    def __init__(self, temperature: float = 0.1,
                 other_margin: float = 0.8,
                 min_domains: int = 2):
        super().__init__()
        self.temperature = temperature
        self.other_margin = other_margin
        self.min_domains = min_domains

    def forward(self, z: torch.Tensor,
                domain_ids: torch.Tensor, eps: float = 1e-4) -> tuple:
        device = z.device
        zero = torch.tensor(0.0, device=device)

        target_mask = domain_ids > 0
        other_mask = domain_ids == 0

        if target_mask.sum() < 2:
            return zero, zero, zero

        z_target = z[target_mask]
        dom_target = domain_ids[target_mask]
        unique_domains = torch.unique(dom_target)
        K = len(unique_domains)

        centroids = {}
        for d in unique_domains:
            mask = dom_target == d
            centroids[d.item()] = z_target[mask].mean(dim=0)

        loss_compact = zero.clone()
        for d in unique_domains:
            mask = dom_target == d
            c = centroids[d.item()].unsqueeze(0)
            loss_compact = loss_compact + ((z_target[mask] - c) ** 2).sum(dim=1).mean()
        loss_compact = loss_compact / K

        c_list = list(centroids.values())
        push = zero.clone()
        n_pairs = 0
        for _i in range(K):
            for _j in range(_i + 1, K):
                sq_dist = ((c_list[_i] - c_list[_j]) ** 2).sum()
                push = push + 1.0 / (sq_dist + eps)
                n_pairs += 1
        loss_push = push / max(n_pairs, 1)

        if K < self.min_domains:
            loss_proto = loss_compact
        else:
            total_proto = zero.clone()
            n_valid = 0
            for d in unique_domains:
                d_int = d.item()
                members = z_target[dom_target == d]
                n_d = members.shape[0]
                for i in range(n_d):
                    zi = members[i]
                    if n_d > 1:
                        others_same = torch.cat([members[:i], members[i+1:]], dim=0)
                        c_pos = others_same.mean(dim=0)
                    else:
                        c_pos = centroids[d_int]

                    d_pos = ((zi - c_pos) ** 2).sum()
                    neg_dists = torch.stack([
                        ((zi - centroids[k]) ** 2).sum()
                        for k in centroids if k != d_int
                    ])
                    all_dists = torch.cat([d_pos.unsqueeze(0), neg_dists])
                    log_probs = F.log_softmax(-all_dists / self.temperature, dim=0)
                    total_proto = total_proto - log_probs[0]
                    n_valid += 1
            loss_proto = total_proto / max(n_valid, 1)

        loss_proto = loss_proto + loss_push

        loss_repel = zero.clone()
        if other_mask.sum() > 0 and len(centroids) > 0:
            z_other = z[other_mask]
            cent_stack = torch.stack(list(centroids.values()))
            dists = torch.cdist(z_other, cent_stack, p=2)
            min_dists, _ = dists.min(dim=1)
            loss_repel = F.relu(self.other_margin - min_dists).mean()

        return loss_compact, loss_proto, loss_repel


class ConvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7,
                 stride: int = 2, padding: int = 3, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if residual and (in_ch != out_ch or stride != 1)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.conv(x)))
        if self.residual:
            skip = self.shortcut(x) if self.shortcut is not None else x
            out = out + skip
        return out


class DeconvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7,
                 stride: int = 2, padding: int = 3, output_padding: int = 1):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_ch, out_ch, kernel,
            stride=stride, padding=padding,
            output_padding=output_padding, bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))


class StereoResidualEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, base_ch: int = 32,
                 dropout: float = 0.0):
        super().__init__()
        ch = [3, base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.blocks = nn.ModuleList([
            ConvBlock1d(ch[i], ch[i + 1], kernel=7, stride=2,
                        padding=3, residual=(i > 0))
            for i in range(len(ch) - 1)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc_mu = nn.Linear(ch[-1], latent_dim)
        self.fc_log_var = nn.Linear(ch[-1], latent_dim)

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        h = self.pool(x).squeeze(-1)
        h = self.dropout(h)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z = F.normalize(mu, p=2, dim=1)
        return z, mu, log_var


class StereoResidualDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, base_ch: int = 32,
                 output_samples: int = 80_000, out_channels: int = 3):
        super().__init__()
        self.output_samples = output_samples
        base_ch_top = base_ch * 16
        self.time_seed = math.ceil(output_samples / (2 ** 5))
        self.fc = nn.Linear(latent_dim, base_ch_top * self.time_seed)

        ch = [base_ch * 16, base_ch * 8, base_ch * 4, base_ch * 2, base_ch, out_channels]
        self.blocks = nn.ModuleList([
            DeconvBlock1d(ch[i], ch[i + 1], kernel=7, stride=2,
                          padding=3, output_padding=1)
            for i in range(len(ch) - 2)
        ])
        self.final_conv = nn.ConvTranspose1d(
            ch[-2], ch[-1], kernel_size=7,
            stride=2, padding=3, output_padding=1,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        bsz = z.shape[0]
        x = self.fc(z)
        x = x.view(bsz, -1, self.time_seed)
        for block in self.blocks:
            x = block(x)
        x = torch.tanh(self.final_conv(x))
        if x.shape[-1] > self.output_samples:
            x = x[..., :self.output_samples]
        elif x.shape[-1] < self.output_samples:
            x = F.pad(x, (0, self.output_samples - x.shape[-1]))
        return x


class DomainAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128, base_ch: int = 32,
                 sample_rate: int = 16_000, max_audio_sec: float = 5.0,
                 dropout: float = 0.0):
        super().__init__()
        output_samples = int(sample_rate * max_audio_sec)
        self.encoder = StereoResidualEncoder(
            latent_dim=latent_dim,
            base_ch=base_ch,
            dropout=dropout,
        )
        self.decoder = StereoResidualDecoder(
            latent_dim=latent_dim,
            base_ch=base_ch,
            output_samples=output_samples,
            out_channels=3,
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        if x.dim() == 3 and x.shape[1] == 1:
            ch1 = x[:, 0, :]
            ch2 = x[:, 0, :]
            res = ch1 - ch2
            x = torch.stack([ch1, ch2, res], dim=1)
        elif x.dim() == 3 and x.shape[1] == 2:
            ch1 = x[:, 0, :]
            ch2 = x[:, 1, :]
            res = ch1 - ch2
            x = torch.stack([ch1, ch2, res], dim=1)
        return x

    def forward(self, x: torch.Tensor):
        x = self._prepare_input(x)
        z, mu, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        return {"z": z, "mu": mu, "log_var": log_var, "x_hat": x_hat}

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["z"]


class StereoReconstructionLoss(nn.Module):
    def __init__(self, fft_sizes: tuple = (512, 1024, 2048)):
        super().__init__()
        self.fft_sizes = fft_sizes

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x_hat.dim() == 2:
            x_hat = x_hat.unsqueeze(1)

        loss_mse = F.mse_loss(x_hat, x)

        bsz, channels, time_steps = x.shape
        x_flat = x.reshape(bsz * channels, time_steps)
        x_hat_flat = x_hat.reshape(bsz * channels, time_steps)

        loss_spec = torch.tensor(0.0, device=x.device)
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            window = torch.hann_window(n_fft, device=x.device)
            x_stft = torch.stft(x_flat, n_fft=n_fft, hop_length=hop,
                                return_complex=True, window=window)
            x_hat_stft = torch.stft(x_hat_flat, n_fft=n_fft, hop_length=hop,
                                    return_complex=True, window=window)
            log_x = torch.log(x_stft.abs() + 1e-7)
            log_x_hat = torch.log(x_hat_stft.abs() + 1e-7)
            loss_spec = loss_spec + F.l1_loss(log_x_hat, log_x)

        return loss_mse + (loss_spec / len(self.fft_sizes))


class StereoTemporalConsistencyLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, model: DomainAutoencoder, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        t_steps = x.shape[-1]
        mid = t_steps // 2
        crop_a = x[..., :mid]
        crop_b = x[..., mid:]

        z_a = model.get_embeddings(crop_a)
        z_b = model.get_embeddings(crop_b)
        cos_sim = (z_a * z_b).sum(dim=1)
        return (1.0 - cos_sim / self.temperature).mean()
