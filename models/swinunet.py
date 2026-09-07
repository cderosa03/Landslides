import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Lookup modelli Swin V2 ────────────────────────────────────────────────
swinv2_size = {
    "tiny":  "swinv2_tiny_window8_256.ms_in1k",
    "small": "swinv2_small_window8_256.ms_in1k",
    "base":  "swinv2_base_window8_256.ms_in1k",
}

swinv2_num_heads_dict = {
    "swinv2_tiny_window8_256.ms_in1k":  [3, 6, 12, 24],
    "swinv2_small_window8_256.ms_in1k": [3, 6, 12, 24],
    "swinv2_base_window8_256.ms_in1k":  [4, 8, 16, 32],
}



# Blocchi condivisi

class DiffFusionModule(nn.Module):
    """
    Fonde la differenza di feature S2 e PlanetScope per uno stage dell'encoder.
    Input:  concatenazione canale di (diff_s2, diff_planet) → (B, 2*C, H, W)
    Output: (B, C, H, W)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.relu(self.fusion(x))


class BasicLayer_up(nn.Module):
    """Uno stage del decoder Transformer."""
    def __init__(self, dim, depth, num_heads):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=4 * dim,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        img_size=128,
        enc_channels=(96, 192, 384, 768),
        depths_decoder=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=1,
    ):
        super().__init__()
        self.encoder_channels = enc_channels
        self.num_layers = len(depths_decoder)

        self.layers_up      = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            dim = enc_channels[-(i_layer + 1)]
            self.layers_up.append(BasicLayer_up(
                dim=dim,
                depth=depths_decoder[-(i_layer + 1)],
                num_heads=num_heads[-(i_layer + 1)],
            ))
            if i_layer < self.num_layers - 1:
                skip_dim = enc_channels[-(i_layer + 2)]
                self.concat_back_dim.append(nn.Linear(dim + skip_dim, skip_dim))
            else:
                self.concat_back_dim.append(nn.Identity())

        self.norm_up = nn.LayerNorm(enc_channels[0])
        self.output  = nn.Conv2d(enc_channels[0], num_classes, kernel_size=1)

        self.img_size   = img_size

    def forward(self, x):
        x, skips = x[-1], x[:-1]

        for i in range(self.num_layers):
            batch, channels, height, width = x.shape
            tokens = x.flatten(2).transpose(1, 2)
            x = self.layers_up[i](tokens).transpose(1, 2).reshape(
                batch, channels, height, width
            )
            if i < len(skips):
                skip = skips[-(i + 1)]
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
                x = torch.cat([x, skip], dim=1)
                batch, channels, height, width = x.shape
                x = self.concat_back_dim[i](x.flatten(2).transpose(1, 2))
                x = x.transpose(1, 2).reshape(batch, -1, height, width)

        batch, channels, height, width = x.shape
        x = self.norm_up(x.flatten(2).transpose(1, 2))
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        x = F.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.output(x)


class SwinEncoder(nn.Module):
    """
    Swin Transformer V2 in features_only mode (timm).
    Restituisce 4 feature map intermedie per le skip connections del decoder.
    """
    def __init__(self, model_name, img_size, in_chans, pretrained, out_indices):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=out_indices,
            img_size=img_size,
        )
        self.out_channels = self.model.feature_info.channels()

    def forward(self, x):
        feats = self.model(x)
        output = []
        for feature, channels in zip(feats, self.out_channels):
            if feature.shape[1] == channels:  # timm already returned NCHW
                output.append(feature)
            elif feature.shape[-1] == channels:  # Swin native NHWC
                output.append(feature.permute(0, 3, 1, 2))
            else:
                raise ValueError(
                    f"Unexpected Swin feature shape {tuple(feature.shape)} for {channels} channels"
                )
        return output


class AuxPyramidEncoder(nn.Module):
    """Lightweight terrain encoder aligned to the four Swin feature scales."""

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride):
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2 if kernel_size == 3 else 0,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def __init__(self, in_channels, channels):
        super().__init__()
        self.stem = self._block(in_channels, channels[0], kernel_size=4, stride=4)
        self.downsamples = nn.ModuleList(
            self._block(channels[index - 1], channels[index], kernel_size=3, stride=2)
            for index in range(1, len(channels))
        )

    def forward(self, aux):
        features = [self.stem(aux)]
        for layer in self.downsamples:
            features.append(layer(features[-1]))
        return features


# Modello principale

class ChangeDetectionSwinUNet(nn.Module):
    """
    Change detection multimodale PlanetScope + Sentinel-2 con serie temporale.

    Sentinel-2 accetta una serie di T immagini (T ≤ N_TEMPORAL):
      s2_t1 / s2_t2 : (B, T, 10, H, W)
      valid_t1 / valid_t2 : (B, T) bool  — True = frame reale, False = padding zero

    Le feature di ogni frame vengono calcolate indipendentemente dallo stesso encoder S2,
    poi aggregate con una media pesata sui soli frame validi (masked temporal mean).

    PlanetScope resta a singola immagine pre/post come in origine:
      p_t1 / p_t2 : (B, C_p, H, W)
    """

    def __init__(
        self, img_size=128, num_classes=1, model_size="small", aux_channels=4,
        pretrained=True,
    ):
        super().__init__()
        self.img_size = img_size

        model_name = swinv2_size[model_size]
        num_heads  = swinv2_num_heads_dict[model_name]
        out_indices = (0, 1, 2, 3)

        # ── Adattatore S2: 10 bande → 3 canali compatibili con Swin ─────
        self.s2_adapter = nn.Conv2d(10, 3, kernel_size=1, bias=False)

        # ── Encoder separati per i due sensori ───────────────────────────
        self.s2_encoder = SwinEncoder(
            model_name=model_name, img_size=img_size,
            in_chans=3, pretrained=pretrained, out_indices=out_indices,
        )
        self.planet_encoder = SwinEncoder(
            model_name=model_name, img_size=img_size,
            in_chans=3, pretrained=pretrained, out_indices=out_indices,
        )

        enc_channels = self.s2_encoder.out_channels  # [96, 192, 384, 768]
        self.aux_channels = aux_channels
        self.aux_encoder = AuxPyramidEncoder(aux_channels, enc_channels)

        # ── Moduli di fusione (uno per stage) ────────────────────────────
        self.fusion_stages = nn.ModuleList([
            DiffFusionModule(3 * ch, ch) for ch in enc_channels
        ])

        # ── Decoder Transformer ──────────────────────────────────────────
        self.decoder = TransformerDecoder(
            img_size=img_size,
            enc_channels=enc_channels,
            num_heads=num_heads,
            num_classes=num_classes,
        )

    # ── Utilità per la media temporale pesata ────────────────────────────
    @staticmethod
    def _temporal_mean(feats_flat, B, T, valid_mask):
        """
        Calcola la media delle feature sui soli frame validi.

        feats_flat : list di 4 tensori shape (B*T, C, H, W)
        valid_mask : (B, T) bool — True = frame reale

        Ritorna: list di 4 tensori shape (B, C, H, W)
        """
        result = []
        # valid_mask come float per la media pesata: (B, T, 1, 1, 1)
        w = valid_mask.float().to(feats_flat[0].device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        for f in feats_flat:
            _, C_f, H_f, W_f = f.shape
            f = f.view(B, T, C_f, H_f, W_f)          # (B, T, C, H, W)
            # Somma pesata / numero frame validi (almeno 1 per evitare div/0)
            f = (f * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)  # (B, C, H, W)
            result.append(f)

        return result

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(self, s2_t1, s2_t2, p_t1, p_t2, aux,
                valid_t1=None, valid_t2=None):
        """
        s2_t1, s2_t2   : (B, T, 10, H, W)  serie temporale Sentinel-2
        p_t1,  p_t2    : (B, C_p,  H, W)   PlanetScope pre/post
        aux             : (B, 4, H, W)      contesto topografico normalizzato
        valid_t1/t2    : (B, T) bool        maschera frame validi (None = tutti validi)
        """
        B, T, C, H, W = s2_t1.shape

        # ── Default: tutti i frame considerati validi ────────────────────
        if valid_t1 is None:
            valid_t1 = torch.ones(B, T, dtype=torch.bool, device=s2_t1.device)
        if valid_t2 is None:
            valid_t2 = torch.ones(B, T, dtype=torch.bool, device=s2_t2.device)

        # ── Adattatore 10→3 canali su ogni frame (batch flattening) ─────
        s2_t1_flat = s2_t1.view(B * T, C, H, W)
        s2_t2_flat = s2_t2.view(B * T, C, H, W)
        s2_t1_flat = self.s2_adapter(s2_t1_flat)   # (B*T, 3, H, W)
        s2_t2_flat = self.s2_adapter(s2_t2_flat)

        # ── Encoding di ogni frame S2 ─────────────────────────────────────
        s2_f1_flat = self.s2_encoder(s2_t1_flat)   # list[4 × (B*T, C, H, W)]
        s2_f2_flat = self.s2_encoder(s2_t2_flat)

        # ── Media temporale pesata (ignora frame di padding) ─────────────
        s2_f1 = self._temporal_mean(s2_f1_flat, B, T, valid_t1)  # list[4 × (B,C,H,W)]
        s2_f2 = self._temporal_mean(s2_f2_flat, B, T, valid_t2)

        # ── Encoding PlanetScope (singola immagine, invariato) ───────────
        p_f1 = self.planet_encoder(p_t1)   # list[4 × (B, C, H, W)]
        p_f2 = self.planet_encoder(p_t2)
        if aux.ndim != 4 or aux.shape[1] != self.aux_channels:
            raise ValueError(
                f"aux atteso (B,{self.aux_channels},H,W), ricevuto {tuple(aux.shape)}"
            )
        aux_features = self.aux_encoder(aux)

        # ── Fusione: differenza S2 + differenza Planet → DiffFusionModule ─
        fused = []
        for i in range(len(s2_f1)):
            diff_s2 = s2_f2[i] - s2_f1[i]          # cambiamento S2
            diff_p  = p_f2[i]  - p_f1[i]           # cambiamento Planet
            if aux_features[i].shape != diff_p.shape:
                raise ValueError(
                    f"AUX stage {i} non allineato: {tuple(aux_features[i].shape)} vs "
                    f"{tuple(diff_p.shape)}"
                )
            f = torch.cat([diff_s2, diff_p, aux_features[i]], dim=1)
            fused.append(self.fusion_stages[i](f))   # (B, C, H, W)

        # ── Decoder → logit per pixel ─────────────────────────────────────
        return self.decoder(fused)   # (B, 1, H, W)
