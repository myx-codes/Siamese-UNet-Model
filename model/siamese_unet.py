# siamese_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


# =====================================================
# --- [1] Gradient Reversal & Domain Discriminator ----
# =====================================================
class GradReverse(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL).
    Passes input unchanged during forward pass.
    Multiplies gradients by -alpha during backward pass.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """
    Domain classifier that tries to distinguish between inhale and exhale features.
    Used for adversarial training.
    """
    def __init__(self, in_ch: int = 512, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Slightly higher dropout for discriminator stability
            nn.Linear(hidden, 1)  # Binary output: 0 (Inhale) vs 1 (Exhale)
        )

    def forward(self, v):
        return self.mlp(v)


# =====================================================
# --- [2] Basic Blocks (SCJAG, Conv, Up) -------------
# =====================================================
class SCJAGGate(nn.Module):
    """
    Spatial-Channel Joint Attention Gate (SCJAG)
    
    Implements the attention mechanism described in the paper:
    1. Channel Attention (SE-style) applied to skip connection (x)
    2. Joint Spatial Attention (additive) using decoder gating signal (g) and attended skip (x_ch)
    """
    def __init__(self, in_ch_x: int, in_ch_g: int, reduction: int = 16):
        super().__init__()
        
        # --- 1. Channel Attention Components ---
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_ch_x, max(8, in_ch_x // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, in_ch_x // reduction), in_ch_x),
            nn.Sigmoid()
        )

        # --- 2. Joint Spatial Attention Components ---
        inter_ch = in_ch_x  # Intermediate channel for additive attention
        
        self.Wx = nn.Conv2d(in_ch_x, inter_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.Wg = nn.Conv2d(in_ch_g, inter_ch, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        """
        Args:
            x: Skip connection feature from encoder [B, Cx, H, W]
            g: Gating signal from decoder           [B, Cg, H, W]
        """
        batch, ch, h, w = x.size()

        # Step 1: Channel Attention
        s = self.avg_pool(x).view(batch, ch)
        s = self.channel_mlp(s).view(batch, ch, 1, 1)
        x_ch = x * s  # Attended skip features

        # Step 2: Joint Spatial Attention (additive)
        g_proj = self.Wg(g)
        x_proj = self.Wx(x_ch)
        
        q = self.relu(g_proj + x_proj)
        psi = self.psi(q)

        # Step 3: Final attended skip connection
        return x_ch * psi


class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv -> BN -> ReLU -> Dropout?) x 2
    Dropout probability controlled via config.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout(dropout_p))
        
        layers.extend([
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling -> SCJAG Attention -> Concatenation -> DoubleConv
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_scjag: bool = True, dropout_p: float = 0.0):
        super().__init__()
        
        # 1. Upsample decoder feature
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        
        # 2. Attention gate (SCJAG)
        self.use_scjag = use_scjag
        if use_scjag:
            self.gate = SCJAGGate(in_ch_x=skip_ch, in_ch_g=out_ch)
        else:
            self.gate = nn.Identity()

        # 3. Convolution after concatenation
        conv_in_ch = out_ch + skip_ch
        self.conv = DoubleConv(conv_in_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x, skip):
        """
        Args:
            x: Feature from lower decoder level
            skip: Corresponding encoder skip connection
        """
        # Upsample gating signal
        g = self.up(x)
        
        # Align spatial dimensions (in case of odd sizes)
        dh = skip.size(2) - g.size(2)
        dw = skip.size(3) - g.size(3)
        if dh != 0 or dw != 0:
            g = F.pad(g, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        
        # Apply attention to skip connection
        if self.use_scjag:
            skip = self.gate(x=skip, g=g)
        
        # Concatenate and convolve
        x = torch.cat([g, skip], dim=1)
        return self.conv(x)


# =====================================================
# --- [3] Encoder, Decoder, and Heads -----------------
# =====================================================
class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        backbone = resnet34(weights=None)
        # Adapt first conv to desired number of input channels (e.g., 3, 6, 9, etc.)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, nn.ReLU(inplace=True))
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4


class UNetDecoder(nn.Module):
    """
    Standard U-Net decoder with SCJAG attention gates.
    """
    def __init__(self, use_scjag: bool = True, dropout_p: float = 0.0):
        super().__init__()
        # Encoder feature channels: x4=512, x3=256, x2=128, x1=64, x0=64
        self.up4 = UpBlock(512, 256, 256, use_scjag, dropout_p)
        self.up3 = UpBlock(256, 128, 128, use_scjag, dropout_p)
        self.up2 = UpBlock(128,  64,  64, use_scjag, dropout_p)
        self.up1 = UpBlock( 64,  64,  32, use_scjag, dropout_p)
        
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, feats):
        x0, x1, x2, x3, x4 = feats
        
        d4 = self.up4(x4, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)
        
        logit = self.head(d1)
        
        # Upsample to original input resolution (512x512 -> 1024x1024? No: ResNet34 downscales by 32,
        # so final feature map is 512/32 = 16x16 -> after 4 upsamples: 256x256 -> final interpolate to 512x512)
        logit = F.interpolate(logit, scale_factor=2, mode='bilinear', align_corners=True)
        
        return logit


class EmbedHead(nn.Module):
    """
    Projection head for contrastive learning (SimCLR-style).
    """
    def __init__(self, in_ch: int, config: dict):
        super().__init__()
        embed_dim = config['model']['embed_dim']
        dropout_val = config['model']['dropout']
        hidden_dim = config['model'].get('proj_hidden_dim', in_ch)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_val),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, f):
        return self.proj(self.pool(f))


# =====================================================
# --- [4] Main Siamese Model --------------------------
# =====================================================
class SiameseUNetSCJAG(nn.Module):
    """
    Siamese U-Net with SCJAG attention gates.
    Supports two modes:
      - Separate decoders for inhale/exhale
      - Shared decoder
    Supports two regularization strategies:
      - Contrastive (default)
      - Domain Adversarial (with GRL)
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        in_channels = config['data']['in_channels']
        decoders    = config['model']['decoders']  # "separate" or "shared"
        dropout     = config['model']['dropout']
        
        # 1. Shared Encoder
        self.encoder = ResNet34Encoder(in_channels=in_channels)
        
        # 2. Embedding head (always active for contrastive learning)
        self.embed = EmbedHead(in_ch=512, config=config)

        # 3. Domain discriminator (conditional)
        self.adv_mode = config['loss'].get('adv_mode', 'contrastive')
        
        if self.adv_mode == 'domain_adversarial':
            self.domain_disc = DomainDiscriminator(in_ch=512, hidden=256)
            print(">> Model initialized in Domain Adversarial mode (GRL enabled).")
        else:
            print(">> Model initialized in Contrastive mode (SimCLR-style).")

        # 4. Decoders
        self.decoders_mode = decoders
        if decoders == "separate":
            self.dec_in = UNetDecoder(use_scjag=True, dropout_p=dropout)
            self.dec_ex = UNetDecoder(use_scjag=True, dropout_p=dropout)
        else:
            self.dec_shared = UNetDecoder(use_scjag=True, dropout_p=dropout)

    def _encode(self, x):
        return self.encoder(x)

    def _decode_in(self, feats):
        if self.decoders_mode == "separate":
            return self.dec_in(feats)
        return self.dec_shared(feats)

    def _decode_ex(self, feats):
        if self.decoders_mode == "separate":
            return self.dec_ex(feats)
        return self.dec_shared(feats)

    def forward(self, xin, xex, alpha=1.0):
        """
        Args:
            xin, xex: Input tensors [B, C, H, W] for inhale and exhale
            alpha: Gradient reversal coefficient (used during training)
        """
        # Inhale branch
        feats_in = self._encode(xin)
        pred_in = self._decode_in(feats_in)
        emb_in = self.embed(feats_in[-1])

        # Exhale branch
        feats_ex = self._encode(xex)
        pred_ex = self._decode_ex(feats_ex)
        emb_ex = self.embed(feats_ex[-1])
        
        # Domain adversarial branch (optional)
        domain_pred_in = None
        domain_pred_ex = None
        
        if self.adv_mode == 'domain_adversarial' and hasattr(self, 'domain_disc'):
            # Global average pooling of deepest features
            f_in_flat = feats_in[-1].mean([2, 3])
            f_ex_flat = feats_ex[-1].mean([2, 3])
            
            # Apply gradient reversal
            feat_in_rev = grad_reverse(f_in_flat, alpha=alpha)
            feat_ex_rev = grad_reverse(f_ex_flat, alpha=alpha)
            
            # Domain classification
            domain_pred_in = self.domain_disc(feat_in_rev)
            domain_pred_ex = self.domain_disc(feat_ex_rev)

        return {
            "pred_mask_in": pred_in,
            "pred_mask_ex": pred_ex,
            "embedding_in": emb_in,
            "embedding_ex": emb_ex,
            "domain_pred_in": domain_pred_in,
            "domain_pred_ex": domain_pred_ex
        }