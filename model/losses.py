import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 1. Component Losses (BCE, Dice, Contrastive)
# =========================================================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten while preserving batch dimension
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class CombinedSegLoss(nn.Module):
    """
    Combined segmentation loss: BCE + Dice
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, pos_weight: float = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        # BCE with optional positive weight for class imbalance
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor(pos_weight))
            self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()
            
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_bce = self.bce(logits, targets)
        l_dice = self.dice(logits, targets)
        return (self.bce_weight * l_bce) + (self.dice_weight * l_dice)

def contrastive_loss(emb1: torch.Tensor, emb2: torch.Tensor, mode: str = 'l2') -> torch.Tensor:
    """
    Contrastive loss to bring positive pairs (inhale-exhale embeddings) closer.
    SimCLR-style.
    """
    if mode == 'l2':
        return F.mse_loss(emb1, emb2)
    elif mode == 'cosine':
        # 1 - cosine similarity to turn it into a distance
        return 1 - F.cosine_similarity(emb1, emb2).mean()
    return F.mse_loss(emb1, emb2)  # default to L2


# =========================================================
# 2. Main Siamese Loss Wrapper (Handles all losses)
# =========================================================

class SiameseLoss(nn.Module):
    """
    Computes total loss from model outputs (dictionary) including:
    - Segmentation loss for both inhale and exhale branches
    - Optional adversarial or contrastive regularization on embeddings
    """
    def __init__(self, cfg: dict):
        super().__init__()
        
        # Read loss parameters from config
        loss_cfg = cfg.get("loss", {})
        
        # 1. Segmentation loss weights
        bce_w = float(loss_cfg.get("bce_weight", 0.5))
        dice_w = float(loss_cfg.get("dice_weight", 0.5))
        pos_w = loss_cfg.get("bce_pos_weight", None)
        if pos_w is not None:
            pos_w = float(pos_w)
        
        self.seg_criterion = CombinedSegLoss(bce_weight=bce_w, dice_weight=dice_w, pos_weight=pos_w)
        
        # 2. Adversarial / Contrastive regularization
        self.lambda_adv = float(loss_cfg.get("lambda_adv", 0.05))
        self.adv_mode = loss_cfg.get("adv_mode", "contrastive")  # "contrastive" or "domain_adversarial"
        self.emb_consistency = loss_cfg.get("emb_consistency", "l2")  # "l2" or "cosine"

    def forward(self, outputs: dict, targets: dict):
        """
        Args:
            outputs: dict from model, e.g.,
                     {'pred_mask_in', 'pred_mask_ex',
                      'embedding_in', 'embedding_ex',
                      'domain_pred_in', 'domain_pred_ex' (optional)}
            targets: dict with {'in': mask_in_tensor, 'ex': mask_ex_tensor}
        """
        device = next(self.parameters()).device if self.parameters() else outputs['pred_mask_in'].device

        # ---------------------------
        # A. Segmentation Loss
        # ---------------------------
        l_seg_in = self.seg_criterion(outputs['pred_mask_in'], targets['in'])
        l_seg_ex = self.seg_criterion(outputs['pred_mask_ex'], targets['ex'])
        
        l_seg_total = l_seg_in + l_seg_ex

        # ---------------------------
        # B. Adversarial / Embedding Regularization Loss
        # ---------------------------
        l_adv = torch.tensor(0.0, device=device)

        # 1. Domain Adversarial mode (with Gradient Reversal Layer)
        if self.adv_mode == 'domain_adversarial' and 'domain_pred_in' in outputs:
            # Targets: Inhale -> 0, Exhale -> 1
            tgt_in = torch.zeros_like(outputs['domain_pred_in'])
            tgt_ex = torch.ones_like(outputs['domain_pred_ex'])
            
            l_dom_in = F.binary_cross_entropy_with_logits(outputs['domain_pred_in'], tgt_in)
            l_dom_ex = F.binary_cross_entropy_with_logits(outputs['domain_pred_ex'], tgt_ex)
            l_adv = l_dom_in + l_dom_ex
            
        # 2. Contrastive mode (default)
        elif self.adv_mode == 'contrastive':
            l_adv = contrastive_loss(
                outputs['embedding_in'], 
                outputs['embedding_ex'], 
                mode=self.emb_consistency
            )

        # ---------------------------
        # C. Total Loss
        # ---------------------------
        total_loss = l_seg_total + (self.lambda_adv * l_adv)

        # Return breakdown for logging / monitoring
        loss_dict = {
            "total": total_loss,
            "seg": l_seg_total,
            "seg_in": l_seg_in,
            "seg_ex": l_seg_ex,
            "adv": l_adv
        }
        
        return total_loss, loss_dict