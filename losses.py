import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020)
    Used in Paper 17 for MD-SupContrast training.

    Input:
        features: tensor of shape [batch_size, feature_dim]
        labels: tensor of shape [batch_size]
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device

        # Normalize feature vectors
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Mask to remove self-comparisons
        logits_mask = torch.ones_like(similarity_matrix) - torch.eye(features.size(0), device=device)
        similarity_matrix = similarity_matrix * logits_mask

        # Build positive mask (same label = positive pair)
        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(device)
        positive_mask = positive_mask * logits_mask  # remove diagonal

        # Compute log-softmax over rows
        log_prob = F.log_softmax(similarity_matrix, dim=1)

        # Only keep positives
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + 1e-8)

        # Final loss
        loss = -mean_log_prob_pos.mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Combines:
        - Supervised Contrastive Loss (SupCon)
        - Cross Entropy Loss (CE)

    Paper 17 uses both during closed-set training.
    """

    def __init__(self, temperature=0.07, ce_weight=1.0, supcon_weight=1.0):
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature)
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.supcon_weight = supcon_weight

    def forward(self, features, logits, labels):
        loss_supcon = self.supcon(features, labels)
        loss_ce = self.ce(logits, labels)
        loss = self.supcon_weight * loss_supcon + self.ce_weight * loss_ce
        return loss, loss_supcon, loss_ce
