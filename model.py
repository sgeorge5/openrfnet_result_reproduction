import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBranch(nn.Module):
    """
    Extracts texture features from spectrograms using ResNet-18
    """

    def __init__(self, pretrained=False, out_dim=512):
        super().__init__()

        resnet = models.resnet18(weights=None if not pretrained else "IMAGENET1K_V1")

        # Modify first conv layer to accept 1-channel input
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove final classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = out_dim

    def forward(self, x):
        feat = self.feature_extractor(x)  # (B, 512, 1, 1)
        return feat.view(feat.size(0), -1)



class TransformerBranch(nn.Module):
    """
    Extracts time-frequency positional features using TransformerEncoder.
    """

    def __init__(self, embed_dim=256, num_layers=4, num_heads=8):
        super().__init__()

        self.embed_dim = embed_dim

        # Linear projection to embedding dimension
        self.proj = nn.Linear(224, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling to get a single feature vector
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, 1, H, W) → flatten frequency dimension
        x = x.squeeze(1)  # (B, H, W)

        # Project each row into embedding space
        x = self.proj(x)  # (B, H, embed_dim)

        # Transformer expects (B, seq_len, embed_dim)
        x = self.transformer(x)  # (B, H, embed_dim)

        # Pool across sequence length
        x = x.permute(0, 2, 1)  # (B, embed_dim, H)
        x = self.pool(x).squeeze(-1)  # (B, embed_dim)

        return x


class FusionHead(nn.Module):
    """
    Fuses ResNet and Transformer features.
    """

    def __init__(self, res_dim=512, trans_dim=256, fused_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(res_dim + trans_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim)
        )
        self.out_dim = fused_dim

    def forward(self, res_feat, trans_feat):
        fused = torch.cat([res_feat, trans_feat], dim=1)
        return self.fc(fused)


class ProjectionHead(nn.Module):
    """
    Projection head for supervised contrastive learning.
    """

    def __init__(self, in_dim=256, proj_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        return self.proj(x)


class ClassifierHead(nn.Module):
    """
    Classification head for closed-set training.
    """

    def __init__(self, in_dim=256, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class OpenRFNet(nn.Module):
    """
    Full model combining:
        - ResNet branch
        - Transformer branch
        - Fusion module
        - Projection head (SupCon)
        - Classifier head (closed-set)
    """

    def __init__(self, num_classes):
        super().__init__()

        self.resnet_branch = ResNetBranch()
        self.transformer_branch = TransformerBranch()
        self.fusion = FusionHead()

        self.projection_head = ProjectionHead()
        self.classifier_head = ClassifierHead(num_classes=num_classes)

    def forward(self, x, return_features=False):
        res_feat = self.resnet_branch(x)
        trans_feat = self.transformer_branch(x)
        fused_feat = self.fusion(res_feat, trans_feat)

        proj = self.projection_head(fused_feat)
        logits = self.classifier_head(fused_feat)

        if return_features:
            return fused_feat, proj, logits

        return proj, logits
