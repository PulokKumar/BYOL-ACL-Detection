import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MLPHead(nn.Module):
    """Simple 2-layer projection/prediction MLP used in BYOL."""

    def __init__(self, in_dim, hidden_dim=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class BYOLNet(nn.Module):
    """Online network: encoder + projection + prediction"""

    def __init__(
            self, encoder, proj_hidden=512, proj_out=256, pred_hidden=512, pred_out=256,
            encoder_dim=512):
        super().__init__()
        self.encoder = encoder
        self.projector = MLPHead(encoder_dim, proj_hidden, proj_out)
        self.predictor = MLPHead(proj_out, pred_hidden, pred_out)

    def forward(self, x):
        feat = self.encoder(x)
        proj = self.projector(feat)
        pred = self.predictor(proj)
        return F.normalize(
            pred, dim=1), F.normalize(
            proj.detach(),
            dim=1)


class Classifier(nn.Module):
    def __init__(self, encoder, feat_dim=512, num_classes=3, head_type="linear"):
        """
        head_type: "linear" | "mlp"
        """
        super().__init__()
        self.encoder = encoder
        self.head_type = head_type.lower()

        if self.head_type == "linear":
            self.head = nn.Linear(feat_dim, num_classes)

        elif self.head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )

        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x):

        feat = self.encoder(x)
        out = self.head(feat)
        return out


def conv1_channel1_modify(resnet):
    old = resnet.conv1
    new_conv = torch.nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                               stride=old.stride, padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new_conv.weight[:] = old.weight.mean(dim=1, keepdim=True)
    resnet.conv1 = new_conv
    return resnet


def get_resnet18_encoder_random():
    resnet = models.resnet18(weights=None)
    resnet = conv1_channel1_modify(resnet)
    resnet.fc = nn.Identity()

    return resnet


def get_resnet18_encoder_imagenet():
    resnet = models.resnet18(weights=None)
    resnet = conv1_channel1_modify(resnet)
    resnet.fc = nn.Identity()

    return resnet
