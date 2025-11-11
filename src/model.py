import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        layers = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feat_dim = resnet.fc.in_features  # 512

    def forward(self, x):
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        return feats


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_layers=1, pretrained=True):
        super().__init__()

        # SAME AS TRAINING
        self.encoder = ResNetEncoder(pretrained=pretrained)

        self.lstm = nn.LSTM(
            self.encoder.feat_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        B, F, C, H, W = x.size()
        x = x.view(B * F, C, H, W)

        feats = self.encoder(x)
        feats = feats.view(B, F, -1)

        lstm_out, _ = self.lstm(feats)
        out = lstm_out[:, -1, :]

        logits = self.classifier(out)
        return logits


def build_model(num_classes):
    # MUST match training (pretrained=False for loading weights)
    return CNN_LSTM(num_classes, hidden_dim=256, num_layers=1, pretrained=False)
