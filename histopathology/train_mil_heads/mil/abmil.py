import torch
import torch.nn.functional as F
from torch import nn


class ABMIL(nn.Module):
    """Attention-based multiple-instance learning head used for slide-level prediction."""
    def __init__(self, in_dim, hidden_dim, num_classes):
        """Initialise projection, gated attention scoring, and slide classifier layers."""
        super(ABMIL, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU()
        )
        self.attention = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        """Aggregate a bag of tile embeddings with learned attention and return slide logits."""
        b, bag_size, d = x.shape

        x = x.view(b * bag_size, d)
        x = self.projection(x)

        att = self.attention(x)
        att = att.view(b, bag_size, -1)
        attention_weights = F.softmax(att, dim=1)

        x = x.view(b, bag_size, -1)
        x = torch.sum(attention_weights * x, dim=1)
        x = self.classifier(x)
        return x, attention_weights
