import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()

        # Load pretrained ResNet18
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        # Heads for multitask classification
        self.fc_skin_type = nn.Linear(512, 3)  # dry, normal, oily
        self.fc_features = nn.Linear(512, 5)   # acne, pimples, wrinkles, pigmentation, dark_circles

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.fc_skin_type(x), self.fc_features(x)
