import torch.nn as nn
import torchvision.models as models

def get_model(num_classes):
    model = models.resnet18(weights="IMAGENET1K_V1")

    # convert to grayscale compatible
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model