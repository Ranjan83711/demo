import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import ImageFile

# Allow loading truncated / partially corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------------
# Config
# ----------------------------
IMG_SIZE = 224
BATCH_SIZE = 16

# ----------------------------
# Transforms
# ----------------------------
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ----------------------------
# Safe Dataset Loader
# ----------------------------
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]

        try:
            sample = self.loader(path)
        except Exception:
            print("⚠️ Skipping corrupted image:", path)
            # Skip to next valid image
            return self.__getitem__((index + 1) % len(self.samples))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


# ----------------------------
# Load Dataset Function
# ----------------------------
def load_dataset(data_path):

    train_dir = os.path.join(data_path, "train")
    test_dir = os.path.join(data_path, "test")

    train_dataset = SafeImageFolder(train_dir, transform=transform_train)
    test_dataset = SafeImageFolder(test_dir, transform=transform_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader, train_dataset.classes