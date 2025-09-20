# utils.py
import torch
from torchvision import transforms
from PIL import Image
import io

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# Mean/std for CIFAR-10 (channel order RGB)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

def preprocess_image_bytes(image_bytes):
    # input: bytes (e.g. uploaded file.read())
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize to 32x32 to match CIFAR (if user uploaded larger image)
    img = img.resize((32, 32), Image.Resampling.LANCZOS)
    tensor = val_transform(img).unsqueeze(0)  # shape 1x3x32x32
    return tensor
