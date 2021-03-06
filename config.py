import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# If it is to run on colab needs to change the location according to the dataset
TRAIN_DIR = "drive/MyDrive/pix2pix/dataset/maps/train"
VAL_DIR = "drive/MyDrive/pix2pix/dataset/maps/val"
EVALUATION_DIR = "pix_2_pix_using_c_gan/evaluation"
LEARNING_RATE = 2e-4 # same as the paper
BATCH_SIZE = 16 # Batch size of 16 in the paper 1
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "drive/MyDrive/pix2pix/pretrained_weight/disc.pth.tar"
CHECKPOINT_GEN = "drive/MyDrive/pix2pix/pretrained_weight/gen.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2), # use a color jitter in 20% of the case
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)