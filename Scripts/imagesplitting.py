import os
import random
import shutil
from tqdm import tqdm

data_dir = "/home/ds3master/college/Neural_network_works/Skin-cancer-detection-on-ISIC-2019-/images"
output_dir = "/home/ds3master/college/Neural_network_works/Skin-cancer-detection-on-ISIC-2019-/Dataset-split"

train_ratio = 0.8

os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)

classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

for cls in classes:
    src_dir = os.path.join(data_dir, cls)
    images = [f for f in os.listdir(src_dir) if f.endswith(".jpg")]
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    os.makedirs(os.path.join(output_dir, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", cls), exist_ok=True)

    for img in tqdm(train_imgs, desc=f"Train {cls}"):
        shutil.copy(os.path.join(src_dir, img), os.path.join(output_dir, "train", cls))

    for img in tqdm(val_imgs, desc=f"Val {cls}"):
        shutil.copy(os.path.join(src_dir, img), os.path.join(output_dir, "val", cls))

print("This shit worked!")
