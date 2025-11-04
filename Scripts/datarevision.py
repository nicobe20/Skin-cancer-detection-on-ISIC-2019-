import os

DATA_DIR = "/home/ds3master/college/Neural_network_works/Skin-cancer-detection-on-ISIC-2019-/Dataset-split"


for subset in ["train", "val"]:
    print(f"\n{subset.upper()}:")
    for cls in os.listdir(os.path.join(DATA_DIR, subset)):
        count = len(os.listdir(os.path.join(DATA_DIR, subset, cls)))
        print(f"  {cls}: {count}")
