import os
import tensorflow as tf


DATA_DIR = "/home/ds3master/college/Neural_network_works/Skin-cancer-detection-on-ISIC-2019-/Dataset-split"

'''
for subset in ["train", "val"]:
    print(f"\n{subset.upper()}:")
    for cls in os.listdir(os.path.join(DATA_DIR, subset)):
        count = len(os.listdir(os.path.join(DATA_DIR, subset, cls)))
        print(f"  {cls}: {count}")
'''


print("GPUs available:", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Simple test operation
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
print("Matrix multiplication done on GPU HELL YEAH BBY")
