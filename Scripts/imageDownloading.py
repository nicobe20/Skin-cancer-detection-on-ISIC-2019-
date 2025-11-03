import kagglehub

# Download latest version
path = kagglehub.dataset_download("salviohexia/isic-2019-skin-lesion-images-for-classification")

print("Path to dataset files:", path)