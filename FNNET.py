from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os 


# Load base model (pretrained on ImageNet)
base_model = EfficientNetB0(
    weights='imagenet',        
    include_top=False,         
    input_shape=(224, 224, 3)
)

#Guys same as before i am telling which image folder it should access, this is the image with the dataset split if you need to split your dataset please use the SCRIPT
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
DATA_DIR = "/home/ds3master/college/Neural_network_works/Skin-cancer-detection-on-ISIC-2019-/Dataset-split"


#Train generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Freeze the convolutional base (don’t train these layers yet)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Unfreeze the top N layers
fine_tune_at = 100  # you can experiment with this number
for layer in base_model.layers[-fine_tune_at:]:
    layer.trainable = True

# Lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
model.save("skin_cancer_finetuned.h5")
print(" Fine-tuned model saved!")
