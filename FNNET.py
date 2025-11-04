#For CNN and finetuning
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
#from sklearn.metrics import classification_report, confusion_matrix for the confusion matrix
#Checkpoints and class weights(the data split is very VERY unbalanced the majority of images are of NV being 10,300)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
#General use
import os 
import numpy as np


# Load base model (pretrained on ImageNet) this is gonna change
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
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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

#Adding class weights as image data set is imbalance

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("best_skin_cancer_model.h5", save_best_only=True)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)


# Unfreeze the top N layers
fine_tune_at = 150
for layer in base_model.layers[-fine_tune_at:]:
    layer.trainable = True


model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks

)

model.save("skin_cancer_finetuned.h5")
print(" Fine-tuned model saved!")
