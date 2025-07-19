# lung_pneumonia.py

import os
import time
import zipfile
from PIL import Image
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam


zip_file_path = '/content/drive/My Drive/chest_xray.zip'
extracted_path = '/tmp/chest_xray_extracted/chest_xray'
os.makedirs(extracted_path, exist_ok=True)

if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('/tmp/chest_xray_extracted')
else:
    raise FileNotFoundError(f"Zip file not found at: {zip_file_path}")


train_dir = os.path.join(extracted_path, 'train')
val_dir = os.path.join(extracted_path, 'val')
test_dir = os.path.join(extracted_path, 'test')


IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)


train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

val_test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
val_data = val_test_gen.flow_from_directory(val_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
test_data = val_test_gen.flow_from_directory(test_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary')


input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
vgg_base.trainable = False

inputs = Input(shape=input_shape)
x = vgg_base(inputs, training=False)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


EPOCHS = 3
start = time.time()

history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_data,
    validation_steps=val_data.samples // BATCH_SIZE)

print(f"Training completed in {time.time() - start:.2f} seconds.")


test_loss, test_acc = model.evaluate(test_data, steps=test_data.samples // BATCH_SIZE)
print(f"Test Accuracy: {test_acc:.4f}")


model.save('multi_hybrid_model.h5')
print("Model saved as multi_hybrid_model.h5")
