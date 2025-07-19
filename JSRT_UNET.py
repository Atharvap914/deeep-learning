


import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model


#


image_dir = r"C:\Users\ath\Desktop\converted\train\images"
mask_dir = r"C:\Users\ath\Desktop\converted\train\masks"
img_size = (256, 256)

images = []
masks = []

for filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    if os.path.exists(mask_path):
        img = load_img(img_path, target_size=img_size, color_mode="grayscale")
        img = img_to_array(img) / 255.0

        mask = load_img(mask_path, target_size=img_size, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0

        images.append(img)
        masks.append(mask)

X = np.array(images)
Y = np.array(masks)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)





inputs = layers.Input(shape=(256, 256, 1))

# Encoder
c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
p1 = layers.MaxPooling2D((2, 2))(c1)

c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
p2 = layers.MaxPooling2D((2, 2))(c2)

# Bottleneck
c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)

# Decoder
u1 = layers.UpSampling2D((2, 2))(c3)
u1 = layers.Concatenate()([u1, c2])
c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c4)

u2 = layers.UpSampling2D((2, 2))(c4)
u2 = layers.Concatenate()([u2, c1])
c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u2)
c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c5)

outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()





model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=21, batch_size=8)





test_image_dir = r"C:\Users\ath\Desktop\converted\test\images"
test_mask_dir = r"C:\Users\ath\Desktop\converted\test\masks"

test_images = []
test_masks = []

for filename in os.listdir(test_image_dir):
    img_path = os.path.join(test_image_dir, filename)
    mask_path = os.path.join(test_mask_dir, filename)

    if os.path.exists(mask_path):
        img = load_img(img_path, target_size=img_size, color_mode="grayscale")
        img = img_to_array(img) / 255.0

        mask = load_img(mask_path, target_size=img_size, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0

        test_images.append(img)
        test_masks.append(mask)

X_test = np.array(test_images)
Y_test = np.array(test_masks)





preds = model.predict(X_test)
preds = (preds > 0.5).astype(np.uint8)  # Binarize predicted masks





import matplotlib.pyplot as plt

n = 5  # Number of samples to show

for i in range(n):
    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.title('Test Image')
    plt.axis('off')

    
    plt.subplot(1, 3, 2)
    plt.imshow(Y_test[i].squeeze(), cmap='gray')
    plt.title('True Mask')
    plt.axis('off')

    # Predicted Mask
    plt.subplot(1, 3, 3)
    plt.imshow(preds[i].squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()







