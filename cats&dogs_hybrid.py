# cats_dogs_hybrid_flat.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import image_dataset_from_directory, load_img, img_to_array

# Load dataset
train_dir = "/content/train"
test_dir = "/content/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

train_ds = image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
test_ds = image_dataset_from_directory(test_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
class_names = train_ds.class_names

# Load ResNet50 model without top layers
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

def extract_features(dataset):
    features, labels = [], []
    for imgs, lbls in dataset:
        imgs = preprocess_input(imgs)
        feats = base_model.predict(imgs)
        features.append(feats)
        labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

# Extract features
X_train, y_train = extract_features(train_ds)
X_test, y_test = extract_features(test_ds)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=99)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train XGBoost classifier
xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y_train)),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    n_estimators=250,
    max_depth=8,
    learning_rate=0.1
)
xgb_model.fit(X_train_pca, y_train)

# Evaluate
y_pred = xgb_model.predict(X_test_pca)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict on new image
img = load_img("/content/images (1).jpeg", target_size=IMG_SIZE)
img_array = img_to_array(img)
img_array = preprocess_input(img_array)
img_array = img_array.reshape(1, 224, 224, 3)

img_feat = base_model.predict(img_array)
img_feat_scaled = scaler.transform(img_feat)
img_feat_pca = pca.transform(img_feat_scaled)

prediction = xgb_model.predict(img_feat_pca)[0]
print("Predicted class:", class_names[prediction])
