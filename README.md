# 🧠 Deep Learning Projects by Atharva Patki

Welcome to my Deep Learning repository! This collection showcases a wide range of AI/ML projects I've built using Python, TensorFlow, Keras, and Scikit-learn. The goal is to solve real-world problems using intelligent models in healthcare, object detection, and image classification.

---

## 📚 What is Deep Learning?

Deep Learning is a subfield of Machine Learning inspired by the human brain. It involves training **artificial neural networks** on large datasets to recognize patterns, make predictions, and classify data. Deep learning models are especially powerful for tasks like:

- Image Classification 📸  
- Object Detection 🎯  
- Medical Diagnosis 🏥  
- Natural Language Processing 🗣️

**Key Concepts:**
- **Neural Networks**: Layers of nodes that learn from data.
- **CNN (Convolutional Neural Networks)**: Specialized for image processing.
- **Transfer Learning**: Using pretrained models like VGG16, ResNet.
- **Segmentation vs Classification**: Classifying pixels vs whole images.

---

## 🧩 Projects Included

### 1. 🔬 JSRT_UNET.py
> **Task**: Lung segmentation from chest X-rays  
> **Model**: U-Net  
> **Dataset**: JSRT (Japanese Society of Radiological Technology)  
> **Goal**: Pixel-wise classification (tumor vs background)

---

### 2. 🐶 Cats & Dogs Hybrid Classifier (`cats&dogs_hybrid.py`)
> **Task**: Binary classification of cat vs dog images  
> **Model**: Hybrid CNN using VGG16 and ResNet  
> **Highlights**: Data Augmentation + Transfer Learning

---

### 3. 🌺 Flowers Classifier (`flowers_data.py`)
> **Task**: Multi-class flower classification  
> **Model**: Simple CNN  
> **Classes**: Rose, Tulip, Sunflower, Daisy, Dandelion  
> **Libraries**: TensorFlow, Keras, Matplotlib

---

### 4. 🫁 Pneumonia Detection (`lung_pneumonia.py`)
> **Task**: Detect pneumonia from chest X-rays  
> **Model**: CNN + VGG16  
> **Dataset**: Kaggle Chest X-ray dataset  
> **Result**: Binary classifier with high accuracy

---

### 5. 🧔 Olivetti Face Classifier (`olivet_hybrid.py`)
> **Task**: Face recognition/classification  
> **Model**: PCA + XGBoost  
> **Dataset**: Olivetti Faces  
> **Pipeline**: Preprocessing → Dimensionality Reduction → Classification

---

### 6. 🏆 Trophy and Medal Detection (`trophy_and_medals.py`)
> **Task**: Object detection for sports trophies & medals  
> **Model**: YOLOv5  
> **Tools**: PyTorch, LabelImg  
> **Application**: Real-time camera-based detection


### 7. 🐾 Oxford Pet Breed Classifier (oxford_vgg_resnet_pca_xgb.py)
Task: Multi-class classification of 37 pet breeds (dogs & cats)
Models Used: VGG16 + ResNet50 (for feature extraction)
Technique: Feature Concatenation → PCA → XGBoost
Dataset: Oxford-IIIT Pet Dataset
Highlights:

Pretrained VGG16 & ResNet used for deep features

PCA applied to reduce feature dimensionality

Final classification using XGBoost
Result: Achieved high accuracy with a lightweight hybrid architecture


---

## ⚙️ Tech Stack

- **Languages**: Python  
- **Libraries**: TensorFlow, Keras, PyTorch, Scikit-learn, OpenCV  
- **Models**: CNN, U-Net, VGG16, ResNet, PCA, XGBoost, YOLOv5  
- **Tools**: Jupyter Notebook, Google Colab, Kaggle, GitHub

 
 
 
 
 ### 8. 🌦️ Weather Forecasting (weather_forecasting_Bidirectional_LSTM.py)
Task: Predict next-hour temperature using past 24 hours of weather data
Model: CNN + Bidirectional LSTM (hybrid architecture)
Dataset: Jena Climate Dataset (2009–2016)
Features: Temperature (T), Pressure (p), Air Density (ρ)
Technique: Time series windowing (24-hour sequence), Conv1D for local trend extraction, BiLSTM for temporal pattern learning
Highlights:

Hybrid deep learning model with both spatial & temporal understanding

Scaled data with MinMaxScaler for faster convergence

Early stopping to avoid overfitting

Final prediction decoded back to Celsius
Result: Smooth temperature prediction with low test loss and close actual vs predicted curve
