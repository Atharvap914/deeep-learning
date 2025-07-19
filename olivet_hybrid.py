

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.datasets import fetch_olivetti_faces


data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = data.data
y = (data.target < 20).astype(int)  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


cumvar = np.cumsum(PCA().fit(X_train_scaled).explained_variance_ratio_)
n_comp = np.argmax(cumvar >= 0.95) + 1


pca = PCA(n_components=n_comp, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


model = XGBClassifier(
    random_state=42,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss'
)
model.fit(X_train_pca, y_train)


cv_scores = cross_val_score(model, X_train_pca, y_train, cv=5)
print(f"CV scores: {cv_scores}\nMean CV score: {cv_scores.mean():.4f}")


test_accuracy = model.score(X_test_pca, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
