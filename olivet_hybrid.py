
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier
from sklearn.datasets import fetch_olivetti_faces

class HybridPCA_XGBoostModel:
    def __init__(self, variance_threshold=0.95):
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = None
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_olivetti_data(self):
        data = fetch_olivetti_faces(shuffle=True, random_state=42)
        X = data.data
        y = (data.target < 20).astype(int)  # binary classification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def _determine_components(self, X):
        temp_pca = PCA()
        temp_pca.fit(X)
        cumvar = np.cumsum(temp_pca.explained_variance_ratio_)
        n_comp = np.argmax(cumvar >= self.variance_threshold) + 1
        return min(n_comp, X.shape[1])

    def fit(self, X_train=None, y_train=None):
        if X_train is None or y_train is None:
            if self.X_train is None:
                print("Loading data first.")
                self.load_olivetti_data()
            X_train, y_train = self.X_train, self.y_train

        print("\nFitting PCA-XGBoost hybrid model.")
        X_scaled = self.scaler.fit_transform(X_train)
        print(f"Features standardized: mean={X_scaled.mean():.3f}, std={X_scaled.std():.3f}")

        n_comp = self._determine_components(X_scaled)
        self.pca = PCA(n_components=n_comp, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"PCA components selected: {n_comp}, explained variance: {self.pca.explained_variance_ratio_.sum():.6f}")

        self.model = XGBClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss'
        )
        self.model.fit(X_pca, y_train)

        scores = cross_val_score(self.model, X_pca, y_train, cv=5)
        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV score: {scores.mean():.4f}")

    def evaluate(self):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not found.")
        X_test_scaled = self.scaler.transform(self.X_test)
        X_test_pca = self.pca.transform(X_test_scaled)
        accuracy = self.model.score(X_test_pca, self.y_test)
        print(f"Final Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    hybrid_model = HybridPCA_XGBoostModel()
    hybrid_model.fit()
    hybrid_model.evaluate()
