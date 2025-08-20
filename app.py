import streamlit as st
import joblib
import numpy as np
from collections import Counter


class SimpleDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.label = None

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split:
            self.label = Counter(y).most_common(1)[0][0]
            return

        n_samples, n_features = X.shape
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False) 

        best_gain = -1
        split_idx, split_thresh = None, None

        for feat in feat_idxs:
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_idx = X[:, feat] <= t
                right_idx = X[:, feat] > t

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                gain = self._information_gain(y, y[left_idx], y[right_idx])
                if gain > best_gain:
                    best_gain = gain
                    split_idx, split_thresh = feat, t

        if best_gain == -1:  # no split found -> leaf
            self.label = Counter(y).most_common(1)[0][0]
            return

        self.feature_index = split_idx
        self.threshold = split_thresh
        left_idx = X[:, split_idx] <= split_thresh
        right_idx = X[:, split_idx] > split_thresh

        self.left = SimpleDecisionTree(max_depth=self.max_depth,
                                       min_samples_split=self.min_samples_split,
                                       n_features=self.n_features)
        self.left.fit(X[left_idx], y[left_idx], depth+1)

        self.right = SimpleDecisionTree(max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split,
                                        n_features=self.n_features)
        self.right.fit(X[right_idx], y[right_idx], depth+1)

    def _gini(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs**2)

    def _information_gain(self, parent, l_child, r_child):
        w_left = len(l_child) / len(parent)
        w_right = 1 - w_left
        return self._gini(parent) - (w_left * self._gini(l_child) + w_right * self._gini(r_child))

    def predict(self, x):
        if self.label is not None:
            return self.label
        if x[self.feature_index] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)


class RandomForestScratch:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.trees = []
        n_features_sub = int(np.sqrt(n_features))  

        for _ in range(self.n_estimators):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_s, y_s = X[idxs], y[idxs]

            tree = SimpleDecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=n_features_sub)
            tree.fit(X_s, y_s)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([[tree.predict(x) for tree in self.trees] for x in X])
        y_pred = [Counter(tree_preds[i]).most_common(1)[0][0] for i in range(len(X))]
        return np.array(y_pred)


scaler = joblib.load("scaler.pkl")
rf_model = joblib.load("RandomForest.pkl")


class_mapping = {0: "Kama Wheat", 1: "Rosa Wheat", 2: "Canadian Wheat"}


st.markdown("<h1 style='text-align: center;'>🌾 Wheat Variety Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict Wheat Varieties using Random Forest (From Scratch)</h4>", unsafe_allow_html=True)
st.write("#### 👉 Enter the wheat seed features below:")


area = st.slider("Area", 10.59, 21.18, 14.85)
perimeter = st.slider("Perimeter", 12.41, 17.25, 14.56)
compactness = st.slider("Compactness", 0.8081, 0.9183, 0.8710)
kernel_length = st.slider("Kernel Length", 4.899, 6.675, 5.6285)
kernel_width = st.slider("Kernel Width", 2.63, 4.033, 3.259)
asymmetry = st.slider("Asymmetry", 0.7651, 8.456, 3.7002)
groove_length = st.slider("Groove Length", 4.519, 6.55, 5.408)


if st.button("🔍 Predict Wheat Variety"):

    features = np.array([[area, perimeter, compactness, kernel_length, kernel_width, asymmetry, groove_length]])
    features_scaled = scaler.transform(features)

    
    prediction = rf_model.predict(features_scaled)[0]
    variety = class_mapping[prediction]

    
    st.success(f"✅ The predicted wheat variety is **{variety}**.")
    st.info(
        f"- **Kama Wheat**: Commonly grown in temperate regions, medium-sized grains.\n"
        f"- **Rosa Wheat**: Known for its reddish hue and high protein content.\n"
        f"- **Canadian Wheat**: Hard wheat with strong gluten, widely used in bread-making."
    )
