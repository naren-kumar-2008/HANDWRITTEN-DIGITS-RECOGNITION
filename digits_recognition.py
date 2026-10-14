"""Handwritten Digits Recognition - Mini Project
Dataset : scikit-learn 'digits' (1797 hand-written digit images, 8 x 8 pixels, classes 0-9)
Models  : K-Nearest Neighbours, Support Vector Machine, Random Forest, Neural Network (MLP)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. Load the data
digits = load_digits()
X, y = digits.data, digits.target
print("Dataset shape :", digits.images.shape)
print("Features per image :", X.shape[1])
print("Classes :", np.unique(y))

# 2. Show some sample images
fig, axes = plt.subplots(2, 5, figsize=(8, 3.6))
for ax, img, label in zip(axes.ravel(), digits.images, digits.target):
    ax.imshow(img, cmap="gray_r"); ax.set_title("Digit: %d" % label); ax.axis("off")
plt.tight_layout(); plt.savefig("out/fig_samples.png", dpi=150); plt.close()

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training samples :", len(X_train), " Testing samples :", len(X_test))

# 4. Train and compare models
models = {
    "K-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Neural Network (MLP)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)),
}
scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    scores[name] = accuracy_score(y_test, model.predict(X_test))
    print("%-24s accuracy = %.4f" % (name, scores[name]))

best_name = max(scores, key=scores.get)
best = models[best_name]
print("\nBest model :", best_name)

# 5. Evaluate the best model
y_pred = best.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=2))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix - " + best_name)
plt.tight_layout(); plt.savefig("out/fig_confusion.png", dpi=150); plt.close()

# 6. Show predictions on test images
idx = np.arange(10)
fig, axes = plt.subplots(2, 5, figsize=(8, 3.8))
for ax, i in zip(axes.ravel(), idx):
    ax.imshow(X_test[i].reshape(8, 8), cmap="gray_r")
    ok = y_pred[i] == y_test[i]
    ax.set_title("Pred: %d (True: %d)" % (y_pred[i], y_test[i]), color="green" if ok else "red", fontsize=9)
    ax.axis("off")
plt.tight_layout(); plt.savefig("out/fig_predictions.png", dpi=150); plt.close()

# 7. Model comparison chart
plt.figure(figsize=(7, 3.6))
plt.barh(list(scores.keys()), [v * 100 for v in scores.values()], color="#2f5d8a")
plt.xlim(90, 100); plt.xlabel("Test accuracy (%)"); plt.title("Model comparison")
for i, v in enumerate(scores.values()): plt.text(v * 100 + 0.1, i, "%.2f%%" % (v * 100), va="center")
plt.tight_layout(); plt.savefig("out/fig_compare.png", dpi=150); plt.close()
