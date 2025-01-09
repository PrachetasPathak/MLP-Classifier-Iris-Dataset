import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
iris = load_iris()
X, y = iris.data[:, :2], iris.target  # First two features only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features Makes sure traing nd test data are on the same scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

# Print accuracy and classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = mlp.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)

# Plot decision boundaries and confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
scatter = axes[0].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=40, cmap='coolwarm')
axes[0].set_title("MLP Decision Boundaries")
axes[0].legend(*scatter.legend_elements(), title="Classes")

# Confusion matrix heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names, ax=axes[1])
axes[1].set_title("Confusion Matrix")
plt.tight_layout()
plt.show()
