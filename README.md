# Assignment-1 : Logistic Regression
# Dataset: data.csv (with columns: SNo, X_1, X_2, y)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("data.csv")

# Rename target column (remove trailing space in 'y ')
df = df.rename(columns={"y ": "y"})

# Features and target
X = df[["X_1", "X_2"]]
y = df["y"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# --- Plot 1: Confusion Matrix ---
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --- Plot 2: Decision Boundary ---
plt.figure(figsize=(6, 5))
x_min, x_max = X["X_1"].min() - 1, X["X_1"].max() + 1
y_min, y_max = X["X_2"].min() - 1, X["X_2"].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X_test["X_1"], X_test["X_2"], c=y_test,
            cmap=plt.cm.Paired, edgecolors="k")
plt.xlabel("X_1")
plt.ylabel("X_2")
plt.title("Logistic Regression Decision Boundary")
plt.show()
