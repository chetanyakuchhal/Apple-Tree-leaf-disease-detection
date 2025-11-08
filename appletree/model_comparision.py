import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# === Paths ===
dataset_path = r'/Users/chetanya/Desktop/train'
output_dir = r'/Users/chetanya/Desktop/project'
os.makedirs(output_dir, exist_ok=True)

# === Step 1: Load Dataset ===
print("Loading dataset...")
images, labels = [], []
for class_name in tqdm(os.listdir(dataset_path)):
    class_dir = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_dir):
        continue
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        images.append(img.flatten())
        labels.append(class_name)

X = np.array(images)
y = np.array(labels)
print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

# === Step 2: Encode + Scale + PCA ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

# === Step 3: Train Models ===
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc = roc_auc_score(y_test, y_prob, multi_class='ovr') if y_prob is not None else np.nan

    results.append([name, acc, prec, rec, f1, roc])

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{name}.png'))
    plt.close()

# === Step 4: Save Results ===
df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
df.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)
print("\nSaved model comparison results to:", os.path.join(output_dir, 'model_comparison_results.csv'))

# === Step 5: Visualization ===
plt.figure(figsize=(10, 6))
bar_width = 0.15
x = np.arange(len(models))

for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score']):
    plt.bar(x + i*bar_width, df[metric], width=bar_width, label=metric)

plt.xticks(x + bar_width*1.5, df['Model'])
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'))
plt.close()

print("\n✅ All plots and results saved in:", output_dir)
