import os
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# --- 1. SETTINGS ---
# Path to your training dataset
dataset_path = r'/Users/chetanya/Desktop/train'
# Path where all models will be saved
output_dir = r'/Users/chetanya/Desktop/appletree'

image_size = (32, 32)  # Must match the app
n_pca_components = 0.95 # Retain 95% of variance

# --- 2. FEATURE EXTRACTION (MUST MATCH APP.PY) ---
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, image_size)
    
    # Histogram features (R,G,B) - 32 bins each
    hist_red = cv2.calcHist([image], [0], None, [32], [0, 256]).flatten()
    hist_green = cv2.calcHist([image], [1], None, [32], [0, 256]).flatten()
    hist_blue = cv2.calcHist([image], [2], None, [32], [0, 256]).flatten()
    
    # Total features = 32 + 32 + 32 = 96
    return np.concatenate([hist_red, hist_green, hist_blue])

# --- 3. LOAD DATASET ---
features, labels = [], []
print("Starting feature extraction...")
for class_dir in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_dir)
    if not os.path.isdir(class_path):
        continue
    
    for idx, image_name in enumerate(os.listdir(class_path)):
        image_path = os.path.join(class_path, image_name)
        feat = extract_features(image_path)
        if feat is not None:
            features.append(feat)
            labels.append(class_dir)
            
        if idx % 500 == 0:
            print(f"Processed {idx} images in class {class_dir}")

features = np.array(features)
labels = np.array(labels)
print(f"Total samples: {len(labels)}, Feature shape: {features.shape}")

# --- 4. ENCODING AND SCALING ---
le = LabelEncoder()
y = le.fit_transform(labels)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features)
print(f"Data scaled. Shape: {X_scaled.shape}")

# --- 5. PCA (Dimensionality Reduction) ---
print("Fitting PCA...")
pca = PCA(n_components=n_pca_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA complete. New shape: {X_pca.shape}")

# --- 6. ANOMALY DETECTOR ---
# Train on the unbalanced, PCA-transformed data to learn what "normal" data is
print("Fitting Anomaly Detector (IsolationForest)...")
anomaly_detector = IsolationForest(contamination=0.15, random_state=42)
anomaly_detector.fit(X_pca)
print("Anomaly Detector trained.")

# --- 7. BALANCING (on PCA-transformed data) ---
print("Balancing data with SMOTE...")
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_pca, y)
print(f"Balanced samples: {len(y_bal)}, Balanced shape: {X_bal.shape}")

# --- 8. TRAIN-TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42
)

# --- 9. MODEL TRAINING (on PCA-balanced data) ---
print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)
print("Model trained.")

# --- 10. EVALUATION ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- 11. SAVE ALL COMPONENTS ---
print(f"Saving all components to: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

joblib.dump(model, os.path.join(output_dir, 'apple_disease_model.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
joblib.dump(le, os.path.join(output_dir, 'label_encoder.pkl'))
joblib.dump(pca, os.path.join(output_dir, 'pca.pkl'))
joblib.dump(anomaly_detector, os.path.join(output_dir, 'anomaly_detector.pkl'))

print(f"\n✅ Training complete. All components saved to '{output_dir}'.")