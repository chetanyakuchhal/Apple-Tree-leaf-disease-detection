from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import joblib
from werkzeug.utils import secure_filename
import tempfile # Use tempfile for saving uploads

app = Flask(__name__) 

# --- 1. SETTINGS ---
# This MUST be the same (32, 32) as in the training script
image_size = (32, 32)
# This MUST point to the folder from your training script
MODEL_DIR = r'/Users/chetanya/Desktop/appletree'

# --- 2. LOAD ALL MODELS ---
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'apple_disease_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    pca = joblib.load(os.path.join(MODEL_DIR, 'pca.pkl'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    anomaly_detector = joblib.load(os.path.join(MODEL_DIR, 'anomaly_detector.pkl'))
    print("✅ All models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models from {MODEL_DIR}: {e}")
    print("--- PLEASE CHECK THE PATH AND ENSURE ALL 5 .pkl FILES EXIST ---")
    model, scaler, pca, label_encoder, anomaly_detector = (None,)*5

# --- 3. FEATURE EXTRACTION (MUST BE *IDENTICAL* TO TRAINING SCRIPT) ---
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

# --- 4. TREATMENT RECOMMENDATIONS ---
treatments = {
    'Apple___Apple_scab': {
        'description': "Fungal disease causing olive-green to black spots on leaves and fruit",
        'treatment': [
            "Chemical: Apply sulfur, myclobutanil (Rally), or fenarimol (Rubigan) fungicides",
            "Organic: Remove fallen leaves, prune for air circulation, use sulfur sprays",
            "Timing: Treat from green tip through first cover spray period",
            "Resistant varieties: 'Liberty', 'Freedom'"
        ],
        'prevention': [
            "Rake and destroy fallen leaves in autumn",
            "Maintain proper tree spacing",
            "Avoid overhead irrigation",
            "Apply dormant sprays in late winter"
        ]
    },
    'Apple___Black_rot': {
        'description': "Fungal disease causing fruit rot and leaf spots",
        'treatment': [
            "Chemical: Use captan, thiophanate-methyl, or mancozeb fungicides",
            "Organic: Remove mummified fruit, prune cankers, use copper sprays",
            "Timing: Begin at petal fall, continue every 10-14 days in wet weather",
            "Critical: Remove infected material immediately"
        ],
        'prevention': [
            "Prune out dead wood 12 inches below cankers",
            "Disinfect pruning tools between cuts",
            "Avoid fruit wounding during handling",
            "Store fruit at 32°F"
        ]
    },
    'Apple___Cedar_apple_rust': {
        'description': "Fungal disease requiring both apple and cedar hosts",
        'treatment': [
            "Chemical: Apply myclobutanil, fenarimol, or trifloxystrobin fungicides",
            "Organic: Remove nearby junipers, use sulfur or copper sprays",
            "Timing: Start at pink bud stage, continue every 7-10 days until petal fall",
            "Resistant varieties: 'Redfree', 'Liberty'"
        ],
        'prevention': [
            "Eliminate junipers within 2 miles if possible",
            "Rake and destroy fallen leaves",
            "Apply dormant oil sprays",
            "Select resistant varieties"
        ]
    },
    'Healthy': {
        'description': "No signs of disease detected",
        'treatment': [
            "Continue regular monitoring",
            "Maintain proper fertilization and irrigation",
            "Prune annually for air circulation",
            "Consider preventive fungicides if disease pressure is high"
        ],
        'prevention': []
    }
}

# --- 5. FLASK ROUTES ---
@app.route('/')
def home():
    # You must have an 'index.html' in a 'templates' folder
    # in the same directory as apple.py
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Check if models loaded correctly
    if not all([model, scaler, pca, label_encoder, anomaly_detector]):
         return jsonify({'error': 'Models are not loaded. Check server logs.'})

    if file:
        filename = secure_filename(file.filename)
        
        # Use a temporary file to save the upload
        temp_dir = tempfile.gettempdir()
        temp_filepath = os.path.join(temp_dir, filename)
        
        try:
            file.save(temp_filepath)
            
            # 1. Extract features (using the *correct* 96-feature function)
            features = extract_features(temp_filepath)
            
            if features is None:
                return jsonify({'error': 'Could not process image'})
                
            # 2. Preprocess features (Scale, then PCA)
            # Input must be 2D, so we use [features]
            features_scaled = scaler.transform([features])
            features_pca = pca.transform(features_scaled)

            # 3. Check for anomalies *first*
            is_anomaly = anomaly_detector.predict(features_pca)[0] == -1

            if is_anomaly:
                return jsonify({
                    'is_anomaly': True,
                    'message': "Unknown Image - This doesn't appear to be an apple leaf or shows symptoms not recognized by our system",
                    'recommendations': [
                        "Verify the image is of an apple leaf",
                        "Check for image quality issues",
                        "Consult with a plant pathologist",
                        "Compare with known apple disease references"
                    ]
                })

            # 4. If not anomaly, proceed with classification
            pred = model.predict(features_pca)[0]
            disease = label_encoder.inverse_transform([pred])[0]

            disease_info = treatments.get(disease, {
                'description': f"Unknown disease category: {disease}",
                'treatment': ["Consult local agricultural extension"],
                'prevention': ["Maintain good plant health"]
            })

            return jsonify({
                'is_anomaly': False,
                'disease': disease,
                'description': disease_info['description'],
                'treatment': disease_info['treatment'],
                'prevention': disease_info['prevention']
            })

        except Exception as e:
            return jsonify({'error': f'An error occurred: {str(e)}'})
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

# --- 6. RUN THE APP ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5007))
    app.run(host='0.0.0.0', port=port, debug=False) # Debug=False is for production