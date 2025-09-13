from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import cv2
import numpy as np
import pandas as pd

# This is the crucial line the server is looking for
app = Flask(__name__)
CORS(app)

model = joblib.load('plastic_classifier_model.pkl')

def extract_features(image):
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'num_contours': 0, 'total_area': 0.0, 'avg_area': 0.0, 'max_area': 0.0, 'avg_circularity': 0.0}
    num_contours = len(contours)
    areas = [cv2.contourArea(c) for c in contours]
    total_area = sum(areas)
    avg_area = total_area / num_contours
    max_area = max(areas)
    perimeters = [cv2.arcLength(c, True) for c in contours]
    circularities = [(4 * np.pi * area) / (perim**2) for area, perim in zip(areas, perimeters) if perim > 0]
    avg_circularity = np.mean(circularities) if circularities else 0
    return {'num_contours': num_contours, 'total_area': total_area, 'avg_area': avg_area, 'max_area': max_area, 'avg_circularity': avg_circularity}

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    features = extract_features(image)
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)
    probability = model.predict_proba(features_df)
    label = 'Plastic' if prediction[0] == 1 else 'Non-Plastic'
    confidence = float(np.max(probability))
    return jsonify({'prediction': label, 'confidence': f"{confidence*100:.2f}%"})

@app.route('/')
def index():
    return "Model server is running!"

# This block is not used by Gunicorn but is good for local testing
if __name__ == '__main__':
    app.run(debug=False)
