{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ae19cdb6-b26c-4a9c-b322-a5451364c564",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "from flask_cors import CORS\n",
    "import joblib\n",
    "import cv2\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "app = Flask(__name__)\n",
    "CORS(app)\n",
    "\n",
    "model = joblib.load('plastic_classifier_model.pkl')\n",
    "\n",
    "def extract_features(image):\n",
    "    if len(image.shape) > 2 and image.shape[2] > 1:\n",
    "        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n",
    "    else:\n",
    "        gray_image = image\n",
    "    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n",
    "    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
    "    if not contours:\n",
    "        return {'num_contours': 0, 'total_area': 0.0, 'avg_area': 0.0, 'max_area': 0.0, 'avg_circularity': 0.0}\n",
    "    num_contours = len(contours)\n",
    "    areas = [cv2.contourArea(c) for c in contours]\n",
    "    total_area = sum(areas)\n",
    "    avg_area = total_area / num_contours\n",
    "    max_area = max(areas)\n",
    "    perimeters = [cv2.arcLength(c, True) for c in contours]\n",
    "    circularities = [(4 * np.pi * area) / (perim**2) for area, perim in zip(areas, perimeters) if perim > 0]\n",
    "    avg_circularity = np.mean(circularities) if circularities else 0\n",
    "    return {'num_contours': num_contours, 'total_area': total_area, 'avg_area': avg_area, 'max_area': max_area, 'avg_circularity': avg_circularity}\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    file = request.files['file']\n",
    "    filestr = file.read()\n",
    "    npimg = np.frombuffer(filestr, np.uint8)\n",
    "    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)\n",
    "    features = extract_features(image)\n",
    "    features_df = pd.DataFrame([features])\n",
    "    prediction = model.predict(features_df)\n",
    "    probability = model.predict_proba(features_df)\n",
    "    label = 'Plastic' if prediction[0] == 1 else 'Non-Plastic'\n",
    "    confidence = float(np.max(probability))\n",
    "    return jsonify({'prediction': label, 'confidence': f\"{confidence*100:.2f}%\"})\n",
    "\n",
    "@app.route('/')\n",
    "def index():\n",
    "    return \"Model server is running!\"\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "07b58799-405d-43c8-9c79-742155ea7fe7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
