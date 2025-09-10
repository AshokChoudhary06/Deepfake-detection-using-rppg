from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import joblib
from ask_feature import final_rppg as fe
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/upload'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model('models/bi-lstm_97_2.keras')
scaler = joblib.load('models/scaler7.pkl')
max_length = 59  # as per your model

@app.route('/')
def home():
    return render_template("frontend.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['video']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Extract features using your rPPG-based pipeline
    bvp = fe.VideoProcessor(filepath)
    bvp.process()
    se = fe.FeatureEngine(bvp.filtered_patch_bvps)
    be = se.create_features()
    feature = np.array(be)
    max_length = 59
    padded = pad_sequences([feature], maxlen=max_length, padding='post', dtype='float32')
    reshaped = padded.reshape(-1, 12)
    scaled = scaler.transform(reshaped).reshape(1, max_length, 12)

    # Predict
    prediction = model.predict(scaled)[0][0]
    label = "FAKE" if prediction > 0.5 else "REAL"
    confidence = round(prediction * 100 if label == "FAKE" else (1 - prediction) * 100, 2)

    return jsonify({'label': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
