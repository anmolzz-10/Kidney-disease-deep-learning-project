from flask import Flask, render_template, request, redirect, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os 
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('kidney_model.h5')

# Define allowed extensions for file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)  # Save file to uploads folder
        file.save(filepath)

        # Preprocess the image
        img = load_img(filepath, target_size=(150, 150))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        # Make prediction
        prediction = model.predict(img)
        class_label = 'Tumor' if prediction > 0.5 else 'Normal'
        accuracy = float(prediction) * 100 if class_label == 'Tumor' else float(1 - prediction) * 100

        return render_template('result.html', result={'class_label': class_label, 'accuracy': accuracy})

    else:
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
