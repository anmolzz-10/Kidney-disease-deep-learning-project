import os
import numpy as np
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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
        result = 'Tumor' if prediction > 0.5 else 'Normal'

        return render_template('result.html', result=result)

    else:
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
