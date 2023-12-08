from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = load_model('model.h5')


class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']


# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']

    # Check if the file has an allowed extension
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file format'})

    # Read file content directly into memory
    img_content = file.read()

    # Preprocess the image
    img_array = preprocess_image_from_memory(img_content)

    # Make prediction
    prediction = model.predict(img_array)

    # Get the predicted class label
    predicted_class = class_labels[np.argmax(prediction)]

    # Return the result
    result = {'class': predicted_class, 'confidence': float(np.max(prediction))}
    return jsonify(result)

# Function to preprocess the image from in-memory content
def preprocess_image_from_memory(img_content):
    img = image.load_img(io.BytesIO(img_content), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# return index.html on GET /
@app.route('/')
def home():
   return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
