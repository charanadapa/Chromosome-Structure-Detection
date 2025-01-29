from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("C:/Users/HOME/Desktop/model.keras")

# Image dimensions
img_height = 128
img_width = 128

def predict_image(img_path):
    class_names = ['Abnormal', 'Normal']
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims to fit the model input
    predictions = model.predict(img_array)
    return class_names[np.argmax(predictions[0])]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Save the uploaded image to a temporary location
    img_path = "uploaded_image.jpg"
    file.save(img_path)

    # Make prediction
    prediction = predict_image(img_path)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
