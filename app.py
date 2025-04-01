from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "brain_tumor_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (modify as per your dataset)
CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Preprocess the image
    img_size = (150, 150)
    img = image.load_img(file_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100  # Convert to percentage

    return jsonify({"class": predicted_class, "confidence": confidence, "image_url": file_path})

if __name__ == "__main__":
    app.run(debug=True)
