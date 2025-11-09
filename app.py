from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model(r"D:\interface\final_cnn_transformer_model_30_epoch.keras")

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Replace with your actual class names
class_names = ['Agro', 'Electronic', 'Garments', 'Medical', 'Packaging']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Load image
    img = image.load_img(filepath, target_size=(128, 128))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 

    # Predict
    prediction = model.predict(img_array)[0] 
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]
    confidence = prediction[class_index] * 100 

    prediction_text = f"Predicted Waste Class: {class_name} ({confidence:.2f}%)"

    return render_template('index.html', prediction_text=prediction_text, img_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
