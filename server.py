from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request 
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os


app = Flask(__name__)

def preprocess_image(image_path):
  img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = np.expand_dims(img, axis=0)

  img /= 255.0
  print(img.shape)
  return img

@app.route('/predict', methods=['POST'])
def predict():
  print(request.files)
  if 'image' not in request.files:
    return jsonify({'error': 'No file part'})

  file = request.files['image']
  if file.filename == '':
    return jsonify({'error':  'No selected file'})

  if file:
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

   
    processed_image = preprocess_image(filepath)
    prediction = model.predict(processed_image)

    predicted_class = np.argmax(prediction)
    if predicted_class == 0:
      predic = "Brown Spot"
    elif predicted_class == 1:
      predic = "Healthy"
    elif predicted_class == 2:
      predic = "White Scale"
    return jsonify({'predicted_class': predic})

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'src/uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    model = tf.keras.models.load_model('src/model/model.h5')
    if os.getenv("ENVIRONMENT") == 'dev':
      app.run(debug=True)
    elif os.getenv("ENVIRONMENT") == 'prod':
      app.run()
    else:
      print("environment not available")