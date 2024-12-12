import os, traceback
from dotenv import load_dotenv
load_dotenv(override=True)

from flask import Flask, jsonify, request 
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename


app = Flask(__name__)
model = tf.keras.models.load_model('src/model/model.keras')
os.makedirs('src/uploads', exist_ok=True)
def preprocess_image(image_path):
  img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = np.expand_dims(img, axis=0)

  img /= 255.0
  return img

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/predict', methods=['POST'])
def predict():
  if 'image' not in request.files:
    return jsonify({'error':400, 'message':   'No file part'}), 400

  file = request.files['image']

  if not allowed_file(file.filename):
    return jsonify({'error': 400, 'message': 'File is not an image'}), 400
  
  if file.filename == '':
    return jsonify({'error': 400, 'message':  'No selected file'}), 400

  if file:
    filename = secure_filename(file.filename)
    filepath = os.path.join('src/uploads', filename)
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
  
@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error':404, 'message':str(e)}), 404

@app.errorhandler(Exception)
def handle_server_error(e):
    print(f"An error occurred on the server: {e}")
    traceback.print_exc()
    return jsonify({'error': 500, 'message': 'An error occurred on the server'}), 500

if __name__ == '__main__':
    if os.getenv("ENVIRONMENT") == 'dev':
      app.run(debug=True)
    elif os.getenv("ENVIRONMENT") == 'prod':
      app.run()
    else:
      raise Exception("Environmet not valid : ",os.getenv("ENVIRONMENT"))