import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, send_file, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import logging

app = Flask(__name__)
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

logging.basicConfig(level=logging.INFO)

def load_and_process_image(image_path):
    logging.info(f"Loading and processing image from {image_path}")
    img = Image.open(image_path).convert("RGB")
    img = img.resize((512, 512))
    img = np.array(img) / 255.0
    return tf.convert_to_tensor(img, dtype=tf.float32)[tf.newaxis, ...]

def tensor_to_image(tensor):
    logging.info("Converting tensor to image")
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

def perform_style_transfer(content_path, style_path):
    try:
        logging.info("Performing style transfer")
        content_image = load_and_process_image(content_path)
        style_image = load_and_process_image(style_path)
        outputs = hub_module(content_image, style_image)
        stylized_image = outputs[0]
        output_path = "output/generated_image.jpg"
        tensor_to_image(stylized_image).save(output_path)
        logging.info(f"Style transfer complete, saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error during style transfer: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/style_transfer', methods=['POST'])
def style_transfer():
    try:
        if 'content_image' not in request.files:
            return jsonify({"error": "Missing content_image"}), 400

        content_image = request.files['content_image']
        style_image = request.files.get('style_image')
        style_url = request.form.get('style_url')

        content_path = "uploads/content_image.jpg"
        os.makedirs("uploads", exist_ok=True)
        content_image.save(content_path)

        if style_image:
            style_path = "uploads/style_image.jpg"
            style_image.save(style_path)
        elif style_url:
            # Download image from URL
            import requests
            response = requests.get(style_url)
            style_path = "uploads/style_image.jpg"
            with open(style_path, 'wb') as f:
                f.write(response.content)
        else:
            return jsonify({"error": "Missing style image or URL"}), 400

        output_path = perform_style_transfer(content_path, style_path)
        if output_path:
            return send_file(output_path, mimetype='image/jpeg')
        else:
            return jsonify({"error": "Style transfer failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    os.makedirs("output", exist_ok=True)
    app.run(debug=True)