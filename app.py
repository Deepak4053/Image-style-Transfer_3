import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import render_template_string
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
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Neural Style Transfer</title>
      <style>
        body { font-family: Arial, sans-serif; background: #f4f4f9; margin: 0; padding: 0; }
        .container { width: 90%; max-width: 800px; margin: 2rem auto; background: #fff; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        h1 { text-align: center; }
        form { display: flex; flex-direction: column; gap: 1rem; }
        label { font-weight: bold; }
        input[type="file"], input[type="text"] { padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px; }
        button { background-color: #007BFF; color: white; border: none; padding: 0.75rem; font-size: 1rem; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .result { margin-top: 2rem; text-align: center; }
        .result img { max-width: 100%; border: 2px solid #007BFF; border-radius: 8px; }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Neural Style Transfer</h1>
        <form id="uploadForm" action="/style_transfer" method="post" enctype="multipart/form-data">
          <div>
            <label for="content_image">Content Image:</label>
            <input type="file" name="content_image" accept="image/*" required>
          </div>
          <div>
            <label for="style_image">Style Image (Upload):</label>
            <input type="file" name="style_image" accept="image/*">
          </div>
          <div>
            <label for="style_url">OR Style Image URL:</label>
            <input type="text" name="style_url" placeholder="Enter style image URL">
          </div>
          <button type="submit">Generate Stylized Image</button>
        </form>
        <div class="result" id="result">
          <!-- Generated image and download link will appear here -->
        </div>
      </div>
      <script>
        document.getElementById("uploadForm").addEventListener("submit", async function(e) {
          e.preventDefault();
          const formData = new FormData(this);
          const styleFile = document.querySelector('input[name="style_image"]').files[0];
          const styleUrl = document.querySelector('input[name="style_url"]').value.trim();
          if (!styleFile && styleUrl !== "") {
            formData.append("style_url", styleUrl);
          }
          try {
            const response = await fetch("/style_transfer", {
              method: "POST",
              body: formData
            });
            if (!response.ok) {
              throw new Error("Style transfer failed");
            }
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            document.getElementById("result").innerHTML = `
              <h2>Generated Image:</h2>
              <img src="${imageUrl}" alt="Stylized Image" /><br>
              <a href="${imageUrl}" download="generated_image.jpg">
                <button>Download Generated Image</button>
              </a>
            `;
          } catch (error) {
            document.getElementById("result").innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
          }
        });
      </script>
    </body>
    </html>
    """
    return render_template_string(html)


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
