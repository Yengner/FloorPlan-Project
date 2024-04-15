from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import cv2
from detectron2dadada.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from model_setup import get_predictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Load model using your utility function and receive both predictor and cfg
model, cfg = get_predictor("/Users/yb/Desktop/FloorPlan-Project/init_config-3k_iter.yaml", "/Users/yb/Desktop/FloorPlan-Project/model_final.pth")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                output_image_path = process_image(filepath)
                return render_template('result.html', image_path=output_image_path)
            except Exception as e:
                return str(e)
    return render_template('upload.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp'}

def process_image(filepath):
    # Load the image with OpenCV
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert color from BGR to RGB

    # Process image through the model
    outputs = model(image)

    # Visualize the results using Detectron2 Visualizer
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_image = out.get_image()[:, :, ::-1]  # Convert color from RGB to BGR

    # Convert numpy array to PIL Image
    output_image_pil = Image.fromarray(output_image)

    # Save the visualized image
    output_image_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(filepath))
    output_image_pil.save(output_image_path)  # Save the image using PIL
    
    return output_image_path

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
