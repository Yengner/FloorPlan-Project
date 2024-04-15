from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from model_setup import get_predictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

model, cfg = get_predictor("/Users/yb/Documents/GitHub/FloorPlan-Project/init_config.yaml", "/Users/yb/Documents/GitHub/FloorPlan-Project/model_final.pth")

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            output_image_path = process_image(filepath)
            return jsonify({'success': True, 'image_path': output_image_path})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    return jsonify({'success': False, 'error': 'Invalid file format'})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp'}

def process_image(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    outputs = model(image)
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_image = out.get_image()[:, :, ::-1]
    output_image_pil = Image.fromarray(output_image)
    output_image_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(filepath))
    output_image_pil.save(output_image_path)
    return output_image_path

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
