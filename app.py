from flask import Flask, render_template, request, session,url_for
import scann
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import time
from cv2 import resize,cvtColor,COLOR_BGR2RGB,imread
import pickle
 
app = Flask(__name__)
app.secret_key = "your_secret_key" 
app.config['OUTPUT_FOLDER'] = 'dataset2/' 
app.config['UPLOAD_DIRECTORY'] = 'input_dir/'
app.config['ALLOWED_EXTENSIONS'] = ['.png']

paths = pickle.load(open("paths_resnet.pkl", "rb"))

searcher = scann.scann_ops_pybind.load_searcher("indexing")
def get_extract_model():
    resnet50_model = ResNet50(weights="imagenet")
    extract_model = Model(inputs=resnet50_model.inputs, outputs=resnet50_model.get_layer("avg_pool").output)
    return extract_model


# Ham tien xu ly, chuyen doi hinh anh thanh tensor
def image_preprocess(img):
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def extract_features(model, img):
    
    img_tensor = image_preprocess(img)

    vector = model.predict(img_tensor)[0]

    vector = vector / np.linalg.norm(vector)
    return vector


@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['file']
    image_path = "image_upload/" + file.filename
    if (image_path == "image_upload/"):
        return render_template("index.html")
    file.save(image_path)
    session['image_path'] = image_path
    return render_template("index.html")

@app.route('/get_Data', methods=['POST'])
def get_Data():
    start = time.time()
    data = request.get_json()

    # Access the variables
    x = data.get('x')
    y = data.get('y')
    width = data.get('width')
    height = data.get('height')
    image_path = session.get('image_path', None)
    fileName = data.get('fileName')
    image_path = "image_upload/" + fileName
    print(x, height, fileName)
    img = imread(image_path)
    # Crop image 
    crop_img = img[int(y):int(y+height), int(x):int(x+width)]

    resnet_model = get_extract_model()

    vector_query = extract_features(resnet_model, crop_img)

    neighbors, distances = searcher.search(vector_query)


    url = [paths[id] for id in neighbors]

    zipped_data = zip(url, distances)
    end = time.time()
    execute_time = round(end - start,2)
    return render_template("index.html", data = zipped_data, time = execute_time)

@app.route("/")
def home():
    return render_template("index.html")
    
if __name__ == '__main__':
    app.run(debug=True)
