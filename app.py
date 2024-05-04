from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from keras.models import load_model
import json
from keras.preprocessing import image

app = Flask(__name__)

class_indices = {0: "Apple___Apple_scab", 1: "Apple___Black_rot", 2: "Apple___Cedar_apple_rust", 3: "Apple___healthy", 4: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 5: "Corn_(maize)___Common_rust_", 6: "Corn_(maize)___healthy"}
indices=class_indices

model = load_model('model.h5')

model.make_predict_function()

# routes





# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name




@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route('/index.html')
def index():
    return render_template('index.html')
@app.route('/plant.html')
def plant():
    return render_template('plant.html')

@app.route('/disease.html')
def Disease():
    return render_template('disease.html')



@app.route("/submit", methods = ['GET', 'POST'])
def upload():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_image_class(model,img_path,class_indices)
		disease_info = get_disease_info(p)
		return render_template("index.html", prediction = p, img_path = img_path,disease_info=disease_info)


def get_disease_info(disease_name):
    parts = disease_name.split("___")
    after_delimiter = parts[-1]
    disease = after_delimiter.rstrip('_')
    disease=disease.replace("_"," ")
    if disease == "healthy":
        return "The leaf is healthy!"
    else:
        with open('C:/Users/93in/plant-disease/disease.json') as f:
            data = json.load(f)
            for item in data['data']:
                if item['Disease'] == disease:
                    return item
                













if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)