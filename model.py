import random
random.seed(0)

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.random.set_seed(0)
from tensorflow import keras
from keras import preprocessing
import os
import json
from zipfile import ZipFile
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

#kaggle_credentails = json.load(open("C:\Users\93in\plant-disease\kaggle.json"))
# Dataset Path
base_dir = 'C:/Users/93in/plant-disease/plantvillage dataset/plantvillage dataset/color'
# Image Parameters
img_size = 224
batch_size = 32
# Image Data Generators
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Use 20% of data for validation
)
# Train Generator
train_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)
# Validation Generator
validation_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)
# Model Definition
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))


model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(train_generator.num_classes, activation='softmax'))
# Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,       # Number of steps per epoch
    epochs=5,                                                    # Number of epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size  # Validation steps
)
model.save('model.h5')

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
# Create a mapping from class indices to class names
class_indices = {v: k for k, v in train_generator.class_indices.items()}

# Example Usage
#image_path = '/content/drive/MyDrive/disease detection/test/corn.jpeg'
#image_path = '/content/drive/MyDrive/disease detection/test/00a20f6f-e8bd-4453-9e25-36ea70feb626___RS_GLSp 4655.JPG'
#image_path = '/content/drive/MyDrive/disease detection/test/test_apple_black_rot.jpeg'
image_path='C:/Users/93in/plant-disease/test_apple_black_rot.jpeg'

predicted_class_name = predict_image_class(model, image_path, class_indices)

# Output the result
print("Predicted Class Name:", predicted_class_name)
parts = predicted_class_name.split("___")
after_delimiter = parts[-1]
disease = after_delimiter.rstrip('_')
disease=disease.replace("_"," ")
print(disease)

import json
# Opening JSON file
f = open('C:/Users/93in/plant-disease/disease.json')

# returns JSON object as a dictionary
data = json.load(f)
c=0
if disease=='healthy':
  print("The leaf is healthy!")
else:
  for i in data['data']:
    if data['data'][c]['Disease']==disease:
      for j in data['data'][c].keys():
        print(j,":",data['data'][c][j])
    c+=1

img = mpimg.imread(image_path)

print(img.shape)
# Display the image
plt.imshow(img)
plt.axis('off')  # Turn off axis numbers
plt.show()
# Closing file
f.close()