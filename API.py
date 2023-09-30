from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import pickle
from PIL import Image
import numpy as np
from torchvision import transforms
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import cv2


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://eyemote-ai.web.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(image):
    image = cv2.resize(image, (500, 500))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 500/10), -4, 128)
    return image


def make_model3():
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(299, 299, 3), pooling=None, classes=5)
    inputs = keras.Input(shape=(299, 299, 3))
    x = base_model(inputs, training=False)    
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(5, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model_inception_resnet = make_model3()

model_inception_resnet.load_weights('model_inception_resnet.h5') 

@app.get("/")
def home():
    return "Welcome"

@app.post("/upload")
async def upload_file(left: UploadFile,right: UploadFile):
    print('uploaded')
    os.makedirs("uploads", exist_ok=True)

    # Save the uploaded file
    with open(os.path.join("uploads", left.filename), 'wb') as f:
        f.write(left.file.read())
    image = cv2.imread('uploads/' + left.filename)
    image = preprocess(image)
    # cv2.imwrite('./OG2.jpeg', image)
    new_size = (299, 299)
    image = np.array([cv2.resize(image, new_size)])
    image = tf.convert_to_tensor(image)
    predL = model_inception_resnet.predict(image)

    with open(os.path.join("uploads", right.filename), 'wb') as f:
        f.write(right.file.read())
    image = cv2.imread('uploads/' + right.filename)
    image = preprocess(image)
    # cv2.imwrite('./OG2.jpeg', image)
    new_size = (299, 299)
    image = np.array([cv2.resize(image, new_size)])
    image = tf.convert_to_tensor(image)
    predR = model_inception_resnet.predict(image)    
    predL = [round(x*100,4) for x in predL.tolist()[0]]
    predR = [round(x*100,4) for x in predR.tolist()[0]]

    return {
        "predL": predL,
        "predR": predR
    }