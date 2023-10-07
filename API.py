import os
import numpy as np
import tensorflow as tf
import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)


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

@app.route("/")
def home():
    return "Welcome"


@app.route("/upload", methods=["POST"])
def upload_file():

    os.makedirs("uploads", exist_ok=True)

    files = request.files.to_dict(flat=False)

    left_file = files['left'][0]
    right_file = files['right'][0]

    # saving the uploaded files
    left_file.save("uploads/uploadL.jpeg")
    right_file.save("uploads/uploadR.jpeg")


    print('uploaded')

    if left_file and right_file:
        left_filename = r"uploads\uploadL.jpeg"
        right_filename = r"uploads\uploadR.jpeg"

        # Process and make predictions on left image
        image_left = cv2.imread(left_filename)
        image_left = preprocess(image_left)
        new_size = (299, 299)
        image_left = np.array([cv2.resize(image_left, new_size)])
        image_left = tf.convert_to_tensor(image_left)
        predL = model_inception_resnet.predict(image_left)

        # Process and make predictions on right image
        image_right = cv2.imread(right_filename)
        image_right = preprocess(image_right)
        new_size = (299, 299)
        image_right = np.array([cv2.resize(image_right, new_size)])
        image_right = tf.convert_to_tensor(image_right)
        predR = model_inception_resnet.predict(image_right)

    #     # Format predictions as needed
        predL = [round(x * 100, 4) for x in predL.tolist()[0]]
        predR = [round(x * 100, 4) for x in predR.tolist()[0]]

        return jsonify({
            "predL": predL,
            "predR": predR
        })
    else:
        return jsonify({"error": "Both left and right files must be provided."})

# if __name__=="__main__":
#     app.run(host='0.0.0.0',port=10000)
