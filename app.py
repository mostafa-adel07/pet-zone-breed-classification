import pickle
from flask import Flask, request
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

app = Flask(__name__)

IMG_SIZE = 224  # Dimensions of the image (224, 224, 3)

BREEDS = ['Abyssinian', 'Egyptian Mau', 'Sphynx - Hairless Cat', 'Himalayan',
          'Persian', 'Siamese', 'Norwegian Forest Cat']  # cat breeds

UNIQUE_LIST = "model/unique_breeds.pkl"  # dog breeds
CAT_MODEL_PATH = "model/catModel2.h5"  # load cat model
DOG_MODEL_PATH = "model/dogModel.h5"  # load dog model
#  load dog breeds from pickle file into list
with open(UNIQUE_LIST, 'rb') as f:
    unique_list = pickle.load(f)


def process_image(image_path):
    """
    Taking path of an image and turn it into tensor
    """

    # Read an image file
    image = tf.io.read_file(image_path)

    # Convert the image into tensor
    image = tf.image.decode_jpeg(image, channels=3)

    # Convert color channel values from 0-255 to 0-1
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image shape (224,224)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image


@app.route('/', methods=['POST'])
def catBreedPredict():
    image = request.files['image']
    image_path = image.filename
    image.save(image_path)
    X = process_image(image_path)
    X2 = tf.expand_dims(X, axis=0)
    model = tf.keras.models.load_model(CAT_MODEL_PATH, custom_objects={"KerasLayer": hub.KerasLayer}, compile=False)
    model2 = tf.keras.models.load_model(DOG_MODEL_PATH, custom_objects={"KerasLayer": hub.KerasLayer}, compile=False)
    y_predict_cat = model.predict(X2)
    y_predict_dog = model2.predict(X2)
    predict = compare(y_predict_cat, y_predict_dog)
    return predict


def compare(catBreed, dogBreed):
    dog = np.max(dogBreed)
    cat = np.max(catBreed)
    print(dog, unique_list[np.argmax(dogBreed)])
    print(cat, BREEDS[np.argmax(catBreed)])
    
    if dog < 0.2 and cat < 0.2:
          return "Cant detect this pet"
    elif dog > cat:
        return unique_list[np.argmax(dogBreed)] + ' dog'
    elif cat > dog:
        return BREEDS[np.argmax(catBreed)] + ' Cat'
    else:
        return "Can not detect this breed"


if __name__ == '__main__':
    app.run(port=3000, debug=True, host = '0.0.0.0')
