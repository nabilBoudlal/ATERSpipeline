import numpy as np
import tensorflow as tf

PRETRAINED_MODEL_PATH = "C:\\Users\\Nabil\\PycharmProjects\\pipelineTesi\\model"


def image_classification(img_path):
    result = " "
    class_names = ['aMano', 'macchina']
    r_model = tf.keras.saving.load_model(PRETRAINED_MODEL_PATH, custom_objects=None, compile=True, safe_mode=True)

    img = tf.keras.utils.load_img(img_path, target_size=(80, 611))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = r_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    result = class_names[np.argmax(score)]
    return result
