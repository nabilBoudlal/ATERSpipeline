import os

import cv2
import numpy as np


def preprocess_image(image_path):
    return get_image_preprocessed(image_path, 1)


def apply_threshold(img, argument):
    switcher = {
        1: cv2.bilateralFilter(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 9, 75, 75)
    }
    return switcher.get(argument, "Invalid method")


def get_image_preprocessed(img_path, method):
    # Read image using opencv
    img = cv2.imread(img_path)

    # Rescale the image, if needed.
    img = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # Apply threshold to get image with only black and white
    img = apply_threshold(img, method)

    return img


def onlyPreprocessImages(image_path, output_dir):
    image_filename = os.path.basename(image_path)
    image_name = os.path.splitext(image_filename)[0]
    output_dir = f"{output_dir}/preprocessed_only_images/{image_name}/"
    os.makedirs(output_dir, exist_ok=True)

    preprocessed_image = preprocess_image(image_path)

    # Salva l'immagine elaborata nella cartella di output
    output_path = os.path.join(output_dir, f"{image_name}_preprocessed.png")
    cv2.imwrite(output_path, preprocessed_image)
