import glob
import os

import pytesseract
from PIL import Image

from imageClassification import image_classification
from textPostprocessing import correct_line
from trocr import trocr_executor

TESSERACT_PATH = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def tesseract(img):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    text = pytesseract.image_to_string(Image.open(img), config=r'--psm 7')
    return text


def recognize_text(image_path):
    classification_result = image_classification(image_path)
    if classification_result == 'aMano':
        return trocr_executor(image_path)
    else:
        return tesseract(image_path)


def process_text_in_directory(base_folder, recognize_text_function):
    crafted_folder = base_folder
    subdirectories = [name for name in os.listdir(crafted_folder) if os.path.isdir(os.path.join(crafted_folder, name))]

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(crafted_folder, subdirectory)
        image_crops_folder = os.path.join(subdirectory_path, "image_crops")
        print(image_crops_folder)

        image_paths = glob.glob(os.path.join(image_crops_folder, '*.png'))
        image_paths = sorted(image_paths)

        output_file = f'results/{os.path.basename(subdirectory_path)}.txt'

        results = ""

        for image_path in image_paths:
            img = Image.open(image_path)
            if recognize_text_function == "handwritten":
                results += recognize_handwritten_text(image_path) + " "
            elif recognize_text_function == "machine":
                results += recognize_machine_text(image_path) + " "
            elif recognize_text_function == "":
                results += recognize_text(image_path) + " "

        with open(output_file, 'w') as file:
            file.write(correct_line(results))


def text_recognition_auto():
    process_text_in_directory("crafted\\", "")


def text_recognition_machine():
    process_text_in_directory("crafted\\", "machine")


def text_recognition_handwritten():
    process_text_in_directory("crafted\\", "handwritten")


def recognize_handwritten_text(image_path):
    return trocr_executor(image_path)


def recognize_machine_text(image_path):
    return tesseract(image_path)
