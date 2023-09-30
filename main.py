import os
import warnings

from imageCrafting import image_crafting
from imagePreprocessing import onlyPreprocessImages
from ocr import text_recognition_handwritten, text_recognition_machine, text_recognition_auto
from parser import get_args


def start_engine_OCR(input_directory, output_directory, method_to_use, options_selected):
    warnings.filterwarnings("ignore")
    for filename in os.listdir(input_directory):
        if options_selected == "preprocess":
            input_image_path = os.path.join(input_directory, filename)
            onlyPreprocessImages(input_image_path, output_directory)
        elif options_selected == "craft":
            image_crafting(os.path.join(input_directory, filename), output_directory)
        else:
            # crafting
            file = image_crafting(os.path.join(input_directory, filename), None)
            # ocr
            if method_to_use == "handwritten":
                text_recognition_handwritten()
            elif method_to_use == "machine":
                text_recognition_machine()
            else:
                # Il metodo  "auto", quindi lascia che il sistema decida
                text_recognition_auto()


if __name__ == '__main__':
    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    method = args.method
    options = args.options

    start_engine_OCR(input_dir, output_dir, method, options)

    print("DONE")
