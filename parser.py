import argparse


def get_args():
    parser = argparse.ArgumentParser(
        prog='OCR text extraction pipeline',
        description='Unicam Thesis',
        epilog=''
    )
    parser.add_argument('input_dir', metavar='input_dir', type=str)
    parser.add_argument('output_dir', metavar='output_dir', type=str)
    parser.add_argument("--method", choices=["auto", "handwritten", "machine"], default="auto",
                        help="Text recognition method")
    parser.add_argument("--options", choices=["preprocess", "craft"], default="",
                        help="Only preprocess image or craft")
    args = parser.parse_args()

    return args
