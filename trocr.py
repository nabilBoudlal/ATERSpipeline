from PIL import Image
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel

MODEL_PATH = 'C:\\Users\\Nabil\\PycharmProjects\\pipelineTesi\\trocr_output'


def trocr_executor(img_path):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    image = Image.open(img_path).convert("RGB")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    result = "" + generated_text

    return result
