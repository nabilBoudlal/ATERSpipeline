import os

from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
)

from imagePreprocessing import preprocess_image


def image_crafting(image_path, output_directory):
    # Preprocess image
    image = preprocess_image(image_path)

    # Read image
    image = read_image(image)

    # Load models
    refine_net = load_refinenet_model(cuda=False)
    craft_net = load_craftnet_model(cuda=False)

    # Perform prediction
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=False,
        long_size=1280
    )

    # Create output folder specific to the original image
    image_filename = os.path.basename(image_path)
    image_name = os.path.splitext(image_filename)[0]
    if output_directory is not None:
        output_dir = f"{output_directory}/{image_name}/"
    else:
        output_dir = f"crafted/{image_name}/"
        os.makedirs(output_dir, exist_ok=True)

    # Export detected text regions
    exported_file_paths = export_detected_regions(
        image=image,
        regions=prediction_result["boxes"],
        output_dir=output_dir,
        rectify=True
    )

    return exported_file_paths
