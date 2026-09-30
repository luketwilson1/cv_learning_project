from pathlib import Path
from typing import Sequence

from PIL import Image
from ultralytics import YOLO

from src.embeddings import find_similar_products
from src.ocr import ProductOCR, calculate_keyword_support, normalize_ocr_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "train-3"
    / "weights"
    / "best.pt"
)

QUERY_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "test"
    / "great_value_black_tea_100ct_8oz__IMG_1867.jpg"
)

CROP_DIRECTORY = PROJECT_ROOT / "data" / "predicted_crops"


def crop_detection(image: Image.Image, coordinates: Sequence[float], output_path: Path) -> Path:
    """Crop one detected object from an image.

    Args:
        image: Original image containing the detected object.
        coordinates: Bounding box represented as `(x1, y1, x2, y2)`.
        output_path: Destination for the cropped image.

    Returns:
        The path to the saved crop.

    Raises:
        ValueError: If the coordinates do not contain four values.
    """
    if len(coordinates) != 4:
        raise ValueError("Expected four bounding-box coordinates.")

    x1, y1, x2, y2 = coordinates

    # Keep the box inside the original image boundaries.
    box = (
        max(0, round(x1)),
        max(0, round(y1)),
        min(image.width, round(x2)),
        min(image.height, round(y2)),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    crop = image.crop(box)
    crop.save(output_path)

    return output_path


def main() -> None:
    """Detect products and identify each crop using catalog retrieval."""
    detector = YOLO(str(MODEL_PATH))
    product_ocr = ProductOCR()
    result = detector.predict(source=str(QUERY_IMAGE), conf=0.25, save=False)[0]

    if len(result.boxes) == 0:
        print("No products detected.")
        return

    with Image.open(QUERY_IMAGE) as source_image:
        source_image = source_image.convert("RGB")

        for detection_index, detection in enumerate(result.boxes):
            coordinates = detection.xyxy[0].cpu().tolist()
            detector_class_id = int(detection.cls[0].item())
            detector_confidence = float(detection.conf[0].item())
            detector_class_name = result.names[detector_class_id]
            crop_path = CROP_DIRECTORY / (f"{QUERY_IMAGE.stem}_{detection_index}.jpg")

            crop_detection(image=source_image, coordinates=coordinates, output_path=crop_path)

            matches = find_similar_products(query_path=crop_path, top_k=3)
            retrieved_class, retrieved_similarity, retrieved_source = matches[0]

            ocr_results = product_ocr.read_text(crop_path)
            detected_words = normalize_ocr_text(ocr_results)
            ocr_support, matching_words = calculate_keyword_support(retrieved_class, detected_words)

            models_agree = detector_class_name == retrieved_class
            final_identity = retrieved_class if models_agree else "review_required"

            print(f"\nDetection {detection_index}:")
            print(f"  YOLO: class={detector_class_name}, confidence={detector_confidence:.4f}")
            print(f"  Retrieval: class={retrieved_class}, similarity={retrieved_similarity:.4f}")
            print(f"  Retrieval source: {Path(retrieved_source).name}")
            print(f"  OCR support: {ocr_support:.2%}")
            print(f"  OCR matching words: {sorted(matching_words)}")
            print(f"  YOLO and retrieval agree: {models_agree}")
            print(f"  Final identity: {final_identity}")

            print("  Top retrieval matches:")

            for rank, (class_name, similarity, image_path) in enumerate(matches, start=1):
                print(f"    {rank}. {class_name}: similarity={similarity:.4f}, source={Path(image_path).name}")


if __name__ == "__main__":
    main()