from pathlib import Path
from typing import Sequence

from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "catalog_crops"

def load_class_names() -> dict[int, str]:
    """Load the product class mapping from the YOLO configuration.

    Returns:
        A mapping from numeric class IDs to product class names.

    Raises:
        FileNotFoundError: If the dataset configuration does not exist.
        KeyError: If the configuration does not contain a `names` mapping.
    """
    with open(DATA_ROOT / "data.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["names"]


def yolo_to_pixels(annotation: Sequence[float], image_width: int, image_height:int) -> tuple[int, tuple[int, int, int]]:
    """Convert a normalized YOLO annotation into pixel coordinates.

    Args:
        annotation: Class ID, normalized center-x, center-y, width, and
            height values.
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.

    Returns:
        The integer class ID and a bounding box represented as
        `(x1, y1, x2, y2)` pixel coordinates.

    Raises:
        ValueError: If the annotation does not contain exactly five values.
    """
    if len(annotation) != 5:
        raise ValueError(f"Expected five annotation values, received {len(annotation)}.")

    class_id, center_x, center_y, width, height = annotation

    x1 = int((center_x - width / 2) * image_width)
    y1 = int((center_y - height / 2) * image_height)
    x2 = int((center_x + width / 2) * image_width)
    y2 = int((center_y + height / 2) * image_height)

    return int(class_id), (x1, y1, x2, y2)


def build_catalog() -> None:
    """Create product crops from the labeled training and validation images.

    Empty annotation files are treated as hard-negative examples and skipped.
    Each product crop is stored in a directory named after its class.

    Raises:
        FileNotFoundError: If an expected image or dataset configuration is
            missing.
        ValueError: If an annotation has an invalid YOLO format.
    """
    class_names = load_class_names()

    for split in ("train", "val"):
        image_directory = DATA_ROOT / "images" / split
        label_directory = DATA_ROOT / "labels" / split

        for image_path in image_directory.glob("*.jpg"):
            label_path = label_directory / f"{image_path.stem}.txt"

            if not label_path.exists() or not label_path.read_text().strip():
                continue

            with Image.open(image_path) as image:
                image = image.convert("RGB")

                for index, line in enumerate(label_path.read_text().splitlines()):
                    annotation = [float(value) for value in line.split()]
                    class_id, box = yolo_to_pixels(annotation, image.width, image.height)

                    class_name = class_names[class_id]
                    output_directory = OUTPUT_ROOT / class_name
                    output_directory.mkdir(parents=True, exist_ok=True)

                    crop = image.crop(box)
                    output_path = output_directory / (f"{image_path.stem}_{index}.jpg")
                    crop.save(output_path)

                    print(f"Saved {output_path}")

if __name__ == "__main__":
    build_catalog()