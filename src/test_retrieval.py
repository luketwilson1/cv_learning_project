from pathlib import Path

from PIL import Image

from src.build_catalog import yolo_to_pixels
from src.embeddings import find_similar_products


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "test"
    / "great_value_black_tea_100ct_8oz__IMG_1867.jpg"
)

TEST_LABEL = (
    PROJECT_ROOT
    / "data"
    / "labels"
    / "test"
    / "great_value_black_tea_100ct_8oz__IMG_1867.txt"
)

QUERY_CROP = (
    PROJECT_ROOT
    / "data"
    / "query_crops"
    / "great_value_black_tea_100ct_8oz__IMG_1867.jpg"
)


def create_query_crop(
    image_path: Path,
    label_path: Path,
    output_path: Path,
) -> Path:
    """Create a query crop using a ground-truth YOLO annotation.

    Args:
        image_path: Path to the complete query image.
        label_path: Path to its YOLO annotation.
        output_path: Destination for the cropped product image.

    Returns:
        The path to the saved query crop.
    """
    annotation = [
        float(value)
        for value in label_path.read_text().strip().split()
    ]

    with Image.open(image_path) as image:
        image = image.convert("RGB")

        _, box = yolo_to_pixels(
            annotation,
            image.width,
            image.height,
        )

        crop = image.crop(box)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)

    return output_path


def main() -> None:
    """Crop the unseen tea image and search the product catalog."""
    query_path = create_query_crop(
        image_path=TEST_IMAGE,
        label_path=TEST_LABEL,
        output_path=QUERY_CROP,
    )

    matches = find_similar_products(
        query_path=query_path,
        top_k=5,
    )

    print(f"Query: {query_path}")

    for rank, (class_name, similarity, image_path) in enumerate(
        matches,
        start=1,
    ):
        print(
            f"{rank}. {class_name}: "
            f"similarity={similarity:.4f}, "
            f"source={Path(image_path).name}"
        )


if __name__ == "__main__":
    main()