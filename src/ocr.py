from pathlib import Path

import easyocr
import torch
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE_PATH = PROJECT_ROOT / "data" / "query_crops" / "great_value_black_tea_100ct_8oz__IMG_1867.jpg"


class ProductOCR:
    """Extract printed text from product images using EasyOCR."""

    def __init__(self) -> None:
        """Initialize the OCR reader."""
        use_gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(["en"], gpu=use_gpu)
    

    def read_text(self, image_path: Path, min_confidence: float = 0.50) -> list[tuple[str, float]]:
        """Extract text from an image.

        Args:
            image_path: Path to the product image.

        Returns:
            Text strings detected in the image.
        """
        results = self.reader.readtext(str(image_path), detail=1)
        filtered_results = [(text, float(confidence)) for _, text, confidence in results if confidence >= min_confidence]
        return filtered_results


def normalize_ocr_text(ocr_results: list[tuple[str, float]]) -> set[str]:
    """Convert OCR results into normalized individual words.

    Args:
        ocr_results: Detected text paired with OCR confidence.

    Returns:
        Unique lowercase alphanumeric words.
    """
    combined_text = " ".join(text for text, _ in ocr_results).lower()
    return set(re.findall(r"[a-z0-9]+", combined_text))


def calculate_keyword_support(class_name: str, detected_words: set[str]) -> tuple[float, set[str]]:
    """Measure how strongly OCR words support a product class.

    Args:
        class_name: Product class written with underscore separators.
        detected_words: Normalized words found by OCR.

    Returns:
        The proportion of class words found and the matching words.
    """
    class_words = set(class_name.lower().split("_"))
    matching_words = class_words.intersection(detected_words)
    support = len(matching_words) / len(class_words)
    return support, matching_words


def main() -> None:
    """Run OCR on a test product crop."""
    product_ocr = ProductOCR()
    detected_text = product_ocr.read_text(TEST_IMAGE_PATH)

    print(f"Image: {TEST_IMAGE_PATH}")
    print("Detected text:")

    for text, confidence in detected_text:
        print(f"- {text}: confidence={confidence:.4f}")

    detected_words = normalize_ocr_text(detected_text)
    class_name = "black_tea"
    support, matching_words = calculate_keyword_support(class_name, detected_words)
    
    print(f"\nNormalized words: {sorted(detected_words)}")
    print(f"OCR support for {class_name}: {support:.2%}")
    print(f"Matching words: {sorted(matching_words)}")

if __name__ == "__main__":
    main()