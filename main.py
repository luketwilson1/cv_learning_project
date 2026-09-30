import cv2

from src.model import load_model
from src.visualization import draw_detections


IMAGE_PATH = "data/raw/great_value_all_purpose_flour_2lb__IMG_1799.jpg"
OUTPUT_PATH = "data/processed/yolo_detections.jpg"


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    model = load_model()

    results = model(IMAGE_PATH)
    result = results[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append(
            {
                "label": result.names[class_id],
                "confidence": confidence,
                "box": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ),
            }
        )

    for detection in detections:
        print(
            detection["label"],
            f'{detection["confidence"]:.2f}',
            detection["box"],
        )

    annotated = draw_detections(image, detections)

    cv2.imwrite(OUTPUT_PATH, annotated)


if __name__ == "__main__":
    main()