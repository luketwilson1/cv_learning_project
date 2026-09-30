import cv2


def draw_detections(image, detections):
    output = image.copy()

    for detection in detections:
        label = f'{detection["label"]} {detection["confidence"]:.2f}'
        x1, y1, x2, y2 = detection["box"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return output