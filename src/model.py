from pathlib import Path

from ultralytics import YOLO

def load_model(model_name="yolo11n.pt"):
    """
    Load a pretrained YOLO object detection model.

    Args:
        model_name: Name or path of the YOLO model weights.

    Returns:
        A loaded YOLO model.
    """
    return YOLO(model_name)


def train_model(data_path, model_name="yolo11n.pt", epochs=20, image_size=640):
    """
    Fine-tune a pretrained YOLO model on a custom dataset.

    Args:
        data_path: Path to the YOLO dataset YAML configuration file.
        model_name: Pretrained YOLO model weights to fine-tune.
        epochs: Number of complete passes through the training dataset.
        image_size: Input image size used during training.

    Returns:
        Training results produced by YOLO.
    """

    model = YOLO(model_name)

    project_root = Path(__file__).resolve().parents[1]
    return model.train(
        data=data_path,
        epochs=epochs,
        imgsz=image_size,
        project=str(project_root / "runs" / "detect"),
        name="train",
        exist_ok=False,
    )
