from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image
from ultralytics import YOLO

from src.detect_and_retrieve import crop_detection
from src.embeddings import CatalogRecognizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train-3" / "weights" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "videos" / "tea_pass.MOV"
CROP_DIRECTORY = PROJECT_ROOT / "data" / "video_crops"

SAMPLE_INTERVAL = 15
MIN_SIMILARITY = 0.70


def summarize_tracks(track_evidence: dict[int, dict[str, list[float]]]) -> None:
    """Print retrieval evidence and the winning identity for each track.

    Args:
        track_evidence: Similarity scores grouped by track ID and retrieved product class.
    """
    print("\nTrack summary:")

    for track_id, class_scores in sorted(track_evidence.items()):
        ranked_classes = sorted(class_scores.items(), key=lambda item: (len(item[1]), mean(item[1])), reverse=True)
        winning_class, winning_scores = ranked_classes[0]

        print(f"\nTrack {track_id}: winner={winning_class}, votes={len(winning_scores)}, average_similarity={mean(winning_scores):.4f}")
        print("  Vote distribution:")

        for class_name, scores in ranked_classes:
            print(f"{class_name}: votes={len(scores)}, average_similarity={mean(scores):.4f}, highest_similarity={max(scores):.4f}")


def main() -> None:
    """Track video products and identify sampled crops using retrieval."""
    detector = YOLO(str(MODEL_PATH))
    recognizer = CatalogRecognizer()
    track_evidence = defaultdict(lambda: defaultdict(list))

    results = detector.track(
        source=str(VIDEO_PATH), 
        tracker="bytetrack.yaml", conf=0.25, stream=True, 
        persist=True, verbose=False, 
        save=True, 
        project=str(PROJECT_ROOT / "runs" / "detect"), name="tea_pass_retrieval"
        )

    for frame_index, result in enumerate(results):
        if frame_index % SAMPLE_INTERVAL != 0:
            continue

        if result.boxes.id is None:
            continue

        # Ultralytics stores OpenCV frames in BGR order.
        rgb_frame = result.orig_img[:, :, ::-1].copy()
        frame_image = Image.fromarray(rgb_frame)

        for detection_index, detection in enumerate(result.boxes):
            if detection.id is None:
                continue

            track_id = int(detection.id[0].item())
            coordinates = detection.xyxy[0].cpu().tolist()
            crop_path = CROP_DIRECTORY / f"track_{track_id}_frame_{frame_index}_detection_{detection_index}.jpg"

            crop_detection(image=frame_image, coordinates=coordinates, output_path=crop_path)
            matches = recognizer.search(query_path=crop_path, top_k=1)

            class_name, similarity, _ = matches[0]
            retrieved_identity = class_name if similarity >= MIN_SIMILARITY else "unknown"

            track_evidence[track_id][retrieved_identity].append(similarity)

            print(f"Frame {frame_index}, track {track_id}: {retrieved_identity}, similarity={similarity:.4f}")

    summarize_tracks(track_evidence)


if __name__ == "__main__":
    main()